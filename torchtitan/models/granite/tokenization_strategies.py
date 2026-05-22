import json
import logging
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable

from filelock import FileLock
from typing import Any

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

logger = logging.getLogger(__name__)

_VALID_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})

# Think tags in content indicate data cleaning failures; start_end are rarer and typically from code snippets.
REJECT_TOKEN_GROUPS: dict[str, tuple[str, ...]] = {
    "think": ("<think>", "</think>"),
    "start_end": ("<|im_start|>", "<|im_end|>"),
}


def _validate_messages(messages: list[dict]) -> None:
    """Validate message list structure."""
    if not messages:
        raise ValueError("messages must not be empty")
    invalid_roles = {m["role"] for m in messages} - _VALID_MESSAGE_ROLES
    if invalid_roles:
        raise ValueError(f"Unknown role(s): {invalid_roles!r}")
    if messages[0]["role"] not in ("system", "user"):
        raise ValueError(
            f"First message must be 'system' or 'user', got '{messages[0]['role']}'"
        )
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("Messages must contain at least one assistant turn")
    system_positions = [i for i, m in enumerate(messages) if m["role"] == "system"]
    if len(system_positions) > 1 or (system_positions and system_positions[0] != 0):
        raise ValueError("system message must be the first message if present")
    for i, msg in enumerate(messages):
        for field in ("content", "reasoning_content"):
            text = msg.get(field)
            if text is None:
                continue
            if not isinstance(text, str):
                raise ValueError(
                    f"Non-string {field} in {msg['role']} message at index {i} "
                    f"(type {type(text).__name__})"
                )


def _fix_empty_thinking(text: str) -> str:
    """Correct no-reasoning rendering for truncate_history_thinking=False.

    When thinking is preserved in context (not truncated), historical no-reasoning
    turns should appear as they were generated: the model received <think>\\n from
    the generation prompt and produced </think>\\n, giving <think>\\n</think>\\n in
    context. The template normalizes these to <think></think> (adjacent, no newlines).
    This fixup restores the inference-time form.

    Not needed for truncate_history_thinking=True: in that mode the orchestrator
    strips thinking using the same template, so <think></think> IS what the model
    sees at inference.
    """
    return text.replace("<think></think>", "<think>\n</think>\n")


def _char_to_token_idx(token_starts: list[int], char_pos: int) -> int:
    """Map a character position to a token index in the BOS-prepended sequence.

    The offset table from the Rust encoder excludes BOS (which the wrapper
    prepends). So bisect gives a raw index; +1 translates to full_tokens space.
    """
    return bisect_left(token_starts, char_pos) + 1


_SPECIAL_TOKEN_PATTERNS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")


def _warn_special_tokens_in_content(messages: list[dict]) -> None:
    """Log a warning if message content contains protocol-level delimiters."""
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not isinstance(content, str):
            logger.warning(
                "Non-string content in %s message at index %d (type %s); "
                "cannot check for special tokens",
                msg["role"], i, type(content).__name__,
            )
            continue
        reasoning = msg.get("reasoning_content", "")
        for pat in _SPECIAL_TOKEN_PATTERNS:
            pos = content.find(pat)
            if pos != -1:
                logger.warning(
                    "Special token %r in content of %s message at index %d: ...%s...",
                    pat, msg["role"], i, content[max(0, pos - 20) : pos + len(pat) + 20],
                )
            if isinstance(reasoning, str):
                pos = reasoning.find(pat)
                if pos != -1:
                    logger.warning(
                        "Special token %r in reasoning_content of %s message at index %d: ...%s...",
                        pat, msg["role"], i, reasoning[max(0, pos - 20) : pos + len(pat) + 20],
                    )


def _prepare_full_sequence(
    messages: list[dict],
    tokenizer: HuggingFaceTokenizer,
    chat_template_kwargs: dict[str, Any],
    *,
    fixup_fn: Callable[[str], str] | None = None,
) -> tuple[list[dict], str, list[int], list[int], list[int]]:
    """Shared preamble for all tokenization strategies.

    Validates message structure, renders the conversation to text via the chat
    template, encodes to token IDs (with BOS prepended and EOS appended if the
    template doesn't produce one), and builds the initial label mask (all IGNORE).

    Returns:
        effective: messages truncated to last assistant turn (trailing user/tool dropped)
        full_text: rendered chat template string (with fixup applied if provided)
        full_tokens: token IDs including BOS and trailing EOS
        input_ids: full_tokens[:-1] (model input)
        label_ids: [IGNORE_INDEX] * len(input_ids) (caller fills in unmasked regions)
    """
    _validate_messages(messages)
    _warn_special_tokens_in_content(messages)
    last_asst_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    effective = messages[: last_asst_idx + 1]
    full_text = tokenizer.apply_chat_template(
        effective, **chat_template_kwargs
    ).rstrip("\n")
    if fixup_fn:
        full_text = fixup_fn(full_text)
    full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
    if full_tokens[-1] != tokenizer.eos_id:
        full_tokens.append(tokenizer.eos_id)
    input_ids = full_tokens[:-1]
    label_ids = [IGNORE_INDEX] * len(input_ids)
    return effective, full_text, full_tokens, input_ids, label_ids


def _tokenize_last_turn(
    messages: list[dict],
    tokenizer: HuggingFaceTokenizer,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, list[int] | int]:
    """Tokenize labeling only the final assistant turn (per-turn encode approach)."""
    effective, _, full_tokens, input_ids, label_ids = _prepare_full_sequence(
        messages, tokenizer, chat_template_kwargs,
    )
    prefix_text = tokenizer.apply_chat_template(
        effective[:-1], add_generation_prompt=True, **chat_template_kwargs,
    )
    prefix_tokens = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
    start = len(prefix_tokens) - 1
    label_ids[start:] = full_tokens[start + 1:]
    return {"input_ids": input_ids, "labels": label_ids, "n_tokens": len(input_ids)}


def _tokenize_all_turns(
    messages: list[dict],
    tokenizer: HuggingFaceTokenizer,
    chat_template_kwargs: dict[str, Any],
    *,
    fixup_fn: Callable[[str], str] | None = None,
) -> dict[str, list[int] | int]:
    """Tokenize a conversation labeling every assistant turn.

    Every turn trains the model to predict <|im_end|> (= EOS) after its content.
    For intermediate turns, loss ends there (the <|im_end|> position is masked
    since predicting the next-turn header is irrelevant). For the last turn,
    <|im_end|> is the final prediction target — it's outside input_ids.

    Uses a single encode + character-offset bisect for all boundary lookups.
    """
    effective, full_text, full_tokens, input_ids, label_ids = _prepare_full_sequence(
        messages, tokenizer, chat_template_kwargs, fixup_fn=fixup_fn,
    )
    last_asst_idx = len(effective) - 1

    full_encoding = tokenizer.tokenizer.encode(full_text)
    token_starts = [o[0] for o in full_encoding.offsets]

    for i, msg in enumerate(effective):
        if msg["role"] != "assistant":
            continue

        prefix_render = tokenizer.apply_chat_template(
            effective[:i], **chat_template_kwargs
        ).rstrip("\n")
        if fixup_fn:
            prefix_render = fixup_fn(prefix_render)
        prefix_end = len(prefix_render)

        think_pos = full_text.index("<think>", prefix_end)
        start = _char_to_token_idx(token_starts, think_pos + len("<think>"))

        if i == last_asst_idx:
            label_ids[start:] = full_tokens[start + 1:]
        else:
            end_text = tokenizer.apply_chat_template(
                effective[:i + 1], **chat_template_kwargs
            ).rstrip("\n")
            if fixup_fn:
                end_text = fixup_fn(end_text)
            # Result equals len(encode(end_text)): -1 for length→index, -1 to skip <|im_end|>
            label_end = _char_to_token_idx(token_starts, len(end_text)) - 2
            end = min(label_end + 1, len(input_ids))
            label_ids[start:end] = full_tokens[start + 1 : end + 1]

    return {"input_ids": input_ids, "labels": label_ids, "n_tokens": len(input_ids)}


def _tokenize_all_turns_reference(
    messages: list[dict],
    tokenizer: HuggingFaceTokenizer,
    chat_template_kwargs: dict[str, Any],
    *,
    fixup_fn: Callable[[str], str] | None = None,
) -> dict[str, list[int] | int]:
    """Reference implementation using per-turn encode calls.

    Kept for correctness testing against the offset-based _tokenize_all_turns.
    """
    effective, full_text, full_tokens, input_ids, label_ids = _prepare_full_sequence(
        messages, tokenizer, chat_template_kwargs, fixup_fn=fixup_fn,
    )
    last_asst_idx = len(effective) - 1

    for i, msg in enumerate(effective):
        if msg["role"] != "assistant":
            continue

        prefix_text = tokenizer.apply_chat_template(
            effective[:i],
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if fixup_fn:
            prefix_text = fixup_fn(prefix_text)
        prefix_tokens = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        start = len(prefix_tokens) - 1

        if i == last_asst_idx:
            label_ids[start:] = full_tokens[start + 1:]
        else:
            up_to_text = tokenizer.apply_chat_template(
                effective[:i + 1], **chat_template_kwargs
            ).rstrip("\n")
            if fixup_fn:
                up_to_text = fixup_fn(up_to_text)
            up_to_tokens = tokenizer.encode(
                up_to_text, add_bos=True, add_eos=False
            )
            end = len(up_to_tokens) - 2
            end = min(end + 1, len(input_ids))
            label_ids[start:end] = full_tokens[start + 1 : end + 1]

    return {"input_ids": input_ids, "labels": label_ids, "n_tokens": len(input_ids)}


def _append_failures(path: str, failures: list[dict]) -> None:
    """Append failure records to a JSONL file, safe for concurrent writers."""
    with FileLock(path + ".lock"), open(path, "a") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")


class TokenizationStrategy(ABC):
    def __init__(
        self,
        tokenizer_path: str,
        *,
        failures_path: str | None = None,
        forbidden_content_tokens: tuple[str, ...] = (),
    ) -> None:
        self._tokenizer_path = tokenizer_path
        self._tokenizer: HuggingFaceTokenizer | None = None
        self.failures_path = failures_path
        self.forbidden_content_tokens = forbidden_content_tokens

    _REQUIRED_SPECIAL_TOKENS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        if self._tokenizer is None:
            tok = HuggingFaceTokenizer(tokenizer_path=self._tokenizer_path)
            if tok.eos_id is None:
                raise ValueError("Tokenizer must have a valid eos_id")
            for token in self._REQUIRED_SPECIAL_TOKENS:
                if tok.tokenizer.token_to_id(token) is None:
                    raise ValueError(
                        f"{token} must be a registered special token; "
                        f"offset-based boundary lookup requires atomic tokenization"
                    )
            self._tokenizer = tok
        return self._tokenizer

    @abstractmethod
    def _tokenize_one(self, messages: list[dict]) -> dict[str, list[int] | int]:
        """Tokenize one sample. Raises on malformed input or tokenization error."""
        ...

    def _check_forbidden_content(self, messages: list[dict]) -> None:
        """Raise ValueError if any message contains forbidden tokens in content."""
        if not self.forbidden_content_tokens:
            return
        for i, msg in enumerate(messages):
            for field in ("content", "reasoning_content"):
                text = msg.get(field)
                if not isinstance(text, str):
                    continue
                for tok in self.forbidden_content_tokens:
                    if tok in text:
                        raise ValueError(
                            f"Forbidden token {tok!r} in {field} of {msg['role']} "
                            f"message at index {i}"
                        )

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        """Tokenize a batch of samples. Malformed samples are dropped and logged."""
        results: dict[str, list] = {k: [] for k in self.column_schema}
        failures: list[dict] = []
        for messages in batch["messages"]:
            try:
                self._check_forbidden_content(messages)
                result = self._tokenize_one(messages)
                for key in results:
                    results[key].append(result[key])
            except Exception as e:
                logger.warning("Dropping sample: %s", e)
                if self.failures_path:
                    failures.append({"messages": messages, "error": str(e)})
        if failures:
            _append_failures(self.failures_path, failures)
        return results

    @property
    @abstractmethod
    def column_schema(self) -> dict:
        """PyArrow column name → pa.DataType for this strategy's output."""
        ...

    @property
    @abstractmethod
    def chat_template_kwargs(self) -> dict:
        """kwargs forwarded to apply_chat_template; recorded in the manifest."""
        ...


class BackboneSuffixStrategy(TokenizationStrategy):
    """Pre-tokenizes multi-turn SFT data with backbone+suffix layout.

    Produces (input_ids, labels, positions, suffix_starts, insertion_limits) where:
    - The backbone is identical to TruncateLastStrategy (truncate_history_thinking=True,
      labels only on last assistant turn).
    - Each suffix recovers the thinking trace from a historical assistant turn group
      (consecutive assistant+tool turns between two user messages).
    - Suffixes are only created when at least one turn in the group has non-empty
      reasoning_content AND the group is followed by a later user message (which
      triggers truncation in the backbone).

    The packing layer expands suffix_starts/insertion_limits into per-token tensors
    at runtime.
    """

    _CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"truncate_history_thinking": True}

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        return self._CHAT_TEMPLATE_KWARGS

    def _tokenize_one(self, messages: list[dict]) -> dict[str, list[int] | int]:
        effective, full_text, full_tokens, input_ids, label_ids = (
            _prepare_full_sequence(
                messages, self.tokenizer, self.chat_template_kwargs,
            )
        )
        last_asst_idx = len(effective) - 1

        # Single encode for the backbone — character offset table for bisect lookups.
        full_encoding = self.tokenizer.tokenizer.encode(full_text)
        token_starts = [o[0] for o in full_encoding.offsets]

        # Backbone last-turn label_start: find <think> after the preceding context
        prefix_end = len(self.tokenizer.apply_chat_template(
            effective[:-1], **self.chat_template_kwargs
        ).rstrip("\n"))
        think_pos = full_text.index("<think>", prefix_end)
        start = _char_to_token_idx(token_starts, think_pos + len("<think>"))
        label_ids[start:] = full_tokens[start + 1:]

        backbone_input_ids = list(input_ids)
        backbone_labels = list(label_ids)
        backbone_positions = list(range(len(backbone_input_ids)))

        # --- Suffix groups ---
        user_indices = [i for i, m in enumerate(effective) if m["role"] == "user"]
        group_boundaries = user_indices + [last_asst_idx]

        suffix_starts: list[int] = []
        insertion_limits: list[int] = []
        all_suffix_input_ids: list[int] = []
        all_suffix_labels: list[int] = []
        all_suffix_positions: list[int] = []

        for bi in range(len(group_boundaries) - 1):
            group_start = group_boundaries[bi] + 1
            group_end = group_boundaries[bi + 1]

            has_reasoning = any(
                m["role"] == "assistant"
                and m.get("reasoning_content", "").strip()
                for m in effective[group_start:group_end]
            )
            if not has_reasoning:
                # No-reasoning group: label directly in backbone via offset lookup
                for turn_idx in range(group_start, group_end):
                    if effective[turn_idx]["role"] != "assistant":
                        continue
                    turn_prefix_end = len(self.tokenizer.apply_chat_template(
                        effective[:turn_idx], **self.chat_template_kwargs
                    ).rstrip("\n"))
                    close_think_pos = full_text.index("</think>", turn_prefix_end)
                    label_start = _char_to_token_idx(token_starts, close_think_pos)

                    end_text = self.tokenizer.apply_chat_template(
                        effective[: turn_idx + 1], **self.chat_template_kwargs
                    ).rstrip("\n")
                    label_end = _char_to_token_idx(token_starts, len(end_text)) - 2

                    end = min(label_end + 1, len(backbone_input_ids))
                    backbone_labels[label_start:end] = full_tokens[label_start + 1 : end + 1]
                continue

            # Reasoning group — build a suffix sequence
            first_asst_in_group = next(
                i for i in range(group_start, group_end)
                if effective[i]["role"] == "assistant"
            )

            suffix_source_text = self.tokenizer.apply_chat_template(
                effective[:group_end], **self.chat_template_kwargs
            ).rstrip("\n")
            suffix_source_tokens = self.tokenizer.encode(
                suffix_source_text, add_bos=True, add_eos=False
            )
            suffix_encoding = self.tokenizer.tokenizer.encode(suffix_source_text)
            suffix_token_starts = [o[0] for o in suffix_encoding.offsets]

            # insertion_limit: token index of <think> for the first assistant in group
            limit_prefix_end = len(self.tokenizer.apply_chat_template(
                effective[:first_asst_in_group], **self.chat_template_kwargs
            ).rstrip("\n"))
            think_pos = suffix_source_text.index("<think>", limit_prefix_end)
            insertion_limit = _char_to_token_idx(suffix_token_starts, think_pos)

            suffix_tokens = suffix_source_tokens[insertion_limit + 1:]

            if not suffix_tokens:
                raise ValueError(
                    f"Empty suffix: insertion_limit={insertion_limit} >= "
                    f"len(suffix_source_tokens)={len(suffix_source_tokens)}; "
                    "coordinate arithmetic is broken for this sample"
                )

            if suffix_tokens[-1] != self.tokenizer.eos_id:
                suffix_tokens.append(self.tokenizer.eos_id)

            suffix_input = suffix_tokens[:-1]
            suffix_label = [IGNORE_INDEX] * len(suffix_input)

            for turn_idx in range(group_start, group_end):
                if effective[turn_idx]["role"] != "assistant":
                    continue

                turn_prefix_end = len(self.tokenizer.apply_chat_template(
                    effective[:turn_idx], **self.chat_template_kwargs
                ).rstrip("\n"))
                # label_start: token after <think> (works for both reasoning and no-reasoning)
                open_think_pos = suffix_source_text.index("<think>", turn_prefix_end)
                label_start_global = _char_to_token_idx(suffix_token_starts, open_think_pos + len("<think>"))
                label_start_in_suffix = label_start_global - (insertion_limit + 1)

                end_text = self.tokenizer.apply_chat_template(
                    effective[: turn_idx + 1], **self.chat_template_kwargs
                ).rstrip("\n")
                label_end_global = _char_to_token_idx(suffix_token_starts, len(end_text)) - 2
                label_end_in_suffix = label_end_global - (insertion_limit + 1)

                assert label_start_in_suffix >= 0, (
                    f"label_start_in_suffix={label_start_in_suffix} < 0 for "
                    f"turn_idx={turn_idx}; coordinate arithmetic is broken"
                )
                assert label_end_in_suffix >= label_start_in_suffix, (
                    f"label_end_in_suffix={label_end_in_suffix} < "
                    f"label_start_in_suffix={label_start_in_suffix} for "
                    f"turn_idx={turn_idx}; coordinate arithmetic is broken"
                )
                for pos in range(label_start_in_suffix, min(label_end_in_suffix + 1, len(suffix_input))):
                    suffix_label[pos] = suffix_tokens[pos + 1]

            suffix_offset = len(backbone_input_ids) + len(all_suffix_input_ids)
            suffix_starts.append(suffix_offset)
            insertion_limits.append(insertion_limit)

            suffix_pos_start = insertion_limit + 1
            suffix_positions = list(range(suffix_pos_start, suffix_pos_start + len(suffix_input)))

            all_suffix_input_ids.extend(suffix_input)
            all_suffix_labels.extend(suffix_label)
            all_suffix_positions.extend(suffix_positions)

        # --- Assemble final output ---
        final_input_ids = backbone_input_ids + all_suffix_input_ids
        final_labels = backbone_labels + all_suffix_labels
        final_positions = backbone_positions + all_suffix_positions

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "positions": final_positions,
            "suffix_starts": suffix_starts,
            "insertion_limits": insertion_limits,
            "n_tokens": len(final_input_ids),
        }

    def _tokenize_one_reference(self, messages: list[dict]) -> dict[str, list[int] | int]:
        """Reference implementation using per-turn encode calls.

        Kept for correctness testing against the offset-based _tokenize_one.
        """
        effective, full_text, full_tokens, input_ids, label_ids = (
            _prepare_full_sequence(
                messages, self.tokenizer, self.chat_template_kwargs,
            )
        )
        last_asst_idx = len(effective) - 1

        prefix_text = self.tokenizer.apply_chat_template(
            effective[:-1],
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        prefix_tokens = self.tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        start = len(prefix_tokens) - 1
        label_ids[start:] = full_tokens[start + 1:]

        backbone_input_ids = list(input_ids)
        backbone_labels = list(label_ids)
        backbone_positions = list(range(len(backbone_input_ids)))

        user_indices = [i for i, m in enumerate(effective) if m["role"] == "user"]
        group_boundaries = user_indices + [last_asst_idx]

        suffix_starts: list[int] = []
        insertion_limits: list[int] = []
        all_suffix_input_ids: list[int] = []
        all_suffix_labels: list[int] = []
        all_suffix_positions: list[int] = []

        for bi in range(len(group_boundaries) - 1):
            group_start = group_boundaries[bi] + 1
            group_end = group_boundaries[bi + 1]

            has_reasoning = any(
                m["role"] == "assistant"
                and m.get("reasoning_content", "").strip()
                for m in effective[group_start:group_end]
            )
            if not has_reasoning:
                for turn_idx in range(group_start, group_end):
                    if effective[turn_idx]["role"] != "assistant":
                        continue
                    prefix_for_turn = self.tokenizer.apply_chat_template(
                        effective[:turn_idx],
                        add_generation_prompt=True,
                        **self.chat_template_kwargs,
                    )
                    prefix_for_turn_tokens = self.tokenizer.encode(
                        prefix_for_turn, add_bos=True, add_eos=False
                    )
                    label_start = len(prefix_for_turn_tokens) - 1

                    end_render_text = self.tokenizer.apply_chat_template(
                        effective[: turn_idx + 1], **self.chat_template_kwargs
                    ).rstrip("\n")
                    end_render_tokens = self.tokenizer.encode(
                        end_render_text, add_bos=True, add_eos=False
                    )
                    label_end = len(end_render_tokens) - 2

                    end = min(label_end + 1, len(backbone_input_ids))
                    backbone_labels[label_start:end] = full_tokens[label_start + 1 : end + 1]
                continue

            first_asst_in_group = next(
                i for i in range(group_start, group_end)
                if effective[i]["role"] == "assistant"
            )
            prefix_for_limit = self.tokenizer.apply_chat_template(
                effective[:first_asst_in_group],
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            prefix_for_limit_tokens = self.tokenizer.encode(
                prefix_for_limit, add_bos=True, add_eos=False
            )
            insertion_limit = len(prefix_for_limit_tokens) - 2

            suffix_source_text = self.tokenizer.apply_chat_template(
                effective[:group_end], **self.chat_template_kwargs
            ).rstrip("\n")
            suffix_source_tokens = self.tokenizer.encode(
                suffix_source_text, add_bos=True, add_eos=False
            )

            suffix_tokens = suffix_source_tokens[insertion_limit + 1:]

            if not suffix_tokens:
                raise ValueError(
                    f"Empty suffix: insertion_limit={insertion_limit} >= "
                    f"len(suffix_source_tokens)={len(suffix_source_tokens)}; "
                    "coordinate arithmetic is broken for this sample"
                )

            if suffix_tokens[-1] != self.tokenizer.eos_id:
                suffix_tokens.append(self.tokenizer.eos_id)

            suffix_input = suffix_tokens[:-1]
            suffix_label = [IGNORE_INDEX] * len(suffix_input)

            for turn_idx in range(group_start, group_end):
                if effective[turn_idx]["role"] != "assistant":
                    continue

                prefix_for_turn = self.tokenizer.apply_chat_template(
                    effective[:turn_idx],
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                prefix_for_turn_tokens = self.tokenizer.encode(
                    prefix_for_turn, add_bos=True, add_eos=False
                )
                label_start_global = len(prefix_for_turn_tokens) - 1
                label_start_in_suffix = label_start_global - (insertion_limit + 1)

                end_render_text = self.tokenizer.apply_chat_template(
                    effective[: turn_idx + 1], **self.chat_template_kwargs
                ).rstrip("\n")
                end_render_tokens = self.tokenizer.encode(
                    end_render_text, add_bos=True, add_eos=False
                )
                label_end_global = len(end_render_tokens) - 2
                label_end_in_suffix = label_end_global - (insertion_limit + 1)

                assert label_start_in_suffix >= 0, (
                    f"label_start_in_suffix={label_start_in_suffix} < 0 for "
                    f"turn_idx={turn_idx}; coordinate arithmetic is broken"
                )
                assert label_end_in_suffix >= label_start_in_suffix, (
                    f"label_end_in_suffix={label_end_in_suffix} < "
                    f"label_start_in_suffix={label_start_in_suffix} for "
                    f"turn_idx={turn_idx}; coordinate arithmetic is broken"
                )
                for pos in range(label_start_in_suffix, min(label_end_in_suffix + 1, len(suffix_input))):
                    suffix_label[pos] = suffix_tokens[pos + 1]

            suffix_offset = len(backbone_input_ids) + len(all_suffix_input_ids)
            suffix_starts.append(suffix_offset)
            insertion_limits.append(insertion_limit)

            suffix_pos_start = insertion_limit + 1
            suffix_positions = list(range(suffix_pos_start, suffix_pos_start + len(suffix_input)))

            all_suffix_input_ids.extend(suffix_input)
            all_suffix_labels.extend(suffix_label)
            all_suffix_positions.extend(suffix_positions)

        final_input_ids = backbone_input_ids + all_suffix_input_ids
        final_labels = backbone_labels + all_suffix_labels
        final_positions = backbone_positions + all_suffix_positions

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "positions": final_positions,
            "suffix_starts": suffix_starts,
            "insertion_limits": insertion_limits,
            "n_tokens": len(final_input_ids),
        }

    @property
    def column_schema(self) -> dict:
        import pyarrow as pa

        return {
            "input_ids": pa.list_(pa.int32()),
            "labels": pa.list_(pa.int32()),
            "positions": pa.list_(pa.int32()),
            "suffix_starts": pa.list_(pa.int32()),
            "insertion_limits": pa.list_(pa.int32()),
            "n_tokens": pa.int32(),
        }


class TruncateLastStrategy(TokenizationStrategy):
    """Pre-tokenizes multi-turn SFT data, labeling only the final assistant turn.

    Produces (input_ids, labels) pairs where only the last assistant turn is unmasked.
    Uses truncate_history_thinking=True: thinking traces from all but the last assistant
    turn are stripped, matching the vLLM/SGLang inference default.

    Intermediate assistant turns are not trained on because they were collected under a
    different context than the one seen at training time. Turn K was generated when T_{K-1}
    was the "last" assistant (thinking preserved), but in the full training sequence T_{K-1}
    has its thinking stripped. Only the final turn is generated under a context identical
    to what the model sees during training.

    Conversations that do not end with an assistant turn (e.g. agentic trajectories cut
    off after a tool result, or after a system-injected follow-up message) are accepted.
    Messages after the last assistant turn are dropped before tokenization. Two reasons:
    (1) they are environment outputs (tool responses) or injected scaffolding, not model
    outputs — no training signal is lost; (2) for user-last conversations specifically,
    retaining the trailing user message shifts the Granite template's last_user_idx past
    the last assistant turn, incorrectly stripping that turn's thinking traces. The last
    assistant turn itself — including any tool-call decisions and reasoning — is fully
    preserved and trained on.

    No seq_len filtering is applied here; that is deferred to training-time packing.
    """

    _CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"truncate_history_thinking": True}

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        return self._CHAT_TEMPLATE_KWARGS

    def _tokenize_one(self, messages: list[dict]) -> dict[str, list[int] | int]:
        return _tokenize_last_turn(messages, self.tokenizer, self.chat_template_kwargs)

    @property
    def column_schema(self) -> dict:
        import pyarrow as pa

        return {
            "input_ids": pa.list_(pa.int32()),
            "labels": pa.list_(pa.int32()),
            "n_tokens": pa.int32(),
        }


class FullThinkingStrategy(TruncateLastStrategy):
    """Pre-tokenizes multi-turn SFT data with full thinking context.

    Uses truncate_history_thinking=False: thinking traces from every assistant turn
    are preserved in the token sequence. All assistant turns are loss-unmasked —
    the model trains on every token it would generate at inference.

    Use case: agentic training where the model should learn to produce tool calls,
    reasoning, and final responses — matching what it sees during multi-turn inference
    with truncate_history_thinking=False.

    Context fidelity:
        The chat template renders no-reasoning turns as <think></think>{content},
        but at inference the model receives <think>\\n from the generation prompt
        and produces </think>\\n{content}. The _fix_empty_thinking fixup corrects
        this before tokenization so that training context is token-for-token
        identical to inference context.

    Data provenance note:
        Training data may have been collected under truncate_history_thinking=True
        (standard vLLM/SGLang inference), meaning intermediate turns were generated
        without seeing prior thinking context. This strategy presents full context at
        training time regardless of collection conditions. This represents a
        train-vs-collection mismatch for intermediate turns whose effects on
        training quality need empirical validation.
    """

    _CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"truncate_history_thinking": False}

    def _tokenize_one(self, messages: list[dict]) -> dict[str, list[int] | int]:
        return _tokenize_all_turns(
            messages, self.tokenizer, self.chat_template_kwargs,
            fixup_fn=_fix_empty_thinking,
        )


class TruncateEveryTurnStrategy(TokenizationStrategy):
    """Decomposes multi-turn conversations into one example per assistant turn.

    For an N-assistant-turn conversation, produces N independent examples. Each
    example is the conversation truncated to assistant turn K, with labels only
    on turn K. Uses truncate_history_thinking=True: historical thinking traces
    are stripped, matching vLLM/SGLang inference behavior.

    This is a simple baseline for BackboneSuffixStrategy: same per-turn training
    signal, no suffix coordinate math, at the cost of redundant context tokens.
    """

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"truncate_history_thinking": True}

    def _tokenize_one(self, messages: list[dict]) -> dict[str, list[int] | int]:
        effective, full_text, full_tokens, input_ids, label_ids = (
            _prepare_full_sequence(
                messages, self.tokenizer, self.chat_template_kwargs,
            )
        )
        full_encoding = self.tokenizer.tokenizer.encode(full_text)
        token_starts = [o[0] for o in full_encoding.offsets]

        prefix_end = len(self.tokenizer.apply_chat_template(
            effective[:-1], **self.chat_template_kwargs
        ).rstrip("\n"))
        think_pos = full_text.index("<think>", prefix_end)
        start = _char_to_token_idx(token_starts, think_pos + len("<think>"))

        label_ids[start:] = full_tokens[start + 1:]
        return {"input_ids": input_ids, "labels": label_ids, "n_tokens": len(input_ids)}

    def _tokenize_one_reference(self, messages: list[dict]) -> dict[str, list[int] | int]:
        """Reference implementation using per-turn encode calls.

        Kept for correctness testing against the offset-based _tokenize_one.
        """
        return _tokenize_last_turn(messages, self.tokenizer, self.chat_template_kwargs)

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        results: dict[str, list] = {k: [] for k in self.column_schema}
        failures: list[dict] = []
        for messages in batch["messages"]:
            try:
                self._check_forbidden_content(messages)
                last_asst_idx = max(
                    i for i, m in enumerate(messages) if m["role"] == "assistant"
                )
                effective = messages[: last_asst_idx + 1]
                asst_indices = [
                    i for i, m in enumerate(effective) if m["role"] == "assistant"
                ]

                # Only split when historical turns have reasoning to truncate.
                # Otherwise a single pass labeling all turns is equivalent.
                has_historical_reasoning = any(
                    effective[i].get("reasoning_content", "").strip()
                    for i in asst_indices[:-1]
                )

                if has_historical_reasoning:
                    for asst_idx in asst_indices:
                        truncated = effective[: asst_idx + 1]
                        result = self._tokenize_one(truncated)
                        for key in results:
                            results[key].append(result[key])
                else:
                    result = _tokenize_all_turns(
                        effective, self.tokenizer, self.chat_template_kwargs
                    )
                    for key in results:
                        results[key].append(result[key])
            except Exception as e:
                logger.warning("Dropping sample: %s", e)
                if self.failures_path:
                    failures.append({"messages": messages, "error": str(e)})
        if failures:
            _append_failures(self.failures_path, failures)
        return results

    @property
    def column_schema(self) -> dict:
        import pyarrow as pa

        return {
            "input_ids": pa.list_(pa.int32()),
            "labels": pa.list_(pa.int32()),
            "n_tokens": pa.int32(),
        }
