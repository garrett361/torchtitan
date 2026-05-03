import fcntl
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

logger = logging.getLogger(__name__)

_VALID_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


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
    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Last message must be 'assistant', got '{messages[-1]['role']}'"
        )
    system_positions = [i for i, m in enumerate(messages) if m["role"] == "system"]
    if len(system_positions) > 1 or (system_positions and system_positions[0] != 0):
        raise ValueError("system message must be the first message if present")


def _append_failures(path: str, failures: list[dict]) -> None:
    """Append failure records to a JSONL file, safe for concurrent writers."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class TokenizationStrategy(ABC):
    def __init__(self, tokenizer_path: str, *, failures_path: str | None = None) -> None:
        self._tokenizer_path = tokenizer_path
        self._tokenizer: HuggingFaceTokenizer | None = None
        self._failures_path = failures_path

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        if self._tokenizer is None:
            tok = HuggingFaceTokenizer(tokenizer_path=self._tokenizer_path)
            if tok.eos_id is None:
                raise ValueError("Tokenizer must have a valid eos_id")
            self._tokenizer = tok
        return self._tokenizer

    @abstractmethod
    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        """Tokenize a batch of samples. Malformed samples are dropped."""
        ...

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


class NaiveStrategy(TokenizationStrategy):
    """Pre-tokenizes multi-turn SFT data with per-turn assistant label masking.

    Produces (input_ids, labels) pairs where non-assistant tokens are masked with
    IGNORE_INDEX. Uses truncate_history_thinking=True: thinking traces from all but
    the last assistant turn are stripped, matching the vLLM/SGLang inference default.

    No seq_len filtering is applied here; that is deferred to training-time packing.
    """

    _CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"truncate_history_thinking": True}

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        return self._CHAT_TEMPLATE_KWARGS

    def _tokenize_one(
        self, messages: list[dict], *, failures: list | None = None
    ) -> dict[str, list[int] | int] | None:
        """Tokenize one sample. Returns None to drop (malformed or tokenization error).

        failures: if provided, failed samples are appended as {"messages": ..., "error": ...}.
        """
        try:
            _validate_messages(messages)
        except (ValueError, KeyError) as e:
            logger.warning("Dropping malformed sample: %s", e)
            if failures is not None:
                failures.append({"messages": messages, "error": f"validation: {e}"})
            return None

        try:
            full_text = self.tokenizer.apply_chat_template(
                messages, **self.chat_template_kwargs
            )
            full_text = full_text.rstrip("\n")
            full_tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=False)
            if full_tokens[-1] != self.tokenizer.eos_id:
                full_tokens.append(self.tokenizer.eos_id)

            input_ids = full_tokens[:-1]
            label_ids = [IGNORE_INDEX] * len(input_ids)

            last_asst_idx = max(
                i for i, m in enumerate(messages) if m["role"] == "assistant"
            )
            im_end_id = self.tokenizer.token_to_id("<|im_end|>")
            for turn_idx, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                prefix_text = self.tokenizer.apply_chat_template(
                    messages[:turn_idx],
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                prefix_tokens = self.tokenizer.encode(
                    prefix_text, add_bos=True, add_eos=False
                )
                start = len(prefix_tokens) - 1
                if turn_idx == last_asst_idx:
                    end = len(label_ids)
                elif im_end_id is not None:
                    # Scan full_tokens for <|im_end|> rather than re-tokenizing the suffix.
                    # Re-tokenizing breaks with truncate_history_thinking=True: calling
                    # apply_chat_template(messages[:turn_idx+1]) treats turn_idx as the last
                    # turn and preserves its thinking, producing a longer span than appears
                    # in full_tokens where that turn's thinking was stripped.
                    end = full_tokens.index(im_end_id, start + 1)
                else:
                    suffix_text = self.tokenizer.apply_chat_template(
                        messages[: turn_idx + 1], **self.chat_template_kwargs
                    )
                    suffix_tokens = self.tokenizer.encode(
                        suffix_text.rstrip("\n"), add_bos=True, add_eos=False
                    )
                    end = len(suffix_tokens) - 1
                label_ids[start:end] = full_tokens[1:][start:end]

            return {
                "input_ids": input_ids,
                "labels": label_ids,
                "n_tokens": len(input_ids),
            }
        except Exception as e:
            logger.warning("Dropping erroring sample: %s", e)
            if failures is not None:
                failures.append({"messages": messages, "error": f"tokenization: {e}"})
            return None

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        results: dict[str, list] = {"input_ids": [], "labels": [], "n_tokens": []}
        failures: list[dict] = []
        for messages in batch["messages"]:
            result = self._tokenize_one(
                messages, failures=failures if self._failures_path else None
            )
            if result is not None:
                for key in results:
                    results[key].append(result[key])
        if failures and self._failures_path:
            _append_failures(self._failures_path, failures)
        return results

    @property
    def column_schema(self) -> dict:
        import pyarrow as pa

        return {
            "input_ids": pa.list_(pa.int32()),
            "labels": pa.list_(pa.int32()),
            "n_tokens": pa.int32(),
        }
