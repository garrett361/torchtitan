import json
import logging
from abc import ABC, abstractmethod

from filelock import FileLock
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
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("Messages must contain at least one assistant turn")
    system_positions = [i for i, m in enumerate(messages) if m["role"] == "system"]
    if len(system_positions) > 1 or (system_positions and system_positions[0] != 0):
        raise ValueError("system message must be the first message if present")


def _append_failures(path: str, failures: list[dict]) -> None:
    """Append failure records to a JSONL file, safe for concurrent writers."""
    with FileLock(path + ".lock"), open(path, "a") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")


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
    def _tokenize_one(self, messages: list[dict]) -> dict[str, list[int] | int]:
        """Tokenize one sample. Raises on malformed input or tokenization error."""
        ...

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        """Tokenize a batch of samples. Malformed samples are dropped and logged."""
        results: dict[str, list] = {k: [] for k in self.column_schema}
        failures: list[dict] = []
        for messages in batch["messages"]:
            try:
                result = self._tokenize_one(messages)
                for key in results:
                    results[key].append(result[key])
            except Exception as e:
                logger.warning("Dropping sample: %s", e)
                if self._failures_path:
                    failures.append({"messages": messages, "error": str(e)})
        if failures:
            _append_failures(self._failures_path, failures)
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
        _validate_messages(messages)
        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        effective = messages[: last_asst_idx + 1]
        full_text = self.tokenizer.apply_chat_template(
            effective, **self.chat_template_kwargs
        ).rstrip("\n")
        full_tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self.tokenizer.eos_id:
            full_tokens.append(self.tokenizer.eos_id)
        input_ids = full_tokens[:-1]
        label_ids = [IGNORE_INDEX] * len(input_ids)
        prefix_text = self.tokenizer.apply_chat_template(
            effective[:-1],
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        prefix_tokens = self.tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        start = len(prefix_tokens) - 1
        label_ids[start:] = full_tokens[start + 1:]
        return {"input_ids": input_ids, "labels": label_ids, "n_tokens": len(input_ids)}

    @property
    def column_schema(self) -> dict:
        import pyarrow as pa

        return {
            "input_ids": pa.list_(pa.int32()),
            "labels": pa.list_(pa.int32()),
            "n_tokens": pa.int32(),
        }
