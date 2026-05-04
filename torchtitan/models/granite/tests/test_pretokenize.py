"""Tests for TruncateLastStrategy tokenization and label masking.

Uses the shared test tokenizer at tests/assets/tokenizer/ (which has a minimal
chat template: bos + role\ncontent + eos per turn). All tests run without GPUs
or model weights.

Tests with the real Granite tokenizer (chat_template.jinja, truncate_history_thinking,
<think> tokens) are skipped if HF_ASSETS_PATH is not set.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.tokenization_strategies import (
    TruncateLastStrategy,
    _validate_messages,
)

load_dotenv()

_REPO_ROOT = Path(__file__).parents[4]
_TEST_TOKENIZER_PATH = str(_REPO_ROOT / "tests" / "assets" / "tokenizer")
_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")
_DATA_PATH_7M_BALANCED = os.environ.get("DATA_PATH_7M_BALANCED")


class TestValidateMessages(unittest.TestCase):
    def test_valid_single_turn(self):
        _validate_messages(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        )

    def test_valid_system_turn(self):
        _validate_messages(
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        )

    def test_valid_multi_turn(self):
        _validate_messages(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        )

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            _validate_messages([])

    def test_rejects_no_assistant(self):
        with self.assertRaises(ValueError):
            _validate_messages([{"role": "user", "content": "hi"}])

    def test_accepts_tool_as_last(self):
        _validate_messages(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "result"},
            ]
        )

    def test_accepts_user_as_last(self):
        _validate_messages(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
        )

    def test_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            _validate_messages(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "robot", "content": "beep"},
                ]
            )

    def test_rejects_system_not_first(self):
        with self.assertRaises(ValueError):
            _validate_messages(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "oops"},
                    {"role": "assistant", "content": "ok"},
                ]
            )

    def test_rejects_multiple_system(self):
        with self.assertRaises(ValueError):
            _validate_messages(
                [
                    {"role": "system", "content": "rule 1"},
                    {"role": "system", "content": "rule 2"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ]
            )


class TestTruncateLastStrategyBasic(unittest.TestCase):
    """Output structure, shift invariant, and last-turn-only masking using the test tokenizer."""
    def setUp(self):
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TEST_TOKENIZER_PATH)
        self.strategy = TruncateLastStrategy(_TEST_TOKENIZER_PATH)

    def _tokenize(self, messages):
        return self.strategy._tokenize_one(messages)

    def test_single_turn_returns_correct_keys(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("n_tokens", result)

    def test_n_tokens_equals_len_input_ids(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)
        self.assertEqual(result["n_tokens"], len(result["input_ids"]))

    def test_labels_length_matches_input_ids(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)
        self.assertEqual(len(result["labels"]), len(result["input_ids"]))

    def test_some_labels_masked(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)
        masked = [lbl for lbl in result["labels"] if lbl == IGNORE_INDEX]
        unmasked = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        self.assertGreater(len(masked), 0, "user tokens should be masked")
        self.assertGreater(len(unmasked), 0, "assistant tokens should be unmasked")

    def test_last_label_is_eos(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)
        self.assertEqual(result["labels"][-1], self.tokenizer.eos_id)

    def test_input_ids_shift(self):
        """input_ids = full_tokens[:-1]; unmasked labels = full_tokens[1:]."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)

        full_text = self.tokenizer.apply_chat_template(msgs, **self.strategy.chat_template_kwargs)
        full_text = full_text.rstrip("\n")
        full_tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self.tokenizer.eos_id:
            full_tokens.append(self.tokenizer.eos_id)

        self.assertEqual(result["input_ids"], full_tokens[:-1])
        for i, (inp, lbl) in enumerate(zip(result["input_ids"], result["labels"])):
            if lbl != IGNORE_INDEX:
                self.assertEqual(lbl, full_tokens[i + 1])

    def test_malformed_raises(self):
        msgs = [{"role": "user", "content": "hi"}]
        with self.assertRaises(ValueError):
            self._tokenize(msgs)

    def test_malformed_drops_from_batch(self):
        good = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        bad = [{"role": "user", "content": "hi"}]
        batch = {"messages": [good, bad, good]}
        output = self.strategy(batch)
        self.assertEqual(len(output["input_ids"]), 2)
        self.assertEqual(len(output["labels"]), 2)

    def test_multi_turn_only_last_assistant_unmasked(self):
        """Only the last assistant turn must be unmasked; earlier turns must be masked."""
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "unique_first_reply"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "unique_last_reply"},
        ]
        result = self._tokenize(msgs)
        unmasked_ids = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        unmasked_text = self.tokenizer.decode(unmasked_ids)
        self.assertIn("unique_last_reply", unmasked_text)
        self.assertNotIn("unique_first_reply", unmasked_text)

    def test_multi_turn_unmasked_is_contiguous_suffix(self):
        """Unmasked labels must form a single contiguous block at the end."""
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = self._tokenize(msgs)

        labels = result["labels"]
        self.assertGreater(
            sum(1 for lbl in labels if lbl != IGNORE_INDEX),
            0,
            "Expected at least one unmasked label",
        )
        # Once unmasked tokens start, no masked token may follow.
        seen_unmasked = False
        for lbl in labels:
            if lbl != IGNORE_INDEX:
                seen_unmasked = True
            elif seen_unmasked:
                self.fail("Masked label found after unmasked label — not a contiguous suffix")

    def test_tool_last_trailing_tokens_excluded(self):
        """Trailing tool message must not appear in input_ids."""
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "unique_asst_reply"},
            {"role": "tool", "content": "unique_tool_result"},
        ]
        result = self._tokenize(msgs)
        decoded = self.tokenizer.decode(result["input_ids"])
        self.assertIn("unique_asst_reply", decoded)
        self.assertNotIn("unique_tool_result", decoded)

    def test_user_last_trailing_tokens_excluded(self):
        """Trailing user message must not appear in input_ids."""
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "unique_asst_reply"},
            {"role": "user", "content": "unique_followup"},
        ]
        result = self._tokenize(msgs)
        decoded = self.tokenizer.decode(result["input_ids"])
        self.assertIn("unique_asst_reply", decoded)
        self.assertNotIn("unique_followup", decoded)

    def test_batched_call_returns_parallel_lists(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        batch = {"messages": [msgs, msgs, msgs]}
        output = self.strategy(batch)
        self.assertEqual(len(output["input_ids"]), 3)
        self.assertEqual(len(output["labels"]), 3)
        self.assertEqual(len(output["n_tokens"]), 3)
        for i in range(3):
            self.assertEqual(output["n_tokens"][i], len(output["input_ids"][i]))


class TestTruncateLastStrategyOrchestrator(unittest.TestCase):
    """Integration test: run the full pre-tokenization pipeline on a tiny JSONL."""

    def setUp(self):
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TEST_TOKENIZER_PATH)
        self.strategy = TruncateLastStrategy(_TEST_TOKENIZER_PATH)

    def test_jsonl_to_shard(self):
        from datasets import load_dataset, load_from_disk

        data = [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "test.jsonl")
            with open(jsonl_path, "w") as f:
                for row in data:
                    f.write(json.dumps(row) + "\n")

            ds = load_dataset("json", data_files=jsonl_path, split="train")
            ds = ds.map(
                self.strategy,
                batched=True,
                batch_size=10,
                remove_columns=ds.column_names,
            )
            shard_path = os.path.join(tmp, "shard")
            ds.save_to_disk(shard_path)

            loaded = load_from_disk(shard_path)
            self.assertEqual(len(loaded), 2)
            self.assertIn("input_ids", loaded.column_names)
            self.assertIn("labels", loaded.column_names)
            self.assertIn("n_tokens", loaded.column_names)
            for row in loaded:
                self.assertEqual(row["n_tokens"], len(row["input_ids"]))
                self.assertEqual(len(row["labels"]), len(row["input_ids"]))
                self.assertEqual(row["labels"][-1], self.tokenizer.eos_id)


class TestTruncateLastStrategyFailureRecording(unittest.TestCase):
    """Tests that failures are flushed to failures_path after each batch."""

    def test_validation_error_written_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "failures.jsonl")
            strategy = TruncateLastStrategy(_TEST_TOKENIZER_PATH, failures_path=path)
            bad = [{"role": "user", "content": "no assistant"}]
            good = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
            result = strategy({"messages": [bad, good]})

            self.assertEqual(len(result["input_ids"]), 1)

            with open(path) as f:
                records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertIn("messages", rec)
            self.assertIsInstance(rec["error"], str)
            self.assertEqual(rec["messages"], bad)

    def test_all_batch_failures_written(self):
        """Every bad example in a batch ends up in the file after __call__ returns."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "failures.jsonl")
            strategy = TruncateLastStrategy(_TEST_TOKENIZER_PATH, failures_path=path)
            bad1 = [{"role": "user", "content": "no assistant"}]
            bad2 = [{"role": "assistant", "content": "starts wrong"}]
            result = strategy({"messages": [bad1, bad2]})

            self.assertEqual(len(result["input_ids"]), 0)

            with open(path) as f:
                records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(isinstance(r["error"], str) for r in records))


@unittest.skipUnless(
    _HF_ASSETS_PATH, "HF_ASSETS_PATH not set — skipping Granite tokenizer tests"
)
class TestTruncateLastStrategyGranite(unittest.TestCase):
    """Tests using the real Granite tokenizer with truncate_history_thinking."""

    def setUp(self):
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        self.strategy = TruncateLastStrategy(_HF_ASSETS_PATH)

    def test_truncate_history_thinking_masks_old_thinking(self):
        """Thinking tokens in earlier turns should be absent from the tokenized sequence."""
        msgs = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "reasoning_content": "let me think",
                "content": "answer1",
            },
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "reasoning_content": "more thinking",
                "content": "answer2",
            },
        ]
        result = self.strategy._tokenize_one(msgs)

        decoded = self.tokenizer.decode(result["input_ids"], skip_special_tokens=False)
        # With truncate=True, "let me think" from turn 1 should be absent
        self.assertNotIn("let me think", decoded)
        # "more thinking" from the last turn should be present
        self.assertIn("more thinking", decoded)

    def test_thinking_tokens_are_unmasked_in_last_turn(self):
        """<think>...</think> tokens in the last turn must be trained on."""
        msgs = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "reasoning_content": "deep thoughts",
                "content": "final answer",
            },
        ]
        result = self.strategy._tokenize_one(msgs)

        unmasked_ids = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        unmasked_text = self.tokenizer.decode(unmasked_ids, skip_special_tokens=False)
        self.assertIn("deep thoughts", unmasked_text)
        self.assertIn("final answer", unmasked_text)

    def test_user_tokens_always_masked(self):
        msgs = [
            {"role": "user", "content": "unique_user_marker_xyz"},
            {"role": "assistant", "content": "response"},
        ]
        result = self.strategy._tokenize_one(msgs)

        masked_ids = [
            result["input_ids"][i]
            for i, lbl in enumerate(result["labels"])
            if lbl == IGNORE_INDEX
        ]
        masked_text = self.tokenizer.decode(masked_ids, skip_special_tokens=False)
        self.assertIn("unique_user_marker_xyz", masked_text)

    def test_multi_turn_tool_with_reasoning_content_not_dropped(self):
        """Tool-use conversation with reasoning_content in intermediate turns must not be dropped."""
        msgs = [
            {"role": "user", "content": "ping"},
            {
                "role": "assistant",
                "reasoning_content": "I will call a tool.",
                "content": "first_reply",
            },
            {"role": "tool", "content": "tool_result"},
            {
                "role": "assistant",
                "reasoning_content": "The tool returned a result.",
                "content": "second_reply",
            },
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(result["n_tokens"], len(result["input_ids"]))
        self.assertEqual(len(result["labels"]), len(result["input_ids"]))

    def test_multi_turn_tool_only_last_assistant_unmasked(self):
        """In a tool-use conversation only the last assistant turn must be unmasked."""
        msgs = [
            {"role": "user", "content": "ping"},
            {
                "role": "assistant",
                "reasoning_content": "thinking1",
                "content": "unique_first_reply",
            },
            {"role": "tool", "content": "ok"},
            {
                "role": "assistant",
                "reasoning_content": "thinking2",
                "content": "unique_second_reply",
            },
        ]
        result = self.strategy._tokenize_one(msgs)

        unmasked_ids = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        unmasked_text = self.tokenizer.decode(unmasked_ids, skip_special_tokens=False)
        self.assertIn("unique_second_reply", unmasked_text)
        self.assertNotIn("unique_first_reply", unmasked_text)

    def test_tool_last_reasoning_preserved(self):
        """Reasoning in the last assistant tool-call turn must survive after trailing tool is dropped."""
        msgs = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "reasoning_content": "unique_tool_call_reasoning",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "unique_tool_result"},
        ]
        result = self.strategy._tokenize_one(msgs)
        decoded = self.tokenizer.decode(result["input_ids"], skip_special_tokens=False)
        self.assertIn("unique_tool_call_reasoning", decoded)
        self.assertNotIn("unique_tool_result", decoded)

    def test_user_last_reasoning_preserved_by_slicing(self):
        """Trailing user shifts last_user_idx, stripping last assistant thinking without the slice.

        Demonstrates why effective = messages[:last_asst_idx + 1] is required.
        """
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "thinking_a1", "content": "reply1"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "reasoning_content": "unique_last_reasoning",
                "content": "reply2",
            },
            {"role": "user", "content": "unique_injected_user"},
        ]
        # Without slicing, the trailing user shifts last_user_idx past asst2, stripping
        # its thinking. This demonstrates the bug the slice is fixing.
        unsliced = self.tokenizer.apply_chat_template(msgs, truncate_history_thinking=True)
        self.assertNotIn("unique_last_reasoning", unsliced)

        # _tokenize_one slices to effective = msgs[:last_asst_idx + 1], restoring thinking.
        result = self.strategy._tokenize_one(msgs)
        decoded = self.tokenizer.decode(result["input_ids"], skip_special_tokens=False)
        self.assertIn("unique_last_reasoning", decoded)
        self.assertNotIn("unique_injected_user", decoded)


@unittest.skipUnless(
    _DATA_PATH_7M_BALANCED and _HF_ASSETS_PATH,
    "DATA_PATH_7M_BALANCED or HF_ASSETS_PATH not set — skipping real data tests",
)
class TestTruncateLastStrategyRealData(unittest.TestCase):
    """Smoke test on real data. Requires DATA_PATH_7M_BALANCED (dir of .jsonl) + HF_ASSETS_PATH."""

    _NUM_SAMPLES = 1000
    _MAX_DROP_RATE = 0.01

    def setUp(self):
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        self.strategy = TruncateLastStrategy(_HF_ASSETS_PATH)

    def _load_samples(self) -> list[list[dict]]:
        import glob

        jsonl_files = sorted(glob.glob(os.path.join(_DATA_PATH_7M_BALANCED, "*.jsonl")))
        if not jsonl_files:
            self.skipTest(f"No .jsonl files found in {_DATA_PATH_7M_BALANCED}")
        samples = []
        with open(jsonl_files[0]) as f:
            for i, line in enumerate(f):
                if i >= self._NUM_SAMPLES:
                    break
                samples.append(json.loads(line)["messages"])
        return samples

    def test_format_and_drop_rate(self):
        samples = self._load_samples()
        valid = []
        for msgs in samples:
            try:
                valid.append(self.strategy._tokenize_one(msgs))
            except Exception:
                pass

        drop_rate = 1 - len(valid) / len(samples)
        self.assertLess(
            drop_rate,
            self._MAX_DROP_RATE,
            f"Drop rate {drop_rate:.1%} exceeds {self._MAX_DROP_RATE:.1%}",
        )

        eos_id = self.tokenizer.eos_id
        for r in valid:
            self.assertEqual(r["n_tokens"], len(r["input_ids"]))
            self.assertEqual(len(r["labels"]), len(r["input_ids"]))
            self.assertEqual(r["labels"][-1], eos_id)
            masked = sum(1 for lbl in r["labels"] if lbl == IGNORE_INDEX)
            self.assertGreater(masked, 0)
            self.assertGreater(len(r["labels"]) - masked, 0)


if __name__ == "__main__":
    unittest.main()
