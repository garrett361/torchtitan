"""Tests for NaiveStrategy tokenization and label masking.

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

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.tokenization_strategies import (
    NaiveStrategy,
    _validate_messages,
)

_REPO_ROOT = Path(__file__).parents[4]
_TEST_TOKENIZER_PATH = str(_REPO_ROOT / "tests" / "assets" / "tokenizer")
_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")


def _make_tokenizer(path: str) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(tokenizer_path=path)


def _make_strategy(tokenizer_path: str) -> NaiveStrategy:
    return NaiveStrategy(tokenizer_path)


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

    def test_rejects_last_not_assistant(self):
        with self.assertRaises(ValueError):
            _validate_messages([{"role": "user", "content": "hi"}])

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


class TestNaiveStrategyBasic(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _make_tokenizer(_TEST_TOKENIZER_PATH)
        self.strategy = _make_strategy(_TEST_TOKENIZER_PATH)

    def _tokenize(self, messages):
        return self.strategy._tokenize_one(messages)

    def test_single_turn_returns_correct_keys(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self._tokenize(msgs)
        self.assertIsNotNone(result)
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

    def test_malformed_returns_none(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self._tokenize(msgs)
        self.assertIsNone(result)

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

    def test_multi_turn_each_assistant_unmasked(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = self._tokenize(msgs)
        self.assertIsNotNone(result)

        full_text = self.tokenizer.apply_chat_template(msgs, **self.strategy.chat_template_kwargs)
        full_tokens = self.tokenizer.encode(
            full_text.rstrip("\n"), add_bos=True, add_eos=False
        )
        if full_tokens[-1] != self.tokenizer.eos_id:
            full_tokens.append(self.tokenizer.eos_id)

        unmasked = [(i, lbl) for i, lbl in enumerate(result["labels"]) if lbl != IGNORE_INDEX]
        self.assertGreater(len(unmasked), 0)
        for i, lbl in unmasked:
            self.assertEqual(lbl, full_tokens[i + 1])

    def test_multi_turn_masked_boundary(self):
        """Tokens between assistant turns (user turns) must all be masked."""
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = self._tokenize(msgs)

        # Find the region between the two assistant turns (after first unmasked block).
        labels = result["labels"]
        in_masked_gap = False
        saw_first_unmasked = False
        saw_second_unmasked = False
        for lbl in labels:
            if lbl != IGNORE_INDEX and not saw_first_unmasked:
                saw_first_unmasked = True
            elif lbl == IGNORE_INDEX and saw_first_unmasked and not in_masked_gap:
                in_masked_gap = True
            elif lbl != IGNORE_INDEX and in_masked_gap:
                saw_second_unmasked = True
                in_masked_gap = False
        self.assertTrue(
            saw_second_unmasked, "Expected a masked gap between two assistant turns"
        )

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


class TestNaiveStrategyOrchestrator(unittest.TestCase):
    """Integration test: run the full pre-tokenization pipeline on a tiny JSONL."""

    def setUp(self):
        self.tokenizer = _make_tokenizer(_TEST_TOKENIZER_PATH)
        self.strategy = _make_strategy(_TEST_TOKENIZER_PATH)

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


@unittest.skipUnless(
    _HF_ASSETS_PATH, "HF_ASSETS_PATH not set — skipping Granite tokenizer tests"
)
class TestNaiveStrategyGranite(unittest.TestCase):
    """Tests using the real Granite tokenizer with truncate_history_thinking."""

    def setUp(self):
        self.tokenizer = _make_tokenizer(_HF_ASSETS_PATH)
        self.strategy = _make_strategy(_HF_ASSETS_PATH)

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
        self.assertIsNotNone(result)

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
        self.assertIsNotNone(result)

        # All unmasked labels should decode to include thinking content
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
        self.assertIsNotNone(result)

        # Tokens decoded from masked positions should not contain the response
        masked_ids = [
            result["input_ids"][i]
            for i, lbl in enumerate(result["labels"])
            if lbl == IGNORE_INDEX
        ]
        masked_text = self.tokenizer.decode(masked_ids, skip_special_tokens=False)
        self.assertIn("unique_user_marker_xyz", masked_text)


if __name__ == "__main__":
    unittest.main()
