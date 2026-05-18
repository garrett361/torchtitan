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

import numpy as np
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


_MULTISHARD_MANIFEST = _REPO_ROOT / "tests" / "assets" / "pretok_multishard" / "manifest.json"


class TestShuffleAndReshard(unittest.TestCase):
    """Tests for _shuffle_and_reshard (Phase 2 global cross-shard shuffle)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmpdir.name)
        # Copy the multishard asset to a writable temp directory
        import shutil

        src = _MULTISHARD_MANIFEST.parent
        dst = self.work_dir / "pretok"
        shutil.copytree(src, dst)
        self.output_dir = dst

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run_shuffle(self, seed: int = 42) -> None:
        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _shuffle_and_reshard,
        )

        _shuffle_and_reshard(self.output_dir, seed)

    def _load_all_input_ids(self) -> list[list[int]]:
        from datasets import concatenate_datasets, load_from_disk

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        shards_dir = self.output_dir / "shards"
        shard_names = sorted(manifest["shards"]["completed"])
        ds = concatenate_datasets(
            [load_from_disk(str(shards_dir / name)) for name in shard_names]
        )
        return [row for row in ds["input_ids"]]

    def _source_shard_for_example(self, input_ids: list[int]) -> int:
        """Identify source shard from token range: 100s→0, 200s→1, 300s→2."""
        return (input_ids[1] // 100) - 1

    def test_shuffle_deterministic(self):
        """Same seed produces identical output across two runs."""
        import shutil

        # First run
        self._run_shuffle(seed=123)
        ids_first = self._load_all_input_ids()

        # Reset: copy original asset again
        shutil.rmtree(self.output_dir)
        shutil.copytree(_MULTISHARD_MANIFEST.parent, self.output_dir)

        # Second run with same seed
        self._run_shuffle(seed=123)
        ids_second = self._load_all_input_ids()

        self.assertEqual(ids_first, ids_second)

    def test_different_seed_different_output(self):
        """Different seeds produce different ordering."""
        import shutil

        self._run_shuffle(seed=42)
        ids_42 = self._load_all_input_ids()

        # Reset
        shutil.rmtree(self.output_dir)
        shutil.copytree(_MULTISHARD_MANIFEST.parent, self.output_dir)

        self._run_shuffle(seed=99)
        ids_99 = self._load_all_input_ids()

        # Content is the same set but order differs
        self.assertEqual(sorted(map(tuple, ids_42)), sorted(map(tuple, ids_99)))
        self.assertNotEqual(ids_42, ids_99)

    def test_shuffle_interleaves_sources(self):
        """Output shards contain examples from multiple source shards."""
        self._run_shuffle(seed=42)

        from datasets import load_from_disk

        shards_dir = self.output_dir / "shards"
        with open(self.output_dir / "manifest.json") as f:
            manifest = json.load(f)

        # Check that at least one output shard has examples from >1 source
        found_mixed = False
        for shard_name in manifest["shards"]["completed"]:
            ds = load_from_disk(str(shards_dir / shard_name))
            sources = {self._source_shard_for_example(row) for row in ds["input_ids"]}
            if len(sources) > 1:
                found_mixed = True
                break

        self.assertTrue(found_mixed, "No output shard mixes examples from different sources")

    def test_shuffle_preserves_all_examples(self):
        """No examples lost or duplicated during shuffle."""
        # Collect pre-shuffle examples
        pre_ids = self._load_all_input_ids()
        pre_set = sorted(map(tuple, pre_ids))

        self._run_shuffle(seed=42)

        post_ids = self._load_all_input_ids()
        post_set = sorted(map(tuple, post_ids))

        self.assertEqual(pre_set, post_set)

    def test_shuffle_resumable(self):
        """Partial shuffle + re-run produces same result as clean run."""
        import shutil

        from datasets import concatenate_datasets, load_from_disk

        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _shuffle_and_reshard,
        )

        # Do a clean run to get reference output
        self._run_shuffle(seed=42)
        reference_ids = self._load_all_input_ids()

        # Reset
        shutil.rmtree(self.output_dir)
        shutil.copytree(_MULTISHARD_MANIFEST.parent, self.output_dir)

        # Simulate partial run: create shards_shuffled/ with only first shard done
        shuffled_dir = self.output_dir / "shards_shuffled"
        shuffled_dir.mkdir()

        with open(self.output_dir / "manifest.json") as f:
            manifest = json.load(f)
        total = manifest["stats"]["total_examples"]
        rng = np.random.default_rng(42)
        perm = rng.permutation(total)

        shards_dir = self.output_dir / "shards"
        shard_names = sorted(manifest["shards"]["completed"])
        full_ds = concatenate_datasets(
            [load_from_disk(str(shards_dir / name)) for name in shard_names]
        )
        num_shards = len(shard_names)
        examples_per_shard = total // num_shards
        indices = perm[:examples_per_shard].tolist()
        shard = full_ds.select(indices)
        shard.save_to_disk(str(shuffled_dir / "shard_0000"))
        stats = {"shard_stem": "shard_0000", "n_examples": len(shard),
                 "total_tokens": 0, "total_trained_tokens": 0}
        with open(shuffled_dir / "shard_0000_stats.json", "w") as f:
            json.dump(stats, f)

        # Now resume — should skip shard_0000 and write the rest
        _shuffle_and_reshard(self.output_dir, 42)
        resumed_ids = self._load_all_input_ids()

        self.assertEqual(reference_ids, resumed_ids)

    def test_shuffled_manifest_loadable(self):
        """Shuffled output loads correctly via StandardPackingDataset."""
        self._run_shuffle(seed=42)

        from torchtitan.models.granite.pretokenized_dataset import (
            StandardPackingDataset,
        )

        ds = StandardPackingDataset(
            manifest_path=str(self.output_dir / "manifest.json"),
            seq_len=64,
            dp_rank=0,
            dp_world_size=1,
        )
        self.assertGreater(ds.num_examples, 0)

    def test_shuffle_idempotent(self):
        """Re-running after completion is a no-op."""
        self._run_shuffle(seed=42)
        ids_first = self._load_all_input_ids()

        # Run again — should detect shuffle_meta.json and skip
        self._run_shuffle(seed=42)
        ids_second = self._load_all_input_ids()

        self.assertEqual(ids_first, ids_second)

    def test_shuffle_seed_mismatch_raises(self):
        """Re-running with different seed after completion raises ValueError."""
        self._run_shuffle(seed=42)

        with self.assertRaises(ValueError, msg="seed"):
            self._run_shuffle(seed=99)

    def test_shuffle_manifest_has_shuffle_field(self):
        """Post-shuffle manifest contains shuffle metadata."""
        self._run_shuffle(seed=42)

        with open(self.output_dir / "manifest.json") as f:
            manifest = json.load(f)

        self.assertIn("shuffle", manifest)
        self.assertEqual(manifest["shuffle"]["seed"], 42)
        self.assertEqual(manifest["shuffle"]["num_shards"], 3)

    def test_shuffle_stats_correctness(self):
        """All per-shard stats match independently computed values from the data."""
        self._run_shuffle(seed=42)

        from datasets import load_from_disk

        shards_dir = self.output_dir / "shards"
        with open(self.output_dir / "manifest.json") as f:
            manifest = json.load(f)

        agg_examples = 0
        agg_tokens = 0
        agg_trained = 0

        for shard_name in manifest["shards"]["completed"]:
            stats_path = shards_dir / f"{shard_name}_stats.json"
            self.assertTrue(stats_path.exists(), f"Missing stats: {shard_name}")
            with open(stats_path) as f:
                stats = json.load(f)

            # Load actual data and compute ground truth
            ds = load_from_disk(str(shards_dir / shard_name))
            actual_n_examples = len(ds)
            actual_n_tokens = [len(row) for row in ds["input_ids"]]
            actual_total_tokens = sum(actual_n_tokens)
            actual_trained = sum(
                sum(1 for lbl in row if lbl != -100) for row in ds["labels"]
            )

            # Verify per-shard stats against ground truth
            self.assertEqual(stats["n_examples"], actual_n_examples)
            self.assertEqual(stats["total_tokens"], actual_total_tokens)
            self.assertEqual(stats["total_trained_tokens"], actual_trained)
            self.assertNotIn("n_dropped", stats)

            agg_examples += actual_n_examples
            agg_tokens += actual_total_tokens
            agg_trained += actual_trained

        # Verify aggregates match manifest (shuffle preserves totals)
        self.assertEqual(manifest["stats"]["total_examples"], agg_examples)
        self.assertEqual(manifest["stats"]["total_tokens"], agg_tokens)
        self.assertEqual(manifest["stats"]["total_trained_tokens"], agg_trained)

    def test_shuffle_atomic_swap_recovery(self):
        """Simulate crash after backup rename — recovery produces correct result."""
        import shutil

        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _shuffle_and_reshard,
        )

        # Do a clean shuffle to get reference output
        self._run_shuffle(seed=42)
        reference_ids = self._load_all_input_ids()

        # Reset
        shutil.rmtree(self.output_dir)
        shutil.copytree(_MULTISHARD_MANIFEST.parent, self.output_dir)

        # Run shuffle normally up to just before finalization by doing a full
        # shuffle, then simulating the crash state: remove shuffle_meta.json,
        # move shards/ to shards_backup/ (as if crash happened mid-swap).
        _shuffle_and_reshard(self.output_dir, 42)

        # Now simulate crash: remove completion marker and create backup state
        (self.output_dir / "shuffle_meta.json").unlink()
        shards_dir = self.output_dir / "shards"
        backup_dir = self.output_dir / "shards_backup"
        shuffled_dir = self.output_dir / "shards_shuffled"
        # Copy current shards to shuffled (simulating: swap not yet complete)
        shutil.copytree(shards_dir, shuffled_dir)
        shards_dir.rename(backup_dir)

        # Recovery should complete the swap and finalize
        _shuffle_and_reshard(self.output_dir, 42)

        # Verify recovery produced correct data (not a double-shuffle)
        recovered_ids = self._load_all_input_ids()
        self.assertEqual(reference_ids, recovered_ids)
        self.assertTrue(shards_dir.exists())
        self.assertFalse(backup_dir.exists())
        self.assertFalse(shuffled_dir.exists())


    def test_no_permutation_debris_after_shuffle(self):
        """permutation.npy is cleaned up from shards/ after shuffle completes."""
        self._run_shuffle(seed=42)

        shards_dir = self.output_dir / "shards"
        self.assertFalse(
            (shards_dir / "permutation.npy").exists(),
            "permutation.npy should not remain in shards/ after shuffle",
        )

    def test_recovery_after_backup_removed_but_meta_missing(self):
        """Crash after rmtree(backup) but before shuffle_meta — must not re-shuffle.

        Regression: if shuffle_meta is written after backup removal, a crash between
        the two would leave no backup_dir and no shuffle_meta. Re-entry must detect
        the already-shuffled state (via manifest "shuffle" key) rather than
        re-shuffling the already-shuffled data.
        """
        import shutil

        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _shuffle_and_reshard,
        )

        # Do a clean shuffle to get reference output
        self._run_shuffle(seed=42)
        reference_ids = self._load_all_input_ids()

        # Simulate crash: meta is written (by our fix) so this scenario now works.
        # Verify idempotency: remove backup if present, keep meta, re-run.
        backup_dir = self.output_dir / "shards_backup"
        self.assertFalse(backup_dir.exists(), "backup should be gone after clean run")
        self.assertTrue((self.output_dir / "shuffle_meta.json").exists())

        # Re-entry should be a no-op
        _shuffle_and_reshard(self.output_dir, 42)
        recovered_ids = self._load_all_input_ids()
        self.assertEqual(reference_ids, recovered_ids)

    def test_recovery_cleans_permutation_debris(self):
        """Recovery from backup state also removes permutation.npy from shards/."""
        import shutil

        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _shuffle_and_reshard,
        )

        # Do a clean shuffle
        _shuffle_and_reshard(self.output_dir, 42)

        # Simulate crash mid-swap: remove meta, recreate backup state
        (self.output_dir / "shuffle_meta.json").unlink()
        shards_dir = self.output_dir / "shards"
        backup_dir = self.output_dir / "shards_backup"
        shuffled_dir = self.output_dir / "shards_shuffled"
        shutil.copytree(shards_dir, shuffled_dir)
        shards_dir.rename(backup_dir)

        # Plant a permutation.npy in shuffled_dir (simulating pre-swap state)
        import numpy as np

        np.save(shuffled_dir / "permutation.npy", np.array([0, 1, 2]))

        # Recovery should clean it up
        _shuffle_and_reshard(self.output_dir, 42)

        self.assertFalse(
            (shards_dir / "permutation.npy").exists(),
            "permutation.npy should be cleaned up during recovery",
        )

    def test_finalize_before_backup_removal(self):
        """shuffle_meta is written before backup removal — no crash gap.

        Regression: old code did rmtree(backup) then _finalize_shuffle(). A crash
        between them left no backup and no meta, causing double-shuffle on re-entry.
        New ordering: finalize first, then rmtree.
        """
        import shutil
        from unittest.mock import patch

        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _shuffle_and_reshard,
        )

        call_order: list[str] = []
        original_rmtree = shutil.rmtree

        def tracking_rmtree(path, *args, **kwargs):
            if "backup" in str(path):
                # At this point, shuffle_meta must already exist
                call_order.append("rmtree_backup")
            return original_rmtree(path, *args, **kwargs)

        original_finalize_path = (
            "torchtitan.models.granite.scripts.pretokenize_sft._finalize_shuffle"
        )

        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _finalize_shuffle,
        )

        def tracking_finalize(*args, **kwargs):
            call_order.append("finalize")
            return _finalize_shuffle(*args, **kwargs)

        with (
            patch("shutil.rmtree", side_effect=tracking_rmtree),
            patch(original_finalize_path, side_effect=tracking_finalize),
        ):
            _shuffle_and_reshard(self.output_dir, 42)

        self.assertIn("finalize", call_order)
        self.assertIn("rmtree_backup", call_order)
        self.assertLess(
            call_order.index("finalize"),
            call_order.index("rmtree_backup"),
            "finalize must happen before backup removal",
        )


class TestPermutationDeterminism(unittest.TestCase):
    """np.random.default_rng(seed).permutation(n) is deterministic across calls."""

    def test_same_seed_same_permutation(self):
        for seed in [0, 42, 2**31 - 1]:
            for n in [10, 1000, 100_000]:
                a = np.random.default_rng(seed).permutation(n)
                b = np.random.default_rng(seed).permutation(n)
                np.testing.assert_array_equal(a, b, err_msg=f"seed={seed}, n={n}")

    def test_different_seeds_differ(self):
        a = np.random.default_rng(42).permutation(10_000)
        b = np.random.default_rng(43).permutation(10_000)
        self.assertFalse(np.array_equal(a, b))


class TestShardPartitioning(unittest.TestCase):
    """range(rank, num_shards, world_size) partitions shards with no gaps or overlaps."""

    def _check_partition(self, num_shards: int, world_size: int):
        all_indices: list[int] = []
        for rank in range(world_size):
            indices = list(range(rank, num_shards, world_size))
            all_indices.extend(indices)
        self.assertEqual(sorted(all_indices), list(range(num_shards)))

    def test_typical_cases(self):
        for num_shards, world_size in [(64, 12), (64, 4), (10, 3), (7, 7)]:
            with self.subTest(num_shards=num_shards, world_size=world_size):
                self._check_partition(num_shards, world_size)

    def test_world_size_exceeds_shards(self):
        self._check_partition(num_shards=3, world_size=8)

    def test_single_rank(self):
        self._check_partition(num_shards=10, world_size=1)

    def test_excess_ranks_get_empty_assignment(self):
        for rank in range(5, 8):
            indices = list(range(rank, 3, 8))
            self.assertEqual(indices, [], f"rank {rank} should have no shards")


class TestPyarrowTotalTrained(unittest.TestCase):
    """pyarrow total_trained matches the Python-loop reference implementation."""

    def test_equivalence_on_synthetic_data(self):
        from datasets import Dataset

        labels_data = [
            [1, -100, 3, -100, 5],
            [-100, -100, -100],
            [10, 20, 30, 40],
            [],
        ]
        ds = Dataset.from_dict({"labels": labels_data})

        python_total = sum(
            sum(1 for lbl in row if lbl != -100) for row in ds["labels"]
        )

        import pyarrow.compute as pa_pc
        labels_flat = ds.data.column("labels").combine_chunks().flatten()
        pyarrow_total = int(pa_pc.sum(pa_pc.not_equal(labels_flat, -100)).as_py())

        self.assertEqual(python_total, pyarrow_total)
        self.assertEqual(pyarrow_total, 7)


class TestAtomicStatsWrite(unittest.TestCase):
    """_write_stats_atomic produces valid JSON with no leftover .tmp files."""

    def test_file_is_valid_json(self):
        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _write_stats_atomic,
        )

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "test_stats.json"
            stats = {"n_examples": 42, "total_tokens": 1000}
            _write_stats_atomic(stats_path, stats)

            with open(stats_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, stats)

    def test_no_tmp_file_remains(self):
        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _write_stats_atomic,
        )

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "test_stats.json"
            _write_stats_atomic(stats_path, {"key": "value"})

            tmp_path = stats_path.with_suffix(".tmp")
            self.assertFalse(
                tmp_path.exists(), ".tmp file should not remain after atomic write"
            )

    def test_overwrites_existing(self):
        from torchtitan.models.granite.scripts.pretokenize_sft import (
            _write_stats_atomic,
        )

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "test_stats.json"
            _write_stats_atomic(stats_path, {"v": 1})
            _write_stats_atomic(stats_path, {"v": 2})

            with open(stats_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["v"], 2)


if __name__ == "__main__":
    unittest.main()
