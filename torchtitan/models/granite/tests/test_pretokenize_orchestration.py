"""Tests for pretokenize_sft.py orchestration: file discovery, assignment,
crash handling, blank-line tolerance, and nested directory support.
"""

import json
import tempfile
import unittest
from pathlib import Path

from datasets import load_from_disk

from torchtitan.models.granite.scripts.pretokenize_sft import (
    _RUN_CONFIG_FILENAME,
    _completed_stems,
    _process_file,
    _shard_stem,
    _write_json_atomic,
)
from torchtitan.models.granite.tokenization_strategies import TruncateLastStrategy

_REPO_ROOT = Path(__file__).parents[4]
_TEST_TOKENIZER_PATH = str(_REPO_ROOT / "tests" / "assets" / "tokenizer")


def _valid_sample():
    return {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    }


def _invalid_sample_no_assistant():
    return {
        "messages": [
            {"role": "user", "content": "hello"},
        ]
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestProcessFileCrashWritesStats(unittest.TestCase):
    def test_missing_messages_key_writes_error_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            jsonl_path = input_dir / "bad.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({"no_messages_key": []}) + "\n")

            strategy = TruncateLastStrategy(
                _TEST_TOKENIZER_PATH, failures_path=str(output_dir / "failures.jsonl")
            )

            _process_file(
                jsonl_path,
                output_dir,
                strategy,
                shard_stem="bad",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            shards_dir = output_dir / "shards"
            stats_path = shards_dir / "bad_stats.json"
            self.assertTrue(stats_path.exists())

            with open(stats_path) as f:
                stats = json.load(f)
            self.assertTrue(stats["skipped"])
            self.assertIn("error", stats)
            self.assertIn("KeyError", stats["error"])

            shard_dir = shards_dir / "bad"
            self.assertFalse(shard_dir.exists())


class TestProcessFileBlankLines(unittest.TestCase):
    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            jsonl_path = input_dir / "data.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(_valid_sample()) + "\n")
                f.write("\n")
                f.write("   \n")
                f.write(json.dumps(_valid_sample()) + "\n")
                f.write("\n")

            strategy = TruncateLastStrategy(
                _TEST_TOKENIZER_PATH, failures_path=str(output_dir / "failures.jsonl")
            )

            _process_file(
                jsonl_path,
                output_dir,
                strategy,
                shard_stem="data",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            shards_dir = output_dir / "shards"
            shard_dir = shards_dir / "data"
            self.assertTrue(shard_dir.exists())

            ds = load_from_disk(str(shard_dir))
            self.assertEqual(len(ds), 2)


class TestProcessFileAllDropped(unittest.TestCase):
    def test_all_invalid_writes_skipped_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            jsonl_path = input_dir / "invalid.jsonl"
            _write_jsonl(jsonl_path, [_invalid_sample_no_assistant()] * 3)

            strategy = TruncateLastStrategy(
                _TEST_TOKENIZER_PATH, failures_path=str(output_dir / "failures.jsonl")
            )

            _process_file(
                jsonl_path,
                output_dir,
                strategy,
                shard_stem="invalid",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            shards_dir = output_dir / "shards"
            stats_path = shards_dir / "invalid_stats.json"
            self.assertTrue(stats_path.exists())

            with open(stats_path) as f:
                stats = json.load(f)
            self.assertTrue(stats["skipped"])
            self.assertEqual(stats["n_dropped"], 3)
            self.assertEqual(stats["n_examples"], 0)

            shard_dir = shards_dir / "invalid"
            self.assertFalse(shard_dir.exists())


class TestNestedDirStructureEndToEnd(unittest.TestCase):
    def test_discovery_and_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"

            # Create nested structure
            (input_dir / "sub_a").mkdir(parents=True)
            (input_dir / "sub_b" / "deep").mkdir(parents=True)

            _write_jsonl(input_dir / "sub_a" / "train.jsonl", [_valid_sample()] * 2)
            _write_jsonl(
                input_dir / "sub_b" / "deep" / "data.jsonl", [_valid_sample()] * 3
            )

            # Output dir is nested inside input dir (realistic scenario)
            output_dir = input_dir / "pretok_test"
            output_dir.mkdir()
            _write_jsonl(output_dir / "failures.jsonl", [{"error": "test"}])

            # Sibling pretok dir from a prior run (has _RUN_CONFIG_FILENAME marker)
            prior_dir = input_dir / "pretok_prior"
            prior_dir.mkdir()
            _write_jsonl(prior_dir / "failures.jsonl", [{"error": "old"}])
            (prior_dir / _RUN_CONFIG_FILENAME).write_text("{}")

            # File discovery excludes current output_dir AND prior pretok dirs
            pretok_dirs = {
                p.parent for p in input_dir.rglob(_RUN_CONFIG_FILENAME)
            }
            pretok_dirs.add(output_dir)
            input_files = sorted(
                f
                for f in input_dir.rglob("*.jsonl")
                if not any(f.is_relative_to(d) for d in pretok_dirs)
            )

            self.assertEqual(len(input_files), 2)
            self.assertTrue(
                all("failures" not in f.name for f in input_files)
            )

            # Shard stem derivation
            stems = [_shard_stem(f, input_dir) for f in input_files]
            self.assertIn("sub_a__train", stems)
            self.assertIn("sub_b__deep__data", stems)
            self.assertEqual(len(stems), len(set(stems)))

            # Process all files
            strategy = TruncateLastStrategy(
                _TEST_TOKENIZER_PATH,
                failures_path=str(output_dir / "failures.jsonl"),
            )

            for f in input_files:
                _process_file(
                    f,
                    output_dir,
                    strategy,
                    shard_stem=_shard_stem(f, input_dir),
                    input_dir=input_dir,
                    num_cpus=1,
                    batch_size=10,
                    rank=0,
                )

            # All shards written and loadable
            shards_dir = output_dir / "shards"
            for stem in stems:
                shard_path = shards_dir / stem
                self.assertTrue(shard_path.exists(), f"Missing shard: {stem}")
                ds = load_from_disk(str(shard_path))
                self.assertGreater(len(ds), 0)
                self.assertIn("input_ids", ds.column_names)
                self.assertIn("labels", ds.column_names)

            # _completed_stems sees all
            completed = _completed_stems(shards_dir)
            self.assertEqual(completed, set(stems))

    def test_shard_stem_collision_detection(self):
        input_dir = Path("/fake/input")
        f1 = input_dir / "a" / "b__c.jsonl"
        f2 = input_dir / "a__b" / "c.jsonl"
        stem1 = _shard_stem(f1, input_dir)
        stem2 = _shard_stem(f2, input_dir)
        # Both produce "a__b__c" — collision. The main script validates this
        # before processing; here we just verify the stems match so the
        # collision check would catch it.
        self.assertEqual(stem1, stem2)


class TestStaticFileAssignment(unittest.TestCase):
    def test_complete_coverage_no_overlap(self):
        files = [Path(f"file_{i}.jsonl") for i in range(9)]
        world_size = 4

        assigned: list[list[Path]] = []
        for rank in range(world_size):
            assigned.append(files[rank::world_size])

        # Complete coverage
        all_assigned = [f for rank_files in assigned for f in rank_files]
        self.assertEqual(sorted(all_assigned), sorted(files))

        # No overlap
        self.assertEqual(len(all_assigned), len(set(all_assigned)))

    def test_deterministic_across_calls(self):
        files = [Path(f"file_{i}.jsonl") for i in range(9)]
        for rank in range(4):
            a = files[rank::4]
            b = files[rank::4]
            self.assertEqual(a, b)


class TestCompletedStemsSkipsExisting(unittest.TestCase):
    def test_completed_stems_returns_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards_dir = Path(tmp) / "shards"
            shards_dir.mkdir()

            _write_json_atomic(
                shards_dir / "sub_a__train_stats.json", {"n_examples": 10}
            )
            _write_json_atomic(
                shards_dir / "sub_b__data_stats.json", {"n_examples": 5}
            )

            completed = _completed_stems(shards_dir)
            self.assertEqual(completed, {"sub_a__train", "sub_b__data"})

    def test_skip_logic_filters_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards_dir = Path(tmp) / "shards"
            shards_dir.mkdir()

            input_dir = Path("/fake")
            all_files = [
                input_dir / "sub_a" / "train.jsonl",
                input_dir / "sub_b" / "data.jsonl",
                input_dir / "sub_c" / "new.jsonl",
            ]

            # Mark first two as completed
            _write_json_atomic(
                shards_dir / "sub_a__train_stats.json", {"n_examples": 10}
            )
            _write_json_atomic(
                shards_dir / "sub_b__data_stats.json", {"n_examples": 5}
            )

            completed = _completed_stems(shards_dir)
            remaining = [
                f for f in all_files if _shard_stem(f, input_dir) not in completed
            ]
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].name, "new.jsonl")

    def test_empty_shards_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards_dir = Path(tmp) / "nonexistent"
            self.assertEqual(_completed_stems(shards_dir), set())


class TestTotalTrainedTokensStats(unittest.TestCase):
    def test_total_trained_tokens_matches_label_count(self):
        """Verify total_trained_tokens stat counts non-IGNORE_INDEX labels correctly.

        This exercises the pc.list_flatten path that replaced the overflow-prone
        combine_chunks().flatten() call.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            samples = [_valid_sample() for _ in range(5)]
            jsonl_path = input_dir / "data.jsonl"
            _write_jsonl(jsonl_path, samples)

            strategy = TruncateLastStrategy(
                _TEST_TOKENIZER_PATH,
                failures_path=str(output_dir / "failures.jsonl"),
            )

            _process_file(
                jsonl_path,
                output_dir,
                strategy,
                shard_stem="data",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            shards_dir = output_dir / "shards"
            stats_path = shards_dir / "data_stats.json"
            self.assertTrue(stats_path.exists())

            with open(stats_path) as f:
                stats = json.load(f)

            ds = load_from_disk(str(shards_dir / "data"))
            expected_trained = sum(
                sum(1 for lbl in row if lbl != -100) for row in ds["labels"]
            )
            self.assertEqual(stats["total_trained_tokens"], expected_trained)
            self.assertGreater(expected_trained, 0)


if __name__ == "__main__":
    unittest.main()
