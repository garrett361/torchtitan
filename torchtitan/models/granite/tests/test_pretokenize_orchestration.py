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
    _write_manifest,
)
from torchtitan.models.granite.tokenization_strategies import (
    BackboneSuffixStrategy,
    TruncateEveryTurnStrategy,
    TruncateLastStrategy,
)

import os

from dotenv import load_dotenv

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")

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


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
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

            _process_file(
                jsonl_path,
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
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


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
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

            _process_file(
                jsonl_path,
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
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


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
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

            _process_file(
                jsonl_path,
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
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


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
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
            for f in input_files:
                _process_file(
                    f,
                    output_dir,
                    TruncateLastStrategy,
                    _HF_ASSETS_PATH,
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


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
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

            _process_file(
                jsonl_path,
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
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


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestProcessFileDropStats(unittest.TestCase):
    """Verify n_dropped is computed from actual failures, not n_lines - n_examples."""

    def test_n_dropped_counts_actual_failures(self):
        """Mix of valid + invalid → n_dropped equals invalid count."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            valid = _valid_sample()
            invalid = _invalid_sample_no_assistant()
            _write_jsonl(input_dir / "mixed.jsonl", [valid, invalid, valid, invalid])

            _process_file(
                input_dir / "mixed.jsonl",
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
                shard_stem="mixed",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            with open(output_dir / "shards" / "mixed_stats.json") as f:
                stats = json.load(f)
            self.assertEqual(stats["n_dropped"], 2)
            self.assertEqual(stats["n_examples"], 2)
            self.assertEqual(stats["n_input_conversations"], 4)

    def test_n_dropped_zero_when_all_valid(self):
        """All valid inputs → n_dropped == 0, no failures file."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            _write_jsonl(input_dir / "good.jsonl", [_valid_sample()] * 3)

            _process_file(
                input_dir / "good.jsonl",
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
                shard_stem="good",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            with open(output_dir / "shards" / "good_stats.json") as f:
                stats = json.load(f)
            self.assertEqual(stats["n_dropped"], 0)
            self.assertEqual(stats["n_examples"], 3)
            failures_path = output_dir / "shards" / "good_failures.jsonl"
            self.assertFalse(failures_path.exists())

    @unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
    def test_expanding_strategy_n_dropped_not_negative(self):
        """TruncateEveryTurnStrategy expands → n_dropped >= 0, n_examples > n_input."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            multi_turn = {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A", "reasoning_content": "R"},
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
                    {"role": "user", "content": "Q3"},
                    {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
                ]
            }
            _write_jsonl(input_dir / "expand.jsonl", [multi_turn] * 2)

            _process_file(
                input_dir / "expand.jsonl",
                output_dir,
                TruncateEveryTurnStrategy,
                _HF_ASSETS_PATH,
                shard_stem="expand",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            with open(output_dir / "shards" / "expand_stats.json") as f:
                stats = json.load(f)
            self.assertEqual(stats["n_dropped"], 0)
            self.assertEqual(stats["n_input_conversations"], 2)
            # 2 conversations × 3 assistant turns each = 6 examples
            self.assertEqual(stats["n_examples"], 6)
            self.assertGreater(stats["n_examples"], stats["n_input_conversations"])

    def test_n_input_conversations_equals_input_lines(self):
        """n_input_conversations matches non-blank JSONL line count."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            jsonl_path = input_dir / "data.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(_valid_sample()) + "\n")
                f.write("\n")  # blank line
                f.write(json.dumps(_valid_sample()) + "\n")
                f.write("   \n")  # whitespace-only line
                f.write(json.dumps(_valid_sample()) + "\n")

            _process_file(
                jsonl_path,
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
                shard_stem="data",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            with open(output_dir / "shards" / "data_stats.json") as f:
                stats = json.load(f)
            # 3 non-blank lines
            self.assertEqual(stats["n_input_conversations"], 3)

    def test_all_dropped_n_dropped_equals_n_lines(self):
        """All invalid inputs → n_dropped == n_lines, shard skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            _write_jsonl(
                input_dir / "bad.jsonl", [_invalid_sample_no_assistant()] * 3
            )

            _process_file(
                input_dir / "bad.jsonl",
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
                shard_stem="bad",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            with open(output_dir / "shards" / "bad_stats.json") as f:
                stats = json.load(f)
            self.assertTrue(stats["skipped"])
            self.assertEqual(stats["n_dropped"], 3)
            self.assertEqual(stats["n_input_conversations"], 3)

    def test_per_shard_failures_file_created(self):
        """Failures file is per-shard and its line count matches n_dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            valid = _valid_sample()
            invalid = _invalid_sample_no_assistant()
            _write_jsonl(input_dir / "data.jsonl", [valid, invalid, invalid, valid])

            _process_file(
                input_dir / "data.jsonl",
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
                shard_stem="data",
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )

            failures_path = output_dir / "shards" / "data_failures.jsonl"
            self.assertTrue(failures_path.exists())
            with open(failures_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

            with open(output_dir / "shards" / "data_stats.json") as f:
                stats = json.load(f)
            self.assertEqual(stats["n_dropped"], len(lines))


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestBackboneSuffixMixedBatchSchema(unittest.TestCase):
    """Regression: empty suffix_starts in early batches must not cause a
    PyArrow schema inference failure when later batches contain non-empty lists.
    Datasets predominantly composed of single-turn conversations (no suffix
    groups) with rare multi-turn samples trigger this when num_proc > 1.
    """

    def test_single_turn_then_multi_turn_with_reasoning(self):
        """When all samples in a worker's first batch are single-turn,
        suffix_starts and insertion_limits are empty lists. Without explicit
        features= in Dataset.map(), HF Datasets infers these as list<null>.
        A later batch containing multi-turn samples with reasoning produces
        non-empty int lists, triggering 'Couldn't cast array of type int64
        to null'. This test reproduces that ordering: 8 single-turn samples
        (filling the first batches at batch_size=4) followed by 2 multi-turn
        samples with reasoning_content that generate suffix groups.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            single_turn = {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
            multi_turn_with_reasoning = {
                "messages": [
                    {"role": "user", "content": "Solve step by step: 17*23"},
                    {
                        "role": "assistant",
                        "reasoning_content": "17*23 = 17*20 + 17*3 = 340 + 51 = 391",
                        "content": "391",
                    },
                    {"role": "user", "content": "Now multiply by 2"},
                    {
                        "role": "assistant",
                        "reasoning_content": "391*2 = 782",
                        "content": "782",
                    },
                ]
            }
            # Many single-turn samples first so batch_size=4 sees only empty
            # suffix_starts in the first batch, then a multi-turn sample later.
            rows = [single_turn] * 8 + [multi_turn_with_reasoning] * 2
            _write_jsonl(input_dir / "mixed.jsonl", rows)

            _process_file(
                input_dir / "mixed.jsonl",
                output_dir,
                BackboneSuffixStrategy,
                _HF_ASSETS_PATH,
                shard_stem="mixed",
                input_dir=input_dir,
                num_cpus=2,
                batch_size=4,
                rank=0,
            )

            ds = load_from_disk(str(output_dir / "shards" / "mixed"))
            self.assertEqual(len(ds), 10)
            n_with_suffix = sum(
                1 for row in ds if len(row["suffix_starts"]) > 0
            )
            self.assertGreater(n_with_suffix, 0)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestWriteManifestPerFileStats(unittest.TestCase):
    """Verify _write_manifest writes per_file_stats.json alongside the manifest."""

    def _setup_shards(self, tmp_path: Path):
        """Process a mix of valid and invalid files, returning (output_dir, input_files)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Valid file
        _write_jsonl(input_dir / "good.jsonl", [_valid_sample()] * 3)
        # All-invalid file (will be skipped)
        _write_jsonl(input_dir / "bad.jsonl", [_invalid_sample_no_assistant()] * 2)

        input_files = sorted(input_dir.glob("*.jsonl"))
        for f in input_files:
            _process_file(
                f,
                output_dir,
                TruncateLastStrategy,
                _HF_ASSETS_PATH,
                shard_stem=_shard_stem(f, input_dir),
                input_dir=input_dir,
                num_cpus=1,
                batch_size=10,
                rank=0,
            )
        return input_dir, output_dir, input_files

    def test_per_file_stats_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir, output_dir, input_files = self._setup_shards(Path(tmp))

            result = _write_manifest(
                output_dir, input_dir, input_files,
                "truncate_last", _HF_ASSETS_PATH,
                {"truncate_history_thinking": True},
            )
            self.assertTrue(result)

            per_file_path = output_dir / "per_file_stats.json"
            self.assertTrue(per_file_path.exists())

            with open(per_file_path) as f:
                per_file = json.load(f)
            self.assertIsInstance(per_file, list)
            self.assertEqual(len(per_file), 2)

    def test_per_file_stats_includes_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir, output_dir, input_files = self._setup_shards(Path(tmp))

            _write_manifest(
                output_dir, input_dir, input_files,
                "truncate_last", _HF_ASSETS_PATH,
                {"truncate_history_thinking": True},
            )

            with open(output_dir / "per_file_stats.json") as f:
                per_file = json.load(f)

            skipped = [s for s in per_file if s.get("skipped")]
            non_skipped = [s for s in per_file if not s.get("skipped")]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(len(non_skipped), 1)
            self.assertIn("bad.jsonl", skipped[0]["input_file"])

    def test_per_file_stats_sums_match_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir, output_dir, input_files = self._setup_shards(Path(tmp))

            _write_manifest(
                output_dir, input_dir, input_files,
                "truncate_last", _HF_ASSETS_PATH,
                {"truncate_history_thinking": True},
            )

            with open(output_dir / "manifest.json") as f:
                manifest = json.load(f)
            with open(output_dir / "per_file_stats.json") as f:
                per_file = json.load(f)

            non_skipped = [s for s in per_file if not s.get("skipped")]
            self.assertEqual(
                sum(s["total_trained_tokens"] for s in non_skipped),
                manifest["stats"]["total_trained_tokens"],
            )
            self.assertEqual(
                sum(s["total_tokens"] for s in non_skipped),
                manifest["stats"]["total_tokens"],
            )

    def test_per_file_stats_not_rewritten_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir, output_dir, input_files = self._setup_shards(Path(tmp))

            _write_manifest(
                output_dir, input_dir, input_files,
                "truncate_last", _HF_ASSETS_PATH,
                {"truncate_history_thinking": True},
            )

            # Second call should not rewrite (manifest already exists)
            result = _write_manifest(
                output_dir, input_dir, input_files,
                "truncate_last", _HF_ASSETS_PATH,
                {"truncate_history_thinking": True},
            )
            self.assertFalse(result)
            # per_file_stats.json still present from first write
            self.assertTrue((output_dir / "per_file_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
