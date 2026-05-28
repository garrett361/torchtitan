"""Unit tests for BackboneSuffixDataset packing layer."""

import json
import unittest
from pathlib import Path

import torch
from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX

_EOS_ID = 2003


def _make_backbone_suffix_shard(
    tmp_path: Path,
    examples: list[dict],
    *,
    shard_name: str = "shard_0000",
) -> Path:
    """Write a backbone_suffix Arrow shard + manifest to tmp_path."""
    import pyarrow as pa

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Explicit schema ensures list columns get int32 element type even when
    # all lists are empty (Arrow would infer null type otherwise).
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("labels", pa.list_(pa.int64())),
        ("positions", pa.list_(pa.int32())),
        ("suffix_starts", pa.list_(pa.int32())),
        ("insertion_limits", pa.list_(pa.int32())),
        ("n_tokens", pa.int64()),
    ])
    table = pa.table(
        {
            "input_ids": [ex["input_ids"] for ex in examples],
            "labels": [ex["labels"] for ex in examples],
            "positions": [ex["positions"] for ex in examples],
            "suffix_starts": [ex["suffix_starts"] for ex in examples],
            "insertion_limits": [ex["insertion_limits"] for ex in examples],
            "n_tokens": [ex["n_tokens"] for ex in examples],
        },
        schema=schema,
    )
    ds = Dataset(table)
    ds.save_to_disk(str(shards_dir / shard_name))

    manifest = {
        "version": 1,
        "strategy": "backbone_suffix",
        "tokenizer": {
            "source_path": "tests/assets/tokenizer",
            "vocab_size": 2009,
            "eos_token_id": _EOS_ID,
            "chat_template_sha256": None,
        },
        "chat_template_kwargs": {"truncate_history_thinking": True},
        "input_files": {"total": 1, "paths": [], "skipped": []},
        "shards": {"completed": [shard_name]},
        "stats": {"total_examples": len(examples), "total_tokens": 0, "total_trained_tokens": 0},
        "created_at": "2026-01-01T00:00:00Z",
        "input_dir": "",
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def _example_no_suffix():
    """Backbone-only example (no suffixes): 6 tokens."""
    return {
        "input_ids": [1, 2, 3, 4, 5, _EOS_ID],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, _EOS_ID],
        "positions": [0, 1, 2, 3, 4, 5],
        "suffix_starts": [],
        "insertion_limits": [],
        "n_tokens": 6,
    }


def _example_one_suffix():
    """Backbone (4 tokens) + one suffix (3 tokens) = 7 tokens total.

    Backbone: positions 0-3, suffix: positions 2-4 (insertion_limit=1).
    """
    return {
        "input_ids": [10, 11, 12, _EOS_ID, 20, 21, _EOS_ID],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, 12, _EOS_ID, IGNORE_INDEX, 21, _EOS_ID],
        "positions": [0, 1, 2, 3, 2, 3, 4],
        "suffix_starts": [4],
        "insertion_limits": [1],
        "n_tokens": 7,
    }


def _example_two_suffixes():
    """Backbone (5 tokens) + suffix_0 (3 tokens) + suffix_1 (2 tokens) = 10 tokens.

    suffix_0: insertion_limit=1, suffix_1: insertion_limit=3.
    """
    return {
        "input_ids": [30, 31, 32, 33, _EOS_ID, 40, 41, _EOS_ID, 50, _EOS_ID],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 33, _EOS_ID,
                   IGNORE_INDEX, 41, _EOS_ID, 50, _EOS_ID],
        "positions": [0, 1, 2, 3, 4, 2, 3, 4, 4, 5],
        "suffix_starts": [5, 8],
        "insertion_limits": [1, 3],
        "n_tokens": 10,
    }


class TestBackboneSuffixDatasetOutputShape(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def _make_dataset(self, examples, seq_len=32, buffer_size=4):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=buffer_size
        )

    def test_output_keys(self):
        """Batch dict has all required keys for backbone_suffix mask."""
        ds = self._make_dataset([_example_one_suffix()])
        batch_dict, labels, stats = next(iter(ds))
        for key in ("input", "positions", "conv_ids", "suffix_ids", "insertion_limits", "attn_cost"):
            self.assertIn(key, batch_dict, f"missing key: {key}")
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIn("n_total_tokens", stats)

    def test_all_tensors_have_seq_len_shape(self):
        """All per-position tensors have shape (seq_len,)."""
        seq_len = 16
        ds = self._make_dataset([_example_one_suffix()], seq_len=seq_len)
        batch_dict, labels, _ = next(iter(ds))
        per_position_keys = ("input", "positions", "conv_ids", "suffix_ids", "insertion_limits")
        for key in per_position_keys:
            self.assertEqual(batch_dict[key].shape, (seq_len,), f"{key} shape mismatch")
        self.assertEqual(labels.shape, (seq_len,))
        self.assertEqual(batch_dict["attn_cost"].shape, (), "attn_cost should be scalar")

    def test_seq_len_drop(self):
        """Examples exceeding seq_len are dropped entirely."""
        seq_len = 5
        big = _example_one_suffix()  # 7 tokens, won't fit
        small = _example_no_suffix()  # 6 tokens, also too big
        tiny = {
            "input_ids": [1, 2, _EOS_ID],
            "labels": [IGNORE_INDEX, 2, _EOS_ID],
            "positions": [0, 1, 2],
            "suffix_starts": [],
            "insertion_limits": [],
            "n_tokens": 3,
        }
        ds = self._make_dataset([big, small, tiny], seq_len=seq_len)
        batches = list(ds)
        self.assertEqual(len(batches), 1)
        # Only the tiny example should survive
        inputs = batches[0][0]["input"].tolist()
        self.assertEqual(inputs[:3], [1, 2, _EOS_ID])

    def test_all_examples_dropped_yields_nothing(self):
        """When all examples exceed seq_len, no batches are produced."""
        seq_len = 3
        ds = self._make_dataset(
            [_example_no_suffix(), _example_one_suffix()], seq_len=seq_len
        )
        batches = list(ds)
        self.assertEqual(len(batches), 0)


class TestBackboneSuffixDatasetConvIds(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def _make_dataset(self, examples, seq_len=32, buffer_size=4):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=buffer_size
        )

    def test_single_example_conv_ids(self):
        """Single example: all real positions get conv_id=1, padding gets 0."""
        seq_len = 16
        ex = _example_one_suffix()  # 7 tokens
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        conv_ids = batch_dict["conv_ids"].tolist()
        self.assertEqual(conv_ids[:7], [1] * 7)
        self.assertEqual(conv_ids[7:], [0] * 9)

    def test_two_packed_examples_get_distinct_conv_ids(self):
        """Two co-packed examples get conv_ids 1 and 2."""
        seq_len = 16
        ex_a = _example_no_suffix()  # 6 tokens
        ex_b = _example_no_suffix()  # 6 tokens, fits together (12 <= 16)
        ds = self._make_dataset([ex_a, ex_b], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        conv_ids = batch_dict["conv_ids"].tolist()
        self.assertEqual(conv_ids[:6], [1] * 6)
        self.assertEqual(conv_ids[6:12], [2] * 6)
        self.assertEqual(conv_ids[12:], [0] * 4)


class TestBackboneSuffixDatasetSuffixIds(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def _make_dataset(self, examples, seq_len=32, buffer_size=4):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=buffer_size
        )

    def test_no_suffix_all_zeros(self):
        """Example with no suffixes: all suffix_ids are 0."""
        seq_len = 16
        ds = self._make_dataset([_example_no_suffix()], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        suffix_ids = batch_dict["suffix_ids"].tolist()
        self.assertEqual(suffix_ids, [0] * seq_len)

    def test_one_suffix_backbone_zero_suffix_nonzero(self):
        """Backbone positions have suffix_id=0, suffix positions have suffix_id=1."""
        seq_len = 16
        ex = _example_one_suffix()  # backbone=4, suffix=3
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        suffix_ids = batch_dict["suffix_ids"].tolist()
        self.assertEqual(suffix_ids[:4], [0, 0, 0, 0])
        self.assertEqual(suffix_ids[4:7], [1, 1, 1])
        self.assertEqual(suffix_ids[7:], [0] * 9)

    def test_two_suffixes_distinct_ids(self):
        """Two suffixes in one example get distinct suffix_ids (1 and 2)."""
        seq_len = 16
        ex = _example_two_suffixes()  # backbone=5, suffix_0=3, suffix_1=2
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        suffix_ids = batch_dict["suffix_ids"].tolist()
        self.assertEqual(suffix_ids[:5], [0] * 5)  # backbone
        self.assertEqual(suffix_ids[5:8], [1, 1, 1])  # suffix_0
        self.assertEqual(suffix_ids[8:10], [2, 2])  # suffix_1
        self.assertEqual(suffix_ids[10:], [0] * 6)  # padding

    def test_two_packed_examples_suffix_ids_unique_per_row(self):
        """Co-packed examples get globally unique suffix_ids within the row."""
        seq_len = 32
        ex_a = _example_one_suffix()  # 7 tokens, 1 suffix
        ex_b = _example_two_suffixes()  # 10 tokens, 2 suffixes
        ds = self._make_dataset([ex_a, ex_b], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        suffix_ids = batch_dict["suffix_ids"].tolist()
        conv_ids = batch_dict["conv_ids"].tolist()
        # All non-zero suffix_ids must be unique per suffix region
        nonzero_ids = set(s for s in suffix_ids if s > 0)
        # 3 total suffixes (1 from ex_a + 2 from ex_b)
        self.assertEqual(len(nonzero_ids), 3)
        # Each distinct suffix_id forms a contiguous run
        for sid in nonzero_ids:
            positions = [i for i, s in enumerate(suffix_ids) if s == sid]
            self.assertEqual(positions, list(range(positions[0], positions[-1] + 1)))
        # Backbone positions of each conv have suffix_id=0
        for conv_id in (1, 2):
            conv_positions = [i for i, c in enumerate(conv_ids) if c == conv_id]
            for i in conv_positions:
                if suffix_ids[i] == 0:
                    continue
                # Once we hit a suffix, all remaining positions in this conv are suffix
                remaining = conv_positions[conv_positions.index(i):]
                self.assertTrue(all(suffix_ids[j] > 0 for j in remaining))


class TestBackboneSuffixDatasetInsertionLimits(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def _make_dataset(self, examples, seq_len=32, buffer_size=4):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=buffer_size
        )

    def test_backbone_insertion_limit_is_end_of_backbone(self):
        """Backbone positions have insertion_limit = offset + backbone_len - 1."""
        seq_len = 16
        ex = _example_one_suffix()  # backbone=4 tokens at offset 0
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        ins_limits = batch_dict["insertion_limits"].tolist()
        # backbone offset=0, backbone_len=4, so limit = 0 + 4 - 1 = 3
        self.assertEqual(ins_limits[:4], [3, 3, 3, 3])

    def test_suffix_insertion_limit_offset_applied(self):
        """Suffix positions get insertion_limit = offset + stored_value."""
        seq_len = 16
        ex = _example_one_suffix()  # insertion_limits=[1], offset=0 → 0+1=1
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        ins_limits = batch_dict["insertion_limits"].tolist()
        # suffix at positions 4-6, insertion_limit = 0 + 1 = 1
        self.assertEqual(ins_limits[4:7], [1, 1, 1])

    def test_padding_insertion_limit_is_negative(self):
        """Padding positions have insertion_limit = -1."""
        seq_len = 16
        ex = _example_one_suffix()  # 7 tokens, 9 padding
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        ins_limits = batch_dict["insertion_limits"].tolist()
        self.assertEqual(ins_limits[7:], [-1] * 9)

    def test_second_example_offset_applied(self):
        """When two examples are co-packed, second example's limits include its offset."""
        seq_len = 32
        ex_a = _example_no_suffix()  # 6 tokens
        ex_b = _example_one_suffix()  # 7 tokens, backbone=4, suffix=3, ins_limit=1
        ds = self._make_dataset([ex_a, ex_b], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        ins_limits = batch_dict["insertion_limits"].tolist()
        conv_ids = batch_dict["conv_ids"].tolist()
        suffix_ids = batch_dict["suffix_ids"].tolist()

        # Determine which example was placed first by checking where the suffix is.
        # ex_b has suffix_ids > 0; ex_a does not.
        suffix_positions = [i for i, s in enumerate(suffix_ids) if s > 0]
        self.assertTrue(suffix_positions, "ex_b must have suffix positions")
        # ex_b's offset is where its conv starts
        ex_b_conv_id = conv_ids[suffix_positions[0]]
        ex_b_start = conv_ids.index(ex_b_conv_id)
        ex_b_backbone_end = suffix_positions[0]  # first suffix pos

        # Verify: backbone limit = offset + backbone_len - 1
        backbone_len = ex_b_backbone_end - ex_b_start
        expected_backbone_limit = ex_b_start + backbone_len - 1
        for j in range(ex_b_start, ex_b_backbone_end):
            self.assertEqual(ins_limits[j], expected_backbone_limit)

        # Verify: suffix limit = offset + stored_insertion_limit (= offset + 1)
        expected_suffix_limit = ex_b_start + 1
        for j in suffix_positions:
            self.assertEqual(ins_limits[j], expected_suffix_limit)

    def test_two_suffixes_different_limits(self):
        """Two suffixes in one example get their respective insertion_limits + offset."""
        seq_len = 16
        ex = _example_two_suffixes()  # insertion_limits=[1, 3], offset=0
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        ins_limits = batch_dict["insertion_limits"].tolist()
        # suffix_0 at [5:8]: limit = 0 + 1 = 1
        self.assertEqual(ins_limits[5:8], [1, 1, 1])
        # suffix_1 at [8:10]: limit = 0 + 3 = 3
        self.assertEqual(ins_limits[8:10], [3, 3])


class TestBackboneSuffixDatasetPositions(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def _make_dataset(self, examples, seq_len=32, buffer_size=4):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=buffer_size
        )

    def test_positions_passthrough(self):
        """Stored positions are used directly (not recomputed)."""
        seq_len = 16
        ex = _example_one_suffix()
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        positions = batch_dict["positions"].tolist()
        self.assertEqual(positions[:7], ex["positions"])

    def test_padding_positions_sequential(self):
        """Padding positions are sequential starting from 0."""
        seq_len = 16
        ex = _example_one_suffix()  # 7 tokens
        ds = self._make_dataset([ex], seq_len=seq_len)
        batch_dict, _, _ = next(iter(ds))
        positions = batch_dict["positions"].tolist()
        self.assertEqual(positions[7:], list(range(9)))


class TestBackboneSuffixDatasetStats(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def _make_dataset(self, examples, seq_len=32, buffer_size=4):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=buffer_size
        )

    def test_stats_token_counts(self):
        """Stats report correct total and trained token counts."""
        seq_len = 16
        ex = _example_one_suffix()
        ds = self._make_dataset([ex], seq_len=seq_len)
        _, _, stats = next(iter(ds))
        self.assertEqual(stats["n_total_tokens"], 7)
        trained = sum(1 for lbl in ex["labels"] if lbl != IGNORE_INDEX)
        self.assertEqual(stats["n_trained_tokens"], trained)
        self.assertEqual(stats["n_examples_packed"], 1)

    def test_stats_two_packed(self):
        """Stats accumulate across co-packed examples."""
        seq_len = 32
        ex_a = _example_no_suffix()  # 6 tokens
        ex_b = _example_one_suffix()  # 7 tokens
        ds = self._make_dataset([ex_a, ex_b], seq_len=seq_len)
        _, _, stats = next(iter(ds))
        self.assertEqual(stats["n_total_tokens"], 6 + 7)
        trained_a = sum(1 for lbl in ex_a["labels"] if lbl != IGNORE_INDEX)
        trained_b = sum(1 for lbl in ex_b["labels"] if lbl != IGNORE_INDEX)
        self.assertEqual(stats["n_trained_tokens"], trained_a + trained_b)
        self.assertEqual(stats["n_examples_packed"], 2)


class TestBackboneSuffixDatasetCheckpointing(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def test_state_dict_roundtrip(self):
        """state_dict → load_state_dict resumes correctly."""
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        seq_len = 16
        # Distinct examples so token content identifies position in iteration
        examples = [
            _example_no_suffix(),
            _example_one_suffix(),
            _example_no_suffix(),
            _example_two_suffixes(),
        ]
        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)

        ds_ref = BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=4
        )
        all_batches = list(ds_ref)
        self.assertGreater(len(all_batches), 1)

        ds_a = BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=4
        )
        it_a = iter(ds_a)
        next(it_a)
        checkpoint = ds_a.state_dict()

        ds_b = BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, buffer_size=4
        )
        ds_b.load_state_dict(checkpoint)

        remaining_a = list(it_a)
        remaining_b = list(ds_b)

        self.assertEqual(len(remaining_a), len(remaining_b))
        for (ba, la, _), (bb, lb, _) in zip(remaining_a, remaining_b):
            self.assertTrue(torch.equal(ba["input"], bb["input"]))
            self.assertTrue(torch.equal(la, lb))


class TestBackboneSuffixGreedyPacking(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil

        self._tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._tmp)

    def _make_dataset(self, examples, seq_len=32):
        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        return BackboneSuffixDataset(
            manifest_path, seq_len=seq_len, infinite=False, packing="buffer_shuffle"
        )

    def test_greedy_produces_valid_output(self):
        """Greedy packing with backbone_suffix format yields correct structure."""
        seq_len = 32
        ds = self._make_dataset([_example_no_suffix(), _example_one_suffix()], seq_len=seq_len)
        batches = list(ds)
        self.assertGreater(len(batches), 0)
        for batch_dict, labels, stats in batches:
            self.assertEqual(batch_dict["input"].shape, (seq_len,))
            self.assertIn("conv_ids", batch_dict)
            self.assertIn("suffix_ids", batch_dict)
            self.assertIn("insertion_limits", batch_dict)
            self.assertGreater(stats["n_examples_packed"], 0)

    def test_greedy_packs_all_examples(self):
        """Greedy packing includes all examples from the dataset."""
        seq_len = 32
        examples = [_example_no_suffix(), _example_one_suffix(), _example_two_suffixes()]
        ds = self._make_dataset(examples, seq_len=seq_len)

        expected_first_tokens = sorted(ex["input_ids"][0] for ex in examples)

        actual_first_tokens = []
        for batch_dict, _, _ in ds:
            conv_ids = batch_dict["conv_ids"].tolist()
            inputs = batch_dict["input"].tolist()
            seen_convs = set()
            for i, cid in enumerate(conv_ids):
                if cid > 0 and cid not in seen_convs:
                    seen_convs.add(cid)
                    actual_first_tokens.append(inputs[i])

        self.assertEqual(sorted(actual_first_tokens), expected_first_tokens)


class TestBackboneSuffixEpochBoundary(unittest.TestCase):
    """Epoch boundary tests for BackboneSuffixDataset."""

    def setUp(self):
        import tempfile
        import shutil

        self._tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._tmp)

    def _make_dataset(self, examples, seq_len=32, **kwargs):
        from torchtitan.models.granite.pretokenized_dataset import BackboneSuffixDataset

        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)
        defaults = dict(infinite=False, packing="buffer_shuffle", buffer_size=4)
        defaults.update(kwargs)
        return BackboneSuffixDataset(manifest_path, seq_len=seq_len, **defaults)

    def test_partial_refill_exhausts_data(self):
        """Fewer examples than buffer_size correctly sets _data_exhausted."""
        examples = [_example_no_suffix(), _example_one_suffix(), _example_two_suffixes()]
        ds = self._make_dataset(examples, buffer_size=10)
        ds._prepare_iter()
        ds._refill_buffer()
        self.assertLessEqual(len(ds._row_indices), 3)
        ds._refill_buffer()
        self.assertTrue(ds._data_exhausted)

    def test_checkpoint_resume(self):
        """Checkpoint round-trip produces identical remaining batches."""
        examples = [_example_no_suffix(), _example_one_suffix(), _example_two_suffixes()]
        ds1 = self._make_dataset(examples, buffer_size=2, packing="longest")
        it1 = iter(ds1)
        next(it1)
        state = ds1.state_dict()

        ds2 = self._make_dataset(examples, buffer_size=2, packing="longest")
        ds2.load_state_dict(state)
        remaining1 = list(it1)
        remaining2 = list(ds2)
        self.assertEqual(len(remaining1), len(remaining2))
        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))

    def test_infinite_epoch_rollover(self):
        """Infinite mode crosses epoch boundary without error."""
        examples = [_example_no_suffix(), _example_one_suffix()]
        ds = self._make_dataset(examples, buffer_size=2, infinite=True)
        batches = []
        for i, batch in enumerate(ds):
            batches.append(batch)
            if i >= 5:
                break
        self.assertEqual(len(batches), 6)
        self.assertGreater(ds._epoch, 0)


class TestReconstructBufferColumnSelection(unittest.TestCase):
    """_reconstruct_buffer selects only metadata columns, avoiding int32 overflow
    on large list columns like input_ids."""

    def setUp(self):
        import tempfile
        import shutil

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._cleanup = lambda: shutil.rmtree(self._tmpdir)

    def tearDown(self):
        self._cleanup()

    def test_restore_produces_identical_buffer(self):
        from torchtitan.models.granite.pretokenized_dataset import BackboneSuffixDataset

        examples = [_example_one_suffix() for _ in range(8)]
        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)

        ds1 = BackboneSuffixDataset(
            manifest_path, seq_len=32, infinite=False, buffer_size=8
        )
        it1 = iter(ds1)
        next(it1)
        state = ds1.state_dict()

        ds2 = BackboneSuffixDataset(
            manifest_path, seq_len=32, infinite=False, buffer_size=8
        )
        ds2.load_state_dict(state)
        ds2._prepare_iter()

        self.assertEqual(sorted(ds2._row_indices), sorted(ds1._row_indices))
        self.assertEqual(sorted(ds2._lengths), sorted(ds1._lengths))

    def test_iteration_matches_after_restore(self):
        from torchtitan.models.granite.pretokenized_dataset import BackboneSuffixDataset

        examples = [
            _example_no_suffix(),
            _example_one_suffix(),
            _example_two_suffixes(),
        ] * 4
        manifest_path = _make_backbone_suffix_shard(self._tmp, examples)

        ds1 = BackboneSuffixDataset(
            manifest_path, seq_len=32, infinite=False, buffer_size=6,
            packing="buffer_shuffle",
        )
        it1 = iter(ds1)
        next(it1)
        state = ds1.state_dict()
        remaining1 = list(it1)

        ds2 = BackboneSuffixDataset(
            manifest_path, seq_len=32, infinite=False, buffer_size=6,
            packing="buffer_shuffle",
        )
        ds2.load_state_dict(state)
        remaining2 = list(ds2)

        self.assertEqual(len(remaining1), len(remaining2))
        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))


if __name__ == "__main__":
    unittest.main()
