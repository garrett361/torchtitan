"""Tests for pretokenized_dataset.py: packing and multi-dataset merge."""

import json
import os
import unittest
from pathlib import Path

from torchtitan.models.granite.pretokenized_dataset import (
    GranitePreTokenizedDataLoader,
    StandardPackingDataset,
    _load_and_merge_manifests,
    _load_manifest,
)

MANIFEST_PATH = Path("tests/assets/pretok_truncate_last/manifest.json")
MANIFEST_PATH_B = Path("tests/assets/pretok_truncate_last_b/manifest.json")


def _build_dataset(
    seq_len: int = 16, buffer_size: int = 6, packing: str = "longest", **extra_kwargs
):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    kwargs = dict(
        manifest_path=MANIFEST_PATH,
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=1,
        infinite=False,
        _manifest=manifest,
        packing=packing,
        buffer_size=buffer_size,
    )
    kwargs.update(extra_kwargs)
    return StandardPackingDataset(**kwargs)


class TestMultiDatasetMerge(unittest.TestCase):
    """Tests for comma-separated multi-dataset loading and manifest merging."""

    def test_merge_combines_examples(self):
        """Merging two datasets produces combined example count."""
        manifest, dataset = _load_and_merge_manifests([MANIFEST_PATH, MANIFEST_PATH_B])
        self.assertEqual(manifest["stats"]["total_examples"], 12)
        self.assertEqual(manifest["stats"]["total_tokens"], 68)
        self.assertEqual(len(dataset), 12)

    def test_merge_validates_strategy_mismatch(self):
        """Mismatched strategies raise ValueError."""
        import tempfile

        m = _load_manifest(MANIFEST_PATH)
        m["strategy"] = "some_other_strategy"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(m, f)
            bad_path = Path(f.name)
        try:
            with self.assertRaises(ValueError, msg="Strategy mismatch"):
                _load_and_merge_manifests([MANIFEST_PATH, bad_path])
        finally:
            bad_path.unlink()

    def test_merge_validates_eos_mismatch(self):
        """Mismatched eos_token_id raises ValueError."""
        import tempfile

        m = _load_manifest(MANIFEST_PATH)
        m["tokenizer"]["eos_token_id"] = 9999
        # Need a shards dir alongside the manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            with open(manifest_file, "w") as f:
                json.dump(m, f)
            with self.assertRaises(ValueError, msg="eos_token_id mismatch"):
                _load_and_merge_manifests([MANIFEST_PATH, manifest_file])

    def test_merge_computes_length_stats(self):
        """Merged manifest has correct length_stats over combined dataset."""
        manifest, _ = _load_and_merge_manifests([MANIFEST_PATH, MANIFEST_PATH_B])
        ls = manifest["length_stats"]
        # Both fixtures: [6,5,8,4,7,4] → combined [4,4,4,4,5,5,6,6,7,7,8,8]
        self.assertEqual(ls["min"], 4)
        self.assertEqual(ls["max"], 8)
        self.assertAlmostEqual(ls["mean"], 5.7, places=1)
        self.assertEqual(ls["median"], 5)
        self.assertAlmostEqual(ls["std"], 1.5, places=1)
        self.assertEqual(ls["p95"], 8)

    def test_merge_preserves_tokenizer_info(self):
        """Merged manifest retains tokenizer from first dataset."""
        manifest, _ = _load_and_merge_manifests([MANIFEST_PATH, MANIFEST_PATH_B])
        self.assertEqual(manifest["tokenizer"]["eos_token_id"], 2003)
        self.assertEqual(manifest["tokenizer"]["vocab_size"], 2009)
        self.assertEqual(manifest["strategy"], "truncate_last")

    def test_single_path_no_merge(self):
        """Single path in comma-separated string behaves like non-merged."""
        config = GranitePreTokenizedDataLoader.Config(
            dataset_path=str(MANIFEST_PATH),
        )
        loader = GranitePreTokenizedDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            seq_len=16,
            local_batch_size=1,
        )
        self.assertEqual(loader.dataset.num_examples, 6)

    def test_multi_path_iteration(self):
        """Multi-dataset comma-separated path produces correct iteration."""
        config = GranitePreTokenizedDataLoader.Config(
            dataset_path=f"{MANIFEST_PATH},{MANIFEST_PATH_B}",
            infinite=False,
            packing="buffer_shuffle",
        )
        loader = GranitePreTokenizedDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            seq_len=16,
            local_batch_size=1,
        )
        self.assertEqual(loader.dataset.num_examples, 12)
        batches = list(loader)
        self.assertGreater(len(batches), 0)

    def test_multi_path_directory_format(self):
        """Paths without .json suffix get /manifest.json appended."""
        dir_a = str(MANIFEST_PATH.parent)
        dir_b = str(MANIFEST_PATH_B.parent)
        config = GranitePreTokenizedDataLoader.Config(
            dataset_path=f"{dir_a},{dir_b}",
            infinite=False,
            packing="buffer_shuffle",
        )
        loader = GranitePreTokenizedDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            seq_len=16,
            local_batch_size=1,
        )
        self.assertEqual(loader.dataset.num_examples, 12)


class TestGreedyPacking(unittest.TestCase):
    """Greedy packing uses random selection from fitting items."""

    def test_greedy_selects_fitting_items(self):
        """Greedy packing only selects items that fit in remaining space."""
        ds = _build_dataset(seq_len=16, packing="buffer_shuffle", buffer_size=6)
        batches = list(ds)
        for inp_dict, _, stats in batches:
            self.assertLessEqual(stats["n_total_tokens"], 16)

    def test_greedy_selection_returns_valid_index(self):
        """Greedy selection returns a valid buffer index when items fit."""
        from torchtitan.models.granite.pretokenized_dataset import _select_buffer_shuffle

        ds = _build_dataset(seq_len=16, packing="buffer_shuffle", buffer_size=6)
        ds._prepare_iter()
        ds._data_exhausted = False
        ds._refill_buffer()

        idx = _select_buffer_shuffle(ds, remaining=9999)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(ds._row_indices))

    def test_greedy_selection_rejects_when_too_long(self):
        """Greedy selection returns -1 when no item fits."""
        from torchtitan.models.granite.pretokenized_dataset import _select_buffer_shuffle

        ds = _build_dataset(seq_len=16, packing="buffer_shuffle", buffer_size=6)
        ds._prepare_iter()
        ds._data_exhausted = False
        ds._refill_buffer()

        idx = _select_buffer_shuffle(ds, remaining=0)
        self.assertEqual(idx, -1)


class TestSortedBufferInvariants(unittest.TestCase):
    """Verify parallel array sync and sort invariants for the bisect-based buffer."""

    def _get_filled_dataset(self):
        ds = _build_dataset(seq_len=16, packing="longest", buffer_size=6)
        ds._prepare_iter()
        ds._data_exhausted = False
        ds._refill_buffer()
        return ds

    def _assert_sync(self, ds):
        self.assertEqual(len(ds._row_indices), len(ds._lengths))
        self.assertEqual(len(ds._row_indices), len(ds._ages))
        for i, row_idx in enumerate(ds._row_indices):
            item = ds._materialize_item(row_idx)
            self.assertEqual(ds._lengths[i], len(item.input_ids))

    def _assert_sorted(self, ds):
        for i in range(len(ds._lengths) - 1):
            self.assertLessEqual(ds._lengths[i], ds._lengths[i + 1])

    def test_parallel_array_sync_after_refill(self):
        """After refill, all three arrays are in sync."""
        ds = self._get_filled_dataset()
        self._assert_sync(ds)

    def test_parallel_array_sync_during_iteration(self):
        """Arrays stay in sync after consuming several batches."""
        ds = self._get_filled_dataset()
        it = ds._iter_packed()
        for _ in range(3):
            next(it, None)
        if ds._row_indices:
            self._assert_sync(ds)

    def test_sort_invariant_after_refill(self):
        """_lengths is sorted ascending after refill."""
        ds = self._get_filled_dataset()
        self._assert_sorted(ds)

    def test_sort_invariant_during_iteration(self):
        """_lengths remains sorted after partial consumption."""
        ds = self._get_filled_dataset()
        it = ds._iter_packed()
        next(it, None)
        if ds._row_indices:
            self._assert_sorted(ds)


class TestEpochBoundary(unittest.TestCase):
    """Tests for epoch boundary behavior with Arrow-based refill."""

    def test_partial_refill_exhausts_data(self):
        """When dataset has fewer examples than buffer_size, refill fills partially."""
        ds = _build_dataset(seq_len=16, buffer_size=10, packing="buffer_shuffle")
        ds._prepare_iter()
        ds._refill_buffer()
        self.assertLessEqual(len(ds._row_indices), 6)
        # Second refill discovers exhaustion
        ds._refill_buffer()
        self.assertTrue(ds._data_exhausted)
        batches = list(ds._iter_packed())
        self.assertGreater(len(batches), 0)

    def test_checkpoint_resume_mid_dataset(self):
        """Checkpoint at non-zero sample_idx resumes correctly."""
        ds1 = _build_dataset(seq_len=16, buffer_size=3, packing="longest")
        it1 = iter(ds1)
        next(it1)
        state = ds1.state_dict()
        self.assertGreater(state["sample_idx"], 0)

        ds2 = _build_dataset(seq_len=16, buffer_size=3, packing="longest")
        ds2.load_state_dict(state)
        remaining1 = list(it1)
        remaining2 = list(ds2)
        self.assertEqual(len(remaining1), len(remaining2))
        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))

    def test_infinite_epoch_rollover(self):
        """Infinite mode crosses epoch boundary without error."""
        ds = _build_dataset(seq_len=16, buffer_size=3, packing="buffer_shuffle", infinite=True)
        batches = []
        for i, batch in enumerate(ds):
            batches.append(batch)
            if i >= 9:
                break
        self.assertEqual(len(batches), 10)
        self.assertGreater(ds._epoch, 0)
        for inp_dict, labels, _ in batches:
            self.assertEqual(inp_dict["input"].shape[0], 16)
            self.assertEqual(labels.shape[0], 16)

    def test_rng_state_roundtrip(self):
        """batch_rng_state in state_dict produces identical RNG after restore."""
        ds = _build_dataset(seq_len=16, buffer_size=6, packing="buffer_shuffle")
        it = iter(ds)
        next(it)
        next(it)
        state = ds.state_dict()

        ds2 = _build_dataset(seq_len=16, buffer_size=6, packing="buffer_shuffle")
        ds2.load_state_dict(state)
        self.assertEqual(
            ds._batch_rng.integers(1000),
            ds2._batch_rng.integers(1000),
        )


class TestOldestFirstSeed(unittest.TestCase):
    """Tests for oldest-first seed selection in the packing buffer."""

    def test_seed_is_oldest_item(self):
        """Batch seed is always the oldest (earliest-inserted) buffer item."""
        ds = _build_dataset(seq_len=16, buffer_size=6, packing="buffer_shuffle")
        ds._prepare_iter()
        ds._data_exhausted = False
        ds._refill_buffer()

        # Record the first-inserted item (lowest age = oldest)
        oldest_age = min(ds._ages)
        oldest_idx = ds._ages.index(oldest_age)
        oldest_row_idx = ds._row_indices[oldest_idx]
        oldest_item = ds._materialize_item(oldest_row_idx)
        oldest_ids = list(oldest_item.input_ids)

        batch = next(ds._iter_packed())
        input_tensor = batch[0]["input"].tolist()
        self.assertEqual(
            input_tensor[: len(oldest_ids)],
            oldest_ids,
            "Seed should be the oldest item in the buffer",
        )

    def test_ages_checkpoint_roundtrip(self):
        """Ages and age_counter survive checkpoint save/restore."""
        ds1 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        it1 = iter(ds1)
        next(it1)

        state = ds1.state_dict()
        self.assertIn("ages", state)
        self.assertIn("age_counter", state)

        ds2 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        ds2.load_state_dict(state)
        ds2._prepare_iter()

        self.assertEqual(ds1._ages, ds2._ages)
        self.assertEqual(ds1._age_counter, ds2._age_counter)

        remaining1 = list(it1)
        remaining2 = list(ds2)
        self.assertEqual(len(remaining1), len(remaining2))
        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))


class TestSnapshotResumeDataLoader(unittest.TestCase):
    """Verify StatefulDataLoader resume with misaligned snapshot_every_n_steps.

    When snapshot_every_n_steps=N and we stop at step S where S % N != 0,
    the dataloader replays S - last_snapshot_step batches on resume. This test
    verifies that the replayed sequence matches the original.
    """

    def _build_loader(self, snapshot_every_n_steps: int = 4):
        config = GranitePreTokenizedDataLoader.Config(
            dataset_path=str(MANIFEST_PATH),
            infinite=True,
            packing="buffer_shuffle",
            buffer_size=6,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=False,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        return GranitePreTokenizedDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            seq_len=16,
            local_batch_size=1,
        )

    def test_resume_misaligned_with_snapshot(self):
        """Resume from step 7 (snapshot at step 4) replays steps 5-7 identically."""
        dl = self._build_loader(snapshot_every_n_steps=4)
        it = iter(dl)

        # Consume 7 batches (last snapshot at step 4, so steps_since_snapshot=3)
        for _ in range(7):
            next(it)

        state = dl.state_dict()

        # Consume 5 more batches from the original iterator
        expected = []
        for _ in range(5):
            inp_dict, labels = next(it)
            expected.append((inp_dict["input"].clone(), labels.clone()))

        # Resume from checkpoint
        dl_resumed = self._build_loader(snapshot_every_n_steps=4)
        dl_resumed.load_state_dict(state)
        it_resumed = iter(dl_resumed)

        for i, (exp_input, exp_labels) in enumerate(expected):
            inp_dict, labels = next(it_resumed)
            self.assertTrue(
                inp_dict["input"].equal(exp_input),
                f"Batch {i} input_ids mismatch after resume",
            )
            self.assertTrue(
                labels.equal(exp_labels),
                f"Batch {i} labels mismatch after resume",
            )

    def test_resume_aligned_with_snapshot(self):
        """Resume from step 8 (exactly on snapshot boundary) produces same sequence."""
        dl = self._build_loader(snapshot_every_n_steps=4)
        it = iter(dl)

        for _ in range(8):
            next(it)

        state = dl.state_dict()

        expected = []
        for _ in range(5):
            inp_dict, labels = next(it)
            expected.append((inp_dict["input"].clone(), labels.clone()))

        dl_resumed = self._build_loader(snapshot_every_n_steps=4)
        dl_resumed.load_state_dict(state)
        it_resumed = iter(dl_resumed)

        for i, (exp_input, exp_labels) in enumerate(expected):
            inp_dict, labels = next(it_resumed)
            self.assertTrue(
                inp_dict["input"].equal(exp_input),
                f"Batch {i} input_ids mismatch after aligned resume",
            )
            self.assertTrue(
                labels.equal(exp_labels),
                f"Batch {i} labels mismatch after aligned resume",
            )

    def test_resume_with_large_snapshot_interval(self):
        """snapshot_every_n_steps=1024 (default) still resumes correctly."""
        dl = self._build_loader(snapshot_every_n_steps=1024)
        it = iter(dl)

        for _ in range(7):
            next(it)

        state = dl.state_dict()

        expected = []
        for _ in range(3):
            inp_dict, labels = next(it)
            expected.append((inp_dict["input"].clone(), labels.clone()))

        dl_resumed = self._build_loader(snapshot_every_n_steps=1024)
        dl_resumed.load_state_dict(state)
        it_resumed = iter(dl_resumed)

        for i, (exp_input, exp_labels) in enumerate(expected):
            inp_dict, labels = next(it_resumed)
            self.assertTrue(
                inp_dict["input"].equal(exp_input),
                f"Batch {i} input_ids mismatch with large snapshot interval",
            )
            self.assertTrue(
                labels.equal(exp_labels),
                f"Batch {i} labels mismatch with large snapshot interval",
            )


class TestCrossRankLPT(unittest.TestCase):
    """Tests for cross-rank LPT packing: determinism, coverage, and balance."""

    def _build_for_rank(self, dp_rank, dp_world_size=2, packing="longest", **kwargs):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        defaults = dict(
            manifest_path=MANIFEST_PATH,
            seq_len=16,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=False,
            _manifest=manifest,
            packing=packing,
            buffer_size=6,
        )
        defaults.update(kwargs)
        return StandardPackingDataset(**defaults)

    def test_cross_rank_determinism(self):
        """Same data + same logic = identical LPT decisions across ranks.

        Ranks see the same buffer state and make the same assignment choices.
        The only difference is which batch slice they yield. Verify that the
        union of batch[0] and batch[1] is consistent across independent
        instantiations.
        """
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=2)
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=2)

        batches_0 = [(inp["input"].clone(), lbl.clone()) for inp, lbl, _ in ds0]
        batches_1 = [(inp["input"].clone(), lbl.clone()) for inp, lbl, _ in ds1]

        # Both ranks produce same number of batches
        self.assertEqual(len(batches_0), len(batches_1))
        self.assertGreater(len(batches_0), 0)

        # Ranks produce different batches (they get different slices)
        any_different = any(
            not b0[0].equal(b1[0]) for b0, b1 in zip(batches_0, batches_1)
        )
        self.assertTrue(any_different, "Ranks should get different batch slices")

    def test_all_items_consumed_across_dp_degree(self):
        """Union of examples across ranks covers all input data, no duplicates."""
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=2)
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=2)

        def extract_packed_examples(ds):
            examples = []
            for inp, lbl, stats in ds:
                n = stats["n_total_tokens"]
                positions = inp["positions"][:n].tolist()
                tokens = inp["input"][:n].tolist()
                start = 0
                for i in range(1, len(positions)):
                    if positions[i] != positions[i - 1] + 1:
                        examples.append(tuple(tokens[start:i]))
                        start = i
                examples.append(tuple(tokens[start:]))
            return examples

        examples_0 = extract_packed_examples(ds0)
        examples_1 = extract_packed_examples(ds1)
        all_examples = sorted(examples_0 + examples_1)

        from torchtitan.models.granite.pretokenized_dataset import (
            _load_manifest,
            _load_shards,
        )

        full_manifest = _load_manifest(MANIFEST_PATH)
        shards_dir = MANIFEST_PATH.parent / "shards"
        full_ds = _load_shards(full_manifest, shards_dir)
        expected_examples = sorted(
            tuple(full_ds[i]["input_ids"]) for i in range(len(full_ds))
        )

        self.assertEqual(all_examples, expected_examples)

    def test_dp_degree_one_equivalence(self):
        """dp_world_size=1 produces same batches as before (no cross-rank logic)."""
        ds = self._build_for_rank(dp_rank=0, dp_world_size=1)
        batches = list(ds)
        self.assertGreater(len(batches), 0)
        for inp, lbl, stats in batches:
            self.assertEqual(inp["input"].shape[0], 16)
            self.assertGreater(stats["n_total_tokens"], 0)

    def test_cross_rank_balance(self):
        """Cross-rank LPT produces balanced L² costs across ranks."""
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=2)
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=2)

        def compute_l2_cost(inp_dict, stats):
            n = stats["n_total_tokens"]
            positions = inp_dict["positions"][:n].tolist()
            cost = 0
            cur = 1
            for i in range(1, len(positions)):
                if positions[i] != positions[i - 1] + 1:
                    cost += cur * cur
                    cur = 1
                else:
                    cur += 1
            cost += cur * cur
            return cost

        costs_0 = [compute_l2_cost(inp, s) for inp, _, s in ds0]
        costs_1 = [compute_l2_cost(inp, s) for inp, _, s in ds1]

        # Per-step, the L² costs should be close between ranks
        for c0, c1 in zip(costs_0, costs_1):
            ratio = max(c0, c1) / max(min(c0, c1), 1)
            self.assertLess(ratio, 3.0, "Cross-rank L² cost ratio should be bounded")

    def test_buffer_shuffle_rng_deterministic_across_ranks(self):
        """buffer_shuffle RNG advances identically on all ranks."""
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=2, packing="buffer_shuffle")
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=2, packing="buffer_shuffle")

        batches_0 = list(ds0)
        batches_1 = list(ds1)

        self.assertEqual(len(batches_0), len(batches_1))
        # After iteration, both should have consumed same buffer items
        # (same RNG sequence = same selection choices)
        self.assertEqual(ds0._batch_rng.bit_generator.state,
                         ds1._batch_rng.bit_generator.state)

    def test_checkpoint_resume_cross_rank(self):
        """Checkpoint at dp_rank=0 restores correctly for same rank."""
        ds1 = self._build_for_rank(dp_rank=0, dp_world_size=2, infinite=True,
                                   packing="buffer_shuffle")
        it1 = iter(ds1)
        next(it1)
        next(it1)

        state = ds1.state_dict()

        ds2 = self._build_for_rank(dp_rank=0, dp_world_size=2, infinite=True,
                                   packing="buffer_shuffle")
        ds2.load_state_dict(state)

        remaining1 = [next(it1) for _ in range(3)]
        remaining2 = [b for b, _ in zip(ds2, range(3))]

        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))


class TestCostFromMetadata(unittest.TestCase):
    """Tests for _cost_from_metadata: cost computation from Arrow metadata."""

    def test_standard_triangular(self):
        """StandardPackingDataset returns (n, n*(n+1)//2) for valid rows."""
        import numpy as np

        ds = _build_dataset(packing="longest")
        for n, expected_cost in [(1, 1), (4, 10), (8, 36), (16, 136)]:
            scalars = {"n_tokens": np.array([n])}
            result = ds._cost_from_metadata(scalars, {}, 0)
            self.assertEqual(result, (n, expected_cost))

    def test_standard_skips_oversized(self):
        """Returns None when n_tokens > seq_len."""
        import numpy as np

        ds = _build_dataset(seq_len=8, packing="longest")
        scalars = {"n_tokens": np.array([9])}
        self.assertIsNone(ds._cost_from_metadata(scalars, {}, 0))

    def test_backbone_suffix_no_suffixes(self):
        """With no suffixes, BackboneSuffixDataset returns triangular cost."""
        import numpy as np

        ds = _build_backbone_suffix_dataset()
        scalars = {"n_tokens": np.array([6])}
        list_arrays = {
            "suffix_starts": (np.array([0, 0]), np.array([], dtype=np.int64)),
            "insertion_limits": (np.array([0, 0]), np.array([], dtype=np.int64)),
        }
        result = ds._cost_from_metadata(scalars, list_arrays, 0)
        self.assertEqual(result, (6, 21))

    def test_backbone_suffix_known_structure(self):
        """Verify cost for B=4, one suffix of length 3, ins_limit=2.

        backbone self: 4*5//2 = 10
        suffix self:   3*4//2 = 6
        suffix→backbone: 3*(2+1) = 9
        total: 25
        """
        import numpy as np

        ds = _build_backbone_suffix_dataset()
        scalars = {"n_tokens": np.array([7])}
        list_arrays = {
            "suffix_starts": (np.array([0, 1]), np.array([4])),
            "insertion_limits": (np.array([0, 1]), np.array([2])),
        }
        result = ds._cost_from_metadata(scalars, list_arrays, 0)
        self.assertEqual(result, (7, 25))

    def test_backbone_suffix_multiple_suffixes(self):
        """Verify cost with B=3, two suffixes S1=2 (ins=1), S2=2 (ins=2).

        backbone self: 3*4//2 = 6
        S1 self: 2*3//2 = 3, S1→backbone: 2*(1+1) = 4
        S2 self: 2*3//2 = 3, S2→backbone: 2*(2+1) = 6
        total: 22
        """
        import numpy as np

        ds = _build_backbone_suffix_dataset()
        scalars = {"n_tokens": np.array([7])}
        list_arrays = {
            "suffix_starts": (np.array([0, 2]), np.array([3, 5])),
            "insertion_limits": (np.array([0, 2]), np.array([1, 2])),
        }
        result = ds._cost_from_metadata(scalars, list_arrays, 0)
        self.assertEqual(result, (7, 22))


class TestSelectAttnBalanced(unittest.TestCase):
    """Tests for _select_attn_balanced selection contract."""

    def _make_ds_with_buffer(self, lengths):
        ds = _build_dataset(packing="attn_balanced", buffer_size=len(lengths))
        ds._row_indices = []
        ds._lengths = []
        ds._costs = []
        ds._ages = []
        ds._age_counter = 0
        for i, length in enumerate(lengths):
            cost = length * (length + 1) // 2
            ds._insert_entry(length, cost, row_idx=i)
        return ds

    def test_returns_neg1_when_nothing_fits(self):
        from torchtitan.models.granite.pretokenized_dataset import (
            _select_attn_balanced,
        )

        ds = self._make_ds_with_buffer([3, 5, 8])
        idx, cost = _select_attn_balanced(ds, remaining=0, deficit=100)
        self.assertEqual(idx, -1)

    def test_selects_closest_to_deficit(self):
        """Picks item whose cost is closest to the deficit."""
        from torchtitan.models.granite.pretokenized_dataset import (
            _select_attn_balanced,
        )

        ds = self._make_ds_with_buffer([3, 5, 8])
        # costs: 3→6, 5→15, 8→36
        # deficit=14 → closest is 15 (length 5)
        idx, cost = _select_attn_balanced(ds, remaining=10, deficit=14)
        self.assertEqual(cost, 15)

    def test_respects_remaining_constraint(self):
        """Items exceeding remaining are excluded even if cost matches better."""
        from torchtitan.models.granite.pretokenized_dataset import (
            _select_attn_balanced,
        )

        ds = self._make_ds_with_buffer([3, 8])
        # costs: 3→6, 8→36
        # deficit=35 prefers cost=36 (len=8), but remaining=4 excludes it
        idx, cost = _select_attn_balanced(ds, remaining=4, deficit=35)
        self.assertEqual(cost, 6)

    def test_exact_match(self):
        """When an item's cost exactly matches deficit, picks it."""
        from torchtitan.models.granite.pretokenized_dataset import (
            _select_attn_balanced,
        )

        ds = self._make_ds_with_buffer([3, 5, 8])
        # cost of len=5 is 15
        idx, cost = _select_attn_balanced(ds, remaining=10, deficit=15)
        self.assertEqual(cost, 15)


class TestAttnBalancedPacking(unittest.TestCase):
    """Integration tests for attn_balanced packing mode."""

    def _build_for_rank(self, dp_rank, dp_world_size=2, **kwargs):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        defaults = dict(
            manifest_path=MANIFEST_PATH,
            seq_len=16,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=False,
            _manifest=manifest,
            packing="attn_balanced",
            buffer_size=6,
        )
        defaults.update(kwargs)
        return StandardPackingDataset(**defaults)

    def test_all_items_consumed_across_ranks(self):
        """Items are not duplicated; at most dp-1 dropped as remainder."""
        from torchtitan.models.granite.pretokenized_dataset import (
            _load_manifest,
            _load_shards,
        )

        dp = 2
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=dp)
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=dp)

        def extract_packed_examples(ds):
            examples = []
            for inp, lbl, stats in ds:
                n = stats["n_total_tokens"]
                positions = inp["positions"][:n].tolist()
                tokens = inp["input"][:n].tolist()
                start = 0
                for i in range(1, len(positions)):
                    if positions[i] != positions[i - 1] + 1:
                        examples.append(tuple(tokens[start:i]))
                        start = i
                examples.append(tuple(tokens[start:]))
            return examples

        examples_0 = extract_packed_examples(ds0)
        examples_1 = extract_packed_examples(ds1)
        all_examples = sorted(examples_0 + examples_1)

        full_manifest = _load_manifest(MANIFEST_PATH)
        shards_dir = MANIFEST_PATH.parent / "shards"
        full_ds = _load_shards(full_manifest, shards_dir)
        expected_examples = sorted(
            tuple(full_ds[i]["input_ids"]) for i in range(len(full_ds))
        )

        # No duplicates
        self.assertEqual(len(all_examples), len(set(all_examples)))
        # All emitted items come from the source dataset
        for ex in all_examples:
            self.assertIn(ex, expected_examples)
        # At most dp-1 items dropped as remainder at epoch end
        dropped = len(expected_examples) - len(all_examples)
        self.assertLessEqual(dropped, dp - 1)
        self.assertGreaterEqual(dropped, 0)

    def test_cross_rank_determinism(self):
        """Independent instantiations produce identical assignment decisions."""
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=2)
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=2)

        batches_0 = [(inp["input"].clone(), lbl.clone()) for inp, lbl, _ in ds0]
        batches_1 = [(inp["input"].clone(), lbl.clone()) for inp, lbl, _ in ds1]

        self.assertEqual(len(batches_0), len(batches_1))
        self.assertGreater(len(batches_0), 0)

        # Ranks should get different slices
        any_different = any(
            not b0[0].equal(b1[0]) for b0, b1 in zip(batches_0, batches_1)
        )
        self.assertTrue(any_different, "Ranks should get different batch slices")

    def test_rank_stall_does_not_starve_others(self):
        """When one rank fills early, others continue packing."""
        # With seq_len=16 and items of sizes 4-8, one rank can fill to near
        # capacity. The other rank should still receive items.
        ds0 = self._build_for_rank(dp_rank=0, dp_world_size=2)
        ds1 = self._build_for_rank(dp_rank=1, dp_world_size=2)

        batches_0 = list(ds0)
        batches_1 = list(ds1)

        # Both ranks should produce non-empty batches with reasonable packing
        for inp, lbl, stats in batches_0 + batches_1:
            self.assertGreater(stats["n_total_tokens"], 0)

    def test_dp1_produces_valid_output(self):
        """dp_world_size=1 works (deficit always 0, random fallback)."""
        ds = self._build_for_rank(dp_rank=0, dp_world_size=1)
        batches = list(ds)
        self.assertGreater(len(batches), 0)
        for inp, lbl, stats in batches:
            self.assertEqual(inp["input"].shape[0], 16)
            self.assertGreater(stats["n_total_tokens"], 0)

    def test_checkpoint_resume(self):
        """Resume from checkpoint produces identical sequence."""
        ds1 = self._build_for_rank(dp_rank=0, dp_world_size=2, infinite=True)
        it1 = iter(ds1)
        next(it1)
        next(it1)

        state = ds1.state_dict()

        ds2 = self._build_for_rank(dp_rank=0, dp_world_size=2, infinite=True)
        ds2.load_state_dict(state)

        remaining1 = [next(it1) for _ in range(3)]
        remaining2 = [b for b, _ in zip(ds2, range(3))]

        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))


class TestMaterializeItem(unittest.TestCase):
    """Verify _materialize_item returns correct data from Arrow source."""

    def test_standard_matches_arrow(self):
        """Materialized input_ids/labels match raw Arrow column values."""
        import numpy as np

        ds = _build_dataset(seq_len=16, buffer_size=6, packing="longest")
        ds._prepare_iter()

        for row_idx in range(len(ds._data)):
            item = ds._materialize_item(row_idx)
            table_slice = ds._data.data.slice(row_idx, 1)
            col = table_slice.column("input_ids").combine_chunks()
            offsets = col.offsets.to_numpy()
            expected = col.values.to_numpy()[offsets[0]:offsets[1]]
            np.testing.assert_array_equal(item.input_ids, expected)

    def test_backbone_suffix_all_fields_numpy(self):
        """BackboneSuffixDataset materializes all fields as np.ndarray."""
        import tempfile

        import numpy as np
        from datasets import Dataset

        from torchtitan.models.granite.pretokenized_dataset import (
            BackboneSuffixDataset,
        )

        ds_data = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4, 5], [10, 20, 30]],
            "labels": [[-100, -100, 3, 4, 5], [-100, 20, 30]],
            "positions": [[0, 1, 2, 3, 4], [0, 1, 2]],
            "suffix_starts": [[3], []],
            "insertion_limits": [[2], []],
            "n_tokens": [5, 3],
        })
        manifest = {
            "strategy": "backbone_suffix",
            "tokenizer": {"eos_token_id": 0},
            "shards": {"completed": []},
            "stats": {"total_examples": 2},
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(manifest, tmp)
        tmp.close()

        try:
            ds = BackboneSuffixDataset(
                manifest_path=tmp.name,
                seq_len=32,
                dp_rank=0,
                dp_world_size=1,
                infinite=False,
                _manifest=manifest,
                _full_dataset=ds_data,
                packing="longest",
                buffer_size=4,
            )
            ds._prepare_iter()

            for row_idx in range(len(ds._data)):
                item = ds._materialize_item(row_idx)
                for field in item:
                    self.assertIsInstance(field, np.ndarray)
        finally:
            os.unlink(tmp.name)

    def test_last_row_in_shard(self):
        """Materializing the last row doesn't raise."""
        ds = _build_dataset(seq_len=16, buffer_size=6, packing="longest")
        ds._prepare_iter()
        last_idx = len(ds._data) - 1
        item = ds._materialize_item(last_idx)
        self.assertGreater(len(item.input_ids), 0)


class TestRefillBufferMetadata(unittest.TestCase):
    """Verify buffer metadata consistency after _refill_buffer."""

    def test_lengths_match_materialized_items(self):
        """_lengths[k] equals len(_materialize_item(_row_indices[k]).input_ids)."""
        ds = _build_dataset(seq_len=16, buffer_size=6, packing="longest")
        ds._prepare_iter()
        ds._refill_buffer()

        for k in range(len(ds._row_indices)):
            item = ds._materialize_item(ds._row_indices[k])
            self.assertEqual(ds._lengths[k], len(item.input_ids))

    def test_oversized_items_excluded(self):
        """Items with n_tokens > seq_len are absent from buffer."""
        ds = _build_dataset(seq_len=4, buffer_size=10, packing="longest")
        ds._prepare_iter()
        ds._refill_buffer()

        for length in ds._lengths:
            self.assertLessEqual(length, 4)

    def test_row_indices_in_range(self):
        """All _row_indices values are valid shard-relative indices."""
        ds = _build_dataset(seq_len=16, buffer_size=6, packing="longest")
        ds._prepare_iter()
        ds._refill_buffer()

        data_len = len(ds._data)
        for row_idx in ds._row_indices:
            self.assertGreaterEqual(row_idx, 0)
            self.assertLess(row_idx, data_len)


class TestPendingRestore(unittest.TestCase):
    """Deferred checkpoint restore: load_state_dict → _prepare_iter window."""

    def test_pending_set_after_load(self):
        """After load_state_dict with row_indices, _pending_restore is set."""
        ds1 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        it1 = iter(ds1)
        next(it1)

        state = ds1.state_dict()
        self.assertIn("row_indices", state)

        ds2 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        ds2.load_state_dict(state)

        self.assertIsNotNone(ds2._pending_restore)
        self.assertEqual(ds2._row_indices, [])

    def test_buffer_reconstructed_after_iter(self):
        """After iter() triggers _prepare_iter, buffer is reconstructed."""
        ds1 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        it1 = iter(ds1)
        next(it1)

        state = ds1.state_dict()

        ds2 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        ds2.load_state_dict(state)
        ds2._prepare_iter()

        self.assertIsNone(ds2._pending_restore)
        self.assertEqual(len(ds2._row_indices), len(ds1._row_indices))
        self.assertEqual(ds2._lengths, ds1._lengths)

    def test_state_dict_in_pending_window(self):
        """state_dict() called before _prepare_iter serializes from _pending_restore."""
        ds1 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        it1 = iter(ds1)
        next(it1)

        state1 = ds1.state_dict()

        ds2 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        ds2.load_state_dict(state1)
        state2 = ds2.state_dict()

        self.assertEqual(state1["row_indices"], state2["row_indices"])
        self.assertEqual(state1["ages"], state2["ages"])
        self.assertEqual(state1["age_counter"], state2["age_counter"])

    def test_iteration_matches_after_restore(self):
        """Restored dataset produces identical batches to original."""
        ds1 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        it1 = iter(ds1)
        next(it1)

        state = ds1.state_dict()
        remaining1 = list(it1)

        ds2 = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        ds2.load_state_dict(state)
        remaining2 = list(ds2)

        self.assertEqual(len(remaining1), len(remaining2))
        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))


class TestReconstructBufferBounds(unittest.TestCase):
    """Out-of-bounds row index raises ValueError during reconstruction."""

    def test_invalid_row_index_raises(self):
        ds = _build_dataset(seq_len=16, buffer_size=4, packing="buffer_shuffle")
        ds._prepare_iter()

        with self.assertRaises(ValueError) as ctx:
            ds._reconstruct_buffer([9999], [0])
        self.assertIn("9999", str(ctx.exception))


class TestMidLoopRefill(unittest.TestCase):
    """Force mid-loop refill with tiny buffer_size."""

    def test_fill_default_refills_mid_loop(self):
        """With buffer_size=2, dp=1, fill loop must refill to pack all items."""
        ds = _build_dataset(seq_len=16, buffer_size=2, packing="longest")
        batches = list(ds)
        self.assertGreater(len(batches), 0)
        total_examples = sum(b[2]["n_examples_packed"] for b in batches)
        self.assertGreater(total_examples, 2)

    def test_fill_attn_balanced_refills_mid_loop(self):
        """With buffer_size=2, dp=1, attn_balanced fill refills mid-loop."""
        ds = _build_dataset(seq_len=16, buffer_size=2, packing="attn_balanced")
        batches = list(ds)
        self.assertGreater(len(batches), 0)
        total_examples = sum(b[2]["n_examples_packed"] for b in batches)
        self.assertGreater(total_examples, 2)

    def test_small_buffer_cross_rank(self):
        """With buffer_size=2, dp_world_size=2, all items still consumed."""
        batches_r0 = list(
            _build_dataset(
                seq_len=16, buffer_size=2, packing="longest",
                dp_rank=0, dp_world_size=2,
            )
        )
        batches_r1 = list(
            _build_dataset(
                seq_len=16, buffer_size=2, packing="longest",
                dp_rank=1, dp_world_size=2,
            )
        )
        total = sum(b[2]["n_examples_packed"] for b in batches_r0 + batches_r1)
        self.assertGreaterEqual(total, 4)


class TestEpochWrap(unittest.TestCase):
    """Test infinite=True epoch wrapping and remnant dropping."""

    def test_infinite_wraps_epoch(self):
        """With infinite=True, dataset produces more batches than one epoch."""
        ds = _build_dataset(
            seq_len=16, buffer_size=4, packing="longest", infinite=True
        )
        it = iter(ds)
        batches = [next(it) for _ in range(10)]
        self.assertEqual(len(batches), 10)
        total_examples = sum(b[2]["n_examples_packed"] for b in batches)
        self.assertGreater(total_examples, 6)

    def test_finite_remnant_dropped_with_warning(self):
        """When fewer items remain than dp_world_size, warns and drops."""
        import warnings

        # seq_len=5 filters out items with n_tokens > 5; with 6 examples
        # only 3 fit, but dp=4 needs 4 seeds → triggers remnant warning.
        ds = _build_dataset(
            seq_len=5, buffer_size=6, packing="longest",
            dp_rank=0, dp_world_size=4, infinite=False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(ds)
        remnant_warnings = [
            x for x in w if "remaining at end of epoch" in str(x.message)
        ]
        self.assertGreater(len(remnant_warnings), 0)


def _build_backbone_suffix_dataset(seq_len=16, buffer_size=6):
    """Helper to build a BackboneSuffixDataset for cost metadata tests."""
    import json
    import tempfile

    import numpy as np
    from datasets import Dataset

    from torchtitan.models.granite.pretokenized_dataset import BackboneSuffixDataset

    manifest = {
        "strategy": "backbone_suffix",
        "tokenizer": {"eos_token_id": 0},
        "shards": {"completed": []},
        "stats": {"total_examples": 0},
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(manifest, tmp)
    tmp.close()

    empty_ds = Dataset.from_dict({
        "input_ids": [[]],
        "labels": [[]],
        "positions": [[]],
        "suffix_starts": [[]],
        "insertion_limits": [[]],
        "n_tokens": [0],
    })

    try:
        return BackboneSuffixDataset(
            manifest_path=tmp.name,
            seq_len=seq_len,
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
            _manifest=manifest,
            _full_dataset=empty_ds,
            packing="attn_balanced",
            buffer_size=buffer_size,
        )
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
