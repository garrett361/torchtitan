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
        self.assertLess(idx, len(ds._buffer))

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
        self.assertEqual(len(ds._buffer), len(ds._lengths))
        self.assertEqual(len(ds._buffer), len(ds._ages))
        for i, item in enumerate(ds._buffer):
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
        if ds._buffer:
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
        if ds._buffer:
            self._assert_sorted(ds)


class TestEpochBoundary(unittest.TestCase):
    """Tests for epoch boundary behavior with Arrow-based refill."""

    def test_partial_refill_exhausts_data(self):
        """When dataset has fewer examples than buffer_size, refill fills partially."""
        ds = _build_dataset(seq_len=16, buffer_size=10, packing="buffer_shuffle")
        ds._prepare_iter()
        ds._refill_buffer()
        self.assertLessEqual(len(ds._buffer), 6)
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
        oldest_ids = list(ds._buffer[oldest_idx].input_ids)

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


if __name__ == "__main__":
    unittest.main()
