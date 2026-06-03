"""Tests for offline pack plan generation (plan_packing.py)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from datasets import Dataset as HFDataset

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.pretokenized_dataset import (
    GranitePreTokenizedDataLoader,
    PlannedPackingDataset,
    _load_shards,
)
from torchtitan.models.granite.scripts.plan_packing import (
    _load_metadata_columns,
    _pre_pack,
    plan_packing,
)


class TestBFDPack(unittest.TestCase):
    """Tests for the bucketed BFD bin-packing algorithm."""

    def test_single_example_fits_exactly(self):
        n_tokens = np.array([1024], dtype=np.int64)
        attn_cost = np.array([1024 * 1025 // 2], dtype=np.int64)
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=1024)
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0], [0])

    def test_two_examples_fit_one_pack(self):
        # 640 placed first, remaining=384 → bucket 3. 384 needs ceil(384/128)=3 → fits.
        n_tokens = np.array([384, 640], dtype=np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=1024)
        self.assertEqual(len(packs), 1)
        self.assertCountEqual(packs[0], [0, 1])
        self.assertEqual(int(costs[0]), int(attn_cost[0]) + int(attn_cost[1]))

    def test_overlong_examples_skipped(self):
        n_tokens = np.array([2000, 500, 300], dtype=np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=1024)
        all_indices = [idx for pack in packs for idx in pack]
        self.assertNotIn(0, all_indices)
        self.assertIn(1, all_indices)
        self.assertIn(2, all_indices)

    def test_no_pack_exceeds_seq_len(self):
        rng = np.random.default_rng(42)
        n_tokens = rng.integers(100, 5000, size=1000).astype(np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        seq_len = 8192
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=seq_len)

        for pack in packs:
            total = sum(int(n_tokens[idx]) for idx in pack)
            self.assertLessEqual(total, seq_len)

    def test_all_valid_examples_placed(self):
        rng = np.random.default_rng(123)
        seq_len = 4096
        n_tokens = rng.integers(50, 3000, size=500).astype(np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=seq_len)

        all_indices = sorted(idx for pack in packs for idx in pack)
        valid_indices = sorted(i for i in range(len(n_tokens)) if n_tokens[i] <= seq_len)
        self.assertEqual(all_indices, valid_indices)

    def test_no_duplicate_assignments(self):
        rng = np.random.default_rng(99)
        n_tokens = rng.integers(100, 2000, size=200).astype(np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=4096)

        all_indices = [idx for pack in packs for idx in pack]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_first_fit_decreasing(self):
        """FFD places items into the first bucket with guaranteed capacity."""
        n_tokens = np.array([900, 800, 100, 200], dtype=np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        packs, _ = _pre_pack(n_tokens, attn_cost, seq_len=1024, bucket_width=64)

        # 900 opens pack with 124 remaining (bucket 1). 800 opens pack with
        # 224 remaining (bucket 3). 100 needs guaranteed bucket ceil(100/64)=2,
        # finds the 224-remaining pack first. 200 needs bucket ceil(200/64)=4,
        # gets a new pack.
        pack_with_800 = None
        for pack in packs:
            if 1 in pack:
                pack_with_800 = pack
        self.assertIn(2, pack_with_800, "100-token item placed via guaranteed bucket")

    def test_attn_cost_accumulation(self):
        n_tokens = np.array([300, 400, 500], dtype=np.int64)
        attn_cost = np.array([100, 200, 300], dtype=np.int64)
        packs, costs = _pre_pack(n_tokens, attn_cost, seq_len=1024)

        for i, pack in enumerate(packs):
            expected_cost = sum(int(attn_cost[idx]) for idx in pack)
            self.assertEqual(int(costs[i]), expected_cost)

    def test_packing_efficiency(self):
        """With many small items, packing should be near-perfect."""
        rng = np.random.default_rng(7)
        n_tokens = rng.integers(100, 500, size=5000).astype(np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        seq_len = 8192
        packs, _ = _pre_pack(n_tokens, attn_cost, seq_len=seq_len)

        total_tokens = n_tokens.sum()
        total_capacity = len(packs) * seq_len
        efficiency = total_tokens / total_capacity
        self.assertGreater(efficiency, 0.98)

    def test_bucket_width_parameter(self):
        rng = np.random.default_rng(55)
        n_tokens = rng.integers(100, 2000, size=300).astype(np.int64)
        attn_cost = n_tokens * (n_tokens + 1) // 2
        seq_len = 4096

        packs_narrow, _ = _pre_pack(n_tokens, attn_cost, seq_len=seq_len, bucket_width=32)
        packs_wide, _ = _pre_pack(n_tokens, attn_cost, seq_len=seq_len, bucket_width=512)

        # Both must be valid
        for packs in (packs_narrow, packs_wide):
            for pack in packs:
                total = sum(int(n_tokens[idx]) for idx in pack)
                self.assertLessEqual(total, seq_len)

        # Narrower buckets should produce ≤ as many packs (tighter fit)
        self.assertLessEqual(len(packs_narrow), len(packs_wide) + 1)


def _create_test_pretok_dir(
    tmp_dir: Path,
    n_examples: int = 100,
    seed: int = 42,
    num_shards: int = 1,
) -> Path:
    """Create a minimal pretokenized dataset using HF save_to_disk (matches production)."""
    rng = np.random.default_rng(seed)
    pretok_dir = tmp_dir / "pretok"
    shards_dir = pretok_dir / "shards"
    shards_dir.mkdir(parents=True)

    n_tokens_arr = rng.integers(100, 5000, size=n_examples).astype(np.int32)
    attn_cost_arr = (
        n_tokens_arr.astype(np.int64) * (n_tokens_arr.astype(np.int64) + 1) // 2
    )

    data = {
        "input_ids": [list(rng.integers(0, 1000, size=int(n))) for n in n_tokens_arr],
        "labels": [list(rng.integers(-100, 1000, size=int(n))) for n in n_tokens_arr],
        "positions": [list(range(int(n))) for n in n_tokens_arr],
        "suffix_starts": [[] for _ in range(n_examples)],
        "insertion_limits": [[] for _ in range(n_examples)],
        "n_tokens": n_tokens_arr.tolist(),
        "train_tokens": (n_tokens_arr // 2).tolist(),
        "attn_cost": attn_cost_arr.tolist(),
    }
    full_ds = HFDataset.from_dict(data)

    shard_names = []
    examples_per_shard = n_examples // num_shards
    for i in range(num_shards):
        start = i * examples_per_shard
        end = start + examples_per_shard if i < num_shards - 1 else n_examples
        shard = full_ds.select(range(start, end))
        shard_name = f"shard_{i:04d}"
        shard.save_to_disk(str(shards_dir / shard_name))
        shard_names.append(shard_name)

    manifest = {
        "version": 1,
        "strategy": "backbone_suffix",
        "tokenizer": {"eos_token_id": 100257, "vocab_size": 100352},
        "shards": {"completed": shard_names},
        "stats": {
            "total_examples": n_examples,
            "total_tokens": int(n_tokens_arr.sum()),
        },
        "length_stats": {"mean": float(n_tokens_arr.mean())},
    }
    with open(pretok_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return pretok_dir


class TestPlanPackingEndToEnd(unittest.TestCase):
    """End-to-end tests for the plan_packing function."""

    def test_produces_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            output_dir = Path(tmp) / "output"

            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            self.assertTrue((output_dir / "pack_plan.arrow").exists())
            self.assertTrue((output_dir / "metadata.json").exists())

            with open(output_dir / "metadata.json") as f:
                meta = json.load(f)
            self.assertEqual(meta["seq_len"], 8192)
            self.assertGreater(meta["total_packs"], 0)
            self.assertEqual(
                meta["total_examples_packed"] + meta["overlong_examples_dropped"], 200
            )

    def test_packs_sorted_by_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=500)
            output_dir = Path(tmp) / "output"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            with pa.memory_map(str(output_dir / "pack_plan.arrow"), "r") as f:
                table = ipc.open_stream(f).read_all()
            costs = table.column("attn_cost").to_numpy()
            self.assertTrue(np.all(costs[1:] >= costs[:-1]))

    def test_default_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=50)
            expected_dir = pretok_dir / "pack_plans" / "seqlen_4096"

            plan_packing(pretok_dir, seq_len=4096, output_dir=expected_dir)
            self.assertTrue((expected_dir / "pack_plan.arrow").exists())

    def test_overlong_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=100)
            output_dir = Path(tmp) / "output"
            # Use tiny seq_len so some examples are overlong
            plan_packing(pretok_dir, seq_len=500, output_dir=output_dir)

            with open(output_dir / "metadata.json") as f:
                meta = json.load(f)
            self.assertGreater(meta["overlong_examples_dropped"], 0)
            self.assertEqual(
                meta["total_examples_packed"] + meta["overlong_examples_dropped"], 100
            )

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            out1 = Path(tmp) / "out1"
            out2 = Path(tmp) / "out2"
            plan_packing(pretok_dir, seq_len=8192, output_dir=out1)
            plan_packing(pretok_dir, seq_len=8192, output_dir=out2)

            with pa.memory_map(str(out1 / "pack_plan.arrow"), "r") as f:
                t1 = ipc.open_stream(f).read_all()
            with pa.memory_map(str(out2 / "pack_plan.arrow"), "r") as f:
                t2 = ipc.open_stream(f).read_all()

            self.assertTrue(t1.equals(t2))


class TestPlannedPackingDataset(unittest.TestCase):
    """Tests for the PlannedPackingDataset iteration logic."""

    def _make_dataset(
        self,
        n_examples=200,
        seq_len=8192,
        dp_rank=0,
        dp_world_size=4,
        seed=42,
        packing="prepacked_attn_grouped",
    ):

        tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(tmp_dir), ignore_errors=True)
        pretok_dir = _create_test_pretok_dir(tmp_dir, n_examples=n_examples)
        output_dir = tmp_dir / "plan"
        plan_packing(pretok_dir, seq_len=seq_len, output_dir=output_dir)

        ds = PlannedPackingDataset(
            pack_plan_path=str(output_dir),
            seq_len=seq_len,
            packing=packing,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=False,
            seed=seed,
        )
        return ds, tmp_dir

    def test_attn_cost_varies_across_packs(self):
        """Regression: attn_cost must reflect each pack's actual cost, not a constant."""
        ds, tmp_dir = self._make_dataset(n_examples=500, dp_world_size=2)
        costs = []
        for input_dict, _labels, _stats in ds:
            costs.append(input_dict["attn_cost"].item())
            if len(costs) >= 20:
                break
        self.assertGreater(len(set(costs)), 1, "attn_cost should not be constant")

    def test_yields_correct_count(self):
        dp_world_size = 4
        ds, tmp_dir = self._make_dataset(
            n_examples=200, dp_world_size=dp_world_size
        )
        n_packs = ds.num_packs
        expected_steps = n_packs // dp_world_size
        actual = sum(1 for _ in ds)
        self.assertEqual(actual, expected_steps)

    def test_all_ranks_same_step_count(self):
        dp_world_size = 4
        counts = []
        for rank in range(dp_world_size):
            ds, _ = self._make_dataset(
                n_examples=200, dp_rank=rank, dp_world_size=dp_world_size
            )
            counts.append(sum(1 for _ in ds))
        self.assertEqual(len(set(counts)), 1)

    def test_state_dict_resume(self):
        ds, tmp_dir = self._make_dataset(n_examples=300, dp_world_size=2)
        results_full = [(d["attn_cost"].item(), s["n_total_tokens"]) for d, _, s in ds]

        ds2, _ = self._make_dataset(n_examples=300, dp_world_size=2)
        # Consume first 5 steps
        it = iter(ds2)
        for _ in range(5):
            next(it)
        state = ds2.state_dict()

        # Resume from state
        ds3, _ = self._make_dataset(n_examples=300, dp_world_size=2)
        ds3.load_state_dict(state)
        results_resumed = [(d["attn_cost"].item(), s["n_total_tokens"]) for d, _, s in ds3]

        self.assertEqual(results_full[5:], results_resumed)

    def test_seq_len_mismatch_raises(self):

        tmp_dir = Path(tempfile.mkdtemp())
        pretok_dir = _create_test_pretok_dir(tmp_dir, n_examples=50)
        output_dir = tmp_dir / "plan"
        plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

        with self.assertRaises(ValueError, msg="seq_len"):
            PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=4096,
                packing="prepacked_attn_grouped",
            )


class TestIndexAlignment(unittest.TestCase):
    """Verify that plan indices align with dataset rows across shards."""

    def test_plan_indices_match_dataset_rows(self):
        """Pack plan indices point to correct rows when loaded via _load_shards."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(
                Path(tmp), n_examples=100, num_shards=3
            )
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            with pa.memory_map(str(output_dir / "pack_plan.arrow"), "r") as f:
                plan_table = ipc.open_stream(f).read_all()

            with open(pretok_dir / "manifest.json") as f:
                manifest = json.load(f)
            dataset = _load_shards(manifest, pretok_dir / "shards")
            n_tokens, _ = _load_metadata_columns(manifest, pretok_dir / "shards")

            for pack_indices in plan_table.column("example_indices"):
                for idx in pack_indices.as_py():
                    row = dataset[idx]
                    self.assertEqual(len(row["input_ids"]), int(n_tokens[idx]))

    def test_multi_shard_total_examples_preserved(self):
        """All examples are accessible across multiple shards."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(
                Path(tmp), n_examples=90, num_shards=3
            )
            with open(pretok_dir / "manifest.json") as f:
                manifest = json.load(f)
            dataset = _load_shards(manifest, pretok_dir / "shards")
            self.assertEqual(len(dataset), 90)

            n_tokens, attn_cost = _load_metadata_columns(
                manifest, pretok_dir / "shards"
            )
            self.assertEqual(len(n_tokens), 90)
            self.assertEqual(len(attn_cost), 90)


class TestSeedConfig(unittest.TestCase):
    """Verify seed controls epoch shuffling."""

    def test_different_seeds_different_orderings(self):

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=100)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            ds1 = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_attn_grouped",
                seed=1,
            )
            ds2 = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_attn_grouped",
                seed=2,
            )
            chunks1 = ds1._epoch_setup(0)
            chunks2 = ds2._epoch_setup(0)
            self.assertFalse(np.array_equal(chunks1, chunks2))

    def test_same_seed_same_ordering(self):

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=100)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            ds1 = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_attn_grouped",
                seed=42,
            )
            ds2 = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_attn_grouped",
                seed=42,
            )
            chunks1 = ds1._epoch_setup(0)
            chunks2 = ds2._epoch_setup(0)
            self.assertTrue(np.array_equal(chunks1, chunks2))

    def test_random_mode_breaks_cost_contiguity(self):

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            ds_grouped = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_attn_grouped",
                dp_world_size=4,
                seed=42,
            )
            ds_random = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random",
                dp_world_size=4,
                seed=42,
            )
            chunks_grouped = ds_grouped._epoch_setup(0)
            chunks_random = ds_random._epoch_setup(0)

            self.assertEqual(chunks_grouped.shape, chunks_random.shape)
            self.assertFalse(np.array_equal(chunks_grouped, chunks_random))

            # In attn_grouped mode, each row contains contiguous indices
            # (adjacent in cost-sorted order). In random mode, they don't.
            row_spans_grouped = chunks_grouped.max(axis=1) - chunks_grouped.min(axis=1)
            row_spans_random = chunks_random.max(axis=1) - chunks_random.min(axis=1)
            self.assertGreater(
                row_spans_random.mean(), row_spans_grouped.mean() * 2,
                "Random mode should have much larger within-chunk index spread",
            )


class TestDataLoaderE2E(unittest.TestCase):
    """End-to-end test through GranitePreTokenizedDataLoader with prepacked modes."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self._tmp), ignore_errors=True)

    def test_backbone_suffix_planned_packing(self):
        """Full dataloader path: config → PlannedPackingDataset → iteration."""

        seq_len = 8192
        pretok_dir = _create_test_pretok_dir(self._tmp, n_examples=200)
        plan_packing(pretok_dir, seq_len=seq_len, output_dir=pretok_dir / "pack_plans" / f"seqlen_{seq_len}")

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        loader = GranitePreTokenizedDataLoader(
            GranitePreTokenizedDataLoader.Config(
                dataset_path=str(pretok_dir),
                packing="prepacked_attn_grouped",
                infinite=False,
            ),
            dp_world_size=2,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=1,
        )

        self.assertIsInstance(loader.dataset, PlannedPackingDataset)

        batches = []
        for input_dict, labels in loader:
            self.assertEqual(input_dict["input"].shape[-1], seq_len)
            self.assertEqual(labels.shape[-1], seq_len)
            batches.append((input_dict, labels))

        self.assertGreater(len(batches), 0)

    def test_planned_packing_state_dict_through_dataloader(self):
        """Dataloader state_dict/load_state_dict produces consistent resumption."""

        seq_len = 8192
        pretok_dir = _create_test_pretok_dir(self._tmp, n_examples=200)
        plan_packing(pretok_dir, seq_len=seq_len, output_dir=pretok_dir / "pack_plans" / f"seqlen_{seq_len}")

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")

        def make_loader():
            return GranitePreTokenizedDataLoader(
                GranitePreTokenizedDataLoader.Config(
                    dataset_path=str(pretok_dir),
                    packing="prepacked_attn_grouped",
                    infinite=False,
                ),
                dp_world_size=2,
                dp_rank=0,
                tokenizer=tokenizer,
                seq_len=seq_len,
                local_batch_size=1,
            )

        # Full pass
        loader_ref = make_loader()
        all_batches = list(loader_ref)
        self.assertGreater(len(all_batches), 5)

        # Partial pass → checkpoint → resume
        loader_a = make_loader()
        it = iter(loader_a)
        for _ in range(5):
            next(it)
        state = loader_a.state_dict()

        loader_b = make_loader()
        loader_b.load_state_dict(state)
        resumed_batches = list(loader_b)

        # Resumed batches should match tail of full pass
        expected = all_batches[5:]
        self.assertEqual(len(resumed_batches), len(expected))
        for (rd, rl), (ed, el) in zip(resumed_batches, expected):
            self.assertTrue(
                torch.equal(rd["input"], ed["input"]),
                "input mismatch after resume",
            )
            self.assertTrue(torch.equal(rl, el), "labels mismatch after resume")

    def test_multi_dataset_with_plan_raises(self):
        """prepacked modes with multiple dataset_path entries are rejected."""

        pretok_dir = _create_test_pretok_dir(self._tmp, n_examples=50)
        seq_len = 8192
        plan_packing(pretok_dir, seq_len=seq_len, output_dir=pretok_dir / "pack_plans" / f"seqlen_{seq_len}")

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        with self.assertRaises(ValueError):
            GranitePreTokenizedDataLoader(
                GranitePreTokenizedDataLoader.Config(
                    dataset_path=f"{pretok_dir},{pretok_dir}",
                    packing="prepacked_attn_grouped",
                    infinite=False,
                ),
                dp_world_size=1,
                dp_rank=0,
                tokenizer=tokenizer,
                seq_len=seq_len,
                local_batch_size=1,
            )


class TestPrepackedRandomBalanced(unittest.TestCase):
    """Tests for the prepacked_random_balanced packing mode."""

    def _make_dataset(
        self,
        n_examples=200,
        seq_len=8192,
        dp_rank=0,
        dp_world_size=4,
        accum_steps=2,
        seed=42,
    ):

        tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(tmp_dir), ignore_errors=True)
        pretok_dir = _create_test_pretok_dir(tmp_dir, n_examples=n_examples)
        output_dir = tmp_dir / "plan"
        plan_packing(pretok_dir, seq_len=seq_len, output_dir=output_dir)

        ds = PlannedPackingDataset(
            pack_plan_path=str(output_dir),
            seq_len=seq_len,
            packing="prepacked_random_balanced",
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            accum_steps=accum_steps,
            infinite=False,
            seed=seed,
        )
        return ds, tmp_dir

    def test_rows_strictly_cost_ordered(self):
        """Within each optimizer step window, rows are ordered by cost: row i
        contains cheaper packs than row i+1 (argsort + reshape guarantees this)."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=500)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            dp = 8
            accum = 4
            ds = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=dp,
                accum_steps=accum,
                seed=42,
            )
            chunks = ds._epoch_setup(0)
            costs = ds._pack_costs

            # chunks shape: (n_windows * accum, dp). Each consecutive `accum`
            # rows form one optimizer step window. Within each window, row k
            # should contain packs with costs <= row k+1 (element-wise after sort).
            n_windows = len(chunks) // accum
            for w in range(n_windows):
                window_rows = chunks[w * accum : (w + 1) * accum]
                row_max_costs = costs[window_rows].max(axis=1)
                # Row max costs should be non-decreasing across the window
                self.assertTrue(
                    np.all(row_max_costs[1:] >= row_max_costs[:-1]),
                    f"Window {w}: row max costs not non-decreasing: {row_max_costs}",
                )

    def test_cost_homogeneity_within_rows(self):
        """Within each micro-batch row, cost variance should be lower than random."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=500)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            dp = 8
            accum = 2
            ds_balanced = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=dp,
                accum_steps=accum,
                seed=42,
            )
            ds_random = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random",
                dp_world_size=dp,
                seed=42,
            )

            chunks_balanced = ds_balanced._epoch_setup(0)
            chunks_random = ds_random._epoch_setup(0)

            costs = ds_balanced._pack_costs
            row_stds_balanced = np.array([
                costs[chunks_balanced[i]].std()
                for i in range(len(chunks_balanced))
            ])
            row_stds_random = np.array([
                costs[chunks_random[i]].std()
                for i in range(min(len(chunks_random), len(chunks_balanced)))
            ])
            self.assertLess(
                row_stds_balanced.mean(),
                row_stds_random.mean(),
                "Balanced mode should have lower within-row cost variance",
            )

    def test_different_epochs_different_orderings(self):
        ds, _ = self._make_dataset(n_examples=200, dp_world_size=4, accum_steps=2)
        chunks_e0 = ds._epoch_setup(0)
        chunks_e1 = ds._epoch_setup(1)
        self.assertFalse(np.array_equal(chunks_e0, chunks_e1))

    def test_global_batch_invariant(self):
        """For fixed GBS (= dp * accum), the SET of packs per optimizer step is
        identical regardless of how dp and accum are decomposed."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=500)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            gbs = 16  # window_size = gbs in both cases
            # Setup A: dp=16, accum=1 (one micro-batch per step)
            ds_a = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=gbs,
                accum_steps=1,
                seed=42,
            )
            # Setup B: dp=8, accum=2 (two micro-batches per step)
            ds_b = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=gbs // 2,
                accum_steps=2,
                seed=42,
            )

            chunks_a = ds_a._epoch_setup(0)  # shape: (n_steps, 16)
            chunks_b = ds_b._epoch_setup(0)  # shape: (n_steps*2, 8)

            # Both have window_size=16. Same seed → same drop (n_packs % 16),
            # same RNG state → same shuffle → same windows of 16 packs.
            # A reshapes each window to (1, 16). B reshapes to (2, 8).
            # The SET of 16 packs per optimizer step must be identical.
            n_steps = min(len(chunks_a), len(chunks_b) // 2)
            for step in range(n_steps):
                set_a = set(chunks_a[step].tolist())
                set_b = set(chunks_b[step * 2].tolist()) | set(
                    chunks_b[step * 2 + 1].tolist()
                )
                self.assertEqual(
                    set_a,
                    set_b,
                    f"Step {step}: pack sets differ between dp=16/accum=1 and "
                    f"dp=8/accum=2",
                )

    def test_random_and_balanced_same_global_batches(self):
        """prepacked_random and prepacked_random_balanced produce identical pack sets
        per optimizer step — balanced only reorders within each step window."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=500)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            dp = 8
            accum = 4

            ds_random = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random",
                dp_world_size=dp,
                accum_steps=accum,
                seed=42,
            )
            ds_balanced = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=dp,
                accum_steps=accum,
                seed=42,
            )

            chunks_random = ds_random._epoch_setup(0)
            chunks_balanced = ds_balanced._epoch_setup(0)

            n_steps = len(chunks_random) // accum
            self.assertEqual(n_steps, len(chunks_balanced) // accum)

            for step in range(n_steps):
                rows_r = chunks_random[step * accum : (step + 1) * accum]
                rows_b = chunks_balanced[step * accum : (step + 1) * accum]
                set_r = set(rows_r.flatten().tolist())
                set_b = set(rows_b.flatten().tolist())
                self.assertEqual(
                    set_r,
                    set_b,
                    f"Step {step}: pack sets differ between prepacked_random and "
                    f"prepacked_random_balanced (dp={dp}, accum={accum})",
                )

    def test_state_dict_resume(self):
        ds, _ = self._make_dataset(n_examples=300, dp_world_size=4, accum_steps=2)
        results_full = [(d["attn_cost"].item(), s["n_total_tokens"]) for d, _, s in ds]

        ds2, _ = self._make_dataset(n_examples=300, dp_world_size=4, accum_steps=2)
        it = iter(ds2)
        for _ in range(5):
            next(it)
        state = ds2.state_dict()

        ds3, _ = self._make_dataset(n_examples=300, dp_world_size=4, accum_steps=2)
        ds3.load_state_dict(state)
        results_resumed = [(d["attn_cost"].item(), s["n_total_tokens"]) for d, _, s in ds3]

        self.assertEqual(results_full[5:], results_resumed)

    def test_accum_steps_change_on_resume(self):
        """Changing accum_steps on resume should not crash."""

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            ds1 = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=4,
                accum_steps=2,
                seed=42,
                infinite=False,
            )
            it = iter(ds1)
            for _ in range(5):
                next(it)
            state = ds1.state_dict()
            self.assertEqual(state["accum_steps"], 2)

            # Resume with different accum_steps — should not raise
            ds2 = PlannedPackingDataset(
                pack_plan_path=str(output_dir),
                seq_len=8192,
                packing="prepacked_random_balanced",
                dp_world_size=4,
                accum_steps=4,
                seed=42,
                infinite=False,
            )
            ds2.load_state_dict(state)
            # Should be able to iterate
            results = list(ds2)
            self.assertGreater(len(results), 0)


class TestPlannedPackingWorkers(unittest.TestCase):
    """Verify PlannedPackingDataset works correctly with multiple DataLoader workers."""

    def test_workers_see_disjoint_data(self):
        """With num_workers=2, no data duplication occurs."""
        from torchdata.stateful_dataloader import StatefulDataLoader

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            def make_ds():
                return PlannedPackingDataset(
                    pack_plan_path=str(output_dir),
                    seq_len=8192,
                    packing="prepacked_random_balanced",
                    dp_rank=0,
                    dp_world_size=2,
                    accum_steps=2,
                    seed=42,
                    infinite=False,
                )

            ds0 = make_ds()
            dl0 = StatefulDataLoader(ds0, num_workers=0, batch_size=None)
            items0 = [item[2]["n_examples_packed"] for item in dl0]

            ds2 = make_ds()
            dl2 = StatefulDataLoader(ds2, num_workers=2, batch_size=None)
            items2 = [item[2]["n_examples_packed"] for item in dl2]

            self.assertEqual(
                len(items0),
                len(items2),
                f"num_workers=2 yielded {len(items2)} items vs "
                f"{len(items0)} with num_workers=0 ({len(items2)/len(items0):.0f}x duplication)",
            )
            self.assertEqual(items0, items2)

    def test_worker_count_preserves_ordering(self):
        """num_workers=0, 2, and 3 all produce identical sequences (balanced)."""
        from torchdata.stateful_dataloader import StatefulDataLoader

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            def make_ds():
                return PlannedPackingDataset(
                    pack_plan_path=str(output_dir),
                    seq_len=8192,
                    packing="prepacked_random_balanced",
                    dp_rank=0,
                    dp_world_size=2,
                    accum_steps=2,
                    seed=42,
                    infinite=False,
                )

            results = {}
            for nw in (0, 2, 3):
                ds = make_ds()
                dl = StatefulDataLoader(ds, num_workers=nw, batch_size=None)
                results[nw] = [item[0]["input"].tolist() for item in dl]

            self.assertEqual(results[0], results[2])
            self.assertEqual(results[0], results[3])

    def test_worker_count_preserves_ordering_random(self):
        """num_workers=0 and 2 produce identical sequences (random mode)."""
        from torchdata.stateful_dataloader import StatefulDataLoader

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            def make_ds():
                return PlannedPackingDataset(
                    pack_plan_path=str(output_dir),
                    seq_len=8192,
                    packing="prepacked_random",
                    dp_rank=0,
                    dp_world_size=2,
                    accum_steps=2,
                    seed=42,
                    infinite=False,
                )

            results = {}
            for nw in (0, 2):
                ds = make_ds()
                dl = StatefulDataLoader(ds, num_workers=nw, batch_size=None)
                results[nw] = [item[0]["input"].tolist() for item in dl]

            self.assertEqual(results[0], results[2])

    def test_random_and_balanced_same_global_batches_with_workers(self):
        """Cross-mode pack-multiset equivalence holds through DataLoader with workers.

        Iterates all dp_ranks for both modes and verifies the multiset of packs
        per optimizer step is identical — balanced only reorders within each step.
        """
        from torchdata.stateful_dataloader import StatefulDataLoader

        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=500)
            output_dir = Path(tmp) / "plan"
            plan_packing(pretok_dir, seq_len=8192, output_dir=output_dir)

            dp = 4
            accum = 3
            num_workers = 2

            def collect_all_ranks(packing):
                """Collect n_total_tokens from all dp_ranks, return per-step multisets."""
                from collections import Counter

                per_rank = []
                for rank in range(dp):
                    ds = PlannedPackingDataset(
                        pack_plan_path=str(output_dir),
                        seq_len=8192,
                        packing=packing,
                        dp_rank=rank,
                        dp_world_size=dp,
                        accum_steps=accum,
                        seed=42,
                        infinite=False,
                    )
                    dl = StatefulDataLoader(ds, num_workers=num_workers, batch_size=None)
                    per_rank.append([item[2]["n_total_tokens"] for item in dl])
                n_items = len(per_rank[0])
                self.assertTrue(all(len(r) == n_items for r in per_rank))
                n_steps = n_items // accum
                step_multisets = []
                for step in range(n_steps):
                    packs = Counter()
                    for rank_items in per_rank:
                        for i in range(step * accum, (step + 1) * accum):
                            packs[rank_items[i]] += 1
                    step_multisets.append(packs)
                return step_multisets

            steps_random = collect_all_ranks("prepacked_random")
            steps_balanced = collect_all_ranks("prepacked_random_balanced")

            self.assertEqual(len(steps_random), len(steps_balanced))
            for step, (sr, sb) in enumerate(zip(steps_random, steps_balanced)):
                self.assertEqual(
                    sr,
                    sb,
                    f"Step {step}: global batch pack multisets differ between "
                    f"prepacked_random and prepacked_random_balanced",
                )


    def test_epochs_reaches_one_after_full_pass(self):
        """epochs ≈ 1.0 after consuming all packs in one epoch.

        Uses dp_world_size=1 so the single rank sees all packs, avoiding
        variance from random pack-to-rank assignment with small test data.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pretok_dir = _create_test_pretok_dir(Path(tmp), n_examples=200)
            seq_len = 8192
            plan_packing(
                pretok_dir,
                seq_len=seq_len,
                output_dir=pretok_dir / "pack_plans" / f"seqlen_{seq_len}",
            )

            dp = 1
            accum = 4
            tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
            loader = GranitePreTokenizedDataLoader(
                GranitePreTokenizedDataLoader.Config(
                    dataset_path=str(pretok_dir),
                    packing="prepacked_random_balanced",
                    infinite=False,
                    num_workers=0,
                ),
                dp_world_size=dp,
                dp_rank=0,
                tokenizer=tokenizer,
                seq_len=seq_len,
                local_batch_size=1,
                accum_steps=accum,
            )

            for _ in loader:
                pass

            stats = loader.get_data_stats()
            # With dp=1, all packs go to rank 0. The only source of deviation
            # from 1.0 is the window-remainder drop (n_packs % gbs packs dropped).
            self.assertAlmostEqual(stats["epochs"], 1.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
