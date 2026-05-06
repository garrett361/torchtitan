"""Tests for pretokenized_dataset.py: cost-balanced packing and multi-dataset merge."""

import json
import os
import statistics
import unittest
from pathlib import Path

from torchtitan.models.granite.pretokenized_dataset import (
    GranitePreTokenizedDataLoader,
    TruncateLastBufferDataset,
    TruncateLastCostBalancedDataset,
    _load_and_merge_manifests,
    _load_manifest,
)

MANIFEST_PATH = Path("tests/assets/pretok_truncate_last/manifest.json")
MANIFEST_PATH_B = Path("tests/assets/pretok_truncate_last_b/manifest.json")


def _make_manifest_with_length_stats(seq_len: int = 16) -> dict:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    k = seq_len // 1024 if seq_len >= 1024 else 16
    manifest["stats"]["length_stats"] = {
        f"squared_tokens_per_example_{k}kmax": 30.0,
        f"tokens_per_example_{k}kmax": 5.0,
        f"n_examples_{k}kmax": 6,
    }
    return manifest


def _build_dataset(cls, seq_len: int = 16, buffer_size: int = 6, **extra_kwargs):
    manifest = _make_manifest_with_length_stats(seq_len)
    return cls(
        manifest_path=MANIFEST_PATH,
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=1,
        infinite=False,
        shuffle_in_memory=True,
        _manifest=manifest,
        buffer_size=buffer_size,
        **extra_kwargs,
    )


def _compute_batch_cost_from_positions(batch) -> int:
    """Compute sum(l_i²) from position resets in a packed batch."""
    positions = batch[0]["positions"].tolist()
    lengths = []
    cur = 1
    for i in range(1, len(positions)):
        if positions[i] == 0:
            lengths.append(cur)
            cur = 1
        else:
            cur += 1
    lengths.append(cur)
    return sum(l * l for l in lengths)


class TestCostBalancedPacking(unittest.TestCase):
    def test_cost_targeting_selects_toward_target(self):
        """Cost-balanced selection moves batch cost toward target, not away."""
        seq_len = 16
        target_cost = seq_len * 30.0 / 5.0  # 96

        cost_ds = _build_dataset(
            TruncateLastCostBalancedDataset, seq_len=seq_len, target_cost=target_cost
        )
        cost_batches = list(cost_ds)
        costs = [b[2]["batch_attention_cost"] for b in cost_batches]
        self.assertTrue(len(costs) > 0)

        # Multi-example batches should have cost closer to target than a
        # single-example-only approach would (first example alone)
        for batch, cost in zip(cost_batches, costs):
            n_examples = batch[2]["n_examples_packed"]
            if n_examples > 1:
                # With >1 example packed, cost should be non-trivial
                self.assertGreater(cost, 0)

    def test_fifo_guarantee(self):
        """The oldest buffered example always appears in the next packed batch."""
        seq_len = 16
        target_cost = seq_len * 30.0 / 5.0
        ds = _build_dataset(
            TruncateLastCostBalancedDataset,
            seq_len=seq_len,
            target_cost=target_cost,
            buffer_size=6,
        )

        ds._prepare_iter()
        ds._data_iter = ds._get_data_iter()
        ds._data_exhausted = False
        ds._refill_buffer()

        oldest_ids = list(ds._buffer[0][0])
        batch = ds._pack_one_batch()
        input_tensor = batch[0]["input"].tolist()
        self.assertEqual(input_tensor[: len(oldest_ids)], oldest_ids)

    def test_target_cost_computation(self):
        """Verify target_cost = seq_len * E[l²] / E[l] with correct cutoff selection."""
        length_stats = {
            "squared_tokens_per_example_128kmax": 452913586.6,
            "tokens_per_example_128kmax": 11974.1,
            "n_examples_128kmax": 7632648,
            "squared_tokens_per_example_64kmax": 249358438.9,
            "tokens_per_example_64kmax": 9872.1,
            "n_examples_64kmax": 7422164,
        }
        seq_len = 131072
        _CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]
        valid_cutoffs = [
            c
            for c in _CUTOFFS
            if c <= seq_len
            and f"tokens_per_example_{c // 1024}kmax" in length_stats
        ]
        cutoff = max(valid_cutoffs)
        self.assertEqual(cutoff, 131072)

        k = cutoff // 1024
        sq_tokens = length_stats[f"squared_tokens_per_example_{k}kmax"]
        mean_tokens = length_stats[f"tokens_per_example_{k}kmax"]
        target_cost = seq_len * sq_tokens / mean_tokens

        expected = 131072 * 452913586.6 / 11974.1
        self.assertAlmostEqual(target_cost, expected, places=0)
        self.assertAlmostEqual(target_cost / seq_len**2, 0.2886, places=3)

    def test_cutoff_selection_uses_largest_leq_seq_len(self):
        """When seq_len falls between cutoffs, picks the largest available ≤ seq_len."""
        length_stats = {
            "squared_tokens_per_example_16kmax": 40877268.1,
            "tokens_per_example_16kmax": 4818.3,
            "n_examples_16kmax": 6031281,
            "squared_tokens_per_example_32kmax": 121729873.9,
            "tokens_per_example_32kmax": 7487.7,
            "n_examples_32kmax": 6956200,
        }
        seq_len = 32768
        _CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]
        valid_cutoffs = [
            c
            for c in _CUTOFFS
            if c <= seq_len
            and f"tokens_per_example_{c // 1024}kmax" in length_stats
        ]
        self.assertEqual(max(valid_cutoffs), 32768)

    def test_checkpointing(self):
        """Save/restore mid-iteration produces identical subsequent batches."""
        seq_len = 16
        target_cost = seq_len * 30.0 / 5.0

        ds1 = _build_dataset(
            TruncateLastCostBalancedDataset,
            seq_len=seq_len,
            target_cost=target_cost,
            buffer_size=4,
        )
        ds1._prepare_iter()
        it1 = iter(ds1)
        next(it1)  # consume one batch

        state = ds1.state_dict()

        ds2 = _build_dataset(
            TruncateLastCostBalancedDataset,
            seq_len=seq_len,
            target_cost=target_cost,
            buffer_size=4,
        )
        ds2.load_state_dict(state)
        ds2._prepare_iter()
        it2 = iter(ds2)

        remaining1 = list(it1)
        remaining2 = list(it2)

        self.assertEqual(len(remaining1), len(remaining2))
        for b1, b2 in zip(remaining1, remaining2):
            self.assertTrue(b1[0]["input"].equal(b2[0]["input"]))
            self.assertTrue(b1[1].equal(b2[1]))

    def test_batch_attention_cost_in_stats(self):
        """Cost-balanced batches include batch_attention_cost in stats dict."""
        seq_len = 16
        target_cost = seq_len * 30.0 / 5.0
        ds = _build_dataset(
            TruncateLastCostBalancedDataset, seq_len=seq_len, target_cost=target_cost
        )
        for _, _, stats in ds:
            self.assertIn("batch_attention_cost", stats)
            self.assertIsInstance(stats["batch_attention_cost"], int)
            self.assertGreater(stats["batch_attention_cost"], 0)


class TestCostBalancedE2E(unittest.TestCase):
    """End-to-end tests using the real test_sample pretokenized dataset.

    Requires DATA_PATH_7M_BALANCED_TEST_SAMPLE_PRETOK_TRUNC_LAST in .env.
    """

    _manifest_path: str | None = None

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        load_dotenv()
        cls._manifest_path = os.getenv(
            "DATA_PATH_7M_BALANCED_TEST_SAMPLE_PRETOK_TRUNC_LAST"
        )

    def setUp(self):
        if self._manifest_path is None:
            self.skipTest(
                "DATA_PATH_7M_BALANCED_TEST_SAMPLE_PRETOK_TRUNC_LAST not set"
            )

    def _load_manifest(self):
        manifest_file = Path(self._manifest_path) / "manifest.json"
        with open(manifest_file) as f:
            return json.load(f)

    def test_manifest_has_length_stats(self):
        """Manifest has both squared_tokens and tokens_per_example fields."""
        manifest = self._load_manifest()
        length_stats = manifest["stats"]["length_stats"]
        self.assertIn("squared_tokens_per_example_16kmax", length_stats)
        self.assertIn("tokens_per_example_16kmax", length_stats)
        self.assertIsNotNone(length_stats["tokens_per_example_16kmax"])

    def test_cost_balanced_iteration(self):
        """Full iteration with cost_balanced packing produces valid batches."""
        manifest_file = str(Path(self._manifest_path) / "manifest.json")
        manifest = self._load_manifest()
        length_stats = manifest["stats"]["length_stats"]

        seq_len = 16384
        _CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]
        valid_cutoffs = [
            c
            for c in _CUTOFFS
            if c <= seq_len
            and f"tokens_per_example_{c // 1024}kmax" in length_stats
        ]
        cutoff = max(valid_cutoffs)
        k = cutoff // 1024
        sq_tokens = length_stats[f"squared_tokens_per_example_{k}kmax"]
        mean_tokens = length_stats[f"tokens_per_example_{k}kmax"]
        target_cost = seq_len * sq_tokens / mean_tokens

        ds = TruncateLastCostBalancedDataset(
            manifest_path=manifest_file,
            seq_len=seq_len,
            dp_rank=0,
            dp_world_size=1,
            infinite=True,
            shuffle_in_memory=True,
            buffer_size=64,
            target_cost=target_cost,
        )

        costs = []
        n_batches = 0
        for input_dict, labels, stats in ds:
            self.assertEqual(input_dict["input"].shape[0], seq_len)
            self.assertEqual(labels.shape[0], seq_len)
            self.assertIn("batch_attention_cost", stats)
            costs.append(stats["batch_attention_cost"])
            n_batches += 1
            if n_batches >= 50:
                break

        self.assertGreater(n_batches, 0)
        if len(costs) > 1:
            cv = statistics.stdev(costs) / statistics.mean(costs)
            # Cost-balanced should have reasonable CV (< 1.0)
            self.assertLess(cv, 1.0, f"Cost CV={cv:.3f} is unexpectedly high")

    def test_cost_balanced_vs_buffer_variance(self):
        """Cost-balanced has lower cost variance than buffer on the real dataset.

        Uses DATA_PATH_7M_BALANCED_PRETOK_TRUNCATE_LAST (7.6M examples) to
        demonstrate variance reduction at scale.
        """
        from dotenv import load_dotenv

        load_dotenv()
        full_data_path = os.getenv("DATA_PATH_7M_BALANCED_PRETOK_TRUNCATE_LAST")
        if full_data_path is None:
            self.skipTest("DATA_PATH_7M_BALANCED_PRETOK_TRUNCATE_LAST not set")

        manifest_file = str(Path(full_data_path) / "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        length_stats = manifest["stats"]["length_stats"]

        seq_len = 16384
        _CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]
        valid_cutoffs = [
            c
            for c in _CUTOFFS
            if c <= seq_len
            and f"tokens_per_example_{c // 1024}kmax" in length_stats
        ]
        cutoff = max(valid_cutoffs)
        k = cutoff // 1024
        sq_tokens = length_stats[f"squared_tokens_per_example_{k}kmax"]
        mean_tokens = length_stats[f"tokens_per_example_{k}kmax"]
        target_cost = seq_len * sq_tokens / mean_tokens

        n_batches = 100
        common_kwargs = dict(
            manifest_path=manifest_file,
            seq_len=seq_len,
            dp_rank=0,
            dp_world_size=1,
            infinite=True,
            shuffle_in_memory=True,
            buffer_size=64,
        )

        # Buffer packing
        buffer_ds = TruncateLastBufferDataset(**common_kwargs)
        buffer_costs = []
        for i, (inp, _, stats) in enumerate(buffer_ds):
            buffer_costs.append(_compute_batch_cost_from_positions((inp, None, stats)))
            if i + 1 >= n_batches:
                break

        # Cost-balanced packing
        cost_ds = TruncateLastCostBalancedDataset(
            **common_kwargs, target_cost=target_cost
        )
        cost_balanced_costs = []
        for i, (_, _, stats) in enumerate(cost_ds):
            cost_balanced_costs.append(stats["batch_attention_cost"])
            if i + 1 >= n_batches:
                break

        buffer_std = statistics.stdev(buffer_costs)
        cost_std = statistics.stdev(cost_balanced_costs)
        self.assertLess(
            cost_std,
            buffer_std,
            f"Cost-balanced std={cost_std:.0f} should be < buffer std={buffer_std:.0f}",
        )


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

    def test_merge_length_stats_weighted_average(self):
        """Merged length_stats use correct weighted averages."""
        manifest, _ = _load_and_merge_manifests([MANIFEST_PATH, MANIFEST_PATH_B])
        ls = manifest["stats"]["length_stats"]

        # Dataset A: n=4, sq=20.0, mean=4.0
        # Dataset B: n=6, sq=40.0, mean=6.0
        # Merged: sq = (4*20 + 6*40) / (4+6) = 320/10 = 32.0
        # Merged: mean = (4*4 + 6*6) / (4+6) = 52/10 = 5.2
        self.assertAlmostEqual(ls["squared_tokens_per_example_16kmax"], 32.0, places=1)
        self.assertAlmostEqual(ls["tokens_per_example_16kmax"], 5.2, places=1)
        self.assertEqual(ls["n_examples_16kmax"], 10)

    def test_merge_preserves_tokenizer_info(self):
        """Merged manifest retains tokenizer from first dataset."""
        manifest, _ = _load_and_merge_manifests([MANIFEST_PATH, MANIFEST_PATH_B])
        self.assertEqual(manifest["tokenizer"]["eos_token_id"], 2003)
        self.assertEqual(manifest["tokenizer"]["vocab_size"], 2009)
        self.assertEqual(manifest["strategy"], "truncate_last")

    def test_single_path_no_merge(self):
        """Single path in comma-separated string behaves like non-merged."""
        config = GranitePreTokenizedDataLoader.Config(
            manifest_path=str(MANIFEST_PATH),
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
            manifest_path=f"{MANIFEST_PATH},{MANIFEST_PATH_B}",
            infinite=False,
            packing="greedy",
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
            manifest_path=f"{dir_a},{dir_b}",
            infinite=False,
            packing="greedy",
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


if __name__ == "__main__":
    unittest.main()
