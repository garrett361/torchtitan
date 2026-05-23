"""Verifies that save_to_disk correctly resolves the _indices mapping from
select() on datasets >= 4.x, making an explicit flatten_indices() unnecessary.
Also exercises the overlapped compute/write pattern used in _shuffle_and_reshard.
"""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk


def _make_dataset(n: int = 100) -> Dataset:
    return Dataset.from_dict(
        {"id": list(range(n)), "value": [f"row_{i}" for i in range(n)]}
    )


class TestSelectSaveToDisk(unittest.TestCase):
    """save_to_disk resolves _indices internally on datasets >= 4.x."""

    def test_select_carries_indices_mapping(self):
        ds = _make_dataset(100)
        perm = np.random.default_rng(42).permutation(len(ds))
        selected = ds.select(perm)

        self.assertIsNotNone(
            selected._indices,
            "select() should attach an _indices mapping",
        )

    def test_save_to_disk_resolves_indices(self):
        """save_to_disk handles the gather internally — no flatten needed."""
        ds = _make_dataset(100)
        perm = np.random.default_rng(42).permutation(len(ds))
        selected = ds.select(perm)

        with tempfile.TemporaryDirectory() as tmpdir:
            selected.save_to_disk(tmpdir)
            reloaded = Dataset.load_from_disk(tmpdir)

        self.assertEqual(reloaded["id"], perm.tolist())

    def test_numpy_indices_accepted(self):
        """select() accepts numpy arrays directly (no .tolist() needed)."""
        ds = _make_dataset(100)
        perm = np.random.default_rng(42).permutation(len(ds))
        selected = ds.select(perm)

        self.assertEqual(selected["id"], perm.tolist())


class TestOverlappedShuffleWrite(unittest.TestCase):
    """Exercises the overlapped compute/write pattern from _shuffle_and_reshard."""

    def test_overlapped_write_produces_correct_shards(self):
        """Drain at the top of each iteration so compute overlaps prior write."""
        n = 200
        num_shards = 4
        ds = _make_dataset(n)
        rng = np.random.default_rng(99)
        permutation = rng.permutation(n)
        examples_per_shard = n // num_shards

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            pending_future: Future | None = None

            with ThreadPoolExecutor(max_workers=1) as pool:
                for i in range(num_shards):
                    if pending_future is not None:
                        pending_future.result()

                    start = i * examples_per_shard
                    end = (
                        start + examples_per_shard
                        if i < num_shards - 1
                        else n
                    )
                    indices = permutation[start:end]
                    shard = ds.select(indices)

                    shard_path = str(out / f"shard_{i:04d}")
                    pending_future = pool.submit(shard.save_to_disk, shard_path)

                if pending_future is not None:
                    pending_future.result()

            all_ids: list[int] = []
            for i in range(num_shards):
                reloaded = load_from_disk(str(out / f"shard_{i:04d}"))
                all_ids.extend(reloaded["id"])

            self.assertEqual(all_ids, permutation.tolist())

    def test_overlapped_write_error_is_swallowed_and_recorded(self):
        """Mirrors _drain_pending: write errors are caught, not re-raised."""
        errors_recorded: list[str] = []

        def _drain_pending(fut: Future | None) -> None:
            if fut is None:
                return
            try:
                fut.result()
            except Exception as e:
                errors_recorded.append(str(e))

        def _failing_write(shard, path):
            raise IOError("disk full")

        ds = _make_dataset(50)
        perm = np.random.default_rng(0).permutation(len(ds))
        shard = ds.select(perm)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_failing_write, shard, "/fake")
            _drain_pending(fut)

        self.assertEqual(len(errors_recorded), 1)
        self.assertIn("disk full", errors_recorded[0])


if __name__ == "__main__":
    unittest.main()
