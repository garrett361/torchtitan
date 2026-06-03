"""DCP (distributed checkpoint) round-trip tests for GranitePreTokenizedDataLoader.

Tests that the full state dict pipeline:
  GranitePreTokenizedDataLoader → ParallelAwareDataloader (pickle + rank keying)
    → dcp.save (filesystem) → dcp.load → ParallelAwareDataloader → resume
survives a round-trip through torch.distributed.checkpoint.

All tests use no_dist=True (single-process, no GPU required).
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch.distributed.checkpoint as dcp
from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.granite.pretokenized_dataset import GranitePreTokenizedDataLoader

_EOS_ID = 2003

# 20 examples with varied lengths (3-8 tokens each) so that packing produces
# distinct batches at different positions, preventing epoch-wrap coincidences.
_EXAMPLES = [
    ([i * 100 + j for j in range(length)] + [_EOS_ID],
     [IGNORE_INDEX] * length + [_EOS_ID])
    for i, length in enumerate([
        3, 5, 4, 7, 3, 6, 4, 5, 8, 3,
        6, 4, 7, 5, 3, 8, 4, 6, 5, 7,
    ])
]


def _make_shard(tmp_path: Path, examples: list[tuple[list[int], list[int]]]) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    n_tokens_list = [len(ids) for ids, _ in examples]
    ds = Dataset.from_dict(
        {
            "input_ids": [ids for ids, _ in examples],
            "labels": [lbls for _, lbls in examples],
            "n_tokens": n_tokens_list,
            "attn_cost": [n * (n + 1) // 2 for n in n_tokens_list],
        }
    )
    ds.save_to_disk(str(shards_dir / "shard_0000"))

    manifest = {
        "version": 1,
        "strategy": "truncate_last",
        "tokenizer": {
            "source_path": "tests/assets/tokenizer",
            "vocab_size": 2009,
            "eos_token_id": _EOS_ID,
            "chat_template_sha256": None,
        },
        "chat_template_kwargs": {"truncate_history_thinking": True},
        "input_files": {"total": 1, "paths": [], "skipped": []},
        "shards": {"completed": ["shard_0000"]},
        "stats": {},
        "created_at": "2026-01-01T00:00:00Z",
        "input_dir": "",
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def _build_loader(
    manifest_path: Path,
    *,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    seq_len: int = 32,
    batch_size: int = 1,
    snapshot_every_n_steps: int = 4,
    **config_kwargs,
) -> GranitePreTokenizedDataLoader:
    config = GranitePreTokenizedDataLoader.Config(
        dataset_path=str(manifest_path),
        infinite=True,
        snapshot_every_n_steps=snapshot_every_n_steps,
        **config_kwargs,
    )
    return GranitePreTokenizedDataLoader(
        config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=None,
        seq_len=seq_len,
        local_batch_size=batch_size,
    )


def _dcp_save(loader, ckpt_dir: str):
    dcp.save({"dataloader": loader}, checkpoint_id=ckpt_dir, no_dist=True)


def _dcp_load(loader, ckpt_dir: str):
    dcp.load({"dataloader": loader}, checkpoint_id=ckpt_dir, no_dist=True)


def _assert_batches_equal(test_case, expected, actual_iter, *, context: str):
    """Assert that actual_iter produces batches matching expected."""
    for i, (exp_inp, exp_lbl) in enumerate(expected):
        inp, lbl = next(actual_iter)
        test_case.assertTrue(
            inp["input"].equal(exp_inp["input"]),
            f"Batch {i} input mismatch ({context})",
        )
        test_case.assertTrue(
            lbl.equal(exp_lbl),
            f"Batch {i} labels mismatch ({context})",
        )


class TestDCPRoundTrip(unittest.TestCase):
    """Full DCP filesystem round-trip for GranitePreTokenizedDataLoader."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._manifest = _make_shard(self._tmp, _EXAMPLES)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_save_load_resumes_data(self):
        """DCP save/load produces identical subsequent batches."""
        dl = _build_loader(self._manifest)
        it = iter(dl)
        for _ in range(5):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_resume")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(5)]

        dl2 = _build_loader(self._manifest)
        _dcp_load(dl2, ckpt_dir)

        _assert_batches_equal(self, expected, iter(dl2), context="DCP resume")

    def test_resumed_differs_from_fresh(self):
        """A resumed loader produces different batches than a fresh one."""
        dl = _build_loader(self._manifest)
        it = iter(dl)
        for _ in range(8):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_neg")
        _dcp_save(dl, ckpt_dir)

        dl_resumed = _build_loader(self._manifest)
        _dcp_load(dl_resumed, ckpt_dir)
        it_resumed = iter(dl_resumed)
        resumed_batch, _ = next(it_resumed)

        dl_fresh = _build_loader(self._manifest)
        it_fresh = iter(dl_fresh)
        fresh_batch, _ = next(it_fresh)

        self.assertFalse(
            resumed_batch["input"].equal(fresh_batch["input"]),
            "Resumed loader must differ from fresh (DCP state must take effect)",
        )

    def test_consumed_stats_survive(self):
        """get_data_stats() matches after DCP round-trip."""
        dl = _build_loader(self._manifest)
        it = iter(dl)
        for _ in range(5):
            next(it)

        stats_before = dl.get_data_stats()
        self.assertGreater(stats_before["n_total_tokens"], 0)

        ckpt_dir = str(self._tmp / "ckpt_stats")
        _dcp_save(dl, ckpt_dir)

        dl2 = _build_loader(self._manifest)
        _dcp_load(dl2, ckpt_dir)

        stats_after = dl2.get_data_stats()
        self.assertEqual(stats_before["n_total_tokens"], stats_after["n_total_tokens"])
        self.assertEqual(
            stats_before["n_trained_tokens"], stats_after["n_trained_tokens"]
        )
        self.assertEqual(
            stats_before["n_examples_packed"], stats_after["n_examples_packed"]
        )

    def test_buffer_packing_longest(self):
        """DCP round-trip with packing='longest' preserves row_indices."""
        dl = _build_loader(self._manifest, packing="longest", buffer_size=10)
        it = iter(dl)
        for _ in range(6):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_longest")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(5)]

        dl2 = _build_loader(self._manifest, packing="longest", buffer_size=10)
        _dcp_load(dl2, ckpt_dir)

        _assert_batches_equal(self, expected, iter(dl2), context="longest packing")

    def test_buffer_shuffle_rng(self):
        """DCP round-trip with packing='buffer_shuffle' preserves RNG state."""
        dl = _build_loader(self._manifest, packing="buffer_shuffle", buffer_size=10)
        it = iter(dl)
        for _ in range(6):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_shuffle")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(5)]

        dl2 = _build_loader(self._manifest, packing="buffer_shuffle", buffer_size=10)
        _dcp_load(dl2, ckpt_dir)

        _assert_batches_equal(self, expected, iter(dl2), context="buffer_shuffle")

    def test_with_num_workers(self):
        """DCP round-trip with num_workers=2 preserves per-worker state."""
        dl = _build_loader(
            self._manifest,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=2,
        )
        it = iter(dl)
        for _ in range(8):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_workers")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(5)]

        # Resumed loader
        dl2 = _build_loader(
            self._manifest,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=2,
        )
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        _assert_batches_equal(self, expected, it2, context="num_workers=2")

        # Fresh loader must produce different first batch (state matters)
        dl_fresh = _build_loader(
            self._manifest,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=2,
        )
        fresh_batch, _ = next(iter(dl_fresh))
        self.assertFalse(
            expected[0][0]["input"].equal(fresh_batch["input"]),
            "num_workers=2: resumed batch must differ from fresh start",
        )


class TestDCPMultiRank(unittest.TestCase):
    """Rank isolation through DCP."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._manifest = _make_shard(self._tmp, _EXAMPLES)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_rank_state_isolation(self):
        """Each rank's DCP checkpoint independently resumes correct data."""
        dl0 = _build_loader(self._manifest, dp_rank=0, dp_world_size=2)
        dl1 = _build_loader(self._manifest, dp_rank=1, dp_world_size=2)

        it0, it1 = iter(dl0), iter(dl1)
        for _ in range(4):
            next(it0)
        for _ in range(8):
            next(it1)

        ckpt0 = str(self._tmp / "ckpt_rank0")
        ckpt1 = str(self._tmp / "ckpt_rank1")
        _dcp_save(dl0, ckpt0)
        _dcp_save(dl1, ckpt1)

        expected0 = [next(it0) for _ in range(5)]
        expected1 = [next(it1) for _ in range(5)]

        # Ranks produce different data (precondition for meaningful isolation test)
        self.assertFalse(
            all(
                e0[0]["input"].equal(e1[0]["input"])
                for e0, e1 in zip(expected0, expected1)
            ),
            "Ranks must produce different data for isolation test to be meaningful",
        )

        dl0_resumed = _build_loader(self._manifest, dp_rank=0, dp_world_size=2)
        _dcp_load(dl0_resumed, ckpt0)

        dl1_resumed = _build_loader(self._manifest, dp_rank=1, dp_world_size=2)
        _dcp_load(dl1_resumed, ckpt1)

        _assert_batches_equal(self, expected0, iter(dl0_resumed), context="rank 0")
        _assert_batches_equal(self, expected1, iter(dl1_resumed), context="rank 1")


class TestSnapshotDCPInteraction(unittest.TestCase):
    """Interaction between snapshot_every_n_steps and DCP save/load."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)
        self._manifest = _make_shard(self._tmp, _EXAMPLES)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_misaligned_step(self):
        """DCP save at step 7 (snapshot at 4) replays steps 5-7 correctly."""
        dl = _build_loader(self._manifest, snapshot_every_n_steps=4)
        it = iter(dl)
        for _ in range(7):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_misaligned")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(5)]

        dl2 = _build_loader(self._manifest, snapshot_every_n_steps=4)
        _dcp_load(dl2, ckpt_dir)

        _assert_batches_equal(self, expected, iter(dl2), context="misaligned snapshot")

    def test_aligned_step(self):
        """DCP save at step 8 (on snapshot boundary) resumes without replay."""
        dl = _build_loader(self._manifest, snapshot_every_n_steps=4)
        it = iter(dl)
        for _ in range(8):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_aligned")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(5)]

        dl2 = _build_loader(self._manifest, snapshot_every_n_steps=4)
        _dcp_load(dl2, ckpt_dir)

        _assert_batches_equal(self, expected, iter(dl2), context="aligned snapshot")


if __name__ == "__main__":
    unittest.main()
