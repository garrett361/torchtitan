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

_EXAMPLES = [
    ([1, 2, 3, _EOS_ID], [IGNORE_INDEX, 2, 3, _EOS_ID]),
    ([4, 5, 6, _EOS_ID], [IGNORE_INDEX, 5, 6, _EOS_ID]),
    ([7, 8, 9, _EOS_ID], [IGNORE_INDEX, 8, 9, _EOS_ID]),
    ([10, 11, 12, _EOS_ID], [IGNORE_INDEX, 11, 12, _EOS_ID]),
    ([13, 14, 15, _EOS_ID], [IGNORE_INDEX, 14, 15, _EOS_ID]),
    ([16, 17, 18, _EOS_ID], [IGNORE_INDEX, 17, 18, _EOS_ID]),
]


def _make_shard(tmp_path: Path, examples: list[tuple[list[int], list[int]]]) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_dict(
        {
            "input_ids": [ids for ids, _ in examples],
            "labels": [lbls for _, lbls in examples],
            "n_tokens": [len(ids) for ids, _ in examples],
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
    seq_len: int = 16,
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

        expected = [next(it) for _ in range(3)]

        dl2 = _build_loader(self._manifest)
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        for i, (exp_inp, exp_lbl) in enumerate(expected):
            inp, lbl = next(it2)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Batch {i} input mismatch after DCP resume",
            )
            self.assertTrue(
                lbl.equal(exp_lbl), f"Batch {i} labels mismatch after DCP resume"
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
        dl = _build_loader(self._manifest, packing="longest", buffer_size=6)
        it = iter(dl)
        for _ in range(4):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_longest")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(3)]

        dl2 = _build_loader(self._manifest, packing="longest", buffer_size=6)
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        for i, (exp_inp, exp_lbl) in enumerate(expected):
            inp, lbl = next(it2)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Batch {i} input mismatch (longest packing)",
            )
            self.assertTrue(lbl.equal(exp_lbl), f"Batch {i} labels mismatch (longest)")

    def test_buffer_shuffle_rng(self):
        """DCP round-trip with packing='buffer_shuffle' preserves RNG state."""
        dl = _build_loader(self._manifest, packing="buffer_shuffle", buffer_size=6)
        it = iter(dl)
        for _ in range(4):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_shuffle")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(3)]

        dl2 = _build_loader(self._manifest, packing="buffer_shuffle", buffer_size=6)
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        for i, (exp_inp, exp_lbl) in enumerate(expected):
            inp, lbl = next(it2)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Batch {i} input mismatch (buffer_shuffle)",
            )
            self.assertTrue(
                lbl.equal(exp_lbl), f"Batch {i} labels mismatch (buffer_shuffle)"
            )

    def test_with_num_workers(self):
        """DCP round-trip with num_workers=2 preserves per-worker state."""
        dl = _build_loader(
            self._manifest,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=2,
        )
        it = iter(dl)
        for _ in range(5):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_workers")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(3)]

        dl2 = _build_loader(
            self._manifest,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=2,
        )
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        for i, (exp_inp, exp_lbl) in enumerate(expected):
            inp, lbl = next(it2)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Batch {i} input mismatch (num_workers=2)",
            )
            self.assertTrue(
                lbl.equal(exp_lbl), f"Batch {i} labels mismatch (num_workers=2)"
            )


class TestDCPMultiRank(unittest.TestCase):
    """Rank isolation and error handling through DCP."""

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
        for _ in range(3):
            next(it0)
        for _ in range(6):
            next(it1)

        ckpt0 = str(self._tmp / "ckpt_rank0")
        ckpt1 = str(self._tmp / "ckpt_rank1")
        _dcp_save(dl0, ckpt0)
        _dcp_save(dl1, ckpt1)

        expected0 = [next(it0) for _ in range(3)]
        expected1 = [next(it1) for _ in range(3)]

        dl0_resumed = _build_loader(self._manifest, dp_rank=0, dp_world_size=2)
        _dcp_load(dl0_resumed, ckpt0)
        it0r = iter(dl0_resumed)

        dl1_resumed = _build_loader(self._manifest, dp_rank=1, dp_world_size=2)
        _dcp_load(dl1_resumed, ckpt1)
        it1r = iter(dl1_resumed)

        for i, (exp_inp, exp_lbl) in enumerate(expected0):
            inp, lbl = next(it0r)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Rank 0 batch {i} input mismatch",
            )
            self.assertTrue(lbl.equal(exp_lbl), f"Rank 0 batch {i} labels mismatch")

        for i, (exp_inp, exp_lbl) in enumerate(expected1):
            inp, lbl = next(it1r)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Rank 1 batch {i} input mismatch",
            )
            self.assertTrue(lbl.equal(exp_lbl), f"Rank 1 batch {i} labels mismatch")


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

        expected = [next(it) for _ in range(3)]

        dl2 = _build_loader(self._manifest, snapshot_every_n_steps=4)
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        for i, (exp_inp, exp_lbl) in enumerate(expected):
            inp, lbl = next(it2)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Batch {i} mismatch (misaligned snapshot)",
            )
            self.assertTrue(
                lbl.equal(exp_lbl),
                f"Batch {i} labels mismatch (misaligned snapshot)",
            )

    def test_aligned_step(self):
        """DCP save at step 8 (on snapshot boundary) resumes without replay."""
        dl = _build_loader(self._manifest, snapshot_every_n_steps=4)
        it = iter(dl)
        for _ in range(8):
            next(it)

        ckpt_dir = str(self._tmp / "ckpt_aligned")
        _dcp_save(dl, ckpt_dir)

        expected = [next(it) for _ in range(3)]

        dl2 = _build_loader(self._manifest, snapshot_every_n_steps=4)
        _dcp_load(dl2, ckpt_dir)
        it2 = iter(dl2)

        for i, (exp_inp, exp_lbl) in enumerate(expected):
            inp, lbl = next(it2)
            self.assertTrue(
                inp["input"].equal(exp_inp["input"]),
                f"Batch {i} mismatch (aligned snapshot)",
            )
            self.assertTrue(
                lbl.equal(exp_lbl),
                f"Batch {i} labels mismatch (aligned snapshot)",
            )


if __name__ == "__main__":
    unittest.main()
