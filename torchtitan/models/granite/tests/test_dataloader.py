"""Unit tests for TruncateLastDataset and GranitePreTokenizedDataLoader."""

import json
import unittest
from pathlib import Path

import torch
from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.granite.pretokenized_dataset import (
    GranitePreTokenizedDataLoader,
    TruncateLastDataset,
)

_EOS_ID = 2003


def _make_shard(
    tmp_path: Path,
    examples: list[tuple[list[int], list[int]]],
    *,
    shard_name: str = "shard_0000",
    strategy: str = "truncate_last",
) -> Path:
    """Write a single Arrow shard + manifest.json to tmp_path; return manifest path."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_dict(
        {
            "input_ids": [ids for ids, _ in examples],
            "labels": [lbls for _, lbls in examples],
            "n_tokens": [len(ids) for ids, _ in examples],
        }
    )
    ds.save_to_disk(str(shards_dir / shard_name))

    manifest = {
        "version": 1,
        "strategy": strategy,
        "tokenizer": {
            "source_path": "tests/assets/tokenizer",
            "vocab_size": 2009,
            "eos_token_id": _EOS_ID,
            "chat_template_sha256": None,
        },
        "chat_template_kwargs": {"truncate_history_thinking": True},
        "shards": {
            "completed": [shard_name],
            "total_expected": 1,
        },
        "stats": {},
        "created_at": "2026-01-01T00:00:00Z",
        "input_dir": "",
        "input_files_sha256": {},
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


class TestTruncateLastDatasetPacking(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir)

    def _make_dataset(self, examples, seq_len=16, **kwargs) -> TruncateLastDataset:
        manifest_path = _make_shard(self._tmp, examples)
        return TruncateLastDataset(manifest_path, seq_len=seq_len, infinite=False, **kwargs)

    def test_output_shape(self):
        """Every yielded batch has tensors of length seq_len."""
        seq_len = 8
        examples = [
            ([1, 2, 3, _EOS_ID], [IGNORE_INDEX, 3, _EOS_ID, IGNORE_INDEX]),
            ([1, 4, 5, 6, _EOS_ID], [IGNORE_INDEX, 5, 6, _EOS_ID, IGNORE_INDEX]),
        ]
        ds = self._make_dataset(examples, seq_len=seq_len)
        for batch_dict, labels in ds:
            self.assertEqual(batch_dict["input"].shape, (seq_len,))
            self.assertEqual(batch_dict["positions"].shape, (seq_len,))
            self.assertEqual(labels.shape, (seq_len,))

    def test_labels_passthrough(self):
        """Unmasked label positions in the output match the pre-tokenized labels."""
        seq_len = 16
        input_ids = [1, 10, 20, 30, _EOS_ID]
        label_ids = [IGNORE_INDEX, IGNORE_INDEX, 30, _EOS_ID, IGNORE_INDEX]
        examples = [(input_ids, label_ids)]
        ds = self._make_dataset(examples, seq_len=seq_len)

        batches = list(ds)
        self.assertEqual(len(batches), 1)
        _, labels = batches[0]

        # First len(input_ids) positions must match pre-tokenized labels exactly.
        for i, expected in enumerate(label_ids):
            self.assertEqual(labels[i].item(), expected)

    def test_seq_len_drop(self):
        """Examples longer than seq_len are silently dropped."""
        seq_len = 4
        long_example = ([1, 2, 3, 4, _EOS_ID], [IGNORE_INDEX, 3, 4, _EOS_ID, IGNORE_INDEX])
        short_example = ([1, 2, _EOS_ID], [IGNORE_INDEX, _EOS_ID, IGNORE_INDEX])
        ds = self._make_dataset([long_example, short_example], seq_len=seq_len)

        batches = list(ds)
        # long_example (5 tokens) dropped; short_example (3 tokens) packed and padded.
        self.assertEqual(len(batches), 1)
        _, labels = batches[0]
        # Padding positions should be IGNORE_INDEX.
        self.assertEqual(labels[-1].item(), IGNORE_INDEX)

    def test_position_resets_at_document_boundary(self):
        """Positions reset to 0 at each packed document boundary."""
        seq_len = 8
        # Two 3-token examples pack into one seq_len=8 sequence with 2 pad tokens.
        ex = ([1, 2, _EOS_ID], [IGNORE_INDEX, _EOS_ID, IGNORE_INDEX])
        ds = self._make_dataset([ex, ex], seq_len=seq_len)

        batches = list(ds)
        self.assertEqual(len(batches), 1)
        positions = batches[0][0]["positions"].tolist()
        # First example: positions [0, 1, 2]; second: [0, 1, 2]; padding: [0, 1].
        self.assertEqual(positions[:3], [0, 1, 2])
        self.assertEqual(positions[3:6], [0, 1, 2])

    def test_padding_uses_eos(self):
        """Padding tokens are filled with EOS_ID; padding labels are IGNORE_INDEX."""
        seq_len = 8
        ex = ([1, 2, _EOS_ID], [IGNORE_INDEX, _EOS_ID, IGNORE_INDEX])
        ds = self._make_dataset([ex], seq_len=seq_len)

        batches = list(ds)
        self.assertEqual(len(batches), 1)
        inputs, labels = batches[0][0]["input"].tolist(), batches[0][1].tolist()
        # Positions 3..7 are padding.
        self.assertTrue(all(t == _EOS_ID for t in inputs[3:]))
        self.assertTrue(all(lbl == IGNORE_INDEX for lbl in labels[3:]))

    def test_stats(self):
        """get_data_stats returns correct token and example counts."""
        seq_len = 16
        input_ids = [1, 10, 20, 30, _EOS_ID]
        label_ids = [IGNORE_INDEX, IGNORE_INDEX, 30, _EOS_ID, IGNORE_INDEX]
        ds = self._make_dataset([(input_ids, label_ids)], seq_len=seq_len)
        list(ds)  # consume

        stats = ds.get_data_stats()
        self.assertEqual(stats["n_total_tokens"], len(input_ids))
        self.assertEqual(stats["n_trained_tokens"], sum(1 for l in label_ids if l != IGNORE_INDEX))
        self.assertEqual(stats["n_examples_packed"], 1)

    def test_dp_sharding_disjoint(self):
        """Rank 0 and rank 1 see disjoint examples with world_size=2."""
        seq_len = 32
        n = 10
        # Unique token IDs per example so we can identify them.
        examples = [
            ([100 + i, _EOS_ID], [_EOS_ID, IGNORE_INDEX])
            for i in range(n)
        ]
        manifest_path = _make_shard(self._tmp, examples)

        def collect_first_tokens(rank):
            ds = TruncateLastDataset(
                manifest_path, seq_len=seq_len, dp_rank=rank, dp_world_size=2, infinite=False
            )
            tokens = set()
            for batch_dict, _ in ds:
                for t in batch_dict["input"].tolist():
                    if 100 <= t < 200:
                        tokens.add(t)
            return tokens

        rank0_tokens = collect_first_tokens(0)
        rank1_tokens = collect_first_tokens(1)

        self.assertFalse(rank0_tokens & rank1_tokens, "Ranks share examples")
        self.assertEqual(rank0_tokens | rank1_tokens, {100 + i for i in range(n)})

    def test_checkpointing_resumes_correctly(self):
        """load_state_dict restores to the exact point so subsequent batches match."""
        seq_len = 8
        examples = [
            ([1, 2, 3, _EOS_ID], [IGNORE_INDEX, 3, _EOS_ID, IGNORE_INDEX]),
            ([1, 4, 5, _EOS_ID], [IGNORE_INDEX, 5, _EOS_ID, IGNORE_INDEX]),
            ([1, 6, 7, _EOS_ID], [IGNORE_INDEX, 7, _EOS_ID, IGNORE_INDEX]),
            ([1, 8, 9, _EOS_ID], [IGNORE_INDEX, 9, _EOS_ID, IGNORE_INDEX]),
        ]
        manifest_path = _make_shard(self._tmp, examples)

        # Reference: collect all batches in one pass.
        ds_ref = TruncateLastDataset(manifest_path, seq_len=seq_len, infinite=False)
        all_batches = list(ds_ref)
        self.assertGreater(len(all_batches), 1, "Need at least 2 batches to test resume")

        # Checkpoint after first batch, restore, then compare remaining batches.
        ds_a = TruncateLastDataset(manifest_path, seq_len=seq_len, infinite=False)
        it_a = iter(ds_a)
        next(it_a)  # consume first batch
        checkpoint = ds_a.state_dict()

        ds_b = TruncateLastDataset(manifest_path, seq_len=seq_len, infinite=False)
        ds_b.load_state_dict(checkpoint)

        remaining_a = list(it_a)
        remaining_b = list(ds_b)

        self.assertEqual(len(remaining_a), len(remaining_b))
        for (ba, la), (bb, lb) in zip(remaining_a, remaining_b):
            self.assertTrue(torch.equal(ba["input"], bb["input"]))
            self.assertTrue(torch.equal(la, lb))


class TestMultiWorkerSharding(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir)

    def test_workers_see_disjoint_data(self):
        """With num_workers=2, no data duplication occurs."""
        seq_len = 32
        n = 20
        examples = [
            ([100 + i, _EOS_ID], [_EOS_ID, IGNORE_INDEX])
            for i in range(n)
        ]
        manifest_path = _make_shard(self._tmp, examples)

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        loader = GranitePreTokenizedDataLoader(
            GranitePreTokenizedDataLoader.Config(
                manifest_path=str(manifest_path),
                infinite=False,
                num_workers=2,
                persistent_workers=False,
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=1,
        )

        all_tokens = set()
        for batch_dict, _ in loader:
            for t in batch_dict["input"].flatten().tolist():
                if 100 <= t < 200:
                    all_tokens.add(t)

        self.assertEqual(all_tokens, {100 + i for i in range(n)})

    def test_shared_stats_nonzero(self):
        """Shared stats are visible to the main process after multi-worker iteration."""
        seq_len = 16
        examples = [
            ([1, 10, 20, _EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, _EOS_ID, IGNORE_INDEX]),
            ([1, 30, 40, _EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, _EOS_ID, IGNORE_INDEX]),
            ([1, 50, 60, _EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, _EOS_ID, IGNORE_INDEX]),
            ([1, 70, 80, _EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, _EOS_ID, IGNORE_INDEX]),
        ]
        manifest_path = _make_shard(self._tmp, examples)

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        loader = GranitePreTokenizedDataLoader(
            GranitePreTokenizedDataLoader.Config(
                manifest_path=str(manifest_path),
                infinite=False,
                num_workers=2,
                persistent_workers=False,
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=1,
        )

        for _ in loader:
            pass

        stats = loader.dataset.get_data_stats()
        self.assertGreater(stats["n_total_tokens"], 0)
        self.assertGreater(stats["n_trained_tokens"], 0)
        self.assertGreater(stats["n_examples_packed"], 0)


class TestGranitePreTokenizedDataLoaderDispatch(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir)

    def test_dispatch_truncate_last(self):
        """DataLoader instantiates TruncateLastDataset for strategy='truncate_last'."""
        examples = [([1, 2, _EOS_ID], [IGNORE_INDEX, _EOS_ID, IGNORE_INDEX])]
        manifest_path = _make_shard(self._tmp, examples, strategy="truncate_last")

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        loader = GranitePreTokenizedDataLoader(
            GranitePreTokenizedDataLoader.Config(
                manifest_path=str(manifest_path),
                infinite=False,
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=16,
            local_batch_size=1,
        )
        self.assertIsInstance(loader.dataset, TruncateLastDataset)

    def test_dispatch_unknown_strategy_raises(self):
        """DataLoader raises ValueError for an unregistered strategy."""
        examples = [([1, 2, _EOS_ID], [IGNORE_INDEX, _EOS_ID, IGNORE_INDEX])]
        manifest_path = _make_shard(self._tmp, examples, strategy="unknown_strategy")

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        with self.assertRaises(ValueError):
            GranitePreTokenizedDataLoader(
                GranitePreTokenizedDataLoader.Config(
                    manifest_path=str(manifest_path),
                    infinite=False,
                ),
                dp_world_size=1,
                dp_rank=0,
                tokenizer=tokenizer,
                seq_len=16,
                local_batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
