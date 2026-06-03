"""Pre-tokenized SFT datasets for offline training.

Reads Arrow shards produced by pretokenize_sft.py. A separate plan_packing.py
step produces pack plans that PlannedPackingDataset consumes at training time.

Class hierarchy:
    PlannedPackingDataset         — reads pre-computed pack plans for cost-balanced
                                    batching across DP ranks
    GranitePreTokenizedDataLoader — dataloader wrapper with stats accumulation
                                    and DCP checkpointing
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path) as f:
        return json.load(f)


def _load_shards(manifest: dict[str, Any], shards_dir: Path) -> Dataset:
    shard_names = sorted(manifest["shards"]["completed"])
    if not shard_names:
        raise ValueError(f"No completed shards in manifest at {shards_dir.parent}")
    return concatenate_datasets(
        [load_from_disk(str(shards_dir / name)) for name in shard_names]
    )


class PlannedPackingDataset(IterableDataset, Stateful):
    """Dataset that reads a pre-computed pack plan produced by plan_packing.py.

    Three packing strategies:
    - prepacked_random_balanced: random global shuffle with local cost-balancing
      within each optimizer step window (dp_degree * accum_steps packs).
      Preserves unbiased sampling while recovering TPS via cost-homogeneous
      micro-batches.
    - prepacked_attn_grouped: cost-sorted chunks ensure all DP ranks in a
      global batch get packs with near-identical attention cost, minimizing
      synchronization idle time.
    - prepacked_random: random chunk assignment gives smoother loss curves at
      the cost of cross-rank sync efficiency.

    The iteration is fully deterministic from (seed, epoch, dp_degree): on
    resume, just recompute the epoch ordering and seek to the saved step.
    """

    def __init__(
        self,
        *,
        pack_plan_path: str | Path,
        seq_len: int,
        packing: Literal[
            "prepacked_attn_grouped", "prepacked_random", "prepacked_random_balanced"
        ],
        dp_rank: int = 0,
        dp_world_size: int = 1,
        accum_steps: int = 1,
        infinite: bool = False,
        seed: int = 42,
    ) -> None:
        import pyarrow.ipc as ipc

        pack_plan_path = Path(pack_plan_path)

        plan_metadata_path = pack_plan_path / "metadata.json"
        with open(plan_metadata_path) as f:
            plan_meta = json.load(f)
        if plan_meta["seq_len"] != seq_len:
            raise ValueError(
                f"Pack plan was built for seq_len={plan_meta['seq_len']} but "
                f"training seq_len={seq_len}. Regenerate the plan."
            )

        source_pretok_dir = Path(plan_meta["source_pretok_dir"])
        if not source_pretok_dir.exists():
            raise FileNotFoundError(
                f"Pack plan references source_pretok_dir={source_pretok_dir} "
                f"which does not exist. Was the dataset moved?"
            )

        manifest_path = source_pretok_dir / "manifest.json"
        manifest = _load_manifest(manifest_path)
        self._data = _load_shards(manifest, source_pretok_dir / "shards")

        plan_arrow_path = pack_plan_path / "pack_plan.arrow"
        import pyarrow as pa

        with pa.memory_map(str(plan_arrow_path), "r") as f:
            reader = ipc.open_stream(f)
            plan_table = reader.read_all()

        self._example_indices: list[list[int]] = [
            row.as_py() for row in plan_table.column("example_indices")
        ]
        self._total_examples: int = sum(len(ex) for ex in self._example_indices)
        self._pack_costs: np.ndarray = plan_table.column("attn_cost").to_numpy()
        self._pack_total_tokens: np.ndarray = (
            plan_table.column("total_tokens").to_numpy()
        )

        if "strategy" not in manifest:
            raise ValueError(
                f"Manifest at {manifest_path} missing required 'strategy' field."
            )
        self._strategy = manifest["strategy"]
        self._packing = packing
        self._accum_steps = accum_steps
        self._eos_id: int = manifest["tokenizer"]["eos_token_id"]
        self.seq_len = seq_len
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self.infinite = infinite
        self._seed = seed

        if self._packing == "prepacked_random_balanced" and accum_steps < 2:
            logger.warning(
                "prepacked_random_balanced with accum_steps=1 provides no throughput "
                "benefit over prepacked_random. Consider using gradient accumulation."
            )

        self._epoch: int = 0
        self._step: int = 0

        length_stats = manifest.get("length_stats", {})
        self._dataset_mean_length: float = float(length_stats.get("mean", 0.0))

    @property
    def num_packs(self) -> int:
        return len(self._example_indices)

    @property
    def num_examples(self) -> int:
        return self._total_examples // self._dp_world_size

    def _epoch_setup(self, epoch: int) -> np.ndarray:
        """Compute the chunk-to-pack assignment for a given epoch.

        Returns chunk_pack_indices: shape (n_chunks, dp_world_size), where each
        row contains the pack indices assigned to dp ranks for that chunk/step.
        """
        n_packs = len(self._example_indices)
        dp = self._dp_world_size
        rng = np.random.default_rng(self._seed + epoch)

        # Drop to align with GBS (= dp * accum_steps). This ensures dp is
        # purely an implementation detail: same GBS + same seed → same data
        # order regardless of dp/accum decomposition.
        gbs = dp * self._accum_steps
        remainder = n_packs % gbs
        if remainder > 0:
            drop_indices = rng.choice(n_packs, size=remainder, replace=False)
            keep_mask = np.ones(n_packs, dtype=bool)
            keep_mask[drop_indices] = False
            pack_indices = np.where(keep_mask)[0]
        else:
            pack_indices = np.arange(n_packs)

        n_chunks = len(pack_indices) // dp
        if self._packing == "prepacked_attn_grouped":
            chunk_pack_indices = pack_indices.reshape(n_chunks, dp)
            rng.shuffle(chunk_pack_indices)
        elif self._packing == "prepacked_random":
            rng.shuffle(pack_indices)
            chunk_pack_indices = pack_indices.reshape(n_chunks, dp)
        elif self._packing == "prepacked_random_balanced":
            rng.shuffle(pack_indices)
            chunk_rows = []
            for w_start in range(0, len(pack_indices), gbs):
                window = pack_indices[w_start : w_start + gbs]
                order = np.argsort(self._pack_costs[window])
                chunk_rows.append(window[order].reshape(self._accum_steps, dp))
            chunk_pack_indices = np.concatenate(chunk_rows, axis=0)
        else:
            raise ValueError(f"Unknown prepacked mode: {self._packing!r}")
        return chunk_pack_indices

    def _materialize_and_pack(self, pack_idx: int) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, int]
    ]:
        """Fetch all examples for a pack, assemble into a padded batch."""
        example_indices = self._example_indices[pack_idx]
        pack_cost = int(self._pack_costs[pack_idx])

        if self._strategy == "backbone_suffix":
            return self._pack_backbone_suffix(example_indices, pack_cost)
        else:
            return self._pack_standard(example_indices, pack_cost)

    def _pack_standard(self, example_indices: list[int], pack_cost: int) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, int]
    ]:
        inputs = np.full(self.seq_len, self._eos_id, dtype=np.int32)
        labels = np.full(self.seq_len, IGNORE_INDEX, dtype=np.int32)
        positions = np.zeros(self.seq_len, dtype=np.int32)
        offset = 0
        n_trained = 0

        for row_idx in example_indices:
            table_slice = self._data.data.slice(row_idx, 1)
            col = table_slice.column("input_ids").combine_chunks()
            offsets = col.offsets.to_numpy()
            input_ids = col.values.to_numpy()[offsets[0]:offsets[1]]

            col = table_slice.column("labels").combine_chunks()
            offsets = col.offsets.to_numpy()
            item_labels = col.values.to_numpy()[offsets[0]:offsets[1]]

            n = len(input_ids)
            inputs[offset:offset + n] = input_ids
            labels[offset:offset + n] = item_labels
            positions[offset:offset + n] = np.arange(n, dtype=np.int32)
            n_trained += int(np.count_nonzero(item_labels != IGNORE_INDEX))
            offset += n

        # Pad positions for remainder
        pad_len = self.seq_len - offset
        if pad_len > 0:
            positions[offset:] = np.arange(pad_len, dtype=np.int32)

        stats = {
            "n_total_tokens": offset,
            "n_trained_tokens": n_trained,
            "n_examples_packed": len(example_indices),
        }
        return (
            {
                "input": torch.from_numpy(inputs.astype(np.int64)),
                "positions": torch.from_numpy(positions.astype(np.int64)),
                "attn_cost": torch.tensor(pack_cost, dtype=torch.int64),
            },
            torch.from_numpy(labels.astype(np.int64)),
            stats,
        )

    def _pack_backbone_suffix(self, example_indices: list[int], pack_cost: int) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, dict[str, int]
    ]:
        inputs = np.full(self.seq_len, self._eos_id, dtype=np.int32)
        labels = np.full(self.seq_len, IGNORE_INDEX, dtype=np.int32)
        positions = np.zeros(self.seq_len, dtype=np.int32)
        conv_ids = np.zeros(self.seq_len, dtype=np.int32)
        suffix_ids = np.zeros(self.seq_len, dtype=np.int32)
        insertion_limits = np.full(self.seq_len, -1, dtype=np.int32)
        offset = 0
        n_trained = 0
        conv_counter = 1
        suffix_counter = 1

        def _read_list(table_slice: "pa.Table", col_name: str) -> np.ndarray:
            col = table_slice.column(col_name).combine_chunks()
            offs = col.offsets.to_numpy()
            if offs[0] == offs[1]:
                return np.array([], dtype=np.int32)
            return col.values.to_numpy()[offs[0]:offs[1]].copy()

        for row_idx in example_indices:
            table_slice = self._data.data.slice(row_idx, 1)

            input_ids = _read_list(table_slice, "input_ids")
            item_labels = _read_list(table_slice, "labels")
            item_positions = _read_list(table_slice, "positions")
            suffix_starts = _read_list(table_slice, "suffix_starts")
            ins_limits = _read_list(table_slice, "insertion_limits")

            n = len(input_ids)
            backbone_len = int(suffix_starts[0]) if len(suffix_starts) > 0 else n

            inputs[offset:offset + n] = input_ids
            labels[offset:offset + n] = item_labels
            positions[offset:offset + n] = item_positions
            conv_ids[offset:offset + n] = conv_counter

            # suffix_ids
            for k in range(len(suffix_starts)):
                s_start = int(suffix_starts[k])
                s_end = (
                    int(suffix_starts[k + 1]) if k + 1 < len(suffix_starts) else n
                )
                suffix_ids[offset + s_start:offset + s_end] = suffix_counter
                suffix_counter += 1

            # insertion_limits
            backbone_limit = offset + backbone_len - 1
            insertion_limits[offset:offset + backbone_len] = backbone_limit
            for k in range(len(suffix_starts)):
                s_start = int(suffix_starts[k])
                s_end = (
                    int(suffix_starts[k + 1]) if k + 1 < len(suffix_starts) else n
                )
                insertion_limits[offset + s_start:offset + s_end] = (
                    offset + int(ins_limits[k])
                )

            n_trained += int(
                np.count_nonzero(labels[offset:offset + n] != IGNORE_INDEX)
            )
            offset += n
            conv_counter += 1

        # Pad positions
        pad_len = self.seq_len - offset
        if pad_len > 0:
            positions[offset:] = np.arange(pad_len, dtype=np.int32)

        stats = {
            "n_total_tokens": offset,
            "n_trained_tokens": n_trained,
            "n_examples_packed": len(example_indices),
        }
        return (
            {
                "input": torch.from_numpy(inputs.astype(np.int64)),
                "positions": torch.from_numpy(positions.astype(np.int64)),
                "conv_ids": torch.from_numpy(conv_ids.astype(np.int64)),
                "suffix_ids": torch.from_numpy(suffix_ids.astype(np.int64)),
                "insertion_limits": torch.from_numpy(
                    insertion_limits.astype(np.int64)
                ),
                "attn_cost": torch.tensor(pack_cost, dtype=torch.int64),
            },
            torch.from_numpy(labels.astype(np.int64)),
            stats,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        epoch = self._epoch
        while True:
            chunk_pack_indices = self._epoch_setup(epoch)
            start_step = self._step if epoch == self._epoch else 0

            for step_idx in range(start_step, len(chunk_pack_indices)):
                if step_idx % num_workers != worker_id:
                    continue
                pack_idx = int(chunk_pack_indices[step_idx, self._dp_rank])
                self._step = step_idx + 1
                yield self._materialize_and_pack(pack_idx)

            epoch += 1
            self._epoch = epoch
            self._step = 0

            if not self.infinite:
                break

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "step": self._step,
            "accum_steps": self._accum_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = state_dict["epoch"]
        self._step = state_dict["step"]
        saved_accum = state_dict.get("accum_steps")
        if saved_accum is not None and saved_accum != self._accum_steps:
            logger.info(
                f"accum_steps changed from {saved_accum} to {self._accum_steps} on "
                f"resume. Pack-to-step groupings will differ for the remainder of "
                f"epoch {self._epoch}; statistical properties (no bias) are preserved."
            )



class GranitePreTokenizedDataLoader(ParallelAwareDataloader):
    """DataLoader for pre-tokenized Arrow shards with planned packing.

    Reads pre-computed pack plans (produced by plan_packing.py) and iterates
    through cost-balanced batches. Data stats (token counts, example counts)
    are accumulated in the main process as batches are consumed via __iter__,
    so they reflect only consumed batches — not prefetched ones.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str
        """Directory containing pretokenized shards, manifest.json, and
        pack_plans/. Also accepts a direct path to manifest.json."""

        infinite: bool = True
        """Loop the dataset indefinitely."""

        packing: Literal[
            "prepacked_random_balanced",
            "prepacked_attn_grouped",
            "prepacked_random",
        ] = "prepacked_random_balanced"
        """Packing algorithm.
        'prepacked_random_balanced' uses offline pack plan with random global
        shuffle plus local cost-balancing within each optimizer step window
        (dp_degree * accum_steps packs). Preserves unbiased sampling while
        recovering TPS via cost-homogeneous micro-batches.
        'prepacked_attn_grouped' uses offline pack plan with cost-balanced chunk
        assignment (packs grouped by attn_cost for sync efficiency).
        'prepacked_random' uses offline pack plan with random chunk assignment
        (smoother loss at the cost of sync efficiency).
        All modes read from {pretok_dir}/pack_plans/seqlen_{seq_len}/
        (produced by plan_packing.py)."""

        seed: int = 42
        """RNG seed for epoch shuffling."""

        snapshot_every_n_steps: int = 16
        """How often StatefulDataLoader snapshots worker state for checkpointing.
        At most N-1 steps are replayed on resume. Lower N reduces replay at
        negligible serialization cost."""

    def __init__(
        self,
        config: "GranitePreTokenizedDataLoader.Config",
        *,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        accum_steps: int = 1,
        **kwargs,
    ) -> None:
        if local_batch_size > 1:
            raise ValueError(
                f"local_batch_size must be 1 for pre-tokenized datasets (got "
                f"{local_batch_size}). Packing handles batching within each "
                f"sequence, and mixed strategies produce heterogeneous tensors "
                f"that cannot be stacked."
            )

        if not config.dataset_path:
            raise ValueError(
                "dataloader.dataset_path must be set (--dataloader.dataset-path on CLI)."
            )
        path = Path(config.dataset_path.strip())
        if path.suffix == ".json":
            pretok_dir = path.parent
        else:
            pretok_dir = path

        pack_plan_path = pretok_dir / "pack_plans" / f"seqlen_{seq_len}"
        dataset = PlannedPackingDataset(
            pack_plan_path=pack_plan_path,
            seq_len=seq_len,
            packing=config.packing,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            accum_steps=accum_steps,
            infinite=config.infinite,
            seed=config.seed,
        )

        self._consumed_n_total_tokens: int = 0
        self._consumed_n_trained_tokens: int = 0
        self._consumed_n_examples_packed: int = 0

        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=config.snapshot_every_n_steps,
        )

    def __iter__(self):
        for input_dict, labels, stats in super().__iter__():
            self._consumed_n_total_tokens += stats["n_total_tokens"].sum().item()
            self._consumed_n_trained_tokens += stats["n_trained_tokens"].sum().item()
            self._consumed_n_examples_packed += stats["n_examples_packed"].sum().item()
            yield input_dict, labels

    def get_data_stats(self) -> dict[str, Any]:
        n_dataset = max(self.dataset.num_examples, 1)
        return {
            "n_total_tokens": self._consumed_n_total_tokens,
            "n_trained_tokens": self._consumed_n_trained_tokens,
            "n_examples_packed": self._consumed_n_examples_packed,
            "epochs": self._consumed_n_examples_packed / n_dataset,
            "dataset_mean_length": self.dataset._dataset_mean_length,
        }

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["_consumed_stats"] = {
            "n_total_tokens": self._consumed_n_total_tokens,
            "n_trained_tokens": self._consumed_n_trained_tokens,
            "n_examples_packed": self._consumed_n_examples_packed,
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        consumed = state_dict.pop("_consumed_stats", None)
        if consumed is not None:
            self._consumed_n_total_tokens = consumed["n_total_tokens"]
            self._consumed_n_trained_tokens = consumed["n_trained_tokens"]
            self._consumed_n_examples_packed = consumed["n_examples_packed"]
        super().load_state_dict(state_dict)
