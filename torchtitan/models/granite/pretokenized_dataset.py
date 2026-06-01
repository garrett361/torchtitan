"""Pre-tokenized SFT datasets for offline training.

Reads Arrow shards produced by pretokenize_sft.py. Strategy dispatch happens
at GranitePreTokenizedDataLoader construction time via manifest["strategy"].

Class hierarchy:
    PreTokenizedDataset           — abstract base: manifest loading, cross-rank
                                    LPT packing, checkpointing
    StandardPackingDataset        — format primitives for (input_ids, labels) shards
    BackboneSuffixDataset         — format primitives for backbone+suffix shards
    GranitePreTokenizedDataLoader — dataloader; dispatches to the right dataset
                                    class based on manifest["strategy"]

Cross-rank LPT packing: all DP ranks jointly form dp_world_size batches per
step from a shared data stream. Each rank materializes only its assigned batch.
Selection mode (longest, buffer_shuffle) is a constructor parameter.
"""

import bisect
import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
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


def _load_and_merge_manifests(
    manifest_paths: list[Path],
) -> tuple[dict[str, Any], Dataset]:
    """Load multiple manifests, validate compatibility, and merge into one Dataset.

    Returns a synthetic merged manifest (with combined stats) and the
    concatenated Dataset ready for shuffle + DP sharding.
    """
    manifests = [_load_manifest(p) for p in manifest_paths]

    ref = manifests[0]
    for i, m in enumerate(manifests[1:], 1):
        if m["strategy"] != ref["strategy"]:
            raise ValueError(
                f"Strategy mismatch: {manifest_paths[0]} has {ref['strategy']!r}, "
                f"{manifest_paths[i]} has {m['strategy']!r}"
            )
        if m["tokenizer"]["eos_token_id"] != ref["tokenizer"]["eos_token_id"]:
            raise ValueError(
                f"eos_token_id mismatch between {manifest_paths[0]} and "
                f"{manifest_paths[i]}"
            )
        if m["tokenizer"]["vocab_size"] != ref["tokenizer"]["vocab_size"]:
            raise ValueError(
                f"vocab_size mismatch between {manifest_paths[0]} and "
                f"{manifest_paths[i]}"
            )
        ref_kwargs = ref.get("chat_template_kwargs", {})
        m_kwargs = m.get("chat_template_kwargs", {})
        if ref_kwargs != m_kwargs:
            logger.warning(
                "chat_template_kwargs differ between %s (%s) and %s (%s)",
                manifest_paths[0],
                ref_kwargs,
                manifest_paths[i],
                m_kwargs,
            )

    all_datasets = []
    for path, manifest in zip(manifest_paths, manifests):
        shards_dir = path.parent / "shards"
        all_datasets.append(_load_shards(manifest, shards_dir))
    combined = concatenate_datasets(all_datasets)

    total_examples = sum(m["stats"]["total_examples"] for m in manifests)
    total_tokens = sum(m["stats"]["total_tokens"] for m in manifests)
    total_trained_tokens = sum(
        m["stats"].get("total_trained_tokens", 0) for m in manifests
    )

    merged_manifest: dict[str, Any] = {
        "version": ref["version"],
        "strategy": ref["strategy"],
        "tokenizer": ref["tokenizer"],
        "stats": {
            "total_examples": total_examples,
            "total_tokens": total_tokens,
            "total_trained_tokens": total_trained_tokens,
        },
    }

    logger.info(
        "Merged %d datasets: %s total examples, %s total tokens",
        len(manifests),
        f"{total_examples:,}",
        f"{total_tokens:,}",
    )

    return merged_manifest, combined


# ---------------------------------------------------------------------------
# Selection functions
# ---------------------------------------------------------------------------

_PACKING_MODES = {"longest", "buffer_shuffle", "attn_balanced"}


def _select_buffer_shuffle(dataset: "PreTokenizedDataset", remaining: int) -> int:
    """Random pick from items that fit (bisect on sorted buffer)."""
    max_idx = bisect.bisect_right(dataset._lengths, remaining) - 1
    if max_idx < 0:
        return -1
    return int(dataset._batch_rng.integers(max_idx + 1))


def _select_longest(dataset: "PreTokenizedDataset", remaining: int) -> int:
    """Binary search the sorted buffer for the longest item that fits."""
    idx = bisect.bisect_right(dataset._lengths, remaining) - 1
    return idx if idx >= 0 else -1


def _select_attn_balanced(
    dataset: "PreTokenizedDataset", remaining: int, deficit: int
) -> tuple[int, int]:
    """Select buffer item to equalize attention cost across DP ranks.

    The caller identifies the rank with the lowest accumulated attention cost
    and computes its deficit: max_rank_cost - this_rank_cost. We pick the item
    whose attention cost is closest to that deficit, bringing the cheapest rank
    toward parity with the most expensive one.

    Returns (idx, cost) or (-1, 0) if nothing fits.
    """
    max_idx = bisect.bisect_right(dataset._lengths, remaining) - 1
    if max_idx < 0:
        return -1, 0

    costs_arr = np.asarray(dataset._costs[: max_idx + 1], dtype=np.int64)
    best_idx = int(np.abs(costs_arr - deficit).argmin())
    return best_idx, int(costs_arr[best_idx])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PreTokenizedDataset(IterableDataset, Stateful):
    """Abstract base for pre-tokenized SFT datasets.

    Handles manifest loading, shard concatenation, cross-rank LPT packing,
    and checkpointing. Subclasses implement format-specific primitives
    (_cost_from_metadata, _materialize_item, _new_batch, _place_item, _pad_and_flush).

    Cross-rank packing: all DP ranks read from the same global data stream.
    Each packing step forms dp_world_size batches simultaneously: seed with
    oldest items, fill by selecting items and assigning to batches using exact
    attention cost for balancing. Each rank yields only its assigned batch.
    No cross-rank communication — determinism is guaranteed by identical input
    order + identical logic.

    Packing modes:
    - "longest": pick longest fitting item, assign to cheapest rank.
    - "buffer_shuffle": pick random fitting item, assign to cheapest rank.
    - "attn_balanced": pick cheapest rank, select item to close attention
      cost gap with the most expensive rank.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int,
        *,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        cp_rank: int = 0,
        infinite: bool = False,
        tokenizer: BaseTokenizer | None = None,
        packing: str = "longest",
        buffer_size: int = 512,
        _manifest: dict[str, Any] | None = None,
        _full_dataset: Dataset | None = None,
        _dataset_id: str | None = None,
    ) -> None:
        if packing not in _PACKING_MODES:
            raise ValueError(
                f"Unknown packing mode: {packing!r}. Supported: {sorted(_PACKING_MODES)}"
            )

        manifest_path = Path(manifest_path)
        manifest = _manifest if _manifest is not None else _load_manifest(manifest_path)

        if _full_dataset is not None:
            full_dataset = _full_dataset
        else:
            shards_dir = manifest_path.parent / "shards"
            full_dataset = _load_shards(manifest, shards_dir)

        # No DP split: all ranks see the same full dataset. Cross-rank LPT
        # forms dp_world_size batches per step; each rank takes its slice.
        self._original_data: Dataset = full_dataset
        self._data: Dataset = self._original_data

        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._eos_id: int = manifest["tokenizer"]["eos_token_id"]
        self._tokenizer = tokenizer
        self._cp_rank = cp_rank
        self._dataset_id = _dataset_id or f"pretok:{manifest_path}"
        self.seq_len = seq_len
        self.infinite = infinite

        self._packing = packing
        self._buffer_size = buffer_size

        self._sample_idx: int = 0
        self._epoch: int = 0
        self._logged_first_sample = False

        self._worker_id: int = 0
        self._num_workers: int = 1

        self._row_indices: list[int] = []
        self._lengths: list[int] = []
        self._costs: list[int] = []
        self._batch_rng = np.random.default_rng(42)
        self._pending_restore: tuple | None = None

        self._data_exhausted: bool = False
        self._expose_all_costs: bool = False
        self._last_all_batch_costs: list[int] = []

        self._precompute_dataset_stats()

    # --- Dataset baseline statistics ---

    def _precompute_dataset_stats(self) -> None:
        """Compute population mean length from Arrow metadata.

        This is the unbiased baseline for detecting packing strategy bias.
        """
        all_lengths = self._original_data.data.column("n_tokens").to_numpy()
        valid = all_lengths[all_lengths <= self.seq_len].astype(np.int64)
        if len(valid) == 0:
            self._dataset_mean_length = 0.0
            return
        self._dataset_mean_length: float = float(valid.mean())
        # TODO: Track dataset trained-token and cost baselines once
        # n_trained_tokens is a scalar column in the Arrow schema.

    # --- Abstract primitives (subclass contract) ---

    _arrow_list_columns: tuple[str, ...] = ()
    _arrow_scalar_columns: tuple[str, ...] = ()
    _cost_list_columns: tuple[str, ...] = ()

    @abstractmethod
    def _cost_from_metadata(
        self,
        scalars: dict[str, np.ndarray],
        list_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
        idx: int,
    ) -> tuple[int, int] | None:
        """Return (length, cost) from metadata columns, or None to skip row."""
        ...

    @abstractmethod
    def _materialize_item(self, row_idx: int) -> Any:
        """Read full item from Arrow table by shard-relative row index."""
        ...

    def _read_list_column(self, table_slice, col_name: str) -> np.ndarray:
        assert table_slice.num_rows == 1
        col = table_slice.column(col_name).combine_chunks()
        offsets = col.offsets.to_numpy()
        return col.values.to_numpy()[offsets[0]:offsets[1]].copy()

    @abstractmethod
    def _new_batch(self) -> dict:
        """Create empty mutable batch accumulator."""
        ...

    @abstractmethod
    def _place_item(self, batch: dict, item) -> None:
        """Append one item into the batch accumulator."""
        ...

    @abstractmethod
    def _pad_and_flush(
        self, batch: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
        """Pad to seq_len and produce final (inputs_dict, labels, stats)."""
        ...

    # --- Shared infrastructure ---

    def _insert_entry(self, length: int, cost: int, row_idx: int) -> None:
        idx = bisect.bisect_right(self._lengths, length)
        self._row_indices.insert(idx, row_idx)
        self._lengths.insert(idx, length)
        self._costs.insert(idx, cost)

    def _remove_at(self, idx: int) -> None:
        del self._row_indices[idx]
        del self._lengths[idx]
        del self._costs[idx]

    def _log_first_sample(self, input_ids: list[int], label_ids: list[int]) -> None:
        """Log the first sample with trained tokens highlighted."""
        if self._logged_first_sample or self._cp_rank != 0 or self._tokenizer is None:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.id != 0:
            return
        RED, RESET = "\033[31m", "\033[0m"
        is_trained = [False] + [
            label_ids[j - 1] != IGNORE_INDEX for j in range(1, len(input_ids))
        ]
        parts = []
        k = 0
        while k < len(input_ids):
            end_k = k + 1
            while end_k < len(input_ids) and is_trained[end_k] == is_trained[k]:
                end_k += 1
            text = (
                self._tokenizer.decode(input_ids[k:end_k], skip_special_tokens=False)
                .encode("unicode_escape")
                .decode("ascii")
            )
            parts.append(f"{RED}{text}{RESET}" if not is_trained[k] else text)
            k = end_k
        logger.info(
            "[%s] First sample (red = not predicted):\n%s",
            type(self).__name__,
            "".join(parts),
        )
        self._logged_first_sample = True

    @property
    def num_examples(self) -> int:
        """Examples this rank will consume (full dataset / dp_world_size)."""
        return len(self._original_data) // self._dp_world_size

    def _prepare_iter(self) -> None:
        """Worker detection and data sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers
        if self._num_workers > 1:
            self._data = cast(
                Dataset,
                split_dataset_by_node(
                    self._original_data, self._worker_id, self._num_workers
                ),
            )
        else:
            self._data = self._original_data
        self._sample_idx = min(self._sample_idx, len(self._data))
        if self._pending_restore is not None:
            row_indices = self._pending_restore
            self._pending_restore = None
            self._reconstruct_buffer(row_indices)

    def _advance_epoch(self) -> None:
        self._sample_idx = 0
        self._epoch += 1
        self._data_exhausted = False
        logger.warning(
            "Dataset '%s' is being re-looped (epoch %d)",
            self._dataset_id,
            self._epoch,
        )

    def _refill_buffer(self) -> None:
        """Batch-read metadata from Arrow tables (no full item materialization)."""
        needed = self._buffer_size - len(self._row_indices)
        if needed <= 0 or self._data_exhausted:
            return
        data_len = len(self._data)
        if self._sample_idx >= data_len:
            self._data_exhausted = True
            return

        chunk_size = min(needed, data_len - self._sample_idx)
        table_slice = self._data.data.slice(self._sample_idx, chunk_size)

        scalars: dict[str, np.ndarray] = {}
        for col_name in self._arrow_scalar_columns:
            scalars[col_name] = table_slice.column(col_name).to_numpy()

        list_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col_name in self._cost_list_columns:
            col = table_slice.column(col_name).combine_chunks()
            list_arrays[col_name] = (col.offsets.to_numpy(), col.values.to_numpy())

        base_idx = self._sample_idx
        for i in range(chunk_size):
            result = self._cost_from_metadata(scalars, list_arrays, i)
            if result is not None:
                length, cost = result
                self._insert_entry(length, cost, base_idx + i)

        self._sample_idx += chunk_size

    # --- Cross-rank LPT packing loop ---

    def _iter_packed(self):
        self._data_exhausted = False
        dp = self._dp_world_size

        while True:
            self._refill_buffer()

            epoch_remnant = len(self._row_indices) < dp
            if epoch_remnant:
                if not self.infinite:
                    if self._row_indices:
                        import warnings
                        warnings.warn(
                            f"Dataset '{self._dataset_id}': {len(self._row_indices)} "
                            f"items remaining at end of epoch but need {dp} to form "
                            f"a step. Dropping.",
                            stacklevel=2,
                        )
                    break
                self._advance_epoch()
                self._refill_buffer()
                if len(self._row_indices) < dp:
                    break

            # Seed: dp random buffer entries
            seed_indices = sorted(
                self._batch_rng.choice(len(self._row_indices), size=dp, replace=False),
                reverse=True,
            )
            seed_lengths = [self._lengths[i] for i in seed_indices]
            seed_costs = [self._costs[i] for i in seed_indices]
            seed_row_indices = [self._row_indices[i] for i in seed_indices]
            for i in seed_indices:
                self._remove_at(i)

            my_batch = self._new_batch()
            batch_remaining = [self.seq_len] * dp
            batch_cost = [0] * dp

            for r in range(dp):
                batch_remaining[r] -= seed_lengths[r]
                batch_cost[r] += seed_costs[r]
                if r == self._dp_rank:
                    item = self._materialize_item(seed_row_indices[r])
                    self._place_item(my_batch, item)

            if self._packing == "attn_balanced":
                self._fill_attn_balanced(my_batch, batch_remaining, batch_cost, dp)
            else:
                self._fill_default(my_batch, batch_remaining, batch_cost, dp)

            my_batch["attn_cost"] = batch_cost[self._dp_rank]
            if self._expose_all_costs:
                self._last_all_batch_costs = list(batch_cost)
            yield self._pad_and_flush(my_batch)

    def _fill_default(
        self,
        my_batch: dict,
        batch_remaining: list[int],
        batch_cost: list[int],
        dp: int,
    ) -> None:
        """Fill loop for longest/buffer_shuffle: pick item, assign to cheapest rank."""
        select = (
            _select_longest if self._packing == "longest"
            else _select_buffer_shuffle
        )
        while True:
            if not self._row_indices:
                self._refill_buffer()
                if not self._row_indices:
                    break

            max_remaining = max(batch_remaining)
            if max_remaining <= 0:
                break

            idx = select(self, max_remaining)
            if idx == -1:
                break

            item_len = self._lengths[idx]
            item_cost = self._costs[idx]
            row_idx = self._row_indices[idx]

            best_rank = -1
            best_cost = float("inf")
            for r in range(dp):
                if batch_remaining[r] >= item_len and batch_cost[r] < best_cost:
                    best_cost = batch_cost[r]
                    best_rank = r

            self._remove_at(idx)

            if best_rank == self._dp_rank:
                item = self._materialize_item(row_idx)
                self._place_item(my_batch, item)

            batch_remaining[best_rank] -= item_len
            batch_cost[best_rank] += item_cost

    def _fill_attn_balanced(
        self,
        my_batch: dict,
        batch_remaining: list[int],
        batch_cost: list[int],
        dp: int,
    ) -> None:
        """Fill loop for attn_balanced: pick cheapest rank, select item to close gap."""
        while True:
            if not self._row_indices:
                self._refill_buffer()
                if not self._row_indices:
                    break

            best_rank = -1
            best_cost_val = float("inf")
            for r in range(dp):
                if batch_remaining[r] > 0 and batch_cost[r] < best_cost_val:
                    best_cost_val = batch_cost[r]
                    best_rank = r
            if best_rank == -1:
                break

            remaining = batch_remaining[best_rank]
            deficit = max(batch_cost) - batch_cost[best_rank]

            if deficit == 0:
                idx = _select_buffer_shuffle(self, remaining)
                if idx == -1:
                    batch_remaining[best_rank] = 0
                    continue
                item_cost = self._costs[idx]
            else:
                idx, item_cost = _select_attn_balanced(self, remaining, deficit)
                if idx == -1:
                    batch_remaining[best_rank] = 0
                    continue

            item_len = self._lengths[idx]
            row_idx = self._row_indices[idx]
            self._remove_at(idx)

            if best_rank == self._dp_rank:
                item = self._materialize_item(row_idx)
                self._place_item(my_batch, item)

            batch_remaining[best_rank] -= item_len
            batch_cost[best_rank] += item_cost

    def __iter__(self):
        self._prepare_iter()
        yield from self._iter_packed()

    # --- Checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "epoch": self._epoch,
            "sample_idx": self._sample_idx,
            "batch_rng_state": self._batch_rng.bit_generator.state,
        }
        if self._pending_restore is not None:
            d["row_indices"] = list(self._pending_restore)
        elif self._row_indices:
            d["row_indices"] = list(self._row_indices)
        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = state_dict["epoch"]
        self._sample_idx = state_dict["sample_idx"]
        if "batch_rng_state" in state_dict:
            self._batch_rng = np.random.default_rng()
            self._batch_rng.bit_generator.state = state_dict["batch_rng_state"]
        if "row_indices" in state_dict:
            self._pending_restore = state_dict["row_indices"]

    def _reconstruct_buffer(self, row_indices: list[int]) -> None:
        """Re-read metadata from Arrow to rebuild buffer state."""
        data_len = len(self._data)
        for row_idx in row_indices:
            if row_idx >= data_len:
                raise ValueError(
                    f"Stored row index {row_idx} exceeds dataset length {data_len}. "
                    f"Checkpoint was likely saved with a different num_workers or "
                    f"dataset shard configuration."
                )

        needed_cols = list(self._arrow_scalar_columns) + list(self._cost_list_columns)
        table_slice = self._data.data.table.select(needed_cols).take(row_indices)

        scalars: dict[str, np.ndarray] = {}
        for col_name in self._arrow_scalar_columns:
            scalars[col_name] = table_slice.column(col_name).to_numpy()

        list_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col_name in self._cost_list_columns:
            col = table_slice.column(col_name).combine_chunks()
            list_arrays[col_name] = (col.offsets.to_numpy(), col.values.to_numpy())

        for i, row_idx in enumerate(row_indices):
            result = self._cost_from_metadata(scalars, list_arrays, i)
            if result is None:
                continue
            length, cost = result
            self._insert_entry(length, cost, row_idx)


# ---------------------------------------------------------------------------
# Format: truncate_last
# ---------------------------------------------------------------------------


class _ChatItem(NamedTuple):
    input_ids: np.ndarray
    labels: np.ndarray


class StandardPackingDataset(PreTokenizedDataset):
    """Pre-tokenized (input_ids, labels, n_tokens) shards.

    Used by strategies that produce the standard (input_ids, labels) schema:
    TruncateLastStrategy, FullThinkingStrategy, TruncateEveryTurnStrategy.
    """

    _arrow_list_columns = ("input_ids", "labels")
    _arrow_scalar_columns = ("n_tokens",)
    _cost_list_columns: tuple[str, ...] = ()

    def _cost_from_metadata(self, scalars, list_arrays, idx):
        n = int(scalars["n_tokens"][idx])
        if n > self.seq_len:
            return None
        return (n, n * (n + 1) // 2)

    def _materialize_item(self, row_idx: int) -> _ChatItem:
        table_slice = self._data.data.slice(row_idx, 1)
        fields = {
            col: self._read_list_column(table_slice, col)
            for col in self._arrow_list_columns
        }
        if not self._logged_first_sample:
            self._log_first_sample(
                fields["input_ids"].tolist(), fields["labels"].tolist()
            )
        return _ChatItem(fields["input_ids"], fields["labels"])

    def _new_batch(self) -> dict:
        return {
            "inputs": np.full(self.seq_len, self._eos_id, dtype=np.int32),
            "labels": np.full(self.seq_len, IGNORE_INDEX, dtype=np.int32),
            "positions": np.zeros(self.seq_len, dtype=np.int32),
            "offset": 0,
            "n_total": 0,
            "n_trained": 0,
            "n_examples": 0,
        }

    def _place_item(self, batch: dict, item: _ChatItem) -> None:
        n = len(item.input_ids)
        off = batch["offset"]
        batch["inputs"][off:off + n] = item.input_ids
        batch["labels"][off:off + n] = item.labels
        batch["positions"][off:off + n] = np.arange(n, dtype=np.int32)
        batch["offset"] = off + n
        batch["n_total"] += n
        batch["n_trained"] += int(np.count_nonzero(item.labels != IGNORE_INDEX))
        batch["n_examples"] += 1

    def _pad_and_flush(
        self, batch: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
        off = batch["offset"]
        pad_len = self.seq_len - off
        if pad_len > 0:
            batch["positions"][off:] = np.arange(pad_len, dtype=np.int32)
        stats: dict[str, int] = {
            "n_total_tokens": batch["n_total"],
            "n_trained_tokens": batch["n_trained"],
            "n_examples_packed": batch["n_examples"],
        }
        return (
            {
                "input": torch.from_numpy(batch["inputs"].astype(np.int64)),
                "positions": torch.from_numpy(batch["positions"].astype(np.int64)),
                "attn_cost": torch.tensor(batch.get("attn_cost", 0), dtype=torch.int64),
            },
            torch.from_numpy(batch["labels"].astype(np.int64)),
            stats,
        )



# ---------------------------------------------------------------------------
# Format: backbone_suffix
# ---------------------------------------------------------------------------


class _BackboneSuffixItem(NamedTuple):
    input_ids: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    suffix_starts: np.ndarray
    insertion_limits: np.ndarray


class BackboneSuffixDataset(PreTokenizedDataset):
    """Pre-tokenized backbone+suffix shards for flex attention.

    Reads (input_ids, labels, positions, suffix_starts, insertion_limits, n_tokens)
    and expands the compressed suffix metadata into per-token [seq_len] tensors
    (conv_ids, suffix_ids, insertion_limits) during packing.
    """

    _arrow_list_columns = (
        "input_ids", "labels", "positions", "suffix_starts", "insertion_limits",
    )
    _arrow_scalar_columns = ("n_tokens",)
    _cost_list_columns = ("suffix_starts", "insertion_limits")

    def _cost_from_metadata(self, scalars, list_arrays, idx):
        n = int(scalars["n_tokens"][idx])
        if n > self.seq_len:
            return None
        offsets, values = list_arrays["suffix_starts"]
        suffix_starts = values[offsets[idx]:offsets[idx + 1]]
        offsets, values = list_arrays["insertion_limits"]
        insertion_limits = values[offsets[idx]:offsets[idx + 1]]
        if len(suffix_starts) == 0:
            return (n, n * (n + 1) // 2)
        backbone_len = int(suffix_starts[0])
        cost = backbone_len * (backbone_len + 1) // 2
        for k in range(len(suffix_starts)):
            s_start = int(suffix_starts[k])
            s_end = int(suffix_starts[k + 1]) if k + 1 < len(suffix_starts) else n
            s_len = s_end - s_start
            cost += s_len * (s_len + 1) // 2
            cost += s_len * (int(insertion_limits[k]) + 1)
        return (n, cost)

    def _materialize_item(self, row_idx: int) -> _BackboneSuffixItem:
        table_slice = self._data.data.slice(row_idx, 1)
        fields = {
            col: self._read_list_column(table_slice, col)
            for col in self._arrow_list_columns
        }
        if not self._logged_first_sample:
            self._log_first_sample(
                fields["input_ids"].tolist(), fields["labels"].tolist()
            )
        return _BackboneSuffixItem(
            fields["input_ids"],
            fields["labels"],
            fields["positions"],
            fields["suffix_starts"],
            fields["insertion_limits"],
        )

    def _new_batch(self) -> dict:
        return {
            "inputs": np.full(self.seq_len, self._eos_id, dtype=np.int32),
            "labels": np.full(self.seq_len, IGNORE_INDEX, dtype=np.int32),
            "positions": np.zeros(self.seq_len, dtype=np.int32),
            "conv_ids": np.zeros(self.seq_len, dtype=np.int32),
            "suffix_ids": np.zeros(self.seq_len, dtype=np.int32),
            "insertion_limits": np.full(self.seq_len, -1, dtype=np.int32),
            "offset": 0,
            "n_total": 0,
            "n_trained": 0,
            "n_examples": 0,
            "conv_counter": 1,
            "suffix_counter": 1,
        }

    def _place_item(self, batch: dict, item: _BackboneSuffixItem) -> None:
        input_ids = item.input_ids
        labels = item.labels
        positions = item.positions
        suffix_starts = item.suffix_starts
        ins_limits = item.insertion_limits
        off = batch["offset"]
        n = len(input_ids)
        backbone_len = suffix_starts[0] if len(suffix_starts) > 0 else n
        conv_id = batch["conv_counter"]

        batch["inputs"][off:off + n] = input_ids
        batch["labels"][off:off + n] = labels
        batch["positions"][off:off + n] = positions
        batch["conv_ids"][off:off + n] = conv_id

        # suffix_ids: 0 for backbone, unique counter for each suffix
        suffix_counter = batch["suffix_counter"]
        for k in range(len(suffix_starts)):
            s_start = suffix_starts[k]
            s_end = suffix_starts[k + 1] if k + 1 < len(suffix_starts) else n
            batch["suffix_ids"][off + s_start:off + s_end] = suffix_counter
            suffix_counter += 1
        batch["suffix_counter"] = suffix_counter

        # insertion_limits: backbone gets off + backbone_len - 1,
        # each suffix gets off + its compressed limit value
        backbone_limit = off + backbone_len - 1
        batch["insertion_limits"][off:off + backbone_len] = backbone_limit
        for k in range(len(suffix_starts)):
            s_start = suffix_starts[k]
            s_end = suffix_starts[k + 1] if k + 1 < len(suffix_starts) else n
            batch["insertion_limits"][off + s_start:off + s_end] = off + ins_limits[k]

        batch["offset"] = off + n
        batch["n_total"] += n
        batch["n_trained"] += int(np.count_nonzero(batch["labels"][off:off + n] != IGNORE_INDEX))
        batch["n_examples"] += 1
        batch["conv_counter"] += 1

    def _pad_and_flush(
        self, batch: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
        off = batch["offset"]
        pad_len = self.seq_len - off
        if pad_len > 0:
            batch["positions"][off:] = np.arange(pad_len, dtype=np.int32)
        stats: dict[str, int] = {
            "n_total_tokens": batch["n_total"],
            "n_trained_tokens": batch["n_trained"],
            "n_examples_packed": batch["n_examples"],
        }
        return (
            {
                "input": torch.from_numpy(batch["inputs"].astype(np.int64)),
                "positions": torch.from_numpy(batch["positions"].astype(np.int64)),
                "conv_ids": torch.from_numpy(batch["conv_ids"].astype(np.int64)),
                "suffix_ids": torch.from_numpy(batch["suffix_ids"].astype(np.int64)),
                "insertion_limits": torch.from_numpy(
                    batch["insertion_limits"].astype(np.int64)
                ),
                "attn_cost": torch.tensor(batch.get("attn_cost", 0), dtype=torch.int64),
            },
            torch.from_numpy(batch["labels"].astype(np.int64)),
            stats,
        )



# ---------------------------------------------------------------------------
# Planned (offline) packing dataset
# ---------------------------------------------------------------------------


class PlannedPackingDataset(IterableDataset, Stateful):
    """Dataset that reads a pre-computed pack plan for cost-balanced batching.

    Instead of online packing (buffer-based, sequential), this class loads a
    pack plan (produced by plan_packing.py) and iterates through cost-sorted
    chunks of dp_degree packs. All FSDP ranks in a global batch get packs with
    near-identical attention cost, minimizing synchronization idle time.

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

        # Dataloader compatibility
        stats = manifest.get("stats", {})
        length_stats = manifest.get("length_stats", {})
        self._dataset_mean_length: float = float(
            length_stats.get("mean", stats.get("tokens_per_example", 0.0))
        )

    @property
    def num_examples(self) -> int:
        return len(self._example_indices)

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
        epoch = self._epoch
        while True:
            chunk_pack_indices = self._epoch_setup(epoch)
            start_step = self._step if epoch == self._epoch else 0

            for step_idx in range(start_step, len(chunk_pack_indices)):
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


# ---------------------------------------------------------------------------
# Dataset registry + DataLoader
# ---------------------------------------------------------------------------

_DATASET_CLASSES: dict[str, type[PreTokenizedDataset]] = {
    "truncate_last": StandardPackingDataset,
    "backbone_suffix": BackboneSuffixDataset,
    "full_thinking": StandardPackingDataset,
    "truncate_every_turn": StandardPackingDataset,
}


class GranitePreTokenizedDataLoader(ParallelAwareDataloader):
    """DataLoader for pre-tokenized Arrow shards.

    Reads manifest["strategy"] to dispatch to the correct dataset class.
    Supports all strategies registered in _DATASET_CLASSES.

    Data stats (token counts, example counts) are accumulated in the main
    process as batches are consumed via __iter__, so they reflect only
    consumed batches — not prefetched ones.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str
        """Directory (or comma-separated directories) containing pretokenized
        shards and manifest.json. Also accepts direct paths to manifest.json
        for backward compatibility."""

        infinite: bool = True
        """Loop the dataset indefinitely."""

        packing: Literal[
            "longest",
            "buffer_shuffle",
            "attn_balanced",
            "prepacked_attn_grouped",
            "prepacked_random",
            "prepacked_random_balanced",
        ] = "buffer_shuffle"
        """Packing algorithm. 'longest' picks longest fitting item from buffer
        (deterministic, ~99.9% efficiency at 128k seq_len). 'buffer_shuffle'
        picks random fitting item (deterministic RNG across ranks).
        'attn_balanced' targets cross-rank attention cost balance by selecting
        items that close the gap between cheapest and most expensive rank.
        'prepacked_attn_grouped' uses offline pack plan with cost-balanced chunk
        assignment (packs grouped by attn_cost for sync efficiency).
        'prepacked_random' uses offline pack plan with random chunk assignment
        (smoother loss at the cost of sync efficiency).
        'prepacked_random_balanced' uses offline pack plan with random global
        shuffle plus local cost-balancing within each optimizer step window
        (dp_degree * accum_steps packs). Preserves unbiased sampling while
        recovering TPS via cost-homogeneous micro-batches.
        All prepacked modes read from {pretok_dir}/pack_plans/seqlen_{seq_len}/
        (produced by plan_packing.py); 'buffer_size' is ignored."""

        buffer_size: int = 512
        """Number of examples held in the lookahead buffer (per worker)."""

        seed: int = 42
        """RNG seed for epoch shuffling in planned packing mode."""

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
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        cp_rank: int = 0,
        accum_steps: int = 1,
    ) -> None:
        if local_batch_size > 1:
            raise ValueError(
                f"local_batch_size must be 1 for pre-tokenized datasets (got "
                f"{local_batch_size}). Packing handles batching within each "
                f"sequence, and mixed strategies produce heterogeneous tensors "
                f"that cannot be stacked."
            )

        # Parse comma-separated paths
        if not config.dataset_path:
            raise ValueError(
                "dataloader.dataset_path must be set (--dataloader.dataset-path on CLI)."
            )
        raw_paths = [p.strip() for p in config.dataset_path.split(",") if p.strip()]
        manifest_paths: list[Path] = []
        for p in raw_paths:
            path = Path(p)
            if path.suffix != ".json":
                path = path / "manifest.json"
            manifest_paths.append(path)

        if config.packing in (
            "prepacked_attn_grouped",
            "prepacked_random",
            "prepacked_random_balanced",
        ):
            if len(manifest_paths) > 1:
                raise ValueError(
                    f"packing={config.packing!r} cannot be used with multiple "
                    "dataset_path entries. Generate a plan from the merged "
                    "pretok dir instead."
                )
            pretok_dir = manifest_paths[0].parent
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
        else:
            if len(manifest_paths) > 1:
                manifest, full_dataset = _load_and_merge_manifests(manifest_paths)
                dataset_id = f"pretok:merged[{','.join(p.parent.parent.name for p in manifest_paths)}]"
            else:
                manifest = _load_manifest(manifest_paths[0])
                full_dataset = None
                dataset_id = None

            strategy = manifest.get("strategy")
            if strategy not in _DATASET_CLASSES:
                raise ValueError(
                    f"Unsupported strategy {strategy!r} in {config.dataset_path}. "
                    f"Supported: {sorted(_DATASET_CLASSES)}"
                )

            dataset = _DATASET_CLASSES[strategy](
                manifest_path=str(manifest_paths[0]),
                seq_len=seq_len,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                cp_rank=cp_rank,
                infinite=config.infinite,
                tokenizer=tokenizer,
                packing=config.packing,
                buffer_size=config.buffer_size,
                _manifest=manifest,
                _full_dataset=full_dataset,
                _dataset_id=dataset_id,
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
