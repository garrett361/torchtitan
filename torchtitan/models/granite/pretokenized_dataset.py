"""Pre-tokenized SFT datasets for offline training.

Reads Arrow shards produced by pretokenize_sft.py. Strategy dispatch happens
at GranitePreTokenizedDataLoader construction time via manifest["strategy"].

Class hierarchy:
    PreTokenizedDataset           — abstract base: manifest loading, DP sharding,
                                    unified packing loop, checkpointing
    TruncateLastDataset           — format primitives for (input_ids, labels) shards
    BackboneSuffixDataset         — format primitives for backbone+suffix shards
    GranitePreTokenizedDataLoader — dataloader; dispatches to the right dataset
                                    class based on manifest["strategy"]

Packing mode (greedy, buffer, cost_balanced) is a constructor parameter on
PreTokenizedDataset, not a separate class hierarchy.
"""

import bisect
import json
import math
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

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


_SEQ_LEN_CUTOFFS_K = [16, 32, 64, 128, 256, 512]


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

    # Merge length_stats via weighted averages (exact for means)
    all_length_stats = [m["stats"].get("length_stats") for m in manifests]
    merged_length_stats: dict[str, Any] | None = None
    if all(ls is not None for ls in all_length_stats):
        merged_length_stats = {}
        for k in _SEQ_LEN_CUTOFFS_K:
            sq_key = f"squared_tokens_per_example_{k}kmax"
            mean_key = f"tokens_per_example_{k}kmax"
            n_key = f"n_examples_{k}kmax"

            total_n = sum(ls.get(n_key, 0) for ls in all_length_stats)
            if total_n > 0:
                merged_length_stats[sq_key] = round(
                    sum(ls.get(n_key, 0) * ls.get(sq_key, 0) for ls in all_length_stats)
                    / total_n,
                    1,
                )
                merged_length_stats[mean_key] = round(
                    sum(
                        ls.get(n_key, 0) * ls.get(mean_key, 0)
                        for ls in all_length_stats
                    )
                    / total_n,
                    1,
                )
                merged_length_stats[n_key] = total_n
            else:
                merged_length_stats[sq_key] = None
                merged_length_stats[mean_key] = None
                merged_length_stats[n_key] = 0

    merged_manifest: dict[str, Any] = {
        "version": ref["version"],
        "strategy": ref["strategy"],
        "tokenizer": ref["tokenizer"],
        "stats": {
            "total_examples": total_examples,
            "total_tokens": total_tokens,
            "total_trained_tokens": total_trained_tokens,
            "length_stats": merged_length_stats,
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

SelectFn = Callable[["PreTokenizedDataset", int, dict], int]


def _select_greedy(dataset: "PreTokenizedDataset", remaining: int, batch: dict) -> int:
    """Take the next buffered item if it fits (FIFO for unsorted buffer)."""
    if dataset._buffer and dataset._lengths[0] <= remaining:
        return 0
    return -1


def _select_largest_fitting(
    dataset: "PreTokenizedDataset", remaining: int, batch: dict
) -> int:
    """Binary search the sorted buffer for the longest item that fits."""
    idx = bisect.bisect_right(dataset._lengths, remaining) - 1
    return idx if idx >= 0 else -1


def _select_cost_balanced(
    dataset: "PreTokenizedDataset", remaining: int, batch: dict
) -> int:
    """Pick the item whose addition brings batch cost closest to target_cost.

    Uses binary search on the sorted _lengths array: optimal length is
    sqrt(target - current_cost), bisect to that point, check neighbors.
    """
    max_idx = bisect.bisect_right(dataset._lengths, remaining) - 1
    if max_idx < 0:
        return -1

    current_cost = batch.get("cost", 0)
    target = dataset._target_cost
    ideal_len = math.isqrt(max(0, int(target - current_cost)))
    ins = bisect.bisect_left(dataset._lengths, ideal_len, hi=max_idx + 1)

    best_idx, best_gap = -1, float("inf")
    for candidate in range(max(0, ins - 1), min(max_idx + 1, ins + 2)):
        L = dataset._lengths[candidate]
        gap = abs(current_cost + L * L - target)
        if gap < best_gap:
            best_gap, best_idx = gap, candidate
    return best_idx


_SELECT_FNS: dict[str, SelectFn] = {
    "greedy": _select_greedy,
    "buffer": _select_largest_fitting,
    "cost_balanced": _select_cost_balanced,
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PreTokenizedDataset(IterableDataset, Stateful):
    """Abstract base for pre-tokenized SFT datasets.

    Handles manifest loading, shard concatenation, DP sharding, the unified
    packing loop, and checkpointing. Subclasses implement format-specific
    primitives (_tokenize_sample, _new_batch, _place_item, _pad_and_flush).

    Packing mode is selected via the ``packing`` constructor parameter:
    - "greedy": packs items in stream order without reordering.
    - "buffer": lookahead buffer with largest-fitting selection.
    - "cost_balanced": lookahead buffer minimizing attention cost variance.
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
        shuffle_in_memory: bool = True,
        tokenizer: BaseTokenizer | None = None,
        packing: str = "buffer",
        buffer_size: int = 64,
        target_cost: float | None = None,
        _manifest: dict[str, Any] | None = None,
        _full_dataset: Dataset | None = None,
        _dataset_id: str | None = None,
    ) -> None:
        if packing not in _SELECT_FNS:
            raise ValueError(
                f"Unknown packing mode: {packing!r}. Supported: {sorted(_SELECT_FNS)}"
            )
        if packing == "cost_balanced" and target_cost is None:
            raise ValueError("target_cost is required for cost_balanced packing")

        manifest_path = Path(manifest_path)
        manifest = _manifest if _manifest is not None else _load_manifest(manifest_path)

        if _full_dataset is not None:
            full_dataset = _full_dataset
        else:
            shards_dir = manifest_path.parent / "shards"
            full_dataset = _load_shards(manifest, shards_dir)

        # Shuffle before sharding so every DP rank gets a representative length
        # distribution. split_dataset_by_node on a Dataset always returns a Dataset.
        self._shuffle_in_memory = shuffle_in_memory
        self._original_data: Dataset = cast(
            Dataset,
            split_dataset_by_node(
                full_dataset.shuffle(seed=42, keep_in_memory=shuffle_in_memory),
                dp_rank,
                dp_world_size,
            ),
        )
        self._data: Dataset = self._original_data

        self._eos_id: int = manifest["tokenizer"]["eos_token_id"]
        self._tokenizer = tokenizer
        self._cp_rank = cp_rank
        self._dataset_id = _dataset_id or f"pretok:{manifest_path}"
        self.seq_len = seq_len
        self.infinite = infinite

        self._packing = packing
        self._select_fn: SelectFn = _SELECT_FNS[packing]
        self._buffer_size = buffer_size
        self._target_cost = target_cost or 0.0

        self._sample_idx: int = 0
        self._epoch: int = 0
        self._logged_first_sample = False

        self._worker_id: int = 0
        self._num_workers: int = 1

        # Sorted buffer with parallel arrays for O(log n) bisect selection.
        # _lengths is a separate int array (~1.6× faster than bisect key=lambda).
        self._buffer: list = []
        self._lengths: list[int] = []
        self._ages: list[int] = []
        self._age_counter: int = 0
        self._sorted = packing != "greedy"

        self._data_iter: Iterator | None = None
        self._data_exhausted: bool = False

    # --- Abstract primitives (subclass contract) ---

    @abstractmethod
    def _tokenize_sample(self, sample: dict) -> Any | None:
        """Parse raw sample into a NamedTuple with an `input_ids` field, or None to skip."""
        ...

    @abstractmethod
    def _new_batch(self) -> dict:
        """Create empty mutable batch accumulator.
        Must include an 'inputs' key (list[int]) for token counting."""
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

    _item_type: type

    def _serialize_buffer(self) -> list:
        return [list(item) for item in self._buffer]

    def _deserialize_buffer(self, data: list) -> list:
        return [self._item_type(*[list(f) for f in fields]) for fields in data]

    def _insert_item(self, item) -> None:
        item_len = len(item.input_ids)
        if self._sorted:
            idx = bisect.bisect_left(self._lengths, item_len)
            self._buffer.insert(idx, item)
            self._lengths.insert(idx, item_len)
            self._ages.insert(idx, self._age_counter)
        else:
            self._buffer.append(item)
            self._lengths.append(item_len)
            self._ages.append(self._age_counter)
        self._age_counter += 1

    def _remove_at(self, idx: int) -> None:
        del self._buffer[idx]
        del self._lengths[idx]
        del self._ages[idx]

    def _get_data_iter(self):
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

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
        """Total examples in this rank's shard (before packing)."""
        return len(self._original_data)

    def _prepare_iter(self) -> None:
        """Worker detection, epoch shuffle, sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers
        if self._epoch > 0:
            self._data = cast(
                Dataset,
                self._original_data.shuffle(
                    seed=42 + self._epoch, keep_in_memory=self._shuffle_in_memory
                ),
            )
        else:
            self._data = self._original_data
        if self._num_workers > 1:
            self._data = cast(
                Dataset,
                split_dataset_by_node(self._data, self._worker_id, self._num_workers),
            )
        self._sample_idx = min(self._sample_idx, len(self._data))

    def _advance_epoch(self) -> None:
        self._sample_idx = 0
        self._epoch += 1
        self._data = cast(
            Dataset,
            self._original_data.shuffle(
                seed=42 + self._epoch, keep_in_memory=self._shuffle_in_memory
            ),
        )
        if self._num_workers > 1:
            self._data = cast(
                Dataset,
                split_dataset_by_node(self._data, self._worker_id, self._num_workers),
            )
        self._data_iter = self._get_data_iter()
        self._data_exhausted = False
        logger.warning(
            "Dataset '%s' is being re-looped (epoch %d)",
            self._dataset_id,
            self._epoch,
        )

    def _refill_buffer(self) -> None:
        while len(self._buffer) < self._buffer_size and not self._data_exhausted:
            sample = next(self._data_iter, None)
            if sample is None:
                self._data_exhausted = True
                break
            result = self._tokenize_sample(sample)
            self._sample_idx += 1
            if result is None:
                continue
            self._insert_item(result)

    # --- Unified packing loop ---

    def _iter_packed(self):
        self._data_iter = self._get_data_iter()
        self._data_exhausted = False

        while True:
            self._refill_buffer()

            if not self._buffer:
                if not self.infinite:
                    break
                self._advance_epoch()
                continue

            batch = self._new_batch()
            if self._sorted:
                oldest_idx = min(range(len(self._ages)), key=self._ages.__getitem__)
            else:
                oldest_idx = 0
            first = self._buffer[oldest_idx]
            self._remove_at(oldest_idx)
            self._place_item(batch, first)
            batch["cost"] = len(first.input_ids) ** 2

            while True:
                remaining = self.seq_len - len(batch["inputs"])
                if remaining <= 0:
                    break
                idx = self._select_fn(self, remaining, batch)
                if idx == -1:
                    break
                picked = self._buffer[idx]
                self._remove_at(idx)
                self._place_item(batch, picked)
                batch["cost"] += len(picked.input_ids) ** 2
                if not self._buffer:
                    self._refill_buffer()

            yield self._pad_and_flush(batch)

    def __iter__(self):
        self._prepare_iter()
        yield from self._iter_packed()

    # --- Checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "epoch": self._epoch,
            "sample_idx": self._sample_idx,
        }
        if self._buffer:
            d["buffer"] = self._serialize_buffer()
            d["ages"] = list(self._ages)
            d["age_counter"] = self._age_counter
        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = state_dict["epoch"]
        self._sample_idx = state_dict["sample_idx"]
        if "buffer" in state_dict:
            self._buffer = self._deserialize_buffer(state_dict["buffer"])
            self._lengths = [len(item.input_ids) for item in self._buffer]
            self._ages = state_dict.get("ages", list(range(len(self._buffer))))
            self._age_counter = state_dict.get("age_counter", len(self._buffer))
            if self._sorted:
                order = sorted(
                    range(len(self._lengths)), key=self._lengths.__getitem__
                )
                self._buffer = [self._buffer[i] for i in order]
                self._lengths = [self._lengths[i] for i in order]
                self._ages = [self._ages[i] for i in order]
        if self._epoch > 0:
            self._data = cast(
                Dataset,
                self._original_data.shuffle(
                    seed=42 + self._epoch, keep_in_memory=self._shuffle_in_memory
                ),
            )


# ---------------------------------------------------------------------------
# Format: truncate_last
# ---------------------------------------------------------------------------


class _ChatItem(NamedTuple):
    input_ids: list[int]
    labels: list[int]


class TruncateLastDataset(PreTokenizedDataset):
    """Pre-tokenized (input_ids, labels, n_tokens) shards.

    Produced by TruncateLastStrategy. Labels only the final assistant turn;
    all earlier turns are IGNORE_INDEX.
    """

    _item_type = _ChatItem

    def _tokenize_sample(self, sample: dict[str, Any]) -> _ChatItem | None:
        input_ids: list[int] = list(sample["input_ids"])
        label_ids: list[int] = list(sample["labels"])

        if len(input_ids) > self.seq_len:
            logger.debug(
                "Dropping pre-tokenized sample %d: %d tokens > seq_len %d",
                self._sample_idx,
                len(input_ids),
                self.seq_len,
            )
            return None

        self._log_first_sample(input_ids, label_ids)
        return _ChatItem(input_ids, label_ids)

    def _new_batch(self) -> dict:
        return {
            "inputs": [],
            "labels": [],
            "positions": [],
            "n_total": 0,
            "n_trained": 0,
            "n_examples": 0,
        }

    def _place_item(self, batch: dict, item: _ChatItem) -> None:
        batch["inputs"].extend(item.input_ids)
        batch["labels"].extend(item.labels)
        batch["positions"].extend(range(len(item.input_ids)))
        batch["n_total"] += len(item.input_ids)
        batch["n_trained"] += sum(1 for lbl in item.labels if lbl != IGNORE_INDEX)
        batch["n_examples"] += 1

    def _pad_and_flush(
        self, batch: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
        pad_len = self.seq_len - len(batch["inputs"])
        if pad_len > 0:
            batch["inputs"].extend([self._eos_id] * pad_len)
            batch["labels"].extend([IGNORE_INDEX] * pad_len)
            batch["positions"].extend(range(pad_len))
        stats: dict[str, int] = {
            "n_total_tokens": batch["n_total"],
            "n_trained_tokens": batch["n_trained"],
            "n_examples_packed": batch["n_examples"],
        }
        if self._packing == "cost_balanced":
            stats["batch_attention_cost"] = batch["cost"]
        return (
            {
                "input": torch.tensor(batch["inputs"], dtype=torch.long),
                "positions": torch.tensor(batch["positions"], dtype=torch.long),
            },
            torch.tensor(batch["labels"], dtype=torch.long),
            stats,
        )


# ---------------------------------------------------------------------------
# Format: backbone_suffix
# ---------------------------------------------------------------------------


class _BackboneSuffixItem(NamedTuple):
    input_ids: list[int]
    labels: list[int]
    positions: list[int]
    suffix_starts: list[int]
    insertion_limits: list[int]


class BackboneSuffixDataset(PreTokenizedDataset):
    """Pre-tokenized backbone+suffix shards for flex attention.

    Reads (input_ids, labels, positions, suffix_starts, insertion_limits, n_tokens)
    and expands the compressed suffix metadata into per-token [seq_len] tensors
    (conv_ids, suffix_ids, insertion_limits) during packing.
    """

    _item_type = _BackboneSuffixItem

    def _tokenize_sample(self, sample: dict) -> _BackboneSuffixItem | None:
        n_tokens = sample["n_tokens"]
        if n_tokens > self.seq_len:
            logger.debug(
                "Dropping backbone_suffix sample %d: %d tokens > seq_len %d",
                self._sample_idx,
                n_tokens,
                self.seq_len,
            )
            return None
        input_ids = list(sample["input_ids"])
        labels = list(sample["labels"])
        positions = list(sample["positions"])
        suffix_starts = list(sample["suffix_starts"])
        insertion_limits = list(sample["insertion_limits"])
        self._log_first_sample(input_ids, labels)
        return _BackboneSuffixItem(
            input_ids, labels, positions, suffix_starts, insertion_limits
        )

    def _new_batch(self) -> dict:
        return {
            "inputs": [],
            "labels": [],
            "positions": [],
            "conv_ids": [],
            "suffix_ids": [],
            "insertion_limits": [],
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
        off = len(batch["inputs"])
        n = len(input_ids)
        backbone_len = suffix_starts[0] if suffix_starts else n
        conv_id = batch["conv_counter"]

        batch["inputs"].extend(input_ids)
        batch["labels"].extend(labels)
        batch["positions"].extend(positions)
        batch["conv_ids"].extend([conv_id] * n)

        # suffix_ids: 0 for backbone, unique counter for each suffix
        local_suffix_ids = [0] * n
        suffix_counter = batch["suffix_counter"]
        for k in range(len(suffix_starts)):
            s_start = suffix_starts[k]
            s_end = suffix_starts[k + 1] if k + 1 < len(suffix_starts) else n
            for j in range(s_start, s_end):
                local_suffix_ids[j] = suffix_counter
            suffix_counter += 1
        batch["suffix_ids"].extend(local_suffix_ids)
        batch["suffix_counter"] = suffix_counter

        # insertion_limits: backbone gets off + backbone_len - 1,
        # each suffix gets off + its compressed limit value
        local_ins_limits = [0] * n
        backbone_limit = off + backbone_len - 1
        for j in range(backbone_len):
            local_ins_limits[j] = backbone_limit
        for k in range(len(suffix_starts)):
            s_start = suffix_starts[k]
            s_end = suffix_starts[k + 1] if k + 1 < len(suffix_starts) else n
            limit_val = off + ins_limits[k]
            for j in range(s_start, s_end):
                local_ins_limits[j] = limit_val
        batch["insertion_limits"].extend(local_ins_limits)

        batch["n_total"] += n
        batch["n_trained"] += sum(1 for lbl in labels if lbl != IGNORE_INDEX)
        batch["n_examples"] += 1
        batch["conv_counter"] += 1

    def _pad_and_flush(
        self, batch: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
        pad_len = self.seq_len - len(batch["inputs"])
        if pad_len > 0:
            batch["inputs"].extend([self._eos_id] * pad_len)
            batch["labels"].extend([IGNORE_INDEX] * pad_len)
            batch["positions"].extend(range(pad_len))
            batch["conv_ids"].extend([0] * pad_len)
            batch["suffix_ids"].extend([0] * pad_len)
            batch["insertion_limits"].extend([-1] * pad_len)
        stats: dict[str, int] = {
            "n_total_tokens": batch["n_total"],
            "n_trained_tokens": batch["n_trained"],
            "n_examples_packed": batch["n_examples"],
        }
        if self._packing == "cost_balanced":
            stats["batch_attention_cost"] = batch["cost"]
        return (
            {
                "input": torch.tensor(batch["inputs"], dtype=torch.long),
                "positions": torch.tensor(batch["positions"], dtype=torch.long),
                "conv_ids": torch.tensor(batch["conv_ids"], dtype=torch.long),
                "suffix_ids": torch.tensor(batch["suffix_ids"], dtype=torch.long),
                "insertion_limits": torch.tensor(
                    batch["insertion_limits"], dtype=torch.long
                ),
            },
            torch.tensor(batch["labels"], dtype=torch.long),
            stats,
        )



# ---------------------------------------------------------------------------
# Dataset registry + DataLoader
# ---------------------------------------------------------------------------

_DATASET_CLASSES: dict[str, type[PreTokenizedDataset]] = {
    "truncate_last": TruncateLastDataset,
    "backbone_suffix": BackboneSuffixDataset,
    "full_thinking": TruncateLastDataset,
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

        shuffle_in_memory: bool = True
        """Keep shuffle index in memory instead of writing a cache file to the shard
        directory. Avoids filesystem contention when many ranks start simultaneously."""

        packing: Literal["greedy", "buffer", "cost_balanced"] = "buffer"
        """Packing algorithm. 'buffer' maintains a lookahead buffer and selects
        largest-fitting examples (~99.9% efficiency at 128k seq_len). 'greedy'
        packs in sequential order (simpler but ~86% efficiency at 128k).
        'cost_balanced' targets uniform attention cost per sequence."""

        buffer_size: int = 64
        """Number of examples held in the lookahead buffer (per worker)."""

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

        # Compute target_cost for cost_balanced packing
        target_cost: float | None = None
        if config.packing == "cost_balanced":
            length_stats = manifest.get("stats", {}).get("length_stats")
            if not length_stats:
                raise ValueError(
                    "cost_balanced packing requires 'length_stats' in manifest(s). "
                    "Re-run pretokenize_sft.py to generate stats."
                )
            _CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]
            valid_cutoffs = [
                c
                for c in _CUTOFFS
                if c <= seq_len
                and f"tokens_per_example_{c // 1024}kmax" in length_stats
            ]
            if not valid_cutoffs:
                raise ValueError(
                    f"No valid cutoff ≤ seq_len={seq_len} with tokens_per_example "
                    f"stats in manifest {config.dataset_path}"
                )
            cutoff = max(valid_cutoffs)
            k = cutoff // 1024
            mean_sq_len = length_stats[f"squared_tokens_per_example_{k}kmax"]
            mean_len = length_stats[f"tokens_per_example_{k}kmax"]
            # Expected items per sequence (seq_len / E[L]) × expected cost per item (E[L²])
            target_cost = seq_len * mean_sq_len / mean_len
            logger.info(
                "Cost-balanced packing: target_cost=%.2e (T/seq²=%.3f, cutoff=%dk)",
                target_cost,
                target_cost / seq_len**2,
                k,
            )

        dataset = _DATASET_CLASSES[strategy](
            manifest_path=str(manifest_paths[0]),
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            cp_rank=cp_rank,
            infinite=config.infinite,
            shuffle_in_memory=config.shuffle_in_memory,
            tokenizer=tokenizer,
            packing=config.packing,
            buffer_size=config.buffer_size,
            target_cost=target_cost,
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
