"""Pre-tokenized SFT datasets for offline training.

Reads Arrow shards produced by pretokenize_sft.py. Strategy dispatch happens
at GranitePreTokenizedDataLoader construction time via manifest["strategy"].

Class hierarchy:
    PreTokenizedDataset        — abstract base: manifest loading, DP sharding,
                                 checkpointing skeleton
    TruncateLastDataset        — greedy packing for (input_ids, labels, n_tokens) shards
    GranitePreTokenizedDataLoader — single dataloader; dispatches to the right dataset
                                    class based on manifest["strategy"]
"""

import json
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

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


class PreTokenizedDataset(IterableDataset, Stateful):
    """Abstract base for pre-tokenized SFT datasets.

    Handles manifest loading, shard concatenation, DP sharding, and the
    checkpointing fields shared across all strategies. Subclasses own the
    packing loop and batch format.

    Subclass contract:
    - Implement __iter__ (typically delegates to a _iter_* packing method).
    - At end-of-epoch when re-looping, re-shuffle _data via
        self._data = cast(Dataset, self._original_data.shuffle(seed=42 + self._epoch, keep_in_memory=self._shuffle_in_memory))
      to maintain per-epoch diversity. Seed 42 covers epoch 0 (applied in __init__);
      subsequent epochs use seed 42+epoch for a deterministic sequence.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        cp_rank: int = 0,
        infinite: bool = False,
        shuffle_in_memory: bool = True,
        tokenizer: BaseTokenizer | None = None,
        _manifest: dict[str, Any] | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        manifest = _manifest if _manifest is not None else _load_manifest(manifest_path)
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
        self._dataset_id = f"pretok:{manifest_path}"
        self.seq_len = seq_len
        self.infinite = infinite

        self._sample_idx: int = 0
        self._epoch: int = 0
        self._logged_first_sample = False

        self._worker_id: int = 0
        self._num_workers: int = 1

    def _get_data_iter(self):
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def _log_first_sample(self, input_ids: list[int], label_ids: list[int]) -> None:
        """Log the first sample with trained tokens highlighted. No-op if already logged
        or tokenizer is unavailable."""
        if self._logged_first_sample or self._cp_rank != 0 or self._tokenizer is None:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.id != 0:
            return
        RED, RESET = "\033[31m", "\033[0m"
        # input_ids[j] is predicted at step j-1; it's "trained on" iff label_ids[j-1] != IGNORE_INDEX.
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

    @abstractmethod
    def __iter__(self): ...


class TruncateLastDataset(PreTokenizedDataset):
    """Greedy-packs pre-tokenized (input_ids, labels, n_tokens) shards.

    Produced by TruncateLastStrategy. Labels only the final assistant turn;
    all earlier turns are IGNORE_INDEX. Packing and checkpointing match
    ChatDataset behavior exactly.
    """

    def _tokenize_sample(
        self, sample: dict[str, Any]
    ) -> tuple[list[int], list[int]] | None:
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
        return input_ids, label_ids

    def _prepare_iter(self) -> None:
        """Common setup for __iter__: worker detection, epoch shuffle, sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers
        # Recompute from _original_data to handle repeated __iter__ calls
        # (persistent_workers=True recreates the iterator on the same instance)
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

    def __iter__(self):
        self._prepare_iter()
        yield from self._iter_greedy_packed()

    def _iter_greedy_packed(self):
        inputs_buffer: list[int] = []
        labels_buffer: list[int] = []
        positions_buffer: list[int] = []
        batch_n_total: int = 0
        batch_n_trained: int = 0
        batch_n_examples: int = 0

        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                if result is None:
                    self._sample_idx += 1
                    continue

                input_ids, label_ids = result
                remaining = self.seq_len - len(inputs_buffer)

                if len(input_ids) > remaining and len(inputs_buffer) > 0:
                    pad_len = remaining
                    inputs_buffer.extend([self._eos_id] * pad_len)
                    labels_buffer.extend([IGNORE_INDEX] * pad_len)
                    positions_buffer.extend(range(pad_len))
                    yield self._flush(
                        inputs_buffer,
                        labels_buffer,
                        positions_buffer,
                        batch_n_total,
                        batch_n_trained,
                        batch_n_examples,
                    )
                    inputs_buffer, labels_buffer, positions_buffer = [], [], []
                    batch_n_total = batch_n_trained = batch_n_examples = 0

                n_trained = sum(1 for lbl in label_ids if lbl != IGNORE_INDEX)
                batch_n_total += len(input_ids)
                batch_n_trained += n_trained
                batch_n_examples += 1

                inputs_buffer.extend(input_ids)
                labels_buffer.extend(label_ids)
                positions_buffer.extend(range(len(input_ids)))
                self._sample_idx += 1

                if len(inputs_buffer) == self.seq_len:
                    yield self._flush(
                        inputs_buffer,
                        labels_buffer,
                        positions_buffer,
                        batch_n_total,
                        batch_n_trained,
                        batch_n_examples,
                    )
                    inputs_buffer, labels_buffer, positions_buffer = [], [], []
                    batch_n_total = batch_n_trained = batch_n_examples = 0

            if len(inputs_buffer) > 0:
                pad_len = self.seq_len - len(inputs_buffer)
                if pad_len > 0:
                    inputs_buffer.extend([self._eos_id] * pad_len)
                    labels_buffer.extend([IGNORE_INDEX] * pad_len)
                    positions_buffer.extend(range(pad_len))
                yield self._flush(
                    inputs_buffer,
                    labels_buffer,
                    positions_buffer,
                    batch_n_total,
                    batch_n_trained,
                    batch_n_examples,
                )
                inputs_buffer, labels_buffer, positions_buffer = [], [], []
                batch_n_total = batch_n_trained = batch_n_examples = 0

            if not self.infinite:
                logger.warning("Dataset '%s' has run out of data", self._dataset_id)
                break
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
                    split_dataset_by_node(
                        self._data, self._worker_id, self._num_workers
                    ),
                )
            logger.warning(
                "Dataset '%s' is being re-looped (epoch %d)",
                self._dataset_id,
                self._epoch,
            )

    def _flush(
        self,
        inputs: list[int],
        labels: list[int],
        positions: list[int],
        n_total_tokens: int,
        n_trained_tokens: int,
        n_examples_packed: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
        return (
            {
                "input": torch.tensor(inputs, dtype=torch.long),
                "positions": torch.tensor(positions, dtype=torch.long),
            },
            torch.tensor(labels, dtype=torch.long),
            {
                "n_total_tokens": n_total_tokens,
                "n_trained_tokens": n_trained_tokens,
                "n_examples_packed": n_examples_packed,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "sample_idx": self._sample_idx,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = state_dict["epoch"]
        self._sample_idx = state_dict["sample_idx"]
        if self._epoch > 0:
            self._data = cast(
                Dataset,
                self._original_data.shuffle(
                    seed=42 + self._epoch, keep_in_memory=self._shuffle_in_memory
                ),
            )


class TruncateLastBufferDataset(TruncateLastDataset):
    """Buffer-packing variant of TruncateLastDataset.

    Maintains a lookahead buffer of pre-tokenized examples. For each batch,
    starts with the oldest buffered example (FIFO) then fills remaining space
    with the largest example that fits. Reduces padding waste compared to greedy
    packing, especially at long sequence lengths.
    """

    def __init__(self, *args, buffer_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_size = buffer_size
        self._buffer: list[tuple[list[int], list[int]]] = []
        self._data_iter: Iterator | None = None
        self._data_exhausted: bool = False

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
            self._buffer.append(result)

    def __iter__(self):
        self._prepare_iter()
        yield from self._iter_buffer_packed()

    def _iter_buffer_packed(self):
        self._data_iter = self._get_data_iter()
        self._data_exhausted = False

        while True:
            self._refill_buffer()

            if not self._buffer:
                if not self.infinite:
                    break
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
                        split_dataset_by_node(
                            self._data, self._worker_id, self._num_workers
                        ),
                    )
                self._data_iter = self._get_data_iter()
                self._data_exhausted = False
                logger.warning(
                    "Dataset '%s' is being re-looped (epoch %d)",
                    self._dataset_id,
                    self._epoch,
                )
                continue

            yield self._pack_one_batch()

    def _start_batch(
        self,
    ) -> tuple[list[int], list[int], list[int], int, int, int]:
        """Pop oldest from buffer (FIFO), return initial batch state."""
        oldest_ids, oldest_lbls = self._buffer.pop(0)
        inputs_buf = list(oldest_ids)
        labels_buf = list(oldest_lbls)
        positions_buf = list(range(len(oldest_ids)))
        batch_n_total = len(oldest_ids)
        batch_n_trained = sum(1 for lbl in oldest_lbls if lbl != IGNORE_INDEX)
        batch_n_examples = 1
        return inputs_buf, labels_buf, positions_buf, batch_n_total, batch_n_trained, batch_n_examples

    def _pack_one_batch(self):
        inputs_buf, labels_buf, positions_buf, batch_n_total, batch_n_trained, batch_n_examples = self._start_batch()

        while True:
            remaining = self.seq_len - len(inputs_buf)
            if remaining <= 0:
                break
            best_idx = -1
            best_len = 0
            for i, (ids, _) in enumerate(self._buffer):
                L = len(ids)
                if L <= remaining and L > best_len:
                    best_idx = i
                    best_len = L
            if best_idx == -1:
                break
            picked_ids, picked_lbls = self._buffer.pop(best_idx)
            inputs_buf.extend(picked_ids)
            labels_buf.extend(picked_lbls)
            positions_buf.extend(range(len(picked_ids)))
            batch_n_total += len(picked_ids)
            batch_n_trained += sum(
                1 for lbl in picked_lbls if lbl != IGNORE_INDEX
            )
            batch_n_examples += 1

        pad_len = self.seq_len - len(inputs_buf)
        if pad_len > 0:
            inputs_buf.extend([self._eos_id] * pad_len)
            labels_buf.extend([IGNORE_INDEX] * pad_len)
            positions_buf.extend(range(pad_len))

        return self._flush(
            inputs_buf,
            labels_buf,
            positions_buf,
            batch_n_total,
            batch_n_trained,
            batch_n_examples,
        )

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["buffer"] = [(ids, lbls) for ids, lbls in self._buffer]
        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        buffer = state_dict.pop("buffer", [])
        super().load_state_dict(state_dict)
        self._buffer = [(list(ids), list(lbls)) for ids, lbls in buffer]


class TruncateLastCostBalancedDataset(TruncateLastBufferDataset):
    """Cost-balanced packing: targets uniform attention cost per sequence.

    Selects buffer examples to minimize |sum(l_i²) - T| where T is the
    expected attention cost derived from manifest length stats.
    """

    def __init__(self, *args, target_cost: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_cost = target_cost

    def _pack_one_batch(self):
        inputs_buf, labels_buf, positions_buf, batch_n_total, batch_n_trained, batch_n_examples = self._start_batch()
        current_cost = len(inputs_buf) ** 2

        while True:
            remaining = self.seq_len - len(inputs_buf)
            if remaining <= 0:
                break
            best_idx, best_gap = -1, float("inf")
            for i, (ids, _) in enumerate(self._buffer):
                L = len(ids)
                if L > remaining:
                    continue
                gap = abs(current_cost + L * L - self._target_cost)
                if gap < best_gap:
                    best_gap = gap
                    best_idx = i
            if best_idx == -1:
                break
            picked_ids, picked_lbls = self._buffer.pop(best_idx)
            current_cost += len(picked_ids) ** 2
            inputs_buf.extend(picked_ids)
            labels_buf.extend(picked_lbls)
            positions_buf.extend(range(len(picked_ids)))
            batch_n_total += len(picked_ids)
            batch_n_trained += sum(1 for lbl in picked_lbls if lbl != IGNORE_INDEX)
            batch_n_examples += 1

        pad_len = self.seq_len - len(inputs_buf)
        if pad_len > 0:
            inputs_buf.extend([self._eos_id] * pad_len)
            labels_buf.extend([IGNORE_INDEX] * pad_len)
            positions_buf.extend(range(pad_len))

        result = self._flush(
            inputs_buf, labels_buf, positions_buf,
            batch_n_total, batch_n_trained, batch_n_examples,
        )
        result[2]["batch_attention_cost"] = current_cost
        return result


_DATASET_CLASSES: dict[str, type[PreTokenizedDataset]] = {
    "truncate_last": TruncateLastDataset,
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
        manifest_path: str
        """Path to manifest.json produced by pretokenize_sft.py."""

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
        """Number of examples held in the lookahead buffer (per worker).
        Only used when packing='buffer'."""

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
        manifest = _load_manifest(Path(config.manifest_path))
        strategy = manifest.get("strategy")
        if strategy not in _DATASET_CLASSES:
            raise ValueError(
                f"Unsupported strategy {strategy!r} in {config.manifest_path}. "
                f"Supported: {sorted(_DATASET_CLASSES)}"
            )
        dataset_kwargs = dict(
            manifest_path=config.manifest_path,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            cp_rank=cp_rank,
            infinite=config.infinite,
            shuffle_in_memory=config.shuffle_in_memory,
            tokenizer=tokenizer,
            _manifest=manifest,
        )
        if config.packing == "buffer":
            if strategy != "truncate_last":
                raise ValueError(
                    f"Buffer packing only supports 'truncate_last' strategy, "
                    f"got {strategy!r}"
                )
            dataset = TruncateLastBufferDataset(
                **dataset_kwargs, buffer_size=config.buffer_size
            )
        elif config.packing == "cost_balanced":
            if strategy != "truncate_last":
                raise ValueError(
                    f"Cost-balanced packing only supports 'truncate_last' strategy, "
                    f"got {strategy!r}"
                )
            length_stats = manifest.get("stats", {}).get("length_stats")
            if not length_stats:
                raise ValueError(
                    f"cost_balanced packing requires 'length_stats' in manifest. "
                    f"Run patch_manifest_stats.py on {config.manifest_path}"
                )
            _CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]
            valid_cutoffs = [
                c for c in _CUTOFFS
                if c <= seq_len and f"tokens_per_example_{c // 1024}kmax" in length_stats
            ]
            if not valid_cutoffs:
                raise ValueError(
                    f"No valid cutoff ≤ seq_len={seq_len} with tokens_per_example stats "
                    f"in manifest {config.manifest_path}"
                )
            cutoff = max(valid_cutoffs)
            k = cutoff // 1024
            sq_tokens = length_stats[f"squared_tokens_per_example_{k}kmax"]
            mean_tokens = length_stats[f"tokens_per_example_{k}kmax"]
            target_cost = seq_len * sq_tokens / mean_tokens
            logger.info(
                "Cost-balanced packing: target_cost=%.2e (T/seq²=%.3f, cutoff=%dk)",
                target_cost, target_cost / seq_len**2, k,
            )
            dataset = TruncateLastCostBalancedDataset(
                **dataset_kwargs, buffer_size=config.buffer_size, target_cost=target_cost
            )
        else:
            dataset = _DATASET_CLASSES[strategy](**dataset_kwargs)

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
