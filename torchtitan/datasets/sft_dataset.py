from dataclasses import dataclass, field
from typing import Iterator, Literal

import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import Dataset
from torchtitan.logging import logger


def next_power_of_2(n):
    out = 2
    while out < n:
        out *= 2
    return out


def _round_up_to_zig_zag_padding(num_toks: int, cp_degree: int) -> int:
    # return 2 * cp_degree * ((num_toks + 2 * cp_degree - 1) // (2 * cp_degree))
    # NOTE: @goon - getting some weird torch ring attn errors when not rounding up to a power of 2.
    # Shouldn't have to. Will figure out why.
    rounded_up = 2 * cp_degree * ((num_toks + 2 * cp_degree - 1) // (2 * cp_degree))
    return next_power_of_2(rounded_up)


class CPDataCollator:
    """
    Collates and adds proper padding given the cp_degree. Does not chunk up the data, as that is
    handled by the torchtitan CP context manager.
    """

    def __init__(
        self,
        cp_degree: int,
        pad_id: int = 0,
        separator_id: int = -100,
        naive_padding_free: bool = False,
    ):
        self.cp_degree = cp_degree
        self.pad_id = pad_id
        self.separator_id = separator_id
        self.naive_padding_free = naive_padding_free

    def __call__(self, features: list[dict[str, torch.Tensor]]):
        """
        Return None if there are non non-trivial preds
        """
        # features is a list[dict[str, Union[list[int], Tensor]]], make it always be list[dict[str,
        # Tensor]]
        if not torch.is_tensor(features[0]["input_ids"]):
            features = [
                {
                    "input_ids": torch.tensor(f["input_ids"]),
                    "labels": torch.tensor(f["labels"]),
                }
                for f in features
            ]

        if self.naive_padding_free:
            return self._collate_with_naive_padding_free(features)
        return self._collate_with_padding(features)

    def _collate_with_padding(self, features) -> dict[str, torch.Tensor]:
        ret = {"input_ids": [], "labels": []}
        # Need padding, both to align all elements in the batch and also to meet CP divisibility
        # requirements
        seqlens = [f["input_ids"].numel() for f in features]
        max_seqlen = max(seqlens)

        # NOTE: @goon - if using zig-zag, the per-rank tok counts also need to be even, hence
        # the factors of two
        padded_numel = _round_up_to_zig_zag_padding(max_seqlen, self.cp_degree)
        for item in features:
            input_ids = item["input_ids"]
            labels = item["labels"]
            # [CP causal shifting]
            # At this point, the input and labels are in exact causal alignment. We want to shift
            # the label indices over so that input_idx[t] is the input at time step t, while
            # labels[t] is the ground-truth tok at time t + 1. Then the loss is computed like
            # (schematically):
            #
            # ```py
            # out = model(inputs_ids)
            # loss = F.cross_entropy(out, labels)
            # ```

            # Shift and mask the final token
            labels = labels.roll(-1)
            labels[-1] = self.separator_id

            n_pad_toks = padded_numel - input_ids.numel()
            if n_pad_toks > 0:
                input_ids_padding = torch.full(
                    (n_pad_toks,),
                    self.pad_id,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels_padding = torch.full(
                    (n_pad_toks,),
                    self.separator_id,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                input_ids = torch.cat([input_ids, input_ids_padding])
                labels = torch.cat([labels, labels_padding])

            # Chunk up and divide among ranks
            ret["input_ids"].append(input_ids)
            ret["labels"].append(labels)

        # Stack and add a batch dimension
        ret["input_ids"] = torch.stack(ret["input_ids"], dim=0)
        ret["labels"] = torch.stack(ret["labels"], dim=0)
        return ret

    def _collate_with_naive_padding_free(self, features) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        for item in features:
            input_ids = item["input_ids"]
            labels = item["labels"]
            # Shift and mask the final token
            labels = labels.roll(-1)
            labels[-1] = self.separator_id
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Concatenate, pad, and chunk
        n_toks = sum(i.numel() for i in input_ids_list)
        padded_numel = _round_up_to_zig_zag_padding(n_toks, self.cp_degree)
        n_pad_toks = padded_numel - n_toks
        if n_pad_toks > 0:
            input_ids_padding = torch.full(
                (n_pad_toks,),
                self.pad_id,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels_padding = torch.full(
                (n_pad_toks,),
                self.separator_id,
                device=labels.device,
                dtype=labels.dtype,
            )
            input_ids_list.append(input_ids_padding)
            labels_list.append(labels_padding)

        # Concatenate, add a batch dimension, and chunk:
        input_ids = torch.cat(input_ids_list, dim=-1)[None]

        labels = torch.cat(labels_list, dim=-1)[None]

        return {"input_ids": input_ids, "labels": labels}


def get_infinite_iter(dataloader: DataLoader):
    """
    Infinite iterator, skipping over the None cases above.
    """
    epoch_idx = 0
    sampler = dataloader.sampler
    should_set_epoch = isinstance(sampler, DistributedSampler)
    while True:
        if should_set_epoch:
            sampler.set_epoch(epoch_idx)
        for item in iter(dataloader):
            if item is not None:
                yield epoch_idx, item
        epoch_idx += 1


@dataclass
class DatasetStats:
    epoch_idx: torch.LongTensor = field(default_factory=list)
    examples_seen: torch.LongTensor = field(default_factory=list)
    tokens_seen: torch.LongTensor = field(default_factory=list)
    pred_tokens_seen: torch.LongTensor = field(default_factory=list)
    dataset_lens: torch.LongTensor = field(default_factory=list)


class InfiniteCPBatchingIter:
    """
    Infinite data iterator.

    If max_out_tokens=True, this class greedily packs full examples from `dataloader_list`, grouping
    up to max_tokens = batch_size * seq_len tokens in a batch. Examples are drawn per-dataset
    according to `weights`examples are split for context-parallel training. If
    `naive_padding_free=True`, the examples are all concatenated together, otherwise they are
    batched and padded. The iterator returns a tuple of:
    0) A DatatsetStats instance
    1) The number of examples packed in the batch
    2) The cp-processed batch, a dict[str, Tensor] with `input_ids`, and `labels` keys in HF style.
       This iterator also performs the causal shifting of the labels.
    """

    def __init__(
        self,
        dataloader_list: list[DataLoader],
        weights: list[float],
        batch_size: int,
        seq_len: int,
        max_out_tokens: bool,
        cp_degree: int,
        cp_rank: int,
        dataset_lens: list[int],
        pad_id: int = 0,
        separator_id: int = -100,
        seed: int = 42,
        naive_padding_free: bool = False,
        weight_by: Literal["example", "token", "pred_token", "epoch"] = "example",
    ) -> None:
        if not all(w > 0 for w in weights):
            raise ValueError(f"{weights=} must all be strictly positive")
        if weight_by not in ["example", "token", "pred_token", "epoch"]:
            raise ValueError(
                f"{weight_by=} must be one of 'example', 'token', 'pred_token', or 'epoch'"
            )
        self.dataloader_list = dataloader_list
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.dataset_lens_t = torch.tensor(dataset_lens, dtype=torch.float32)
        # Normalize:
        self.weights /= self.weights.sum()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_out_tokens = max_out_tokens
        self.max_tokens = batch_size * seq_len
        self.cp_degree = cp_degree
        self.cp_rank = cp_rank
        self.pad_id = pad_id
        self.separator_id = separator_id
        self.seed = seed
        self.naive_padding_free = naive_padding_free
        self.weight_by = weight_by

        self._stats = DatasetStats(
            epoch_idx=torch.zeros(len(dataloader_list), dtype=torch.int64),
            examples_seen=torch.zeros(len(dataloader_list), dtype=torch.int64),
            tokens_seen=torch.zeros(len(dataloader_list), dtype=torch.int64),
            pred_tokens_seen=torch.zeros(len(dataloader_list), dtype=torch.int64),
            dataset_lens=torch.tensor(dataset_lens, dtype=torch.int64),
        )
        self._infinite_iters = [get_infinite_iter(dl) for dl in self.dataloader_list]
        self._batch = []

        self._cp_collator = CPDataCollator(
            cp_degree=cp_degree,
            pad_id=pad_id,
            separator_id=separator_id,
            naive_padding_free=naive_padding_free,
        )

    def __iter__(self) -> Iterator[tuple[DatasetStats, int, dict[str, torch.Tensor]]]:
        return self

    def __next__(self) -> tuple[DatasetStats, int, dict[str, torch.Tensor]]:
        while True:
            # Select a dataloader per the given weights.
            if self.weight_by == "example":
                iter_idx = torch.multinomial(self.weights, 1).item()
            elif self.weight_by == "token":
                # Choose the most under-represented dataset by total token.
                expected_tokens = self._stats.tokens_seen.sum() * self.weights
                diff_tokens = self._stats.tokens_seen - expected_tokens
                iter_idx = diff_tokens.argmin().item()
            elif self.weight_by == "pred_token":
                # Choose the most under-represented dataset by total pred token.
                expected_pred_tokens = self._stats.pred_tokens_seen.sum() * self.weights
                diff_pred_tokens = self._stats.pred_tokens_seen - expected_pred_tokens
                iter_idx = diff_pred_tokens.argmin().item()
            elif self.weight_by == "epoch":
                # Choose the most under-represented dataset by epochs seen.
                epochs_seen = self._stats.examples_seen / self.dataset_lens_t
                # Normalize and compare to weights
                epochs_seen_normalized = epochs_seen / epochs_seen.sum()
                iter_idx = (epochs_seen_normalized - self.weights).argmin().item()
            else:
                raise ValueError(f"Unexpected {self.weight_by=} value")
            rand_iter = self._infinite_iters[iter_idx]
            epoch_idx, item = next(rand_iter)
            assert isinstance(item, list), f"{item=}"
            assert len(item) == 1, (
                f"Expected batch size 1 inputs, received {len(item)=}"
            )
            n_tok_next_item = item[0]["input_ids"].numel()
            n_pred_tok_next_item = (item[0]["labels"] != self.separator_id).sum()
            if n_tok_next_item > self.seq_len:
                if not self.cp_rank:
                    logger.warning(
                        f"Skipping data example with {n_tok_next_item} tokens > {self.seq_len=} "
                    )
                continue
            self._stats.epoch_idx[iter_idx] = epoch_idx
            self._stats.examples_seen[iter_idx] += 1
            self._stats.tokens_seen[iter_idx] += n_tok_next_item
            self._stats.pred_tokens_seen[iter_idx] += n_pred_tok_next_item
            if not self._should_yield_batch(n_tok_next_item):
                self._batch.extend(item)
            else:
                self.cp_processed_batch = self._cp_collator(self._batch)
                batch_size = len(self._batch)
                self._batch.clear()
                self._batch.extend(item)
                return self._stats, batch_size, self.cp_processed_batch

    def _should_yield_batch(self, n_tok_next_item: int) -> bool:
        if not self._batch:
            return False

        if not self.max_out_tokens:
            return len(self._batch) == self.batch_size

        if self.naive_padding_free:
            current_tok_in_batch = sum(b["input_ids"].numel() for b in self._batch)
            tok_in_batch_with_new_input = _round_up_to_zig_zag_padding(
                current_tok_in_batch + n_tok_next_item, self.cp_degree
            )
        else:
            current_max_tok_example = max(
                _round_up_to_zig_zag_padding(ex["input_ids"].numel(), self.cp_degree)
                for ex in self._batch
            )
            tok_in_new_input = _round_up_to_zig_zag_padding(
                n_tok_next_item, self.cp_degree
            )
            tok_in_batch_with_new_input = (len(self._batch) + 1) * max(
                current_max_tok_example, tok_in_new_input
            )
        return tok_in_batch_with_new_input > self.max_tokens

    # TODO: @goon - proper state handling.
    def load_state_dict(self, state_dict):
        return

    def state_dict(self):
        return {}


class PretokenizedCollator:
    """
    For handling pre-tokenized data
    """

    def __call__(self, examples: list[dict[str, list[int] | int]]):
        out = []
        for ex in examples:
            item = {}
            for k, v in ex.items():
                if torch.torch.is_tensor(v):
                    item[k] = v
                elif isinstance(v, list):
                    item[k] = torch.tensor(v)
            out.append(item)

        return out


def build_sft_data_loader(
    datasets: str,
    dataset_weights: str,
    dp_rank: int,
    dp_degree: int,
    cp_rank: int,
    cp_degree: int,
    batch_size: int,
    seq_len: int,
    naive_padding_free: bool,
    max_out_tokens: bool,
    seed: int = 42,
    num_data_workers: int = 1,
    weight_by: str = "epoch",
    pad_id: int = 0,
    separator_id: int = -100,
    pin_memory=True,
):
    dataset_paths = datasets.split(",")
    dataset_weights = [float(w) for w in dataset_weights.split(",")]
    datasets = [
        Dataset.load_from_disk(path, keep_in_memory=False) for path in dataset_paths
    ]
    dataset_lens = [len(d) for d in datasets]

    samplers = [
        DistributedSampler(
            d,
            num_replicas=dp_degree,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        for d in datasets
    ]
    train_loader_list = [
        DataLoader(
            dset,
            sampler=sampler,
            collate_fn=PretokenizedCollator(),
            # NOTE: @goon -  batch size is intentionally one. InfiniteCPBatchingIter handles forming
            # batches of the appropriate size.
            batch_size=1,
            num_workers=num_data_workers,
            pin_memory=pin_memory,
        )
        for dset, sampler in zip(datasets, samplers)
    ]
    if naive_padding_free and max_out_tokens and batch_size != 1:
        raise ValueError(
            "Use batch_size=1 when naive_padding_free and max_out_tokens are True. "
            "With these fields True, the dataloader will pull data until approximately seq_len "
            "toks in each sequence, flattened and concatenated into a batch size of 1."
        )

    train_loader = InfiniteCPBatchingIter(
        train_loader_list,
        weights=dataset_weights,
        batch_size=batch_size,
        seq_len=seq_len,
        max_out_tokens=max_out_tokens,
        cp_rank=cp_rank,
        cp_degree=cp_degree,
        dataset_lens=dataset_lens,
        pad_id=pad_id,
        separator_id=separator_id,
        seed=seed,
        naive_padding_free=naive_padding_free,
        weight_by=weight_by,
    )
    return train_loader
