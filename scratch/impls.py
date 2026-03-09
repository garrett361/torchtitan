"""Shared implementations for MoE token combining methods."""

from collections.abc import Callable

import torch
from torch import Tensor

MODEL_CFGS = {
    # DeepSeek V3
    "dsv3_16B": {"top_k": 6, "dim": 2048},
    "dsv3_236B": {"top_k": 6, "dim": 5120},
    "dsv3_671B": {"top_k": 8, "dim": 7168},
    # Llama4
    "llama4_17bx16e": {"top_k": 1, "dim": 5120},
    "llama4_17bx128e": {"top_k": 1, "dim": 5120},
    # Qwen3 MoE
    "qwen3_30B_A3B": {"top_k": 8, "dim": 2048},
    "qwen3_235B_A22B": {"top_k": 8, "dim": 4096},
    # GPT-OSS
    "gpt_oss_20b": {"top_k": 4, "dim": 2880},
    "gpt_oss_120b": {"top_k": 4, "dim": 2880},
}


def bmm(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via batched matrix multiply: (T, 1, K) @ (T, K, D) -> (T, 1, D)."""
    return torch.bmm(top_scores.unsqueeze(1), routed_output.float()).squeeze(1)


def bcast_sum(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via bcast multiply and sum: (T, K, 1) * (T, K, D) -> sum -> (T, D)."""
    return (top_scores.unsqueeze(-1) * routed_output.float()).sum(dim=1)


@torch.compile
def bcast_sum_compiled(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via bcast multiply and sum: (T, K, 1) * (T, K, D) -> sum -> (T, D)."""
    return (top_scores.unsqueeze(-1) * routed_output.float()).sum(dim=1)


def einsum(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via einsum: tk,tkd->td."""
    return torch.einsum("tk,tkd->td", top_scores, routed_output.float())


IMPLS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "bmm": bmm,
    "bcast_sum": bcast_sum,
    "bcast_sum_compiled": bcast_sum_compiled,
    "einsum": einsum,
}
