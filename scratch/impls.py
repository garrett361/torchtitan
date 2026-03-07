"""Shared implementations for MoE token combining methods."""

from collections.abc import Callable

import torch
from torch import Tensor

DEEPSEEK_CONFIGS = {
    "16B": {"top_k": 6, "dim": 2048},
    "236B": {"top_k": 6, "dim": 5120},
    "671B": {"top_k": 8, "dim": 7168},
}


def combine_bmm(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via batched matrix multiply: (T, 1, K) @ (T, K, D) -> (T, 1, D)."""
    return torch.bmm(top_scores.unsqueeze(1), routed_output).squeeze(1)


def combine_broadcast_sum(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via broadcast multiply and sum: (T, K, 1) * (T, K, D) -> sum -> (T, D)."""
    return (top_scores.unsqueeze(-1) * routed_output).sum(dim=1)


def combine_einsum(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via einsum: tk,tkd->td."""
    return torch.einsum("tk,tkd->td", top_scores, routed_output)


IMPLS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "bmm": combine_bmm,
    "broadcast_sum": combine_broadcast_sum,
    "einsum": combine_einsum,
}
