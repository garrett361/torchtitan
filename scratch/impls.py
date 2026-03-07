"""Shared implementations for MoE token combining methods."""

from collections.abc import Callable

import torch
from torch import Tensor

DEEPSEEK_CONFIGS = {
    "16B": {"top_k": 6, "dim": 2048},
    "236B": {"top_k": 6, "dim": 5120},
    "671B": {"top_k": 8, "dim": 7168},
}


def bmm(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via batched matrix multiply: (T, 1, K) @ (T, K, D) -> (T, 1, D)."""
    return torch.bmm(top_scores.unsqueeze(1), routed_output).squeeze(1)


def bcast_sum(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via bcast multiply and sum: (T, K, 1) * (T, K, D) -> sum -> (T, D)."""
    return (top_scores.unsqueeze(-1) * routed_output).sum(dim=1)

@torch.compile
def bcast_sum_compiled(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via bcast multiply and sum: (T, K, 1) * (T, K, D) -> sum -> (T, D)."""
    return (top_scores.unsqueeze(-1) * routed_output).sum(dim=1)

def einsum(top_scores: Tensor, routed_output: Tensor) -> Tensor:
    """Combine via einsum: tk,tkd->td."""
    return torch.einsum("tk,tkd->td", top_scores, routed_output)


IMPLS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "bmm": bmm,
    "bcast_sum": bcast_sum,
    "bcast_sum_compiled": bcast_sum_compiled,
    "einsum": einsum,
}
