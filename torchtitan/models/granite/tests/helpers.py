"""Shared test utilities for the granite test suite."""

import torch
import torch.nn.functional as F

from torchtitan.components.loss import IGNORE_INDEX


def has_fa4() -> bool:
    """Return True if flash_attn.cute (FA4) is importable."""
    try:
        import cutlass.cute  # noqa: F401
        from flash_attn.cute import flash_attn_func  # noqa: F401

        return True
    except ImportError:
        return False


def ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SDPA reference attention with explicit bool mask. Handles GQA."""
    # q: (B, S_q, H, D), k/v: (B, S_kv, HKV, D), mask: (S_q, S_kv) bool
    # SDPA expects (B, H, S, D)
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    batch = q.shape[0]
    attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1)
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=attn_mask, scale=scale, enable_gqa=True,
    )
    return out.transpose(1, 2).to(dtype)


def find_unmasked_regions(
    labels: list[int], ignore_index: int = IGNORE_INDEX,
) -> list[tuple[int, int]]:
    """Return (start, end) spans where labels != ignore_index."""
    regions: list[tuple[int, int]] = []
    start = None
    for i, lbl in enumerate(labels):
        if lbl != ignore_index and start is None:
            start = i
        elif lbl == ignore_index and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(labels)))
    return regions
