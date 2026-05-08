"""FA4 custom_op registration for fullgraph torch.compile compatibility.

Registers forward and backward as opaque ops so dynamo never traces into
the CuTe DSL CUDA kernels. Backward is wired via register_autograd.

The mask_mod (a @cute.jit function) closed over at registration time must
not capture per-step values. Only q_offset (fixed per rank) and scalar
constants are permitted in the closure.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


# ─── Plain causal ────────────────────────────────────────────────────────────


@torch.library.custom_op("torchtitan::fa4_causal_fwd", mutates_args=())
def _fa4_causal_fwd(
    q: Tensor, k: Tensor, v: Tensor, softmax_scale: float
) -> tuple[Tensor, Tensor]:
    # return_lse=True: inputs lack requires_grad inside custom_op body, so LSE
    # won't be allocated otherwise — but backward needs it.
    return _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=True, return_lse=True)


@_fa4_causal_fwd.register_fake
def _(q, k, v, softmax_scale):
    lse = torch.empty(
        q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device
    )
    return torch.empty_like(q), lse


@torch.library.custom_op("torchtitan::fa4_causal_bwd", mutates_args=())
def _fa4_causal_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    softmax_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return _flash_attn_bwd(
        q, k, v, out, dout, lse, softmax_scale=softmax_scale, causal=True
    )


@_fa4_causal_bwd.register_fake
def _(q, k, v, out, dout, lse, softmax_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _causal_setup_context(ctx, inputs, output):
    q, k, v, softmax_scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.softmax_scale = softmax_scale


def _causal_backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = torch.ops.torchtitan.fa4_causal_bwd(
        q, k, v, out, dout, lse, ctx.softmax_scale
    )
    return dq, dk, dv, None


_fa4_causal_fwd.register_autograd(
    _causal_backward, setup_context=_causal_setup_context
)


# ─── Masked (registered lazily, once per process) ────────────────────────────


def register_fa4_masked_ops(mask_mod: Callable, variant: str) -> None:
    """Register masked FA4 ops with mask_mod closed over.

    Each variant gets its own op pair (fa4_{variant}_fwd/bwd), so multiple
    mask types can coexist in a single process.

    Idempotent per variant — checks torch.ops namespace to skip if already registered.
    """
    fwd_name = f"fa4_{variant}_fwd"
    bwd_name = f"fa4_{variant}_bwd"
    if hasattr(torch.ops.torchtitan, fwd_name):
        return

    @torch.library.custom_op(f"torchtitan::{fwd_name}", mutates_args=())
    def _fwd(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        softmax_scale: float,
        aux0: Tensor,
        aux1: Tensor | None,
        aux2: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        aux_tensors = [t for t in (aux0, aux1, aux2) if t is not None]
        # return_lse=True: see causal_fwd comment
        return _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            return_lse=True,
        )

    @_fwd.register_fake
    def _(q, k, v, softmax_scale, aux0, aux1, aux2):
        lse = torch.empty(
            q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device
        )
        return torch.empty_like(q), lse

    @torch.library.custom_op(f"torchtitan::{bwd_name}", mutates_args=())
    def _bwd(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        dout: Tensor,
        lse: Tensor,
        softmax_scale: float,
        aux0: Tensor,
        aux1: Tensor | None,
        aux2: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        aux_tensors = [t for t in (aux0, aux1, aux2) if t is not None]
        return _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=softmax_scale,
            causal=False,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
        )

    @_bwd.register_fake
    def _(q, k, v, out, dout, lse, softmax_scale, aux0, aux1, aux2):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _setup_context(ctx, inputs, output):
        q, k, v, softmax_scale, aux0, aux1, aux2 = inputs
        out, lse = output
        aux_to_save = [t for t in (aux0, aux1, aux2) if t is not None]
        ctx.save_for_backward(q, k, v, out, lse, *aux_to_save)
        ctx.softmax_scale = softmax_scale
        ctx.n_aux = len(aux_to_save)

    def _backward(ctx, dout, dlse):
        saved = ctx.saved_tensors
        q, k, v, out, lse = saved[:5]
        aux_saved = saved[5:]
        aux0 = aux_saved[0]
        aux1 = aux_saved[1] if ctx.n_aux > 1 else None
        aux2 = aux_saved[2] if ctx.n_aux > 2 else None
        bwd_op = getattr(torch.ops.torchtitan, bwd_name)
        dq, dk, dv = bwd_op(
            q, k, v, out, dout, lse, ctx.softmax_scale, aux0, aux1, aux2
        )
        return dq, dk, dv, None, None, None, None

    _fwd.register_autograd(_backward, setup_context=_setup_context)
