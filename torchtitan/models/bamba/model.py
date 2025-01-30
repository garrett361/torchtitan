# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.llama.model import (
    Attention,
    FeedForward,
    ModelArgs,
    precompute_freqs_cis,
)
from torchtitan.models.norms import build_norm


@dataclass
class BambaModelArgs(ModelArgs):
    attn_layer_indices: Optional[list[int]] = None
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    n_groups: int = 1
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    chunk_size: int = 256
    use_mamba_kernels: bool = False

    def __post_init__(self) -> None:
        if self.attn_layer_indices is None:
            self.attn_layer_indices = [
                n for n in range(self.n_layers) if not (n + 1) % 8
            ]
        self.d_inner = self.expand * self.dim
        assert self.d_inner == self.expand * self.dim
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.d_in_proj = (
            2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        )


def segsum(x):
    """More stable segment sum calculation."""
    x_shape = x.shape
    T = x_shape[-1]
    x = x[..., None].repeat(*(1 for _ in x_shape), T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def torch_chunk_scan_combined(
    x: torch.Tensor,  # (batch_size, seq_len, n_heads, d_head)
    dt: torch.Tensor,  # (batch_size, seq_len, n_heads)
    A: torch.Tensor,  # (n_heads,)
    B: torch.Tensor,  # (batch_size, seq_len, n_groups, d_state)
    C: torch.Tensor,  # (batch_size, seq_len, n_groups, d_state)
    chunk_size: int,
    D: Optional[torch.Tensor] = None,  # (n_heads,)
):
    """
    Chunked O(chunk_size * seq_len) solution to the mamba2 scan. Modified from mamba_ssm, with
    einops removed.

    Signature mimics mamba_chunk_scan_combined.
    """
    X = x * dt[..., None]
    A = A * dt

    # Chunk seq_len dim
    X, A, B, C = [
        t.reshape(t.shape[0], t.shape[1] // chunk_size, chunk_size, *t.shape[2:])
        for t in (X, A, B, C)
    ]

    A = torch.einsum("bclh->bhcl", A)  # (B, h, c, l)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclgn,bcsgn,bhcls,bcshp->bclhp", C, B, L, X)  # O(L S B C D)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum(
        "bclgn,bhcl,bclhp->bcghpn", B, decay_states, X
    )  # O(B S D G N)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bcghpn->bzghpn", decay_chunk, states)
    states = new_states[:, :-1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclgn,bcghpn,bhcl->bclhp", C, states, state_decay_out)

    Y = Y_diag + Y_off

    # Unchunk
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], *Y.shape[3:])

    # Residual
    if D is not None:
        Y = Y + D[:, None] * x
    return Y


def torch_scan(
    x: torch.Tensor,  # (batch_size, seq_len, n_heads, d_head)
    dt: torch.Tensor,  # (batch_size, seq_len, n_heads)
    A: torch.Tensor,  # (n_heads,)
    B: torch.Tensor,  # (batch_size, seq_len, n_groups, d_state)
    C: torch.Tensor,  # (batch_size, seq_len, n_groups, d_state)
    chunk_size: int,
    D: Optional[torch.Tensor] = None,  # (n_heads,)
):
    """
    Minimal O(seq_len) core pytorch solution to the mamba2 scan. Not as performant as the other
    torch impls under compile.

    Signature mimics mamba_chunk_scan_combined.

    NOTE: Appears too numerically unstable to use.
    """
    X = x * dt[..., None]
    A = A * dt
    A_cs = A.cumsum(dim=1)
    Y = torch.einsum("bsh,bsgn,bshp->bsghpn", (-A_cs).exp(), B, X).cumsum(dim=1)
    Y = torch.einsum("bsgn,bsghpn->bshp", C, Y)
    Y = torch.einsum("bsh,bshp->bshp", A_cs.exp(), Y)

    if D is not None:
        Y = Y + D[:, None] * x

    return Y


def torch_chunk_scan_combined_linear(
    x: torch.Tensor,  # (batch_size, seq_len, n_heads, d_head)
    dt: torch.Tensor,  # (batch_size, seq_len, n_heads)
    A: torch.Tensor,  # (n_heads,)
    B: torch.Tensor,  # (batch_size, seq_len, n_groups, d_state)
    C: torch.Tensor,  # (batch_size, seq_len, n_groups, d_state)
    chunk_size: int,
    D: Optional[torch.Tensor] = None,  # (n_heads,)
):
    """
    Alternative chunked O(chunk_size * seq_len) solution to the mamba2 scan which does not create
    O((seq_len / chunk_size)^2) intermediates.

    Signature mimics mamba_chunk_scan_combined.

    NOTE: Appears too numerically unstable to use.
    """
    X = x * dt[..., None]
    A = A * dt

    # Chunk seq_len dim
    X, A, B, C = [
        t.reshape(t.shape[0], t.shape[1] // chunk_size, chunk_size, *t.shape[2:])
        for t in (X, A, B, C)
    ]

    A = torch.einsum("bclh->bhcl", A)  # (B, h, c, l)
    A_sum = A.sum(dim=-1)  # (b, h, c)
    A_cs = A.cumsum(dim=-1)  # (b, h, c, l)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclgn,bcsgn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    T = (A_sum[..., None] - A_cs).exp()
    right_factor = torch.einsum("bhcl,bclgn,bclhp->bcghpn", T, B, X)

    # 3. Center-factor. (A terms)
    A_sum_cs = A_sum.cumsum(dim=-1)
    center_right = (-A_sum_cs).exp()  # (b, h, c)
    center_right = (
        torch.einsum("bhc->bch", center_right)[:, :, None, :, None, None] * right_factor
    )
    center_right = center_right.cumsum(dim=1) - center_right  # (b, c, g, h, p n)
    center_factor = (
        torch.einsum("bhc->bch", (A_sum_cs - A_sum).exp())[:, :, None, :, None, None]
        * center_right
    )  # (b, c, g, h, p, n)

    # 4. Left-factor (C terms)
    Y_off = torch.einsum("bclgn,bcghpn,bhcl->bclhp", C, center_factor, A_cs.exp())

    Y = Y_diag + Y_off

    # Unchunk
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], *Y.shape[3:])

    # Residual
    if D is not None:
        Y = Y + D[:, None] * x
    return Y


class Mamba2(nn.Module):
    def __init__(self, model_args: BambaModelArgs) -> None:
        super().__init__()
        self.dim = model_args.dim
        self.d_state = model_args.d_state
        self.d_conv = model_args.d_conv
        self.expand = model_args.expand
        self.d_inner = model_args.d_inner
        self.d_in_proj = model_args.d_in_proj
        self.conv_dim = model_args.conv_dim
        self.n_heads = model_args.n_heads
        self.n_groups = model_args.n_groups
        self.dt_min = model_args.dt_min
        self.dt_max = model_args.dt_max
        self.dt_init_floor = model_args.dt_init_floor
        # Fused kernel and sharding options
        self.chunk_size = model_args.chunk_size

        self.in_proj = nn.Linear(self.dim, self.d_in_proj)
        self.out_proj = nn.Linear(self.d_inner, self.dim)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.conv_dim,
            padding=self.d_conv - 1,
        )
        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=self.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(0, 16)
        # A = (
        #     1.0 + torch.randn(self.n_heads, dtype=torch.float32).abs()
        # )  # default init giving me infs
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = build_norm(
            model_args.norm_type, dim=self.d_inner, eps=model_args.norm_eps
        )

        if model_args.use_mamba_kernels:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

            self.scan_impl = mamba_chunk_scan_combined
        else:
            self.scan_impl = torch_chunk_scan_combined

    def forward(self, inputs: torch.Tensor):
        batch, seqlen, _ = inputs.shape
        A = -torch.exp(self.A_log)  # (n_heads) or (d_inner, d_state)
        zxbcdt = self.in_proj(inputs)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.n_groups * self.d_state,
                self.n_heads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, n_heads)
        dt = dt.clamp(0, float("inf"))
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        xBC = xBC[:, :seqlen, :]

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )

        # Scan input and output shapes chosen to mimic mamba_ssm's mamba_chunk_scan_combined

        # Reshape to (batch_size, seqlen, n_heads, d_head)
        x = x.reshape(*x.shape[:-1], self.n_heads, x.shape[-1] // self.n_heads)
        # Reshape to (batch_size, seqlen, n_groups, d_state)
        B = B.reshape(*B.shape[:-1], self.n_groups, B.shape[-1] // self.n_groups)
        C = C.reshape(*C.shape[:-1], self.n_groups, C.shape[-1] // self.n_groups)

        y = self.scan_impl(x, dt, A, B, C, self.chunk_size, self.D)
        # Join heads
        y = y.reshape(*y.shape[:-2], -1)
        out = self.out_proj(self.norm(y * self.act(z)))

        return out

    def init_weights(self, init_std: float):
        nn.init.uniform_(self.conv1d.weight)


class BambaBlock(nn.Module):
    """
    BambaBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (BambaModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: BambaModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.is_attn_layer = layer_id in model_args.attn_layer_indices
        self.attention = (
            Attention(model_args) if self.is_attn_layer else Mamba2(model_args)
        )
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        if self.is_attn_layer:
            h = x + self.attention(self.attention_norm(x), freqs_cis)
        else:
            h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Bamba(nn.Module):
    """
    Transformer Module

    Args:
        model_args (BambaModelArgs): Model configuration arguments.

    Attributes:
        model_args (BambaModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: BambaModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = BambaBlock(layer_id, model_args)

        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: BambaModelArgs) -> "Bamba":
        """
        Initialize a Transformer model from a BambaModelArgs object.

        Args:
            model_args (BambaModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
