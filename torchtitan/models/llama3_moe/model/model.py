# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import math

import torch
from einops import rearrange
from torch import nn

from torchtitan.models.llama3.model.model import Attention, FeedForward
from torchtitan.models.llama3_moe.model.args import TransformerModelArgs
from torchtitan.models.moe import MoE, MoEArgs
from torchtitan.protocols.train_spec import ModelProtocol


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
# NOTE: @goon - this was taken from DSv3 and was used for our 70B Llama context extension, but
# actual Llama3 uses a different RoPE function that was only properly added to titan after we
# started this work: https://github.com/pytorch/torchtitan/pull/1839
def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    original_seq_len: int,
    rope_theta: float = 10000.0,
    rope_factor: float = 20,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        dim (int): Dimension of the frequency tensor.
        seqlen (int): the seqlen being trained on.
        original_seq_len (int): the original training seqlen, if using YaRN to extend.
        rope_theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        rope_factor (float): YaRN
        beta_fast (int): YaRN
        beta_slow (int): YaRN

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.

    NOTE: DSv3 arg defaults:
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1

    NOTE: @goon -  I omitted mscale which doesn't do anything when left at its default 1.0 value.
    """

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Basic RoPE frequency calculation
    freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
    if seq_len > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, rope_theta, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    # Create position indices
    t = torch.arange(seq_len)

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class _CustomMoE(MoE):
    """
    Customizable MoE class. Primarily for implementing different router weight initializations.
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args=moe_args, dim=dim, hidden_dim=hidden_dim)
        self.moe_args = moe_args
        self.hidden_dim = hidden_dim

    name: str | None = None


class VirtualGroupMoE(_CustomMoE):
    """
    Implements the "Virtual Group Initialization" from 2410.07524.

    [Virtual Group Initialization]
    # Assumptions

    The first FFN weight (say, wlog) W_hd, with h the internal hidden index, is used to initialize
    the first MoE weight M_efd through the following steps:
    1. Replicate the FFN weight R times to form M_rhd = W_hd (component-wise), with r in {0, ...,
       R-1} and similar for other indices.
    2. Split the hidden dimension (size H) into G sequential chunks (groups) of size F = H / G.
    3. Reshape the replication and group dimensions into a single expert-index dimension of size
       E = R * G. Full
       progression:
       W_hd --> M_rhd --> M_rgfd --> M_efd

    Nice divisibility assumed everywhere. The above MoE weight-creation strategy is implemented by
    ReplicateMoETransform.

    # Requirements for FFN-MoE equivalence

    The original FFN output is (using a simple FFN; swiGLU doesn't change the argument)

    o_d = sum_hd'(W^(2)_dh phi(W_hd' x_d')) === sum_h(z_hd)

    Letting p_e the router weight for expert e with sum_e(p_e) = 1 and route_score an overall router
    scaling, the MoE output is:

    o^(moe)_d = route_score * sum_efd'(p_e M^(2)_edf phi(M_efd' x_d')
              === route_score * sum_ef(p_e z^(moe)_efd)

    The assumptions above let us reshape z^(moe)_efd --> z^(moe)_rgfd and p_e --> p_rg, leading to:

    o^(moe)_d = route_score * sum_rgf(p_rg z^(moe)_rgfd)

    Our assumptions also allow us to reshape z^(moe)_rgfd --> z^(moe)_rhd where we further have
    z^(moe)_rhd = z_hd, component-wise. If p_rg were such that its component values are independent
    of g, say p_rg = q_r, i.e. each FFN weight slice within a group gets the same routing weight.
    Then the above becomes:

    o^(moe)_d = route_score * sum_rgf(q_r z^(moe)_rgfd)
              = route_score * sum_rh(q_r z^(moe)_rhd)
              = route_score * sum_rh(q_r z_hd)
              = route_score * sum_r(q_r) * o_d

    Therefore we can get exact FFN-MoE equivalence if the expert weights meet the two requirements:
    1. The router weights p_e = p_rg is constant across the r-index
    2. The following normalization condition holds: route_score * sum_r(p_rg) = 1, which uniquely
       fixes route_score = G, due to the normalization sum_e(p_e) = 1 and p_rg's independence of g.

    In the paradigm where the router weights are determined from a top_k-then-softmax approach, as
    in

    p_e = soft_e(top_k(sum_d(P_ed x_d))) ,

    the above can be achieved by initializing the weight P_ed = P_rgd so that it is independent of
    g, i.e. P_rgd = Q_rd for some Q_rd, and also taking the k in top_k to be a multiple of G so that
    entire groups are activated together. The same conclusion also holds for softmax-then-top_k.
    """

    name = "virtual_group"

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args=moe_args, dim=dim, hidden_dim=hidden_dim)
        # Replicated initialization only makes sense if the MoE hidden dim cleanly divides the HF
        # FFN hidden dim
        if self.moe_args.hf_ffn_hidden_dim is None:
            raise ValueError(
                f"{self.__class__.__name__} requires hf_ffn_hidden_dim to be specified"
            )
        self.n_groups, remainder = divmod(
            self.moe_args.hf_ffn_hidden_dim,
            self.hidden_dim,
        )
        if remainder:
            raise ValueError(
                f"{self.hidden_dim=} must be divisible by {self.moe_args.hf_ffn_hidden_dim=}"
            )
        self.n_replicas, remainder = divmod(self.moe_args.num_experts, self.n_groups)
        if remainder:
            raise ValueError(
                f"{self.moe_args.num_experts=} must be divisible by {self.hidden_dim // self.moe_args.hf_ffn_hidden_dim=}"
            )
        if self.moe_args.route_scale != self.n_groups:
            raise ValueError(
                f"{self.moe_args.route_scale=} must be divisible by {self.moe_args.hf_ffn_hidden_dim // self.hidden_dim =}"
            )
        if not self.moe_args.route_norm:
            raise ValueError(f"{self.moe_args.route_norm=} must be True")

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        super().init_weights(init_std=init_std, buffer_device=buffer_device)
        if self.n_groups > 1:
            with torch.no_grad():
                # Weight shape: (num_experts, dim) = (n_replicas * n_groups, dim)
                router_weights = self.router.gate.weight
                router_weights_slice = router_weights[: self.n_replicas]
                replicated_router_weights = torch.cat(
                    [router_weights_slice for _ in range(self.n_groups)], dim=0
                )
                replicated_router_weights = rearrange(
                    replicated_router_weights,
                    "(g r) d -> (r g) d",
                    r=self.n_replicas,
                    g=self.n_groups,
                )
                self.router.gate.weight.copy_(replicated_router_weights)

                # For completeness, though the expert bias should start as zeros, if it exists.
                if (expert_bias := self.expert_bias) is not None:
                    expert_bias_slice = expert_bias[: self.n_replicas]
                    replicated_expert_bias = torch.cat(
                        [expert_bias_slice for _ in range(self.n_groups)], dim=0
                    )
                    replicated_expert_bias = rearrange(
                        replicated_expert_bias,
                        "(g r) -> (r g)",
                        r=self.n_replicas,
                        g=self.n_groups,
                    )
                    self.expert_bias.copy_(replicated_expert_bias)


def get_moe_impl_cls(name: str | None = None) -> type[MoE]:
    if name is None:
        return MoE
    moe_map = {sc.name: sc for sc in _CustomMoE.__subclasses__() if hasattr(sc, "name")}
    return moe_map[name]


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: TransformerModelArgs, is_moe: bool):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        if is_moe:
            self.feed_forward = None
            self.moe = get_moe_impl_cls(model_args.custom_moe_impl)(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
            self.moe_enabled = True

        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=4 * model_args.dim,
                multiple_of=model_args.multiple_of,
                ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            )
            self.moe = None
            self.moe_enabled = False

        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        h = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            out = h + self.moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, buffer_device: torch.device | None = None):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (Linear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        # NOTE: @goon - defaulting to no MoE layers at all for now for testing purposes
        is_moe_list = model_args.is_moe_list or [
            False for _ in range(model_args.n_layers)
        ]
        for layer_id, is_moe in enumerate(is_moe_list):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args, is_moe)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
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
                layer.init_weights(buffer_device=buffer_device)
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
            dim=self.model_args.dim // self.model_args.n_heads,
            seq_len=self.model_args.max_seq_len,
            original_seq_len=self.model_args.original_seq_len,
            rope_theta=self.model_args.rope_theta,
            rope_factor=self.model_args.rope_factor,
            beta_fast=self.model_args.beta_fast,
            beta_slow=self.model_args.beta_slow,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """

        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
