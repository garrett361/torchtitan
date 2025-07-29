# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from mamba_ssm.modules.mamba2 import Mamba2
from torch import nn

from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.models.deepseek_v3.model.model import (
    apply_rotary_emb,
    precompute_freqs_cis,
)
from torchtitan.models.hybrid_moe.model.args import HybridMoEModelArgs
from torchtitan.models.hybrid_moe.model.moe import FeedForward, MoE
from torchtitan.protocols.train_spec import ModelProtocol


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, model_args: HybridMoEModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (Optional[torch.Tensor]): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        if freqs_cis is not None:
            q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        if freqs_cis is not None:
            k_pe = apply_rotary_emb(
                k_pe.unsqueeze(2), freqs_cis
            )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(
            self.kv_norm(kv)
        )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        output = self.sdpa(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(1, 2)  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        return self.wo(output)  # (bsz, seqlen, dim)

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class LinearAttention(Mamba2):
    """
    Wrapper around Mamba2.
    """

    def __init__(self, model_args: HybridMoEModelArgs):
        super().__init__(
            d_model=model_args.dim,
            d_state=model_args.d_state,
            d_conv=model_args.d_conv,
            conv_init=model_args.conv_init,
            expand=model_args.expand,
            headdim=model_args.headdim,
            d_ssm=model_args.d_ssm,
            ngroups=model_args.ngroups,
            A_init_range=model_args.A_init_range,
            D_has_hdim=model_args.D_has_hdim,
            rmsnorm=model_args.rmsnorm,
            norm_before_gate=model_args.norm_before_gate,
            dt_min=model_args.dt_min,
            dt_max=model_args.dt_max,
            dt_init_floor=model_args.dt_init_floor,
            dt_limit=model_args.dt_limit,
            bias=model_args.bias,
            conv_bias=model_args.conv_bias,
            # Fused kernel and sharding options
            chunk_size=model_args.chunk_size,
            use_mem_eff_path=model_args.use_mem_eff_path,
        )

    def init_weights(self, init_std: float):
        return
        # TODO: @goon -
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: HybridMoEModelArgs):
        super().__init__()
        self.attention = (
            Attention(model_args)
            if layer_id in model_args.mha_layer_idxs
            else LinearAttention(model_args)
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            self.moe = MoE(model_args)
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (Optional[torch.Tensor]): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class HybridMoEModel(nn.Module, ModelProtocol):
    """
    Hybrid MoE Transformer model with mamba, attention and feed-forward layers.
    """

    def __init__(self, model_args: HybridMoEModelArgs):
        super().__init__()
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=True
        )
        # TODO: @goon - set freqs_cis to None when using NoPE. Must be accompanied by a change in
        # init_weights, replacing the call to self.freq_cis.device.

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        if not self.model_args.nope:
            with torch.device(buffer_device):
                self.freqs_cis = precompute_freqs_cis(self.model_args)
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

    def forward(
        self,
        tokens: torch.Tensor,
        eos_id: int | None = None,
        input_batch: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer model.

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
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        if self.model_args.use_flex_attn:
            init_attention_mask(
                input_batch if input_batch is not None else tokens, eos_id=eos_id
            )

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer_idx, layer in self.layers.items():
            if int(layer_idx) in self.model_args.mha_layer_idxs:
                h = layer(h, self.freqs_cis)
            else:
                h = layer(h)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
