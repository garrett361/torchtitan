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

from torchtitan.models.attention import init_attention_mask
from torchtitan.models.deepseek_v3.model.model import Attention, precompute_freqs_cis
from torchtitan.models.hybrid_moe.model.args import HybridMoEModelArgs
from torchtitan.models.hybrid_moe.model.moe import FeedForward, MoE
from torchtitan.protocols.train_spec import ModelProtocol


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
        # Some init args aren't saved as attrs but are needed for init, so safe a ref to model_args
        self.model_args = model_args

    def init_weights(self, init_std: float):
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        with torch.no_grad():
            nn.init.uniform_(self.dt_bias.data)
            self.dt_bias.data *= math.log(self.model_args.dt_max) - math.log(
                self.model_args.dt_min
            )
            self.dt_bias.data += math.log(self.model_args.dt_min)
            self.dt_bias.data.exp_()
            self.dt_bias.data.clamp_(min=self.model_args.dt_init_floor)
            self.dt_bias.data = self.dt_bias.data + torch.log(
                -torch.expm1(-self.dt_bias.data)
            )

            self.A_log.data = self.A_log.data.to(torch.float32)
            self.A_log.data.uniform_(*self.model_args.A_init_range)
            # NOTE: @goon -  mamba.ssm has a self.A_log.data.to(dtype) here after the log, but I'm not
            # sure what dtype to use?
            self.A_log.data = torch.log(self.A_log.data)

        nn.init.ones_(self.D)
        if self.rmsnorm:
            self.norm.reset_parameters()

        # NOTE: @goon do we really want this hard-coded init for the in-weights?
        nn.init.trunc_normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: HybridMoEModelArgs):
        super().__init__()
        self.attention = (
            Attention(model_args)
            if (layer_id + 1) % model_args.mha_layer_interval == 0
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
            if (
                self.model_args.mha_layer_interval is not None
                and (int(layer_idx) + 1) % self.model_args.mha_layer_interval == 0
            ):
                h = layer(h, self.freqs_cis)
            else:
                h = layer(h)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
