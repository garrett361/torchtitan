# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


class GraniteTransformerBlock(TransformerBlock):
    """Granite transformer block with residual scaling.

    Identical to Llama3TransformerBlock except each residual stream is scaled
    by ``residual_multiplier`` before addition:
        h = x + residual_multiplier * attn(norm(x))
        out = h + residual_multiplier * ffn(norm(h))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        residual_multiplier: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.residual_multiplier * self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        return h + self.residual_multiplier * self.feed_forward(self.ffn_norm(h))


class GraniteModel(Decoder):
    """Granite 4.1 decoder model.

    Extends Decoder with three per-model scaling multipliers plus per-layer attention scaling:
    - embedding_multiplier: scales token embeddings after lookup
    - residual_multiplier: scales each residual contribution in every block (GraniteTransformerBlock)
    - logits_scaling: scales the final logit projection output
    - attn_scale (1/head_dim): set per-block via GQAttention.Config, not a model-level field

    Weight tying (tok_embeddings.weight == output.weight) is always enabled.
    Pipeline Parallel is not supported due to weight tying.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 100352
        embedding_multiplier: float = 1.0
        logits_scaling: float = 1.0
        enable_weight_tying: bool = True

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )
                if self.enable_weight_tying:
                    raise NotImplementedError(
                        "GraniteModel: Tensor Parallel is not supported with weight tying. "
                        "apply_tp assigns tok_embeddings and output different sharding plans, "
                        "silently breaking the weight identity and producing wrong gradients."
                    )

            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "GraniteModel always uses weight tying, which is not compatible "
                    "with Pipeline Parallel."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def __init__(self, config: Config):
        if not config.enable_weight_tying:
            raise ValueError(
                "GraniteModel requires enable_weight_tying=True: tok_embeddings uses "
                "skip_param_init and relies on output.weight for initialization."
            )
        super().__init__(config)
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling
        self.enable_weight_tying = config.enable_weight_tying
        self.tok_embeddings.weight = self.output.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            # Re-tie before param init so tok_embeddings.weight (skipped by
            # skip_param_init) and output.weight share the same tensor after
            # output is initialized.
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight
        super().init_states(buffer_device=buffer_device)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) * self.embedding_multiplier
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)
        return self.output(self.norm(h)) / self.logits_scaling
