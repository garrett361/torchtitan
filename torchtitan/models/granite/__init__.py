# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraniteModel, GraniteTransformerBlock
from .parallelize import parallelize_granite
from .state_dict_adapter import GraniteStateDictAdapter

__all__ = [
    "parallelize_granite",
    "GraniteModel",
    "granite_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
# Weight-tied tok_embeddings must use skip_param_init: output.weight is
# initialized in init_states and tok_embeddings.weight is re-tied to it.
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_EMBEDDING_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_granite_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    residual_multiplier: float,
    n_kv_heads: int | None = None,
    attn_backend: str = "sdpa",
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer GraniteTransformerBlock configs."""
    inner_attention, mask_type = get_attention_config(attn_backend)
    head_dim = dim // n_heads
    # Granite uses 1/head_dim instead of 1/sqrt(head_dim) for QK scaling.
    attn_scale = 1.0 / head_dim
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            GraniteTransformerBlock.Config(
                residual_multiplier=residual_multiplier,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    attn_scale=attn_scale,
                    mask_type=mask_type,
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _make_untied(
    tied_builder: Callable[[str], GraniteModel.Config],
) -> Callable[[str], GraniteModel.Config]:
    """Wrap a tied-model config builder to produce an untied variant."""

    def builder(attn_backend: str = "sdpa") -> GraniteModel.Config:
        config = tied_builder(attn_backend=attn_backend)
        return dataclasses.replace(
            config,
            enable_weight_tying=False,
            layers=list(config.layers),
            tok_embeddings=Embedding.Config(
                num_embeddings=config.vocab_size,
                embedding_dim=config.dim,
                param_init=_EMBEDDING_INIT,
            ),
        )

    return builder


def _debugmodel(attn_backend: str = "sdpa") -> GraniteModel.Config:
    dim = 256
    n_heads = 16
    n_layers = 4
    vocab_size = 2048
    return GraniteModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="none",
        ),
        layers=_build_granite_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=512,
            residual_multiplier=0.22,
            attn_backend=attn_backend,
        ),
    )


def _debugmodel_fa4(attn_backend: str = "fa4") -> GraniteModel.Config:
    """Debug model with head_dim=64 (FA4 backward hits an ICE — internal compiler
    error in the CuTe DSL kernel compiler — on head_dim<64)."""
    dim = 256
    n_heads = 4
    n_kv_heads = 2
    n_layers = 4
    vocab_size = 2048
    return GraniteModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="none",
        ),
        layers=_build_granite_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=512,
            residual_multiplier=0.22,
            attn_backend=attn_backend,
        ),
    )


def _3b(attn_backend: str = "sdpa") -> GraniteModel.Config:
    dim = 2560
    n_heads = 40
    n_kv_heads = 8
    n_layers = 40
    vocab_size = 100352
    return GraniteModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        embedding_multiplier=12.0,
        logits_scaling=10.0,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=10_000_000,
            backend="complex",
            scaling="none",
        ),
        layers=_build_granite_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=8192,
            residual_multiplier=0.22,
            attn_backend=attn_backend,
        ),
    )


def _8b(attn_backend: str = "sdpa") -> GraniteModel.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 40
    vocab_size = 100352
    return GraniteModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=10_000_000,
            backend="complex",
            scaling="none",
        ),
        layers=_build_granite_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=12800,
            residual_multiplier=0.22,
            attn_backend=attn_backend,
        ),
    )


def _30b(attn_backend: str = "sdpa") -> GraniteModel.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 64
    vocab_size = 100352
    return GraniteModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        embedding_multiplier=12.0,
        logits_scaling=16.0,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=50_000_000,
            backend="complex",
            scaling="none",
        ),
        layers=_build_granite_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=32768,
            residual_multiplier=0.175,
            attn_backend=attn_backend,
        ),
    )


granite_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_fa4": _debugmodel_fa4,
    "3B": _3b,
    "8B": _8b,
    "30B": _30b,
    "debugmodel_untied": _make_untied(_debugmodel),
    "3B_untied": _make_untied(_3b),
    "8B_untied": _make_untied(_8b),
    "30B_untied": _make_untied(_30b),
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    config = granite_configs[flavor](attn_backend=attn_backend)
    return ModelSpec(
        name="granite",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_granite,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=GraniteStateDictAdapter,
    )
