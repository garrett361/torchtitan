# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class RoPEScalingArgs:
    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class GDNLayerArgs:
    hidden_size: int = 2048
    expand_v: float = 2
    head_dim: int = 256
    num_heads: int = 6
    num_v_heads: int = None
    mode: str = "chunk"
    use_gate: bool = True
    use_short_conv: bool = False
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    layer_idx: int = None
    norm_eps: float = 1e-5


@dataclass
class Llama3GDNModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    rope_scaling_args: RoPEScalingArgs = field(default_factory=RoPEScalingArgs)
    gdn_layer_args: GDNLayerArgs = field(default_factory=GDNLayerArgs)

    max_seq_len: int = 131072
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    attn_freq: int = 1

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )
        self.attn_freq = job_config.gdn_args.attn_freq

        self.gdn_layer_args.hidden_size = self.dim

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_dense_model_nparams_and_flops(self, model, seq_len)


@dataclass
class GDNArgs:
    attn_freq: int = 1


@dataclass
class Llama3GDNJobConfig(JobConfig):
    gdn_args: GDNArgs = field(default_factory=GDNArgs)
