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
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    moe_inter_dim: int = 14336
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    # TODO: node-limited routing is not supported yet
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    is_moe_list: list[bool] | None = None

    max_seq_len: int = 131072
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0

    # yarn: https://arxiv.org/pdf/2309.00071
    original_seq_len: int = 8192
    rope_factor: float = 20  # s in 2309.00071 (I believe); see eq (25)
    beta_fast: int = 32  # \alpha in 2309.00071; see around eq (23)
    beta_slow: int = 1  # \beta in 2309.00071; see around eq (23)

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

        # NOTE: @goon - custom args we've added are processed here
        if job_config.custom_args.load_balance_coeff is not None:
            self.moe_args.load_balance_coeff = job_config.custom_args.load_balance_coeff

        if self.is_moe_list is not None and len(is_moe_list) != self.n_layers:
            raise ValueError(
                f"{self.is_moe_list=} must be None or have {self.n_layers=} elements."
            )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_dense_model_nparams_and_flops(self, model, seq_len)
