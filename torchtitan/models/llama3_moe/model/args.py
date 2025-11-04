# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import asdict, dataclass, field

from torch import nn

from torchtitan.models.llama3_moe.custom_args import Llama3MoEJobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Llama3MoEModelArgs(BaseModelArgs):
    dim: int = 4096
    moe_inter_dim: int = 14336
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    custom_moe_impl: str | None = None

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
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
    rope_impl: str = "llama"  # llama or dsv3

    def update_from_config(self, job_config: Llama3MoEJobConfig, **kwargs) -> None:
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

        # NOTE: @goon - processing overrides
        for k, v in asdict(job_config.model_overrides).items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        for k, v in asdict(job_config.moe_overrides).items():
            if v is not None and hasattr(self.moe_args, k):
                setattr(self.moe_args, k, v)
        # Special arg handling:
        if (n_moe_layers := job_config.model_overrides.n_moe_layers) is not None:
            if n_moe_layers > self.n_layers - 1:
                raise ValueError(
                    f"Must have {n_moe_layers=} less than or equal to {self.n_layers-1=}"
                    "n_moe_layers inserts MoE layers starting from the second to last layer"
                    " following the advice of https://arxiv.org/pdf/2403.17887 sec 4.4"
                )
            self.is_moe_list = (
                (self.n_layers - n_moe_layers - 1) * [False]
                + n_moe_layers * [True]
                + [False]
            )

        if self.is_moe_list is not None and len(self.is_moe_list) != self.n_layers:
            raise ValueError(
                f"{self.is_moe_list=} must be None or have {self.n_layers=} elements."
            )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(self, model, seq_len)
