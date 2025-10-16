# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig


@dataclass
class CustomArgs:
    load_balance_coeff: float | None = None
    hf_weight_transform: str = "replicate"
    hf_ffn_hidden_dim: int | None = None 


@dataclass
class JobConfig(JobConfig):
    custom_args: CustomArgs = field(default_factory=CustomArgs)
