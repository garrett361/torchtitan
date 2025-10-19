# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig


@dataclass
class Llama3MoECustomArgs:
    load_balance_coeff: float | None = None
    hf_weight_transform: str = "replicate"
    hf_ffn_hidden_dim: int | None = None
    top_k: int | None = None


@dataclass
class TopKSchedulerArgs:
    name: str = "no_op"
    min_top_k: int | None = None
    step_interval: int | None = None
    warmup_steps: int | None = None


@dataclass
class Llama3MoEJobConfig(JobConfig):
    custom_args: Llama3MoECustomArgs = field(default_factory=Llama3MoECustomArgs)
    top_k_args: TopKSchedulerArgs = field(default_factory=TopKSchedulerArgs)
