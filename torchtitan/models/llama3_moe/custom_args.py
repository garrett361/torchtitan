# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.job_config import JobConfig


@dataclass
class Llama3MoECustomArgs:
    """
    Catch-all, misc. cfg.
    """

    hf_weight_transform: str = "replicate"


@dataclass
class TopKSchedulerArgs:
    name: str = "no_op"
    min_top_k: int | None = None
    step_interval: int | None = None
    warmup_steps: int | None = None


# Classes for overriding model architecture configurations, which can't be done via the CLI or toml
# files otherwise. When non-None values are supplied here, a value will be overridden. Some
# convenience fields are also added:
# - n_moe_layers: int, add this many moe layers starting from the final layer.


@dataclass
class ModelOverrides:
    moe_inter_dim: int | None = None
    n_layers: int | None = None
    custom_moe_impl: str | None = None
    n_moe_layers: int | None = None


@dataclass
class MoEOverrides:
    num_experts: int | None = None
    num_shared_experts: int | None = None
    score_func: Literal["softmax", "sigmoid"] | None = None
    route_norm: bool | None = None
    route_scale: float | None = None
    score_before_experts: bool | None = None
    top_k: int | None = None
    use_grouped_mm: bool | None = None
    load_balance_coeff: float | None | None = None
    hf_ffn_hidden_dim: int | None | None = None


@dataclass
class Llama3MoEJobConfig(JobConfig):
    custom_args: Llama3MoECustomArgs = field(default_factory=Llama3MoECustomArgs)
    top_k_args: TopKSchedulerArgs = field(default_factory=TopKSchedulerArgs)
    model_overrides: ModelOverrides = field(default_factory=ModelOverrides)
    moe_overrides: MoEOverrides = field(default_factory=MoEOverrides)
