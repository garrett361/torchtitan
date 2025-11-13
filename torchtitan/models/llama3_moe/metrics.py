# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3_moe.custom_args import Llama3MoEJobConfig

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


class Hook:
    """
    Class for forward hooks
    """

    def __init__(
        self, module: nn.Module, fqn: str, parallel_dims: ParallelDims
    ) -> None:
        self.module = module
        self.module.register_forward_hook(self)
        self.fqn = fqn.replace("_checkpoint_wrapped_module.moe", "")
        self.parallel_dims = parallel_dims

    def __call__(self, module: nn.Module, args, output) -> None:
        raise NotImplementedError

    def get_stats_dict(self) -> dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class RouterHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inputs_abs_mean = []
        self.inputs_std = []
        self.expert_bias_abs_mean = []
        self.expert_bias_std = []
        self.scores_abs_mean = []
        self.scores_std = []

    @torch.no_grad
    def __call__(self, module: nn.Module, args, output) -> None:
        inputs, expert_bias = args
        scores = output[0]
        self.inputs_abs_mean.append(inputs.detach().abs().mean().item())
        self.inputs_std.append(inputs.detach().std().item())
        self.scores_abs_mean.append(scores.detach().abs().mean().item())
        self.scores_std.append(scores.detach().std().item())
        if expert_bias is not None:
            self.expert_bias_abs_mean.append(expert_bias.detach().abs().mean().item())
            self.expert_bias_std.append(expert_bias.detach().std().item())

    def get_stats_dict(self) -> dict[str, float]:
        stats_dict = {}
        stats_dict[f"moe_router_hook/{self.fqn} inputs abs mean"] = sum(
            self.inputs_abs_mean
        ) / len(self.inputs_abs_mean)
        stats_dict[f"moe_router_hook/{self.fqn} inputs std"] = sum(
            self.inputs_std
        ) / len(self.inputs_std)
        stats_dict[f"moe_router_hook/{self.fqn} scores abs mean"] = sum(
            self.scores_abs_mean
        ) / len(self.scores_abs_mean)
        stats_dict[f"moe_router_hook/{self.fqn} scores std"] = sum(
            self.scores_std
        ) / len(self.scores_std)
        if self.expert_bias_std:
            stats_dict[f"moe_router_hook/{self.fqn} expert_bias abs mean"] = sum(
                self.expert_bias_abs_mean
            ) / len(self.expert_bias_abs_mean)
            stats_dict[f"moe_router_hook/{self.fqn} expert_bias std"] = sum(
                self.expert_bias_std
            ) / len(self.expert_bias_std)

        return stats_dict

    def reset(self) -> None:
        self.inputs_abs_mean.clear()
        self.inputs_std.clear()
        self.expert_bias_abs_mean.clear()
        self.expert_bias_std.clear()
        self.scores_abs_mean.clear()
        self.scores_std.clear()


class CustomMetricsProcessor(MetricsProcessor):
    eps = 1e-10
    # Bad mutable default, but field(default_factory=list) is erroring, maybe b/c of subclassing?
    hooks: list[Hook] = []

    @torch.no_grad
    def get_moe_metrics(self) -> dict[str, Any]:
        moe_metrics = {}
        # Get locally-available MoE stats on relevant ranks
        for model_part in self.model_parts:
            for block_idx, transformer_block in model_part.layers.items():
                if not transformer_block.moe_enabled:
                    continue
                moe_metrics[
                    f"moe_entropy/layer_{block_idx}"
                ] = self.get_normalized_entropy(transformer_block)
                if (
                    n_expert_groups := model_part.model_args.moe_args.n_expert_groups
                ) > 1:
                    moe_metrics[
                        f"moe_group_entropy/layer_{block_idx}"
                    ] = self.get_expert_group_normalized_group_entropy(
                        transformer_block, n_expert_groups
                    )
                # Reset
                transformer_block.moe.tokens_per_expert_cumulative.zero_()
                router_weight = transformer_block.moe.router.gate.weight
                if isinstance(router_weight, DTensor):
                    router_weight = router_weight.full_tensor()
                moe_metrics[f"moe_router/layer_{block_idx} abs mean"] = (
                    router_weight.abs().mean().item()
                )
                moe_metrics[
                    f"moe_router/layer_{block_idx} std"
                ] = router_weight.std().item()

        for hook in self.hooks:
            moe_metrics = {**moe_metrics, **hook.get_stats_dict()}
            hook.reset()

        return moe_metrics

    def get_normalized_entropy(self, transformer_block: nn.Module) -> float:
        tokens_per_expert_cumulative = (
            transformer_block.moe.tokens_per_expert_cumulative + self.eps
        )
        tokens_per_expert_cumulative_prob = (
            tokens_per_expert_cumulative
        ) / tokens_per_expert_cumulative.sum()
        entropy = (
            (
                -tokens_per_expert_cumulative_prob.log()
                * tokens_per_expert_cumulative_prob
            )
            .sum()
            .item()
        )
        max_entropy = math.log(tokens_per_expert_cumulative_prob.numel())
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def get_expert_group_normalized_group_entropy(
        self, transformer_block: nn.Module, n_expert_groups: int
    ) -> float:
        tokens_per_expert_group_cumulative = (
            transformer_block.moe.tokens_per_expert_cumulative.reshape(
                n_expert_groups, -1
            ).sum(dim=-1)
            + self.eps
        )
        tokens_per_expert_group_cumulative_prob = (
            tokens_per_expert_group_cumulative
        ) / tokens_per_expert_group_cumulative.sum()
        entropy = (
            (
                -tokens_per_expert_group_cumulative_prob.log()
                * tokens_per_expert_group_cumulative_prob
            )
            .sum()
            .item()
        )
        max_entropy = math.log(tokens_per_expert_group_cumulative_prob.numel())
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        moe_metrics = self.get_moe_metrics()
        if extra_metrics is None:
            extra_metrics = {**moe_metrics}
        else:
            extra_metrics = {**extra_metrics, **moe_metrics}

        super().log(
            step=step,
            global_avg_loss=global_avg_loss,
            global_max_loss=global_max_loss,
            grad_norm=grad_norm,
            extra_metrics=extra_metrics,
        )


def build_custom_metrics_processor(
    job_config: Llama3MoEJobConfig,
    parallel_dims: ParallelDims,
    model_args: "BaseModelArgs | None" = None,
    tag: str | None = None,
) -> MetricsProcessor:
    """Create a metrics processor.

    Args:
        job_config (Llama3MoEJobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        model_args (BaseModelArgs | None): Model-specific arguments. Defaults to None.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.

    Returns:
        MetricsProcessor: A metrics processor.
    """
    return CustomMetricsProcessor(job_config, parallel_dims, tag)
