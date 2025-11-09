# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3_moe.custom_args import Llama3MoEJobConfig

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


class CustomMetricsProcessor(MetricsProcessor):
    eps = 1e-10

    @torch.no_grad
    def get_moe_metrics(self) -> dict[str, Any]:
        moe_metrics = {}
        # Get locally-available MoE stats on relevant ranks
        for model_part in self.model_parts:
            for block_idx, transformer_block in model_part.layers.items():
                if not transformer_block.moe_enabled:
                    continue
                moe_metrics[f"moe/layer_{block_idx} moe normalized entropy"] = (
                    self.get_normalized_entropy(transformer_block)
                )
                if model_part.model_args.moe_args.n_expert_groups > 1:
                    moe_metrics[
                        f"moe/layer_{block_idx} moe expert_group normalized entropy"
                    ] = self.get_expert_group_normalized_group_entropy(
                        transformer_block
                    )
                # Reset
                transformer_block.moe.tokens_per_expert_cumulative.zero_()
                router_weight = transformer_block.moe.router.gate.weight
                if isinstance(router_weight, DTensor):
                    router_weight = router_weight.full_tensor()
                moe_metrics[f"moe/layer_{block_idx} router abs mean"] = (
                    router_weight.abs().mean().item()
                )
                moe_metrics[f"moe/layer_{block_idx} router std"] = (
                    router_weight.std().item()
                )
                moe_metrics[f"moe/layer_{block_idx} router min"] = (
                    router_weight.min().item()
                )
                moe_metrics[f"moe/layer_{block_idx} router max"] = (
                    router_weight.max().item()
                )

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
        self, transformer_block: nn.Module
    ) -> float:
        tokens_per_expert_group_cumulative_prob = (
            transformer_block.moe.tokens_per_expert_group_cumulative.reshape(
                n_expert_group_groups, -1
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
