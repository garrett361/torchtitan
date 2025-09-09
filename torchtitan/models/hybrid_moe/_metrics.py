import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from torchtitan.components.metrics import MetricsProcessor, _get_metrics_rank
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


class HybridMoEMetricsProcessor(MetricsProcessor):
    eps = 1e-5
    max_entropy: float | None = None

    @cached_property
    def rank(self) -> int:
        return dist.get_rank()

    @cached_property
    def metrics_rank(self) -> int:
        return _get_metrics_rank(self.parallel_dims, self.job_config)

    def _get_moe_balancing_metrics(self) -> dict[str, Any]:
        moe_metrics = {}
        # Get locally-available MoE stats
        for model_part in self.model_parts:
            for block_idx, transformer_block in model_part.layers.items():
                if not transformer_block.moe_enabled:
                    continue
                if transformer_block.moe.load_balance_coeff is None:
                    return
                tokens_per_expert_cumulative = (
                    transformer_block.moe.tokens_per_expert_cumulative
                )

                tokens_per_expert_cumulative_prob = (
                    tokens_per_expert_cumulative + self.eps
                ) / (tokens_per_expert_cumulative + self.eps).sum()
                entropy = (
                    -tokens_per_expert_cumulative_prob.log()
                    * tokens_per_expert_cumulative_prob
                ).sum()
                self.max_entropy = self.max_entropy or math.log(
                    tokens_per_expert_cumulative_prob.numel()
                )
                moe_metrics[int(block_idx)] = entropy / self.max_entropy
                # NOTE: @goon - per-expert stats are pretty noisy, just the entropy seems to
                # give a good view.
                # for exp_idx, tok in enumerate(tokens_per_expert_cumulative.tolist()):
                #     moe_metrics[f"moe/layer_{block_idx}:exp_{exp_idx} tokens"] = tok

                transformer_block.moe.tokens_per_expert_cumulative.zero_()

        if self.parallel_dims.pp_enabled:
            # If using PP, need to also gather stats from other PP ranks.
            # Two steps:
            # 1. Send the number of results to expect
            # 2. Send results
            # TODO: @goon - precompute this based on PP setup.
            pp_ranks = self.parallel_dims.world_mesh["pp"].mesh.tolist()
            # Only other members of the metrics rank's PP group need to send info.
            if self.metrics_rank in pp_ranks:
                for send_rank in pp_ranks:
                    ops = []
                    if send_rank == self.metrics_rank:
                        continue

                    # Send num results to expect
                    if send_rank == self.rank:
                        num_results = torch.tensor(
                            [len(moe_metrics)], device="cuda", dtype=torch.int32
                        )

                        ops.append(
                            dist.P2POp(dist.isend, num_results, self.metrics_rank)
                        )
                    elif self.rank == self.metrics_rank:
                        num_results = torch.empty(1, device="cuda", dtype=torch.int32)
                        ops.append(dist.P2POp(dist.irecv, num_results, send_rank))
                    if ops:
                        for op in dist.batch_isend_irecv(ops):
                            op.wait()

                    # Send results
                    if send_rank == self.rank:
                        layer_idxs = torch.tensor(
                            list(moe_metrics), device="cuda", dtype=torch.int32
                        )
                        results = torch.stack(list(moe_metrics.values()), dim=-1)
                        ops.append(
                            dist.P2POp(dist.isend, layer_idxs, self.metrics_rank)
                        )
                        ops.append(dist.P2POp(dist.isend, results, self.metrics_rank))
                    elif self.rank == self.metrics_rank:
                        num_results_cpu = int(num_results.item())
                        layer_idxs = torch.empty(
                            num_results_cpu, device="cuda", dtype=torch.int32
                        )
                        results = torch.empty(num_results_cpu, device="cuda")
                        ops.append(dist.P2POp(dist.irecv, layer_idxs, send_rank))
                        ops.append(dist.P2POp(dist.irecv, results, send_rank))
                    if ops:
                        for op in dist.batch_isend_irecv(ops):
                            op.wait()

                    if self.rank == self.metrics_rank:
                        for idx, res in zip(layer_idxs.tolist(), results):
                            moe_metrics[idx] = res

        return {
            f"moe/layer_{block_idx} moe normalized entropy": e.item()
            for block_idx, e in moe_metrics.items()
        }

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        moe_metrics = self._get_moe_balancing_metrics()
        if extra_metrics is None:
            extra_metrics = moe_metrics
        else:
            extra_metrics = {**extra_metrics, **moe_metrics}

        super().log(
            step=step,
            global_avg_loss=global_avg_loss,
            global_max_loss=global_max_loss,
            grad_norm=grad_norm,
            extra_metrics=extra_metrics,
        )


def build_hybrid_moe_metrics_processor(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    model_args: "BaseModelArgs | None" = None,
    tag: str | None = None,
) -> MetricsProcessor:
    """Create a metrics processor.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        model_args (BaseModelArgs | None): Model-specific arguments. Defaults to None.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.

    Returns:
        MetricsProcessor: A metrics processor.
    """
    return HybridMoEMetricsProcessor(job_config, parallel_dims, tag)
