import math
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d

from torchtitan.components.metrics import MetricsProcessor, _get_metrics_rank
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


class HybridMoEMetricsProcessor(MetricsProcessor):
    eps = 1e-5
    device = "cuda"
    max_entropy: float | None = None

    @cached_property
    def rank(self) -> int:
        return dist.get_rank()

    @cached_property
    def metrics_rank(self) -> int:
        return _get_metrics_rank(self.parallel_dims, self.job_config)

    @cached_property
    def moe_layer_idxs(self) -> list[int]:
        """
        The MoE layer idxs on the present rank.
        """
        moe_layer_idxs = []
        for model_part in self.model_parts:
            for block_idx, transformer_block in model_part.layers.items():
                if transformer_block.moe_enabled:
                    moe_layer_idxs.append(int(block_idx))
        return moe_layer_idxs

    def send_tensor_to_metrics_rank(
        self, rank: int, tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Send the given tensor from rank to the metrics_rank process. metrics_rank passes in its
        receive buffer. The recv buffer will be modified in-place, and a copy returned.
        """
        if self.rank not in (rank, self.metrics_rank):
            return None

        ops = []
        if rank == self.rank:
            ops.append(dist.P2POp(dist.isend, tensor, self.metrics_rank))
        else:
            ops.append(dist.P2POp(dist.irecv, tensor, rank))
        for op in dist.batch_isend_irecv(ops):
            op.wait()

        if self.rank == self.metrics_rank:
            return tensor.detach().clone()
        return None

    @cache
    def send_moe_layer_idxs_to_metrics_rank(self, rank: int) -> list[int] | None:
        """
        Send MoE layer idxs on `rank` to the metrics rank. Returns the list of idxs on the metric
        rank, and None on all others.

        Two steps:
        1. Send the number of idxs to expect
        2. Send the idxs
        """
        if self.rank not in (rank, self.metrics_rank):
            return None

        # Send num idxs to expect
        num_results = (
            torch.tensor(
                [len(self.moe_layer_idxs)], device=self.device, dtype=torch.int32
            )
            if rank == self.rank
            else torch.empty(1, device=self.device, dtype=torch.int32)
        )
        num_results = self.send_tensor_to_metrics_rank(rank, num_results)

        # Send idxs
        layer_idxs = (
            torch.tensor(self.moe_layer_idxs, device=self.device, dtype=torch.int32)
            if rank == self.rank
            else torch.empty(
                int(num_results.item()), device=self.device, dtype=torch.int32
            )
        )
        layer_idxs = self.send_tensor_to_metrics_rank(rank, layer_idxs)
        if self.rank == self.metrics_rank:
            return layer_idxs.tolist()
        return None

    def get_moe_balancing_metrics(self) -> dict[str, Any]:
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
            pp_ranks = self.parallel_dims.world_mesh["pp"].mesh.tolist()
            # Only other members of the metrics rank's PP group need to send info.
            if self.metrics_rank in pp_ranks:
                for send_rank in pp_ranks:
                    if send_rank == self.metrics_rank:
                        continue
                    # Send/get cached moe idxs from send_rank
                    send_rank_idxs = self.send_moe_layer_idxs_to_metrics_rank(send_rank)
                    # Send results
                    results = (
                        torch.stack(list(moe_metrics.values()), dim=-1)
                        if send_rank == self.rank
                        else torch.empty(
                            len(send_rank_idxs),
                            device=self.device,
                        )
                    )
                    results = self.send_tensor_to_metrics_rank(send_rank, results)

                    if self.rank == self.metrics_rank:
                        for idx, res in zip(send_rank_idxs, results, strict=True):
                            moe_metrics[idx] = res

        if self.rank != self.metrics_rank:
            return {}
        return {
            f"moe/layer_{block_idx} moe normalized entropy": e.item()
            for block_idx, e in moe_metrics.items()
        }

    def get_pp_memory_metrics(self) -> dict[str, Any]:
        """
        Get the CUDA memory metrics on each PP rank, max-reduced over the dp_cp group, if it exists.
        """
        if not self.parallel_dims.pp_enabled:
            return {}

        device_mem_stats = self.device_memory_monitor.get_peak_stats()
        # Put mem stats in a tensor. Order
        # 0: "memory/max_active(GiB)": device_mem_stats.max_active_gib,
        # 1: "memory/max_active(%)": device_mem_stats.max_active_pct,
        # 2: "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
        # 3: "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
        # 4: "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
        # 5: "memory/num_ooms": device_mem_stats.num_ooms,
        mem_stat_prefixes = [
            "memory/max_active(GiB)",
            "memory/max_active(%)",
            "memory/max_reserved(GiB)",
            "memory/max_reserved(%)",
            "memory/num_alloc_retries",
            "memory/num_ooms",
        ]
        mem_stats_t = torch.tensor(
            [
                device_mem_stats.max_active_gib,
                device_mem_stats.max_active_pct,
                device_mem_stats.max_reserved_gib,
                device_mem_stats.max_reserved_pct,
                device_mem_stats.num_alloc_retries,
                device_mem_stats.num_ooms,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if self.rank == self.metrics_rank:
            recv_mem_stats_buffer_t = mem_stats_t.clone()

        if self.parallel_dims.dp_cp_enabled:
            mem_stats_t = funcol.all_reduce(
                mem_stats_t,
                reduceOp=c10d.ReduceOp.MAX.name,
                group=self.parallel_dims.world_mesh["dp_cp"],
            )

        pp_ranks = self.parallel_dims.world_mesh["pp"].mesh.tolist()
        # Only other members of the metrics rank's PP group need to send info.
        # Use the enumerated idx of the rank as the PP stage. TODO: @goon - this doesn't necessarily
        # correspond to the actual PP stage, can improve.
        pp_mem_metrics_t = {}
        if self.metrics_rank in pp_ranks:
            for stage_idx, send_rank in enumerate(pp_ranks):
                if send_rank == self.metrics_rank:
                    pp_mem_metrics_t[stage_idx] = mem_stats_t
                    continue
                recv_mem_stats_t = self.send_tensor_to_metrics_rank(
                    send_rank,
                    mem_stats_t if self.rank == send_rank else recv_mem_stats_buffer_t,
                )
                pp_mem_metrics_t[stage_idx] = recv_mem_stats_t
        if self.rank != self.metrics_rank:
            return {}
        # Turn the tensorial mem metrics into nicely formatted ones
        pp_mem_metrics = {}
        for stage_idx, mem_t in pp_mem_metrics_t.items():
            for metric_name, metric_val in zip(
                mem_stat_prefixes, mem_t.tolist(), strict=True
            ):
                pp_mem_metrics[metric_name + f" pp stage {stage_idx}"] = metric_val
        return pp_mem_metrics

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        moe_metrics = self.get_moe_balancing_metrics()
        pp_mem_metrics = self.get_pp_memory_metrics()
        if extra_metrics is None:
            extra_metrics = {**moe_metrics, **pp_mem_metrics}
        else:
            extra_metrics = {**extra_metrics, **moe_metrics, **pp_mem_metrics}

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
