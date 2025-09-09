import math
from typing import TYPE_CHECKING, Any

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from torchtitan.components.metrics import MetricsProcessor, _get_metrics_rank
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


# for MoE auxiliary-loss-free load balancing
def _is_recomputation_enabled(module):
    return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT


class HybridMoEMetricsProcessor(MetricsProcessor):
    eps = 1e-5
    max_entropy: float | None = None

    def _get_metrics_rank(self) -> int:
        return _get_metrics_rank(self.parallel_dims, self.job_config)

    def _get_moe_balancing_metrics(self) -> dict[str, Any]:
        moe_metrics = {}
        for model_part in self.model_parts:
            for block_idx, transformer_block in model_part.layers.items():
                if not transformer_block.moe_enabled:
                    continue
                if transformer_block.moe.load_balance_coeff is None:
                    return
                tokens_per_expert_cumulative = (
                    transformer_block.moe.tokens_per_expert_cumulative
                )
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert_cumulative is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert_cumulative = tokens_per_expert_cumulative // 2

                tokens_per_expert_cumulative_prob = (
                    tokens_per_expert_cumulative + self.eps
                ) / (tokens_per_expert_cumulative + self.eps).sum()
                entropy = (
                    (
                        -tokens_per_expert_cumulative_prob.log()
                        * tokens_per_expert_cumulative_prob
                    )
                    .sum()
                    .item()
                )
                self.max_entropy = self.max_entropy or math.log(
                    tokens_per_expert_cumulative_prob.numel()
                )
                moe_metrics[f"moe/layer_{block_idx} moe normalized entropy"] = (
                    entropy / self.max_entropy
                )
                for exp_idx, tok in enumerate(tokens_per_expert_cumulative.tolist()):
                    moe_metrics[f"moe/layer_{block_idx}:exp_{exp_idx} tokens"] = tok

                transformer_block.moe.tokens_per_expert_cumulative.zero_()
        return moe_metrics

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
