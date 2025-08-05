from typing import TYPE_CHECKING

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


class HybridMoEMetricsProcessor(MetricsProcessor):
    pass


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
