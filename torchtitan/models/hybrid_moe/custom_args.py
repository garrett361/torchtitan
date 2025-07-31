from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig


@dataclass
class CustomArgs:
    load_balance_coeff: float | None = None


@dataclass
class JobConfig(JobConfig):
    custom_args: CustomArgs = field(default_factory=CustomArgs)
