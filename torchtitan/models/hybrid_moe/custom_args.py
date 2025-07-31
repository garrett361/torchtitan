from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig


@dataclass
class CustomArgs:
    how_is_your_day: str = "good"
    """Just an example."""


@dataclass
class JobConfig(JobConfig):
    custom_args: CustomArgs = field(default_factory=CustomArgs)
