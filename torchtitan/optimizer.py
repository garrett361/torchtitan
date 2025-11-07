# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig


# consider split between PP and non-PP
def build_optimizers(model_parts, job_config: JobConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])

class LambdaLRCustom(LambdaLR):
    """
    Customized lr scheduler which lets the scheduler lambda take on two args.
    """

    def step(self, epoch: int | None = None, num_steps: int = 10000) -> None:
        # HACK: @goon - the num_steps default is a placeholder needed so that the initial step
        # during init doesn't error. Only affects the very first step.
        self.num_steps = num_steps
        super().step(epoch=epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        return [
            base_lr * lmbda(self.last_epoch, self.num_steps)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]


def annealing(
    current_step: int, num_steps: int, warmup_steps: int, final_lr_ratio: float
) -> float:
    if current_step <= warmup_steps:
        factor =  1 - (1 - current_step / warmup_steps) ** 2
    else:
        if num_steps <= warmup_steps:
            raise ValueError(f"{warmup_steps=} is larger than {num_steps=}, reduce warmup_steps")
        factor = 1 - (1 - final_lr_ratio) * (current_step - warmup_steps) / (num_steps - warmup_steps)
    return factor


def build_lr_schedulers(optimizers, job_config: JobConfig):
    def _build_lr_scheduler(optimizer):
        warmup_steps = int(job_config.training.warmup_steps)
        lr_lambda = functools.partial(
            annealing, warmup_steps=warmup_steps, final_lr_ratio=job_config.training.final_lr_ratio
        )
        warmup_scheduler = LambdaLRCustom(optimizer, lr_lambda=lr_lambda)
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self, num_steps: int):
            for schedulers in self.schedulers:
                schedulers.step(num_steps=num_steps)

    return SchedulersContainer(
        [_build_lr_scheduler(optimizer) for optimizer in optimizers]
    )
