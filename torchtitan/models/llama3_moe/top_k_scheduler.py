# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.models.llama3_moe.custom_args import (
    TopKSchedulerArgs,
)
from torchtitan.models.llama3_moe.model.args import Llama3MoEModelArgs
from torchtitan.models.moe import MoE
from torchtitan.tools.logging import logger


class _TopKScheduler(Stateful, ABC):
    def __init__(
        self,
        model_args: Llama3MoEModelArgs,
        top_k_args: TopKSchedulerArgs,
        model_parts: list[torch.nn.Module],
    ) -> None:
        self.model_args = model_args
        self.top_k_args = top_k_args
        self.model_parts = model_parts
        self._step = 0
        if self.model_args.is_moe_list is None:
            self.layer_idx_to_top_k: dict[int, int] = {}
        else:
            self.layer_idx_to_top_k: dict[int, int] = {
                layer_idx: self.model_args.moe_args.top_k
                for layer_idx, val in enumerate(self.model_args.is_moe_list)
                if val
            }

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

    @abstractmethod
    def step(self, loss: torch.Tensor) -> None: ...


class NoOpScheduler(_TopKScheduler):
    name = "no_op"

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return

    def step(self, loss: torch.Tensor) -> None:
        return


class ConstantScheduler(_TopKScheduler):
    name = "constant"

    def __init__(
        self,
        model_args: Llama3MoEModelArgs,
        top_k_args: TopKSchedulerArgs,
        model_parts: list[torch.nn.Module],
    ) -> None:
        super().__init__(
            model_args=model_args, top_k_args=top_k_args, model_parts=model_parts
        )
        assert self.top_k_args.min_top_k is not None
        assert self.top_k_args.step_interval is not None
        assert self.top_k_args.warmup_steps is not None

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "layer_idx_to_top_k": self.layer_idx_to_top_k}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)

    def _get_idx_and_moe(self) -> tuple[int, MoE] | tuple[None, None]:
        layer_top_k_pairs = [
            (layer_idx, k)
            for layer_idx, k in self.layer_idx_to_top_k.items()
            if k > self.top_k_args.min_top_k
        ]
        if not layer_top_k_pairs:
            return None, None
        layer_idx = max(layer_top_k_pairs, key=lambda x: x[0])[0]
        layer_idx_str = str(layer_idx)
        for mp in self.model_parts:
            if layer_idx_str in mp.layers:
                return layer_idx, mp.layers[layer_idx_str].moe

    def step(self, loss: torch.Tensor) -> None:
        self._step += 1
        done_warmup = self._step > self.top_k_args.warmup_steps
        if done_warmup:
            should_step = (
                (self._step - self.top_k_args.warmup_steps)
                % self.top_k_args.step_interval
            ) == 0
            if should_step:
                layer_idx, moe = self._get_idx_and_moe()
                if moe is not None:
                    moe.router.top_k -= 1
                    moe.reorderer.top_k -= 1
                    self.layer_idx_to_top_k[layer_idx] -= 1
                    new_top_k = self.layer_idx_to_top_k[layer_idx]
                    logger.info(
                        f"Reducing top_k on {layer_idx=} MoE from {new_top_k + 1} -> {new_top_k}."
                    )


def get_top_k_scheduler(
    model_args: Llama3MoEModelArgs,
    top_k_args: TopKSchedulerArgs,
    model_parts: list[torch.nn.Module],
) -> type[_TopKScheduler]:
    scheduler_dict = {
        sc.name: sc for sc in _TopKScheduler.__subclasses__() if hasattr(sc, "name")
    }
    return scheduler_dict[top_k_args.name](
        model_args=model_args, top_k_args=top_k_args, model_parts=model_parts
    )
