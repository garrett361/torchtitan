from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.models.llama3_moe.custom_args import Llama3MoEJobConfig
from torchtitan.models.moe import MoE
from torchtitan.tools.logging import logger


class _TopKScheduler(Stateful, ABC):
    def __init__(
        self,
        job_config: Llama3MoEJobConfig,
        model_parts: list[torch.nn.Module],
    ) -> None:
        self.job_config = job_config
        self.model_parts = model_parts
        self._step = 0
        if job_config.model.is_moe_list is None:
            self.layer_idx_to_top_k: dict[int, int] = {}
        else:
            self.layer_idx_to_top_k: dict[int, int] = {
                layer_idx: job_config.model.moe_args.top_k
                for layer_idx, val in enumerate(job_config.model.is_moe_list)
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
        job_config: Llama3MoEJobConfig,
        model_parts: list[torch.nn.Module],
    ) -> None:
        super().__init__(job_config=job_config, model_parts=model_parts)
        self.top_k_args = job_config.top_k_args
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
    job_config: Llama3MoEJobConfig, model_parts: list[torch.nn.Module]
) -> type[_TopKScheduler]:
    top_k_args = job_config.top_k_args
    scheduler_dict = {
        sc.name: sc for sc in _TopKScheduler.__subclasses__() if hasattr(sc, "name")
    }
    return scheduler_dict[top_k_args.name](job_config, model_parts)
