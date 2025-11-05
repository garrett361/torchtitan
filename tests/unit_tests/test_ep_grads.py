# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import dtest
import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.deepseek_v3 import (
    deepseekv3_args,
    DeepSeekV3Model,
    parallelize_deepseekv3,
)


def _check_grads_close(
    model_fsdp: nn.Module,
    model_ep: nn.Module,
    atol=None,
    rtol=None,
) -> None:
    fails = []
    for (n, p_fsdp), (_, p_ep) in zip(
        model_fsdp.named_parameters(), model_ep.named_parameters(), strict=True
    ):
        if p_fsdp.grad is None:
            assert p_ep.grad is None
        else:
            g_fsdp = (
                p_fsdp.grad.full_tensor()
                if isinstance(p_fsdp.grad, DTensor)
                else p_fsdp.grad
            )
            g_ep = (
                p_ep.grad.full_tensor() if isinstance(p_ep.grad, DTensor) else p_ep.grad
            )
            # Very simple test: the abs sum of all grads should be close:
            try:
                g_fsdp_abs_sum = g_fsdp.abs().sum()
                g_ep_abs_sum = g_ep.abs().sum()
                torch.testing.assert_close(
                    g_fsdp_abs_sum, g_ep_abs_sum, atol=atol, rtol=rtol
                )
            except AssertionError:
                fails.append(f"Failed on {n=}: {g_ep_abs_sum/g_fsdp_abs_sum=}")
    if fails:
        raise AssertionError("\n".join(fails))


class TestGrads(dtest.DTest):
    bsz = 2
    seqlen = 256
    tol = 1e-1
    model_args = deepseekv3_args["debugmodel"]

    def _get_fsdp_ep_models_with_grads(
        self, ep_degree: int, dp_replicate: int = 1
    ) -> tuple[nn.Module, nn.Module]:
        torch.manual_seed(42)
        # Create equivalent FSDP and EP debug models:
        self.model_args.max_seq_len = self.seqlen
        model_fsdp = DeepSeekV3Model(self.model_args)
        model_fsdp.init_weights(buffer_device=self.device)
        model_ep = deepcopy(model_fsdp)

        pd_kwargs = {
            "dp_shard": -1,
            "cp": 1,
            "tp": 1,
            "pp": 1,
            "etp": 1,
            "world_size": self.world_size,
        }
        parallel_dims_fsdp = ParallelDims(**pd_kwargs, ep=1, dp_replicate=dp_replicate)
        parallel_dims_ep = ParallelDims(
            **pd_kwargs, ep=ep_degree, dp_replicate=dp_replicate
        )

        # Default JobConfig is fine for parallelization.
        job_config = JobConfig()
        model_fsdp = parallelize_deepseekv3(model_fsdp, parallel_dims_fsdp, job_config)
        model_ep = parallelize_deepseekv3(model_ep, parallel_dims_ep, job_config)

        # Run backwards:
        inputs = torch.randint(
            self.model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        out_fsdp = model_fsdp(inputs)
        out_ep = model_ep(inputs)
        torch.testing.assert_close(out_fsdp, out_ep, atol=self.tol, rtol=self.tol)
        out_fsdp.pow(2).mean().backward()
        out_ep.pow(2).mean().backward()

        return model_fsdp, model_ep

    @pytest.mark.world_size([2, 4, 8])
    def test_grads_world_ep(self, world_size: int) -> None:
        model_fsdp, model_ep = self._get_fsdp_ep_models_with_grads(
            ep_degree=self.world_size
        )
        _check_grads_close(
            model_fsdp,
            model_ep,
            atol=self.tol,
            rtol=self.tol,
        )

    @pytest.mark.world_size([4, 8])
    def test_grads_partial_ep(self, world_size: int) -> None:
        model_fsdp, model_ep = self._get_fsdp_ep_models_with_grads(
            ep_degree=self.world_size // 2
        )
        _check_grads_close(
            model_fsdp,
            model_ep,
            atol=self.tol,
            rtol=self.tol,
        )

    @pytest.mark.world_size(8)
    @pytest.mark.parametrize("dp_replicate", [2, 4], ids=lambda x: f"dp_replicate={x}")
    def test_grads_replicated(self, world_size: int, dp_replicate: int) -> None:
        model_fsdp, model_ep = self._get_fsdp_ep_models_with_grads(
            ep_degree=2, dp_replicate=dp_replicate
        )
        _check_grads_close(
            model_fsdp,
            model_ep,
            atol=self.tol,
            rtol=self.tol,
        )
