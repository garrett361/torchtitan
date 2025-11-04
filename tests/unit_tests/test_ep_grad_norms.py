from copy import deepcopy

import dtest
import pytest
import torch

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.deepseek_v3 import (
    DeepSeekV3Model,
    deepseekv3_args,
    parallelize_deepseekv3,
)


class TestGradNorm(dtest.DTest):
    bsz = 2
    seqlen = 256

    @pytest.mark.world_size(2)
    def test_grad_norm(self, world_size: int) -> None:
        # Create equivalent FSDP and EP debug models:
        model_args = deepseekv3_args["debugmodel"]
        model_args.max_seq_len = self.seqlen
        model_fsdp = DeepSeekV3Model(model_args)
        model_fsdp.init_weights(buffer_device=self.device)
        model_ep = deepcopy(model_fsdp)

        pd_kwargs = {
            "dp_shard": -1,
            "dp_replicate": 1,
            "cp": 1,
            "tp": 1,
            "pp": 1,
            "etp": 1,
            "world_size": self.world_size,
        }
        parallel_dims_fsdp = ParallelDims(**pd_kwargs, ep=1)
        parallel_dims_ep = ParallelDims(**pd_kwargs, ep=self.world_size)

        # Default JobConfig is fine for parallelization.
        job_config = JobConfig()
        model_fsdp = parallelize_deepseekv3(model_fsdp, parallel_dims_fsdp, job_config)
        model_ep = parallelize_deepseekv3(model_ep, parallel_dims_ep, job_config)

        inputs = torch.randint(
            model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        out_fsdp = model_fsdp(inputs)
        out_ep = model_ep(inputs)
        out_fsdp.pow(2).mean().backward()
        out_ep.pow(2).mean().backward()

        grad_norm_fsdp = dist_utils.clip_grad_norm_(
            [p for p in model_fsdp.parameters()],
            1.0,
            foreach=True,
            pp_mesh=None,
            ep_enabled=parallel_dims_fsdp.ep_enabled,
        )
        grad_norm_ep = dist_utils.clip_grad_norm_(
            [p for p in model_ep.parameters()],
            1.0,
            foreach=True,
            pp_mesh=None,
            ep_enabled=parallel_dims_ep.ep_enabled,
        )

        grad_norm_ep
