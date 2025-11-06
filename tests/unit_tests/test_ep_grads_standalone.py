import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.deepseek_v3 import (
    DeepSeekV3Model,
    deepseekv3_args,
    parallelize_deepseekv3,
)


def check_grads_close(
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


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    bsz = 2
    seqlen = 256
    tol = 1e-1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    model_args = deepseekv3_args["debugmodel"]
    torch.manual_seed(42)
    # Create equivalent FSDP and EP debug models:
    model_args.max_seq_len = seqlen
    model_fsdp = DeepSeekV3Model(model_args)
    model_fsdp.init_weights(buffer_device=device)
    model_ep = deepcopy(model_fsdp)

    pd_kwargs = {
        "dp_shard": -1,
        "dp_replicate": 1,
        "cp": 1,
        "tp": 1,
        "pp": 1,
        "etp": 1,
        "world_size": world_size,
    }
    parallel_dims_fsdp = ParallelDims(**pd_kwargs, ep=1)
    parallel_dims_ep = ParallelDims(**pd_kwargs, ep=world_size)

    # Default JobConfig is fine for parallelization.
    job_config = JobConfig()
    model_fsdp = parallelize_deepseekv3(model_fsdp, parallel_dims_fsdp, job_config)
    model_ep = parallelize_deepseekv3(model_ep, parallel_dims_ep, job_config)

    # Run backwards:
    inputs = torch.randint(model_args.vocab_size, size=(bsz, seqlen), device=device)
    out_fsdp = model_fsdp(inputs)
    out_ep = model_ep(inputs)
    torch.testing.assert_close(out_fsdp, out_ep, atol=tol, rtol=tol)
    out_fsdp.pow(2).mean().backward()
    out_ep.pow(2).mean().backward()

    check_grads_close(model_fsdp, model_ep, atol=tol, rtol=tol)
