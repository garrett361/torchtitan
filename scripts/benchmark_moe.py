#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal benchmark script for dense vs MoE model throughput.

Usage:
    torchrun --nproc_per_node=8 scripts/benchmark_moe.py [OPTIONS]
"""

import argparse
import os

import torch
import torch.distributed as dist

from torchtitan.components.metrics import DeviceMemoryMonitor, DeviceMemStats
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Compile,
    Model,
    Parallelism,
    Training,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3_moe import (
    Llama3MoE,
    Llama3MoEJobConfig,
    Llama3MoEModelArgs,
    MoEArgs,
    apply_custom_init,
    llama3_moe_configs,
    parallelize_llama_moe,
)
from torchtitan.models.llama3_moe.custom_args import MoEOverrides
from torchtitan.tools.logging import init_logger, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE Benchmark")

    # Core args
    parser.add_argument(
        "--flavor",
        type=str,
        default="1B",
        choices=list(llama3_moe_configs.keys()),
        help="Model size from llama3_moe_configs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Local batch size per GPU"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--bench-iters", type=int, default=20, help="Timed benchmark iterations"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel degree")

    # MoE config (n-moe-layers=0 means dense baseline)
    parser.add_argument(
        "--n-moe-layers", type=int, default=0, help="Number of MoE layers (0 = dense)"
    )
    parser.add_argument(
        "--n-replicas", type=int, default=2, help="Number of FFN replicas"
    )
    parser.add_argument("--n-groups", type=int, default=64, help="Groups per replica")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k routing. Defautls to n-groups if not provided",
    )
    parser.add_argument(
        "--force-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force load balance",
    )
    parser.add_argument(
        "--moe-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reshard MoE params after forward (default False to match train config)",
    )
    parser.add_argument(
        "--ac-mode",
        type=str,
        default="selective",
        choices=["selective", "full", "none"],
        help="Activation checkpointing mode",
    )
    parser.add_argument(
        "--ac-option",
        type=str,
        default="2",
        help="Selective AC option: integer for every nth layer, or 'op' for op-level AC",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON results (in addition to stdout)",
    )
    parser.add_argument(
        "--custom-moe-impl",
        type=str,
        default="virtual_group",
        choices=["virtual_group", "sonic", "sonic_virtual_group"],
    )

    return parser.parse_args()


def compute_ffn_hidden_dim(model_args: Llama3MoEModelArgs) -> int:
    """Derive FFN hidden dim from model args, matching FeedForward class."""
    hidden_dim = int(2 * 4 * model_args.dim / 3)
    if model_args.ffn_dim_multiplier is not None:
        hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = model_args.multiple_of * (
        (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
    )
    return hidden_dim


def validate_ep(ep: int, world_size: int, num_experts: int) -> None:
    """Validate expert parallelism configuration."""
    if ep > 1:
        if world_size % ep != 0:
            raise ValueError(f"EP={ep} must divide world_size={world_size}")
        if num_experts % ep != 0:
            raise ValueError(f"EP={ep} must divide num_experts={num_experts}")


def check_numerical_health(tensor: torch.Tensor, name: str) -> None:
    """Raise if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any():
        raise RuntimeError(f"{name} contains NaN - benchmark results invalid")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"{name} contains Inf - benchmark results invalid")


def build_model_args(args: argparse.Namespace) -> Llama3MoEModelArgs:
    """Build model args from CLI, handling dense vs MoE configuration."""
    model_args = llama3_moe_configs[args.flavor]

    # Compute derived values
    hf_ffn_hidden_dim = compute_ffn_hidden_dim(model_args)

    if args.n_moe_layers > 0:
        # MoE mode
        num_experts = args.n_replicas * args.n_groups
        top_k = args.top_k or args.n_groups  # Activate full replica
        moe_inter_dim = hf_ffn_hidden_dim // args.n_groups

        # Build is_moe_list: MoE layers from end, excluding last
        n_layers = model_args.n_layers
        if args.n_moe_layers > n_layers - 1:
            raise ValueError(
                f"n_moe_layers={args.n_moe_layers} must be <= n_layers-1={n_layers - 1}"
            )
        is_moe_list = (
            (n_layers - args.n_moe_layers - 1) * [False]
            + args.n_moe_layers * [True]
            + [False]
        )

        # Override model args
        model_args.moe_inter_dim = moe_inter_dim
        model_args.is_moe_list = is_moe_list
        model_args.custom_moe_impl = args.custom_moe_impl 
        model_args.moe_args = MoEArgs(
            num_experts=num_experts,
            num_shared_experts=0,
            top_k=top_k,
            score_func="softmax",
            route_norm=True,
            route_scale=args.n_groups,
            score_before_experts=False,  # Required for virtual_group
            hf_ffn_hidden_dim=hf_ffn_hidden_dim,
            _debug_force_load_balance=args.force_balance,
        )
    else:
        # Dense mode: ensure is_moe_list is None
        model_args.is_moe_list = None

    # Set seq_len
    model_args.max_seq_len = args.seq_len

    return model_args


def build_job_config(args: argparse.Namespace) -> Llama3MoEJobConfig:
    """Build job config for benchmark with explicit settings."""
    return Llama3MoEJobConfig(
        model=Model(
            name="llama3_moe",
            flavor=args.flavor,
        ),
        training=Training(
            seq_len=args.seq_len,
            local_batch_size=args.batch_size,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=Parallelism(
            data_parallel_shard_degree=-1,  # Auto
            expert_parallel_degree=args.ep,
        ),
        activation_checkpoint=ActivationCheckpoint(
            mode=args.ac_mode,
            selective_ac_option=args.ac_option,
        ),
        compile=Compile(
            enable=False,
        ),
        moe_overrides=MoEOverrides(
            moe_reshard_after_forward=args.moe_reshard_after_forward,
        ),
    )


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed and return rank, world_size, device."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def build_parallel_dims(world_size: int, ep: int) -> ParallelDims:
    """Build ParallelDims for FSDP (+ optional EP)."""
    return ParallelDims(
        dp_replicate=1,
        dp_shard=-1,  # Auto-compute
        cp=1,
        tp=1,
        pp=1,
        ep=ep,
        etp=1,
        world_size=world_size,
    )


def create_model(
    model_args: Llama3MoEModelArgs,
    parallel_dims: ParallelDims,
    job_config: Llama3MoEJobConfig,
    device: torch.device,
) -> Llama3MoE:
    """Create and initialize model following train.py order."""
    # 1. Create on meta device
    with torch.device("meta"):
        model = Llama3MoE(model_args)

    # 2. Apply parallelization FIRST (before materialization)
    model = parallelize_llama_moe(model, parallel_dims, job_config)

    # Print model structure (rank 0 only)
    if dist.get_rank() == 0:
        logger.info(f"Job config:\n{job_config}")
        logger.info(f"Model structure post-parallelization:\n{model}")

    # 3. Materialize on device
    model.to_empty(device=device)

    # 4. Init weights
    with torch.no_grad():
        model.init_weights(buffer_device=device)

    # 5. Apply custom init (router std, etc.)
    apply_custom_init(model, job_config)

    # 6. Set train mode
    model.train()

    return model


def run_benchmark(
    model: Llama3MoE,
    input_ids: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    memory_monitor: DeviceMemoryMonitor,
) -> tuple[float, list[torch.Tensor], DeviceMemStats]:
    """Run warmup and timed benchmark, return elapsed_ms, outputs, and memory stats."""
    # Warmup
    for _ in range(warmup_iters):
        model(input_ids).sum().backward()
        model.zero_grad(set_to_none=True)

    # Reset memory stats after warmup to measure steady-state
    memory_monitor.reset_peak_stats()

    # Timed benchmark with CUDA events
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    out_sum_list = []
    start_event.record()
    for _ in range(bench_iters):
        out_sum = model(input_ids).sum()
        out_sum_list.append(out_sum.detach().clone())
        out_sum.backward()
        del out_sum
        model.zero_grad(set_to_none=True)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    mem_stats = memory_monitor.get_peak_stats()
    return elapsed_ms, out_sum_list, mem_stats


def validate_results(model: Llama3MoE, outputs: list[torch.Tensor]) -> None:
    """Check all outputs and gradients for NaN/Inf."""
    # Check all outputs
    for i, out in enumerate(outputs):
        check_numerical_health(out, f"output[{i}]")

    # Check all gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            check_numerical_health(param.grad, f"grad/{name}")


def print_results(
    args: argparse.Namespace,
    model_args: Llama3MoEModelArgs,
    world_size: int,
    elapsed_ms: float,
    passed_check: bool,
    model_param_count: int,
    num_flops_per_token: float,
    mem_stats: DeviceMemStats,
) -> None:
    """Print benchmark results (rank 0 only)."""
    tokens_per_iter = args.batch_size * args.seq_len
    tps_per_gpu = (tokens_per_iter * args.bench_iters) / (elapsed_ms / 1000)
    tps_total = tps_per_gpu * world_size
    time_per_iter = elapsed_ms / args.bench_iters

    n_layers = model_args.n_layers
    n_moe = sum(model_args.is_moe_list) if model_args.is_moe_list else 0

    print("\n=== MoE Benchmark ===")
    if args.n_moe_layers > 0:
        num_experts = args.n_replicas * args.n_groups
        print(f"Model: {args.flavor} ({n_layers} layers, {n_moe} MoE)")
        print(
            f"Experts: {num_experts} ({args.n_replicas} replicas x {args.n_groups} groups), top_k={args.n_groups}"
        )
        print(f"Parallelism: FSDP (dp_shard={world_size // args.ep}), EP={args.ep}")
    else:
        print(f"Model: {args.flavor} ({n_layers} layers, 0 MoE - dense baseline)")
        print(f"Parallelism: FSDP (dp_shard={world_size})")

    print(f"Parameters: {model_param_count:,} total")
    print(f"FLOPs/token: {num_flops_per_token:,.0f}")
    print(f"Batch: {args.batch_size} x {args.seq_len} = {tokens_per_iter:,} tokens/GPU")
    print(f"\nWarmup: {args.warmup_iters} iters")
    print(f"Benchmark: {args.bench_iters} iters")
    print(f"NaN/Inf check: {'PASS' if passed_check else 'FAIL'}")
    print("\nMemory (per GPU):")
    print(
        f"  Peak active: {mem_stats.max_active_gib:.2f} GiB ({mem_stats.max_active_pct:.1f}%)"
    )
    print(
        f"  Peak reserved: {mem_stats.max_reserved_gib:.2f} GiB ({mem_stats.max_reserved_pct:.1f}%)"
    )
    if mem_stats.num_alloc_retries > 0:
        print(f"  Alloc retries: {mem_stats.num_alloc_retries}")
    if mem_stats.num_ooms > 0:
        print(f"  OOMs: {mem_stats.num_ooms}")
    print("\nResults:")
    print(f"  Time/iter: {time_per_iter:.1f} ms")
    print(f"  TPS (per GPU): {tps_per_gpu:,.0f}")
    print(f"  TPS (total): {tps_total:,.0f}")


def write_json_results(
    path: str,
    args: argparse.Namespace,
    model_args: Llama3MoEModelArgs,
    world_size: int,
    elapsed_ms: float,
    passed_check: bool,
    model_param_count: int,
    num_flops_per_token: float,
    mem_stats: DeviceMemStats,
) -> None:
    """Write structured JSON results to file (rank 0 only)."""
    import json

    tokens_per_iter = args.batch_size * args.seq_len
    tps_per_gpu = (tokens_per_iter * args.bench_iters) / (elapsed_ms / 1000)
    tps_total = tps_per_gpu * world_size
    time_per_iter = elapsed_ms / args.bench_iters

    result = {
        "status": "success" if passed_check else "error",
        "error": None if passed_check else "NaN/Inf detected",
        "config": {
            "flavor": args.flavor,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_groups": args.n_groups,
            "n_moe_layers": args.n_moe_layers,
            "n_replicas": args.n_replicas,
            "top_k": args.top_k or args.n_groups,
            "ep": args.ep,
            "ac_mode": args.ac_mode,
            "ac_option": args.ac_option,
            "moe_reshard_after_forward": args.moe_reshard_after_forward,
            "force_balance": args.force_balance,
        },
        "metrics": {
            "tps_per_gpu": tps_per_gpu,
            "tps_total": tps_total,
            "time_per_iter_ms": time_per_iter,
            "peak_active_gib": mem_stats.max_active_gib,
            "peak_active_pct": mem_stats.max_active_pct,
            "peak_reserved_gib": mem_stats.max_reserved_gib,
            "peak_reserved_pct": mem_stats.max_reserved_pct,
            "num_alloc_retries": mem_stats.num_alloc_retries,
            "num_ooms": mem_stats.num_ooms,
        },
        "model_info": {
            "param_count": model_param_count,
            "flops_per_token": num_flops_per_token,
        },
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    args = parse_args()
    init_logger()

    # Initialize distributed
    rank, world_size, device = init_distributed()

    try:
        # Build configurations
        model_args = build_model_args(args)
        job_config = build_job_config(args)

        # Validate EP if enabled
        if args.ep > 1 and args.n_moe_layers > 0:
            num_experts = args.n_replicas * args.n_groups
            validate_ep(args.ep, world_size, num_experts)

        # Build parallel dims and mesh
        parallel_dims = build_parallel_dims(world_size, args.ep)
        _ = parallel_dims.world_mesh  # Initialize mesh

        if rank == 0:
            logger.info(
                f"Building model: {args.flavor}, n_moe_layers={args.n_moe_layers}"
            )

        # Create model
        model = create_model(model_args, parallel_dims, job_config, device)

        # Calculate parameter counts and FLOPs
        model_param_count, num_flops_per_token = model_args.get_nparams_and_flops(
            model, args.seq_len
        )

        # Create synthetic input
        input_ids = torch.randint(
            0,
            model_args.vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
        )

        # Initialize memory monitor
        memory_monitor = DeviceMemoryMonitor(device)

        # Run benchmark
        elapsed_ms, outputs, mem_stats = run_benchmark(
            model, input_ids, args.warmup_iters, args.bench_iters, memory_monitor
        )

        # Validate results (after timing)
        passed_check = True
        try:
            validate_results(model, outputs)
        except RuntimeError as e:
            passed_check = False
            if rank == 0:
                logger.error(str(e))

        # Report results
        if rank == 0:
            print_results(
                args,
                model_args,
                world_size,
                elapsed_ms,
                passed_check,
                model_param_count,
                num_flops_per_token,
                mem_stats,
            )
            if args.output_json:
                write_json_results(
                    args.output_json,
                    args,
                    model_args,
                    world_size,
                    elapsed_ms,
                    passed_check,
                    model_param_count,
                    num_flops_per_token,
                    mem_stats,
                )
                logger.info(f"Results written to {args.output_json}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
