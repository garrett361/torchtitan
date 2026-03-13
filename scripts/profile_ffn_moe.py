"""Profile kernel dispatch for FFN vs MoE layers.

Provides kernel-level visibility into where time is spent - which CUDA kernels dominate
(grouped_mm, scatter/gather, router matmul), kernel launch overhead, and memory operations.
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from torchtitan.models.moe import FeedForward, MoE, MoEArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Layer selection
    parser.add_argument(
        "--layer",
        choices=["ffn", "moe"],
        default="moe",
        help="Layer to profile (default: moe)",
    )

    # Model dimensions (1B defaults)
    parser.add_argument("--dim", type=int, default=2048, help="Model dimension")
    parser.add_argument(
        "--hidden-dim", type=int, default=8192, help="FFN hidden dimension"
    )

    # Input shape
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)

    # MoE config
    parser.add_argument(
        "--n-replicas", type=int, default=2, help="Number of FFN replicas"
    )
    parser.add_argument("--n-groups", type=int, default=64, help="Groups per replica")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k routing. Defaults to n-groups (full replica)",
    )

    # Profiling config
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=3,
        help="Profile iterations (default: 3)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Derived values
    num_experts = args.n_replicas * args.n_groups
    top_k = args.top_k or args.n_groups
    moe_inter_dim = args.hidden_dim // args.n_groups
    tokens = args.batch_size * args.seq_len

    # Input tensor (same shape for both layers)
    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # Create selected layer
    if args.layer == "ffn":
        layer = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim).cuda().bfloat16()
    else:
        moe_args = MoEArgs(
            num_experts=num_experts,
            num_shared_experts=0,
            top_k=top_k,
            score_func="softmax",
            route_norm=True,
            route_scale=args.n_groups,
            score_before_experts=False,
            use_grouped_mm=True,
        )
        layer = MoE(moe_args, dim=args.dim, hidden_dim=moe_inter_dim).cuda().bfloat16()

    # Warmup
    # Use .mean() not .sum() - sonic-moe's CuTe kernels reject stride-0 broadcast
    # tensors that .sum().backward() creates
    for _ in range(args.warmup):
        layer(x).mean().backward()
        x.grad = None
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(args.profile_iters):
            layer(x).mean().backward()
            x.grad = None
        torch.cuda.synchronize()

    # Print header
    print("FFN/MoE Layer Profile")
    print("=" * 80)
    print(f"Layer: {args.layer}")
    print(
        f"Config: dim={args.dim}, hidden_dim={args.hidden_dim}, "
        f"batch={args.batch_size}, seq={args.seq_len}"
    )
    if args.layer == "moe":
        print(
            f"MoE: n_replicas={args.n_replicas}, n_groups={args.n_groups} "
            f"({num_experts} experts), top_k={top_k}"
        )
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Print kernels by op
    print("CUDA Kernels by PyTorch Op:")
    print("=" * 80)
    for event in prof.events():
        if event.kernels:
            print(f"\n{event.name}:")
            for kernel in event.kernels:
                print(f"  {kernel.name:<60} {kernel.duration:>8.1f} us")
    print()

    # Print summary table
    print("PyTorch Op Summary (sorted by CUDA time):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
