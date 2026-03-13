"""Benchmark comparing dense FFN vs MoE layer performance.

Isolates layer-level overhead from full-model effects to diagnose whether the MoE layer
itself is the bottleneck in throughput reduction vs dense baseline.
"""

import argparse

import torch
from triton.testing import do_bench

from torchtitan.models.moe import FeedForward, MoE, MoEArgs
from torchtitan.models.sonic_moe import SONIC_MOE_AVAILABLE, SonicMoE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

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
    parser.add_argument(
        "--n-groups", type=int, default=64, help="Groups per replica"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k routing. Defaults to n-groups (full replica)",
    )
    parser.add_argument(
        "--sonic",
        action="store_true",
        help="Benchmark SonicMoE (requires sonic-moe package)",
    )

    return parser.parse_args()


def ffn_flops(tokens: int, dim: int, hidden_dim: int) -> int:
    """FLOPs for SwiGLU FFN forward pass.

    w1: (tokens, dim) @ (dim, hidden_dim) = 2 * tokens * dim * hidden_dim
    w3: (tokens, dim) @ (dim, hidden_dim) = 2 * tokens * dim * hidden_dim
    w2: (tokens, hidden_dim) @ (hidden_dim, dim) = 2 * tokens * hidden_dim * dim
    Total: 6 * tokens * dim * hidden_dim
    """
    return 6 * tokens * dim * hidden_dim


def moe_flops(tokens: int, dim: int, moe_inter_dim: int, top_k: int) -> int:
    """FLOPs for MoE forward pass.

    Router: (tokens, dim) @ (dim, num_experts) - negligible vs experts

    Each token activates top_k experts, each expert is an FFN:
    Per expert: 6 * 1 * dim * moe_inter_dim (for one token)
    Total: 6 * tokens * top_k * dim * moe_inter_dim
    """
    return 6 * tokens * top_k * dim * moe_inter_dim


def compute_tflops(flops: int, time_ms: float) -> float:
    """Convert FLOPs and time to TFLOPS."""
    return flops / (time_ms * 1e9)  # ms -> s, then / 1e12 for tera


def bench_layer(
    layer: torch.nn.Module, x: torch.Tensor
) -> tuple[float, float]:
    """Benchmark forward and forward+backward timing using Triton's do_bench."""

    def fwd():
        layer(x)

    def fwd_bwd():
        # Use .mean() not .sum() - sonic-moe's CuTe kernels reject stride-0 broadcast
        # tensors that .sum().backward() creates
        layer(x).mean().backward()
        x.grad = None

    fwd_ms = do_bench(fwd)
    fwd_bwd_ms = do_bench(fwd_bwd)
    return fwd_ms, fwd_bwd_ms


def profile_memory(layer: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Profile peak memory usage for forward+backward pass.

    Returns (peak_allocated_gib, peak_reserved_gib).
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    layer(x).mean().backward()
    x.grad = None
    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

    return peak_allocated, peak_reserved


def main():
    args = parse_args()

    # Derived values
    num_experts = args.n_replicas * args.n_groups
    top_k = args.top_k or args.n_groups  # Full replica by default
    moe_inter_dim = args.hidden_dim // args.n_groups
    tokens = args.batch_size * args.seq_len

    # Create layers
    ffn = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim).cuda().bfloat16()

    moe_args = MoEArgs(
        num_experts=num_experts,
        num_shared_experts=0,
        top_k=top_k,
        score_func="softmax",
        route_norm=True,
        route_scale=args.n_groups,
        score_before_experts=False,  # Required for virtual_group
        use_grouped_mm=True,
    )
    moe = MoE(moe_args, dim=args.dim, hidden_dim=moe_inter_dim).cuda().bfloat16()

    # Create SonicMoE if requested
    sonic_moe = None
    if args.sonic:
        if not SONIC_MOE_AVAILABLE:
            print("Warning: --sonic requested but sonic-moe not available, skipping")
        else:
            sonic_moe = SonicMoE.from_moe(moe).cuda().bfloat16()

    # Input tensor (same 3D shape for both layers)
    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # Profile memory first (before timing to avoid interference)
    ffn_peak, ffn_reserved = profile_memory(ffn, x)
    moe_peak, moe_reserved = profile_memory(moe, x)
    sonic_peak, sonic_reserved = (
        profile_memory(sonic_moe, x) if sonic_moe else (None, None)
    )

    # Compute FLOPs
    ffn_f = ffn_flops(tokens, args.dim, args.hidden_dim)
    moe_f = moe_flops(tokens, args.dim, moe_inter_dim, top_k)

    # Print config
    print("FFN vs MoE Layer Benchmark")
    print("==========================")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(
        f"Config: dim={args.dim}, hidden_dim={args.hidden_dim}, "
        f"batch={args.batch_size}, seq={args.seq_len}"
    )
    print(
        f"MoE: n_replicas={args.n_replicas}, n_groups={args.n_groups} "
        f"({num_experts} experts), top_k={top_k}"
    )
    print(f"Tokens: {tokens:,}")
    print()
    print(
        f"FFN FLOPs:  {ffn_f / 1e9:.1f} GFLOPs "
        f"(6 * {tokens} * {args.dim} * {args.hidden_dim})"
    )
    print(
        f"MoE FLOPs:  {moe_f / 1e9:.1f} GFLOPs "
        f"(6 * {tokens} * {top_k} * {args.dim} * {moe_inter_dim})"
    )
    print()

    # Memory results
    print("### Memory (Forward + Backward)")
    print("| Layer | Peak (GiB) | Reserved (GiB) |")
    print("|-------|------------|----------------|")
    print(f"| FFN   | {ffn_peak:.2f}       | {ffn_reserved:.2f}           |")
    print(f"| MoE   | {moe_peak:.2f}       | {moe_reserved:.2f}           |")
    if sonic_peak is not None:
        print(f"| Sonic | {sonic_peak:.2f}       | {sonic_reserved:.2f}           |")
    print()

    # Benchmark FFN
    ffn_fwd_ms, ffn_fwd_bwd_ms = bench_layer(ffn, x)
    del ffn
    torch.cuda.empty_cache()

    # Benchmark MoE
    moe_fwd_ms, moe_fwd_bwd_ms = bench_layer(moe, x)
    del moe
    torch.cuda.empty_cache()

    # Benchmark SonicMoE
    sonic_fwd_ms, sonic_fwd_bwd_ms = None, None
    if sonic_moe is not None:
        sonic_fwd_ms, sonic_fwd_bwd_ms = bench_layer(sonic_moe, x)
        del sonic_moe
        torch.cuda.empty_cache()

    # Forward results
    print("### Forward")
    print("| Layer | ms    | TFLOPS | vs FFN (TFLOPS) |")
    print("|-------|-------|--------|-----------------|")
    ffn_tflops = compute_tflops(ffn_f, ffn_fwd_ms)
    moe_tflops = compute_tflops(moe_f, moe_fwd_ms)
    print(f"| FFN   | {ffn_fwd_ms:.2f}  | {ffn_tflops:.1f}  | 1.00x   |")
    print(f"| MoE   | {moe_fwd_ms:.2f}  | {moe_tflops:.1f}  | {moe_tflops / ffn_tflops:.2f}x   |")
    if sonic_fwd_ms is not None:
        sonic_tflops = compute_tflops(moe_f, sonic_fwd_ms)
        print(f"| Sonic | {sonic_fwd_ms:.2f}  | {sonic_tflops:.1f}  | {sonic_tflops / ffn_tflops:.2f}x   |")
    print()

    # Forward + backward results (3x FLOPs for fwd+bwd)
    print("### Forward + Backward")
    print("| Layer | ms    | TFLOPS | vs FFN (TFLOPS) |")
    print("|-------|-------|--------|-----------------|")
    ffn_fwd_bwd_tflops = compute_tflops(3 * ffn_f, ffn_fwd_bwd_ms)
    moe_fwd_bwd_tflops = compute_tflops(3 * moe_f, moe_fwd_bwd_ms)
    print(f"| FFN   | {ffn_fwd_bwd_ms:.2f}  | {ffn_fwd_bwd_tflops:.1f}  | 1.00x   |")
    print(f"| MoE   | {moe_fwd_bwd_ms:.2f}  | {moe_fwd_bwd_tflops:.1f}  | {moe_fwd_bwd_tflops / ffn_fwd_bwd_tflops:.2f}x   |")
    if sonic_fwd_bwd_ms is not None:
        sonic_fwd_bwd_tflops = compute_tflops(3 * moe_f, sonic_fwd_bwd_ms)
        print(f"| Sonic | {sonic_fwd_bwd_ms:.2f}  | {sonic_fwd_bwd_tflops:.1f}  | {sonic_fwd_bwd_tflops / ffn_fwd_bwd_tflops:.2f}x   |")


if __name__ == "__main__":
    main()
