"""Profile torch.bmm kernel dispatch for MoE token combining."""

import argparse

import torch
from torch.profiler import profile, ProfilerActivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=8192, help="Batch size (bs * seqlen). Default: 8192"
    )
    parser.add_argument(
        "--k", type=int, default=6, help="Top-k experts. Default: 6"
    )
    parser.add_argument(
        "--d", type=int, default=2048, help="Hidden dimension. Default: 2048"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations. Default: 5"
    )
    parser.add_argument(
        "--profile-iters", type=int, default=3, help="Profile iterations. Default: 3"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    N, K, D = args.n, args.k, args.d

    device = torch.device("cuda")

    # A: (N, 1, K) top_scores, B: (N, K, D) routed_output
    A = torch.randn(N, 1, K, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(N, K, D, device=device, dtype=torch.float32, requires_grad=True)

    # Warmup
    for _ in range(args.warmup):
        C = torch.bmm(A, B)
        C.sum().backward()
        A.grad = None
        B.grad = None

    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(args.profile_iters):
            C = torch.bmm(A, B)
            C.sum().backward()
            A.grad = None
            B.grad = None
        torch.cuda.synchronize()

    # Print CUDA kernels from FunctionEvent.kernels
    print(f"BMM Shapes: A=({N}, 1, {K}), B=({N}, {K}, {D})")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print("CUDA Kernels by PyTorch Op:")
    print("=" * 100)
    for event in prof.events():
        if event.kernels:
            print(f"\n{event.name}:")
            for kernel in event.kernels:
                print(f"  {kernel.name:<70} {kernel.duration:>8.1f} us")
    print()

    # Also print summary table
    print("PyTorch Op Summary (sorted by CUDA time):")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
