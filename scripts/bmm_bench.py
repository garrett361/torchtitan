"""Benchmark torch.bmm forward and fwd+bwd for MoE token combining."""

import argparse

import torch
from triton.testing import do_bench


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
    return parser.parse_args()


def main():
    args = parse_args()
    N, K, D = args.n, args.k, args.d

    device = torch.device("cuda")

    # A: (N, 1, K) top_scores, B: (N, K, D) routed_output
    A = torch.randn(N, 1, K, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(N, K, D, device=device, dtype=torch.float32, requires_grad=True)

    def fwd():
        return torch.bmm(A, B)

    def fwd_bwd():
        C = torch.bmm(A, B)
        C.sum().backward()
        A.grad = None
        B.grad = None

    # do_bench returns (median, min, max) in ms when quantiles provided
    fwd_ms = do_bench(fwd, quantiles=[0.5, 0.0, 1.0])
    fwd_bwd_ms = do_bench(fwd_bwd, quantiles=[0.5, 0.0, 1.0])

    # FLOPs calculation
    # Forward: 2 * N * 1 * K * D (standard matmul)
    # Backward: ~2x forward (dA and dB gradients)
    fwd_flops = 2 * N * 1 * K * D
    bwd_flops = 2 * fwd_flops  # approximate
    total_flops = fwd_flops + bwd_flops

    def tflops(flops, ms):
        return (flops / 1e12) / (ms / 1e3) if ms > 0 else 0

    print(f"BMM Shapes: A=({N}, 1, {K}), B=({N}, {K}, {D})")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print(f"{'Operation':<12} {'Median (ms)':>12} {'Min (ms)':>10} {'Max (ms)':>10} {'TFLOP/s':>10}")
    print("-" * 56)
    print(f"{'forward':<12} {fwd_ms[0]:>12.3f} {fwd_ms[1]:>10.3f} {fwd_ms[2]:>10.3f} {tflops(fwd_flops, fwd_ms[0]):>10.2f}")
    print(f"{'fwd+bwd':<12} {fwd_bwd_ms[0]:>12.3f} {fwd_bwd_ms[1]:>10.3f} {fwd_bwd_ms[2]:>10.3f} {tflops(total_flops, fwd_bwd_ms[0]):>10.2f}")


if __name__ == "__main__":
    main()
