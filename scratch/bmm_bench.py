"""Benchmark token combining methods for MoE."""

import argparse

import torch
from triton.testing import do_bench

from impls import DEEPSEEK_CONFIGS, IMPLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        type=str,
        default="bmm",
        choices=list(IMPLS.keys()),
        help=f"Combine method. Choices: {list(IMPLS.keys())}. Default: bmm",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DEEPSEEK_CONFIGS.keys()),
        help="DeepSeek model config. Sets top_k and dim.",
    )
    parser.add_argument(
        "--tokens", "-t", type=int, default=65536, help="Number of tokens. Default: 65536"
    )
    parser.add_argument(
        "--top_k", "-k", type=int, default=None, help="Top-k experts."
    )
    parser.add_argument(
        "--dim", "-d", type=int, default=None, help="Model dimension."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokens = args.tokens
    combine_fn = IMPLS[args.method]

    # Resolve top_k and dim from model config or defaults
    if args.model:
        config = DEEPSEEK_CONFIGS[args.model]
        top_k = config["top_k"]
        dim = config["dim"]

        # Warn if CLI overrides model config
        if args.top_k is not None and args.top_k != top_k:
            print(f"Warning: --top_k overrides model {args.model} config ({top_k} -> {args.top_k})")
            top_k = args.top_k
        if args.dim is not None and args.dim != dim:
            print(f"Warning: --dim overrides model {args.model} config ({dim} -> {args.dim})")
            dim = args.dim
    else:
        # Defaults when no model specified
        top_k = args.top_k if args.top_k is not None else 6
        dim = args.dim if args.dim is not None else 2048

    device = torch.device("cuda")

    # top_scores: (tokens, top_k), routed_output: (tokens, top_k, dim)
    top_scores = torch.randn(tokens, top_k, device=device, dtype=torch.float32, requires_grad=True)
    routed_output = torch.randn(
        tokens, top_k, dim, device=device, dtype=torch.float32, requires_grad=True
    )

    def fwd():
        return combine_fn(top_scores, routed_output)

    def fwd_bwd():
        out = combine_fn(top_scores, routed_output)
        out.sum().backward()
        top_scores.grad = None
        routed_output.grad = None

    fwd_ms = do_bench(fwd, quantiles=[0.5, 0.0, 1.0])
    fwd_bwd_ms = do_bench(fwd_bwd, quantiles=[0.5, 0.0, 1.0])

    # FLOPs: tokens * top_k * dim multiplies + tokens * top_k * dim adds
    fwd_flops = 2 * tokens * top_k * dim
    bwd_flops = 2 * fwd_flops
    total_flops = fwd_flops + bwd_flops

    def tflops(flops, ms):
        return (flops / 1e12) / (ms / 1e3) if ms > 0 else 0

    print(f"Method: {args.method}")
    print(f"Shapes: top_scores=({tokens}, {top_k}), routed_output=({tokens}, {top_k}, {dim})")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print(
        f"{'Operation':<12} {'Median (ms)':>12} {'Min (ms)':>10} {'Max (ms)':>10} {'TFLOP/s':>10}"
    )
    print("-" * 56)
    print(
        f"{'forward':<12} {fwd_ms[0]:>12.3f} {fwd_ms[1]:>10.3f} {fwd_ms[2]:>10.3f} {tflops(fwd_flops, fwd_ms[0]):>10.2f}"
    )
    print(
        f"{'fwd+bwd':<12} {fwd_bwd_ms[0]:>12.3f} {fwd_bwd_ms[1]:>10.3f} {fwd_bwd_ms[2]:>10.3f} {tflops(total_flops, fwd_bwd_ms[0]):>10.2f}"
    )


if __name__ == "__main__":
    main()
