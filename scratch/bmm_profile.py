"""Profile kernel dispatch for MoE token combining methods."""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

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
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations. Default: 5"
    )
    parser.add_argument(
        "--profile-iters", type=int, default=3, help="Profile iterations. Default: 3"
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

    # Warmup
    for _ in range(args.warmup):
        out = combine_fn(top_scores, routed_output)
        out.sum().backward()
        top_scores.grad = None
        routed_output.grad = None

    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(args.profile_iters):
            out = combine_fn(top_scores, routed_output)
            out.sum().backward()
            top_scores.grad = None
            routed_output.grad = None
        torch.cuda.synchronize()

    print(f"Method: {args.method}")
    print(f"Shapes: top_scores=({tokens}, {top_k}), routed_output=({tokens}, {top_k}, {dim})")
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

    print("PyTorch Op Summary (sorted by CUDA time):")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
