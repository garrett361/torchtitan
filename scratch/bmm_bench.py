"""Benchmark token combining methods for MoE."""

import argparse

import torch
from triton.testing import do_bench

from impls import DEEPSEEK_CONFIGS, IMPLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bmm"],
        choices=list(IMPLS.keys()),
        help=f"Combine methods. Choices: {list(IMPLS.keys())}. Default: bmm",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEEPSEEK_CONFIGS.keys()),
        choices=list(DEEPSEEK_CONFIGS.keys()),
        help=f"DeepSeek model configs. Default: all ({list(DEEPSEEK_CONFIGS.keys())})",
    )
    parser.add_argument(
        "--tokens", "-t", type=int, default=65536, help="Number of tokens. Default: 65536"
    )
    return parser.parse_args()


def print_table(
    results: dict[str, dict[str, tuple[float, float]]], methods: list[str]
) -> None:
    """Print benchmark results as markdown table with ratios to baseline."""
    baseline = methods[0]
    base_abbrev = "bcast" if baseline == "broadcast_sum" else baseline

    def abbrev(m: str) -> str:
        return "bcast" if m == "broadcast_sum" else m

    # Build header: baseline fwd ms, ratios for fwd, baseline bwd ms, ratios for bwd
    header_parts = ["Model", f"{base_abbrev} fwd ms"]
    for method in methods[1:]:
        header_parts.append(f"{abbrev(method)}/{base_abbrev} fwd")
    header_parts.append(f"{base_abbrev} fwd+bwd ms")
    for method in methods[1:]:
        header_parts.append(f"{abbrev(method)}/{base_abbrev} fwd+bwd")

    # Build separator
    sep_parts = ["---"] * len(header_parts)

    # Build rows
    rows = []
    for model, method_results in results.items():
        base_fwd, base_fwd_bwd = method_results[baseline]
        row = [model, f"{base_fwd:.2f}"]
        for method in methods[1:]:
            fwd_ms, _ = method_results[method]
            ratio = fwd_ms / base_fwd if base_fwd > 0 else 0
            row.append(f"{ratio:.2f}x")
        row.append(f"{base_fwd_bwd:.2f}")
        for method in methods[1:]:
            _, fwd_bwd_ms = method_results[method]
            ratio = fwd_bwd_ms / base_fwd_bwd if base_fwd_bwd > 0 else 0
            row.append(f"{ratio:.2f}x")
        rows.append(row)

    # Print table
    print(f"| {' | '.join(header_parts)} |")
    print(f"| {' | '.join(sep_parts)} |")
    for row in rows:
        print(f"| {' | '.join(row)} |")


def main():
    args = parse_args()
    tokens = args.tokens
    device = torch.device("cuda")

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Tokens: {tokens}")
    print()

    # {model: {method: (fwd_ms, fwd_bwd_ms)}}
    results: dict[str, dict[str, tuple[float, float]]] = {}

    for model in args.models:
        config = DEEPSEEK_CONFIGS[model]
        top_k = config["top_k"]
        dim = config["dim"]

        # Create tensors once per model
        top_scores = torch.randn(
            tokens, top_k, device=device, dtype=torch.float32, requires_grad=True
        )
        routed_output = torch.randn(
            tokens, top_k, dim, device=device, dtype=torch.float32, requires_grad=True
        )

        results[model] = {}

        for method in args.methods:
            combine_fn = IMPLS[method]

            def fwd():
                return combine_fn(top_scores, routed_output)

            def fwd_bwd():
                out = combine_fn(top_scores, routed_output)
                out.sum().backward()
                top_scores.grad = None
                routed_output.grad = None

            fwd_ms = do_bench(fwd)
            fwd_bwd_ms = do_bench(fwd_bwd)
            results[model][method] = (fwd_ms, fwd_bwd_ms)

    print_table(results, args.methods)


if __name__ == "__main__":
    main()
