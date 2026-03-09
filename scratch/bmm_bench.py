"""Benchmark token combining impls for MoE."""

import argparse

import torch
from impls import IMPLS, MODEL_CFGS
from triton.testing import do_bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--impls",
        nargs="+",
        default=list(IMPLS.keys()),
        choices=list(IMPLS.keys()),
        help=f"Impls Choices: {list(IMPLS.keys())}. Default: bmm",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CFGS.keys()),
        choices=list(MODEL_CFGS.keys()),
        help=f"DeepSeek model configs. Default: all ({list(MODEL_CFGS.keys())})",
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=int,
        default=65536,
        help="Number of tokens. Default: 65536",
    )
    return parser.parse_args()


def print_tables(
    results: dict[str, dict[str, tuple[float, float]]], impls: list[str]
) -> None:
    """Print benchmark results as two markdown tables (fwd and fwd+bwd)."""
    baseline = impls[0]

    def print_single_table(title: str, idx: int) -> None:
        suffix = "fwd" if idx == 0 else "fwd+bwd"
        header = ["Model", f"{baseline} {suffix} ms"]
        for method in impls[1:]:
            header.append(f"{method}/{baseline}")
        sep = ["---"] * len(header)

        print(f"### {title}")
        print(f"| {' | '.join(header)} |")
        print(f"| {' | '.join(sep)} |")
        for model, method_results in results.items():
            base_val = method_results[baseline][idx]
            row = [model, f"{base_val:.2f}"]
            for method in impls[1:]:
                val = method_results[method][idx]
                ratio = val / base_val if base_val > 0 else 0
                rocket = " 🚀" if ratio < 0.999 else ""
                row.append(f"{ratio:.2f}x{rocket}")
            print(f"| {' | '.join(row)} |")
        print()

    print_single_table("Forward", 0)
    print_single_table("Forward + Backward", 1)


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
        config = MODEL_CFGS[model]
        top_k = config["top_k"]
        dim = config["dim"]

        # Create tensors once per model. For the MoE BMM setup the scores are in float32 and the
        # routed_output is usually in low-precision.
        top_scores = torch.randn( 
            tokens, top_k, device=device, dtype=torch.float32, requires_grad=True
        )
        routed_output = torch.randn(
            tokens, top_k, dim, device=device, dtype=torch.bfloat16, requires_grad=True
        )

        results[model] = {}

        for impl_str in args.impls:
            impl = IMPLS[impl_str]

            def fwd():
                return impl(top_scores, routed_output)

            def fwd_bwd():
                out = impl(top_scores, routed_output)
                out.sum().backward()
                top_scores.grad = None
                routed_output.grad = None

            fwd_ms = do_bench(fwd)
            fwd_bwd_ms = do_bench(fwd_bwd)
            results[model][impl_str] = (fwd_ms, fwd_bwd_ms)

    print_tables(results, args.impls)


if __name__ == "__main__":
    main()
