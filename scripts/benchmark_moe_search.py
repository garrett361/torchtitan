#!/usr/bin/env python3
"""
Grid search over MoE benchmark configurations.

Usage:
    python scripts/benchmark_moe_search.py --tag ng64 --n-groups 64 --n-moe-layers 14
"""

import argparse
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE Benchmark Grid Search")

    # Search identification
    parser.add_argument(
        "--tag", type=str, required=True, help="Tag for this search (e.g., ng64)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )

    # Pass-through args (fixed for the search)
    parser.add_argument("--flavor", type=str, default="1B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-groups", type=int, default=64)
    parser.add_argument("--n-moe-layers", type=int, default=14)
    parser.add_argument("--n-replicas", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--force-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)

    # Grid dimensions (comma-separated)
    parser.add_argument("--ep", type=str, default="1,8", help="EP degrees to search")
    parser.add_argument(
        "--ac-mode",
        type=str,
        default="none,selective,full",
        help="AC modes to search",
    )
    parser.add_argument(
        "--ac-option", type=str, default="1,2,op", help="AC options to search"
    )
    parser.add_argument(
        "--moe-reshard-after-forward",
        type=str,
        default="true,false",
        help="Reshard modes to search",
    )

    # Runtime options
    parser.add_argument("--nproc-per-node", type=int, default=8, help="GPUs per node")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs without running",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run dense baseline (n-moe-layers=0) for comparison",
    )
    parser.add_argument(
        "--custom-moe-impl",
        type=str,
        default="virtual_group,sonic_virtual_group",
    )

    return parser.parse_args()


@dataclass
class SearchConfig:
    """A single configuration to benchmark."""

    ep: int
    ac_mode: str
    ac_option: str | None
    moe_reshard_after_forward: bool
    custom_moe_impl: str | None

    def is_valid(self) -> bool:
        """Check if this config combination is valid."""
        # ac_option only matters for selective mode
        if self.ac_mode != "selective" and self.ac_option is not None:
            return False
        if self.ac_mode == "selective" and self.ac_option is None:
            return False
        if (
            self.custom_moe_impl is not None
            and ("sonic" in self.custom_moe_impl)
            and self.ep > 1
        ):
            return False
        return True

def generate_grid(args: argparse.Namespace) -> list[SearchConfig]:
    """Generate all valid config combinations from CLI args."""
    ep_values = [int(x) for x in args.ep.split(",")]
    ac_modes = args.ac_mode.split(",")
    ac_options = args.ac_option.split(",")
    custom_moe_impls = args.custom_moe_impl.split(",")
    reshard_values = [
        x.lower() == "true" for x in args.moe_reshard_after_forward.split(",")
    ]

    configs = []
    for ep in ep_values:
        # moe_reshard_after_forward only affects EP > 1 (separate expert FSDP wrapping)
        # For EP=1, skip reshard variations to avoid redundant runs
        ep_reshard_values = reshard_values if ep > 1 else [False]

        for moe_impl in custom_moe_impls:
            for ac_mode in ac_modes:
                for reshard in ep_reshard_values:
                    if ac_mode == "selective":
                        for ac_opt in ac_options:
                            cfg = SearchConfig(ep, ac_mode, ac_opt, reshard, moe_impl)
                            if cfg.is_valid():
                                configs.append(cfg)
                    else:
                        cfg = SearchConfig(ep, ac_mode, None, reshard, moe_impl)
                        if cfg.is_valid():
                            configs.append(cfg)

    return configs


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    config: SearchConfig
    status: str  # "success", "oom", "error", "skipped"
    error: str | None
    metrics: dict | None
    model_info: dict | None


def run_benchmark(
    args: argparse.Namespace,
    config: SearchConfig,
    tmp_dir: Path,
) -> BenchmarkResult:
    """Run a single benchmark config via torchrun subprocess."""
    result_file = tmp_dir / f"result_{uuid.uuid4().hex}.json"

    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        "scripts/benchmark_moe.py",
        f"--flavor={args.flavor}",
        f"--batch-size={args.batch_size}",
        f"--seq-len={args.seq_len}",
        f"--n-groups={args.n_groups}",
        f"--n-moe-layers={args.n_moe_layers}",
        f"--n-replicas={args.n_replicas}",
        f"--warmup-iters={args.warmup_iters}",
        f"--bench-iters={args.bench_iters}",
        f"--ep={config.ep}",
        f"--ac-mode={config.ac_mode}",
        f"--output-json={result_file}",
        f"--custom-moe-impl={config.custom_moe_impl}",
    ]

    if args.top_k is not None:
        cmd.append(f"--top-k={args.top_k}")

    if args.force_balance:
        cmd.append("--force-balance")
    else:
        cmd.append("--no-force-balance")

    if config.ac_option:
        cmd.append(f"--ac-option={config.ac_option}")

    if config.moe_reshard_after_forward:
        cmd.append("--moe-reshard-after-forward")
    else:
        cmd.append("--no-moe-reshard-after-forward")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            # Check for OOM
            stderr = result.stderr.lower()
            if "out of memory" in stderr:
                return BenchmarkResult(
                    config=config,
                    status="oom",
                    error="CUDA out of memory",
                    metrics=None,
                    model_info=None,
                )
            else:
                return BenchmarkResult(
                    config=config,
                    status="error",
                    error=result.stderr if result.stderr else "Unknown error",
                    metrics=None,
                    model_info=None,
                )

        # Read JSON result
        if not result_file.exists():
            return BenchmarkResult(
                config=config,
                status="error",
                error="Result file not created",
                metrics=None,
                model_info=None,
            )

        with open(result_file) as f:
            data = json.load(f)

        return BenchmarkResult(
            config=config,
            status=data.get("status", "success"),
            error=data.get("error"),
            metrics=data.get("metrics"),
            model_info=data.get("model_info"),
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            config=config,
            status="error",
            error="Timeout (600s)",
            metrics=None,
            model_info=None,
        )
    except Exception as e:
        return BenchmarkResult(
            config=config,
            status="error",
            error=str(e),
            metrics=None,
            model_info=None,
        )
    finally:
        if result_file.exists():
            result_file.unlink()


@dataclass
class BaselineResult:
    """Result from baseline (dense) benchmark run."""

    status: str  # "success", "oom", "error"
    error: str | None
    metrics: dict | None
    model_info: dict | None


def run_baseline(args: argparse.Namespace, tmp_dir: Path) -> BaselineResult:
    """Run dense baseline (n-moe-layers=0) with selective AC."""
    result_file = tmp_dir / f"result_{uuid.uuid4().hex}.json"

    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        "scripts/benchmark_moe.py",
        f"--flavor={args.flavor}",
        f"--batch-size={args.batch_size}",
        f"--seq-len={args.seq_len}",
        "--n-moe-layers=0",  # Dense baseline
        f"--warmup-iters={args.warmup_iters}",
        f"--bench-iters={args.bench_iters}",
        "--ac-mode=none",
        f"--output-json={result_file}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            stderr = result.stderr.lower()
            if "out of memory" in stderr or "cuda error" in stderr:
                return BaselineResult(
                    status="oom",
                    error="CUDA out of memory",
                    metrics=None,
                    model_info=None,
                )
            else:
                return BaselineResult(
                    status="error",
                    error=result.stderr[:500] if result.stderr else "Unknown error",
                    metrics=None,
                    model_info=None,
                )

        if not result_file.exists():
            return BaselineResult(
                status="error",
                error="Result file not created",
                metrics=None,
                model_info=None,
            )

        with open(result_file) as f:
            data = json.load(f)

        return BaselineResult(
            status=data.get("status", "success"),
            error=data.get("error"),
            metrics=data.get("metrics"),
            model_info=data.get("model_info"),
        )

    except subprocess.TimeoutExpired:
        return BaselineResult(
            status="error",
            error="Timeout (600s)",
            metrics=None,
            model_info=None,
        )
    except Exception as e:
        return BaselineResult(
            status="error",
            error=str(e),
            metrics=None,
            model_info=None,
        )
    finally:
        if result_file.exists():
            result_file.unlink()


def write_jsonl_metadata(
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Write metadata header to JSONL file."""
    metadata = {
        "type": "metadata",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "base_config": {
            "flavor": args.flavor,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_groups": args.n_groups,
            "n_moe_layers": args.n_moe_layers,
            "n_replicas": args.n_replicas,
            "top_k": args.top_k,
            "force_balance": args.force_balance,
            "warmup_iters": args.warmup_iters,
            "bench_iters": args.bench_iters,
        },
    }
    with open(path, "w") as f:
        f.write(json.dumps(metadata) + "\n")


def write_jsonl_result(
    path: Path,
    result: BenchmarkResult,
    is_baseline: bool = False,
) -> None:
    """Append a single result to JSONL file."""
    record = {
        "type": "result",
        "is_baseline": is_baseline,
        "config": {
            "ep": result.config.ep,
            "ac_mode": result.config.ac_mode,
            "ac_option": result.config.ac_option,
            "moe_reshard_after_forward": result.config.moe_reshard_after_forward,
            "custom_moe_impl": result.config.custom_moe_impl,
        },
        "status": result.status,
        "error": result.error,
        "metrics": result.metrics,
        "model_info": result.model_info,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_jsonl_baseline(path: Path, result: BaselineResult) -> None:
    """Append baseline result to JSONL file."""
    record = {
        "type": "result",
        "is_baseline": True,
        "config": {
            "ep": None,
            "ac_mode": "selective",
            "ac_option": "2",
            "moe_reshard_after_forward": None,
        },
        "status": result.status,
        "error": result.error,
        "metrics": result.metrics,
        "model_info": result.model_info,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def generate_report(
    jsonl_path: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    """Generate markdown report from JSONL results."""
    # Read all results
    metadata = None
    baseline: dict | None = None
    results: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "metadata":
                metadata = record
            elif record["type"] == "result":
                if record.get("is_baseline"):
                    baseline = record
                else:
                    results.append(record)

    # Sort by TPS/GPU (successful runs first, then by TPS descending)
    def sort_key(r):
        if r["status"] != "success" or r["metrics"] is None:
            return (1, 0)  # Failed runs at end
        return (0, -r["metrics"]["tps_per_gpu"])

    results.sort(key=sort_key)

    # Get baseline TPS for comparison
    baseline_tps: float | None = None
    if baseline and baseline["status"] == "success" and baseline["metrics"]:
        baseline_tps = baseline["metrics"]["tps_per_gpu"]

    # Find best config
    best = next((r for r in results if r["status"] == "success"), None)

    # Generate report
    lines = []
    lines.append(f"# MoE Throughput Search: {args.tag}\n")
    lines.append(f"**Date:** {metadata['timestamp']}")
    lines.append(f"**Git commit:** {metadata['git_commit']}")

    base = metadata["base_config"]
    lines.append(
        f"**Base config:** flavor={base['flavor']}, batch_size={base['batch_size']}, "
        f"seq_len={base['seq_len']}, n_groups={base['n_groups']}, "
        f"n_moe_layers={base['n_moe_layers']}\n"
    )

    # Baseline section
    if baseline:
        lines.append("## Baseline (Dense)\n")
        if baseline["status"] == "success" and baseline["metrics"]:
            m = baseline["metrics"]
            lines.append(
                f"**TPS/GPU:** {m['tps_per_gpu']:,.0f} | "
                f"**Memory:** {m['peak_active_gib']:.1f} GiB\n"
            )
        else:
            lines.append(
                f"**Status:** {baseline['status'].upper()} - {baseline['error']}\n"
            )

    # Best configuration
    lines.append("## Best Configuration\n")
    if best:
        cfg = best["config"]
        ac_opt_str = f" --ac-option {cfg['ac_option']}" if cfg["ac_option"] else ""
        reshard_flag = (
            "--moe-reshard-after-forward"
            if cfg["moe_reshard_after_forward"]
            else "--no-moe-reshard-after-forward"
        )
        balance_flag = (
            "--force-balance"
            if base.get("force_balance", True)
            else "--no-force-balance"
        )
        lines.append("```bash")
        lines.append(
            f"torchrun --nproc_per_node=8 scripts/benchmark_moe.py \\\n"
            f"    --flavor {base['flavor']} --batch-size {base['batch_size']} "
            f"--seq-len {base['seq_len']} \\\n"
            f"    --n-groups {base['n_groups']} --n-moe-layers {base['n_moe_layers']} "
            f"{balance_flag} \\\n"
            f"    --ep {cfg['ep']} --ac-mode {cfg['ac_mode']}{ac_opt_str} {reshard_flag} \\\n"
            f"    --custom-moe-impl {cfg['custom_moe_impl']} "
        )
        lines.append("```\n")
        m = best["metrics"]
        lines.append(
            f"**TPS/GPU:** {m['tps_per_gpu']:,.0f} | "
            f"**Memory:** {m['peak_active_gib']:.1f} GiB active / "
            f"{m['peak_reserved_gib']:.1f} GiB reserved\n"
        )
    else:
        lines.append("No successful configurations.\n")

    # Results table - sort by TPS/GPU
    lines.append("## All Results (by TPS/GPU)\n")
    if baseline_tps:
        lines.append(
            "| Rank | EP | AC Mode | AC Opt | Reshard | TPS/GPU | vs Baseline | "
            "Active GiB | Reserved GiB | Status |"
        )
        lines.append(
            "|------|----|---------|--------|---------|---------|-------------|"
            "------------|--------------|--------|"
        )
    else:
        lines.append(
            "| Rank | EP | AC Mode | AC Opt | Reshard | TPS/GPU | "
            "Active GiB | Reserved GiB | Status |"
        )
        lines.append(
            "|------|----|---------|--------|---------|---------|"
            "------------|--------------|--------|"
        )

    for i, r in enumerate(results, 1):
        cfg = r["config"]
        ac_opt = cfg["ac_option"] or "-"
        reshard = "yes" if cfg["moe_reshard_after_forward"] else "no"

        if r["status"] == "success" and r["metrics"]:
            m = r["metrics"]
            if baseline_tps:
                pct_diff = (m["tps_per_gpu"] - baseline_tps) / baseline_tps * 100
                vs_baseline = f"{pct_diff:+.0f}%"
                lines.append(
                    f"| {i} | {cfg['ep']} | {cfg['ac_mode']} | {ac_opt} | {reshard} | "
                    f"{m['tps_per_gpu']:,.0f} | {vs_baseline} | "
                    f"{m['peak_active_gib']:.1f} | {m['peak_reserved_gib']:.1f} | ok |"
                )
            else:
                lines.append(
                    f"| {i} | {cfg['ep']} | {cfg['ac_mode']} | {ac_opt} | {reshard} | "
                    f"{m['tps_per_gpu']:,.0f} | "
                    f"{m['peak_active_gib']:.1f} | {m['peak_reserved_gib']:.1f} | ok |"
                )
        else:
            if baseline_tps:
                lines.append(
                    f"| {i} | {cfg['ep']} | {cfg['ac_mode']} | {ac_opt} | {reshard} | "
                    f"- | - | - | - | {r['status']} |"
                )
            else:
                lines.append(
                    f"| {i} | {cfg['ep']} | {cfg['ac_mode']} | {ac_opt} | {reshard} | "
                    f"- | - | - | {r['status']} |"
                )

    # Failed configurations
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        lines.append("\n## Failed Configurations\n")
        lines.append("| Config | Error |")
        lines.append("|--------|-------|")
        for r in failed:
            cfg = r["config"]
            ac_str = cfg["ac_mode"]
            if cfg["ac_option"]:
                ac_str += f"/{cfg['ac_option']}"
            reshard_str = (
                "reshard" if cfg["moe_reshard_after_forward"] else "no-reshard"
            )
            config_str = f"ep={cfg['ep']}, ac={ac_str}, {reshard_str}"
            error = r["error"] or r["status"]
            lines.append(f"| {config_str} | {error} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    configs = generate_grid(args)

    print(f"Generated {len(configs)} configurations:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg}")

    if args.dry_run:
        if args.include_baseline and args.n_moe_layers > 0:
            print("\n[Baseline] Would run dense model (n-moe-layers=0)")
        print("\n--dry-run specified, exiting.")
        return

    # Setup output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    jsonl_path = output_dir / f"moe_search_{args.tag}_{date_str}.jsonl"
    report_path = output_dir / f"moe_search_{args.tag}_{date_str}.md"

    # Create temp directory for intermediate results
    tmp_dir = Path("/tmp/moe_search")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    write_jsonl_metadata(jsonl_path, args)
    print(f"\nResults will be written to: {jsonl_path}")
    print(f"Report will be written to: {report_path}\n")

    # Run baseline if requested and MoE layers > 0
    baseline_result = None
    if args.include_baseline and args.n_moe_layers > 0:
        print("[Baseline] Running dense model (n-moe-layers=0)...")
        baseline_result = run_baseline(args, tmp_dir)
        write_jsonl_baseline(jsonl_path, baseline_result)
        if baseline_result.status == "success":
            m = baseline_result.metrics
            print(
                f"    -> OK: {m['tps_per_gpu']:,.0f} TPS/GPU, "
                f"{m['peak_active_gib']:.1f} GiB"
            )
        else:
            print(f"    -> {baseline_result.status.upper()}: {baseline_result.error}")
        print()

    # Run search
    results: list[BenchmarkResult] = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Running: {config}")

        result = run_benchmark(args, config, tmp_dir)
        results.append(result)

        # Write result immediately (crash-resilient)
        write_jsonl_result(jsonl_path, result)

        if result.status == "success":
            m = result.metrics
            print(
                f"    -> OK: {m['tps_per_gpu']:,.0f} TPS/GPU, {m['peak_active_gib']:.1f} GiB"
            )
        else:
            print(f"    -> {result.status.upper()}: {result.error}")

    # Generate report
    print(f"\nGenerating report: {report_path}")
    generate_report(jsonl_path, report_path, args)

    # Summary
    success_count = sum(1 for r in results if r.status == "success")
    print(f"\nComplete: {success_count}/{len(results)} configurations succeeded")
    print(f"Results: {jsonl_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
