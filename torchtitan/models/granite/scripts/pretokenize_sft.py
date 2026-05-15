"""Pre-tokenize SFT JSONL data for offline training.

Each input JSONL file produces one output shard (HF datasets directory) under
output_dir/shards/. Completion is tracked by the presence of a per-shard stats
file, so the script is safe to resume with any number of workers.

Usage (single node):
    python -m torchtitan.models.granite.scripts.pretokenize_sft \\
        --input-dir /path/to/jsonl/ \\
        --output-dir /path/to/output/ \\
        --tokenizer-path /path/to/tokenizer/ \\
        --strategy truncate_last

Usage (multi-node, each node runs this with different --rank):
    # Node 0:
    python -m ... --rank 0 --world-size 4
    # Node 1:
    python -m ... --rank 1 --world-size 4

Resumable and idempotent.  The last rank to finish writes manifest.json once all shards are present.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from filelock import FileLock

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.tokenization_strategies import (
    BackboneSuffixStrategy,
    FullThinkingStrategy,
    TokenizationStrategy,
    TruncateEveryTurnStrategy,
    TruncateLastStrategy,
)

logger = logging.getLogger(__name__)

_STRATEGIES: dict[str, type[TokenizationStrategy]] = {
    "truncate_last": TruncateLastStrategy,
    "backbone_suffix": BackboneSuffixStrategy,
    "full_thinking": FullThinkingStrategy,
    "truncate_every_turn": TruncateEveryTurnStrategy,
}


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


_RUN_CONFIG_FILENAME = "run_config.json"


def _save_run_config(
    input_dir: Path,
    output_dir: Path,
    strategy: str,
    chat_template_kwargs: dict[str, Any],
) -> None:
    with open(output_dir / _RUN_CONFIG_FILENAME, "w") as f:
        json.dump(
            {
                "strategy": strategy,
                "chat_template_kwargs": chat_template_kwargs,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
            f,
            indent=2,
        )


def _load_run_config(output_dir: Path) -> dict[str, Any]:
    path = output_dir / _RUN_CONFIG_FILENAME
    if not path.exists():
        raise ValueError(f"No {_RUN_CONFIG_FILENAME} found in {output_dir}.")
    with open(path) as f:
        return json.load(f)


def _completed_stems(shards_dir: Path) -> set[str]:
    """Return stems of completed shards (identified by their stats sidecar file)."""
    if not shards_dir.exists():
        return set()
    return {p.stem.removesuffix("_stats") for p in shards_dir.glob("*_stats.json")}


def _process_file(
    input_file: Path,
    output_dir: Path,
    strategy: TokenizationStrategy,
    *,
    num_cpus: int,
    batch_size: int,
    rank: int,
) -> None:
    """Tokenize one JSONL file and write its shard."""
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    final_path = shards_dir / stem
    stats_path = shards_dir / f"{stem}_stats.json"

    n_lines = _count_lines(str(input_file))
    logger.info("[rank %d] %s: %d input lines", rank, input_file.name, n_lines)

    t0 = time.monotonic()
    ds = load_dataset("json", data_files=str(input_file), split="train")
    ds = ds.map(
        strategy,
        batched=True,
        batch_size=batch_size,
        num_proc=num_cpus,
        remove_columns=ds.column_names,
        desc=f"[rank {rank}] {input_file.name}",
    )
    ds.save_to_disk(str(final_path))
    elapsed = time.monotonic() - t0

    import numpy as np
    import pyarrow.compute as pc

    n_examples = len(ds)
    n_tokens_arr = np.array(ds["n_tokens"], dtype=np.int64)
    total_tokens = int(n_tokens_arr.sum())
    sum_tokens_squared = int((n_tokens_arr**2).sum())
    labels_flat = ds.data.column("labels").combine_chunks().flatten()
    total_trained = int(pc.sum(pc.not_equal(labels_flat, -100)).as_py())

    stats: dict[str, Any] = {
        "input_file": input_file.name,
        "shard_stem": stem,
        "n_examples": n_examples,
        "n_dropped": n_lines - n_examples,
        "total_tokens": total_tokens,
        "sum_tokens_squared": sum_tokens_squared,
        "total_trained_tokens": total_trained,
        "elapsed_seconds": round(elapsed, 2),
        "examples_per_second": round(n_examples / max(elapsed, 1e-6), 1),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "[rank %d] %s done: %d examples, %d tokens, %.1f ex/s",
        rank,
        input_file.name,
        n_examples,
        total_tokens,
        stats["examples_per_second"],
    )


_SEQ_LEN_CUTOFFS = [16384, 32768, 65536, 131072, 262144, 524288]


def _compute_length_stats(shards_dir: Path, completed_shards: list[str]) -> dict[str, Any]:
    """Load n_tokens from all shards and compute distribution stats."""
    import numpy as np
    from datasets import load_from_disk

    all_lengths: list[int] = []
    for shard_name in completed_shards:
        ds = load_from_disk(str(shards_dir / shard_name))
        all_lengths.extend(ds["n_tokens"])

    arr = np.array(all_lengths, dtype=np.int64)
    squared_tokens_per_example = float((arr**2).mean())

    length_stats: dict[str, Any] = {
        "squared_tokens_per_example": round(squared_tokens_per_example, 1),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 1),
        "median": int(np.median(arr)),
        "std": round(float(arr.std()), 1),
        "p95": int(np.percentile(arr, 95)),
    }

    for cutoff in _SEQ_LEN_CUTOFFS:
        filtered = arr[arr <= cutoff]
        k = cutoff // 1024
        if len(filtered) > 0:
            length_stats[f"squared_tokens_per_example_{k}kmax"] = round(
                float((filtered**2).mean()), 1
            )
            length_stats[f"tokens_per_example_{k}kmax"] = round(
                float(filtered.mean()), 1
            )
            length_stats[f"n_examples_{k}kmax"] = int(len(filtered))
        else:
            length_stats[f"squared_tokens_per_example_{k}kmax"] = None
            length_stats[f"tokens_per_example_{k}kmax"] = None
            length_stats[f"n_examples_{k}kmax"] = 0

    return length_stats


def _write_manifest(
    output_dir: Path,
    input_files: list[Path],
    strategy_name: str,
    tokenizer_path: str,
    chat_template_kwargs: dict[str, Any],
) -> None:
    """Aggregate per-shard stats into manifest.json."""
    tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)
    shards_dir = output_dir / "shards"
    all_stats: list[dict[str, Any]] = []
    for p in sorted(shards_dir.glob("*_stats.json")):
        with open(p) as f:
            all_stats.append(json.load(f))

    total_examples = sum(s["n_examples"] for s in all_stats)
    total_dropped = sum(s["n_dropped"] for s in all_stats)
    total_tokens = sum(s["total_tokens"] for s in all_stats)
    total_trained = sum(s["total_trained_tokens"] for s in all_stats)

    completed_shards = sorted(
        d.name
        for d in shards_dir.iterdir()
        if d.is_dir() and not d.name.endswith("_stats")
    )

    logger.info("Computing length distribution stats from %d shards...", len(completed_shards))
    length_stats = _compute_length_stats(shards_dir, completed_shards)

    chat_template_sha256 = None
    jinja_path = Path(tokenizer_path) / "chat_template.jinja"
    if jinja_path.exists():
        chat_template_sha256 = _sha256_file(str(jinja_path))

    manifest: dict[str, Any] = {
        "version": 1,
        "strategy": strategy_name,
        "tokenizer": {
            "source_path": tokenizer_path,
            "vocab_size": tokenizer.get_vocab_size(),
            "eos_token_id": tokenizer.eos_id,
            "chat_template_sha256": chat_template_sha256,
        },
        "chat_template_kwargs": chat_template_kwargs,
        "shards": {
            "completed": completed_shards,
            "total_expected": len(input_files),
        },
        "stats": {
            "total_examples": total_examples,
            "examples_dropped": total_dropped,
            "total_tokens": total_tokens,
            "total_trained_tokens": total_trained,
            "tokens_per_example": round(total_tokens / total_examples, 1),
            "trained_tokens_per_example": round(total_trained / total_examples, 1),
            "trained_to_total_tokens_ratio": total_trained / total_tokens,
            "length_stats": length_stats,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": str(input_files[0].parent) if input_files else "",
        "input_files": [str(f) for f in sorted(input_files)],
    }

    manifest_path = output_dir / "manifest.json"
    with FileLock(str(manifest_path) + ".lock"):
        if manifest_path.exists():
            logger.info("Manifest already written by another rank, skipping")
            return
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize SFT JSONL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing .jsonl files"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write shards and manifest (default: <input-dir>/pretok_<strategy>)",
    )
    parser.add_argument(
        "--tokenizer-path", required=True, help="Path to HF tokenizer directory"
    )
    parser.add_argument(
        "--strategy",
        choices=list(_STRATEGIES),
        default="truncate_last",
        help="Tokenization strategy",
    )
    parser.add_argument("--rank", type=int, default=0, help="Worker rank (0-indexed)")
    parser.add_argument(
        "--world-size", type=int, default=1, help="Total number of workers"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="CPUs for intra-file parallelism (default: all available // 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for datasets.map tokenization",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s rank={args.rank}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(
        args.output_dir if args.output_dir else input_dir / f"pretok_{args.strategy}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    num_cpus: int = args.num_cpus or (multiprocessing.cpu_count() // 2)

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        raise ValueError(f"No .jsonl files found in {input_dir}")

    strategy = _STRATEGIES[args.strategy](
        args.tokenizer_path,
        failures_path=str(output_dir / "failures.jsonl"),
    )

    run_config_path = output_dir / _RUN_CONFIG_FILENAME
    if run_config_path.exists():
        run_config = _load_run_config(output_dir)
        if run_config["strategy"] != args.strategy:
            raise ValueError(
                f"Resume mismatch: existing shards used strategy {run_config['strategy']!r} "
                f"but current invocation specifies {args.strategy!r}."
            )
        if run_config["chat_template_kwargs"] != strategy.chat_template_kwargs:
            raise ValueError(
                f"Resume mismatch: existing shards used chat_template_kwargs "
                f"{run_config['chat_template_kwargs']} but current invocation has "
                f"{strategy.chat_template_kwargs}."
            )
    elif args.rank == 0:
        _save_run_config(
            input_dir,
            output_dir,
            args.strategy,
            strategy.chat_template_kwargs,
        )

    shards_dir = output_dir / "shards"
    completed = _completed_stems(shards_dir)
    remaining = [f for f in input_files if f.stem not in completed]
    if completed:
        logger.info(
            "Resuming: %d/%d files done, %d remaining",
            len(completed),
            len(input_files),
            len(remaining),
        )

    my_files = remaining[args.rank :: args.world_size]
    logger.info(
        "[rank %d/%d] processing %d files",
        args.rank,
        args.world_size,
        len(my_files),
    )

    for input_file in my_files:
        _process_file(
            input_file,
            output_dir,
            strategy,
            num_cpus=num_cpus,
            batch_size=args.batch_size,
            rank=args.rank,
        )

    completed = _completed_stems(shards_dir)
    if len(completed) == len(input_files):
        _write_manifest(
            output_dir,
            input_files,
            args.strategy,
            args.tokenizer_path,
            strategy.chat_template_kwargs,
        )
    else:
        logger.info(
            "rank %d: %d/%d shards complete, manifest will be written by the last rank to finish",
            args.rank,
            len(completed),
            len(input_files),
        )


if __name__ == "__main__":
    main()
