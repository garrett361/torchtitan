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
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow as pa
from datasets import (
    Dataset,
    Features,
    concatenate_datasets,
    disable_caching,
    load_from_disk,
)

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


def _shard_stem(input_file: Path, input_dir: Path) -> str:
    """Derive a unique shard stem from the file's path relative to input_dir.

    Replaces '/' with '__' so nested files get flat, collision-safe shard names.
    Example: input_dir/a/b/train.jsonl → 'a__b__train'
    """
    rel = input_file.relative_to(input_dir).with_suffix("")
    return str(rel).replace("/", "__")


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


_RUN_CONFIG_FILENAME = "run_config.json"


def _save_run_config(
    input_dir: Path,
    output_dir: Path,
    strategy: str,
    chat_template_kwargs: dict[str, Any],
) -> None:
    _write_json_atomic(
        output_dir / _RUN_CONFIG_FILENAME,
        {
            "strategy": strategy,
            "chat_template_kwargs": chat_template_kwargs,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
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


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via tmp+rename so concurrent readers never see a partial file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def _count_lines(path: Path) -> int:
    """Count lines in a file, returning 0 if it doesn't exist."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def _tokenize_batch(
    batch: dict[str, list], *, strategy: TokenizationStrategy
) -> dict[str, list]:
    return strategy({"messages": [json.loads(m) for m in batch["messages_json"]]})


def _process_file(
    input_file: Path,
    output_dir: Path,
    strategy_cls: type[TokenizationStrategy],
    tokenizer_path: str,
    *,
    shard_stem: str,
    input_dir: Path,
    num_cpus: int,
    batch_size: int,
    rank: int,
) -> None:
    """Tokenize one JSONL file and write its shard."""
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    final_path = shards_dir / shard_stem
    stats_path = shards_dir / f"{shard_stem}_stats.json"
    failures_path = shards_dir / f"{shard_stem}_failures.jsonl"
    strategy = strategy_cls(tokenizer_path, failures_path=str(failures_path))

    try:
        _tokenize_file(
            input_file,
            final_path,
            stats_path,
            strategy,
            failures_path=failures_path,
            input_dir=input_dir,
            shard_stem=shard_stem,
            num_cpus=num_cpus,
            batch_size=batch_size,
            rank=rank,
        )
    except Exception as e:
        logger.error(
            "[rank %d] %s: crashed: %s", rank, input_file.name, e, exc_info=True
        )
        stats: dict[str, Any] = {
            "input_file": str(input_file.relative_to(input_dir)),
            "shard_stem": shard_stem,
            "n_examples": 0,
            "n_dropped": 0,
            "total_tokens": 0,
            "total_trained_tokens": 0,
            "elapsed_seconds": 0.0,
            "examples_per_second": 0.0,
            "skipped": True,
            "error": f"{type(e).__name__}: {e}",
        }
        _write_json_atomic(stats_path, stats)


def _tokenize_file(
    input_file: Path,
    final_path: Path,
    stats_path: Path,
    strategy: TokenizationStrategy,
    *,
    failures_path: Path,
    input_dir: Path,
    shard_stem: str,
    num_cpus: int,
    batch_size: int,
    rank: int,
) -> None:
    messages_json: list[str] = []
    with open(input_file) as f:
        for line in f:
            if not line.strip():
                continue
            messages_json.append(json.dumps(json.loads(line)["messages"]))
    n_lines = len(messages_json)
    logger.info("[rank %d] %s: %d input lines", rank, input_file.name, n_lines)

    t0 = time.monotonic()
    ds = Dataset.from_dict({"messages_json": messages_json})
    features = Features.from_arrow_schema(pa.schema(strategy.column_schema))
    ds = ds.map(
        _tokenize_batch,
        fn_kwargs={"strategy": strategy},
        batched=True,
        batch_size=batch_size,
        num_proc=num_cpus,
        remove_columns=ds.column_names,
        features=features,
        desc=f"[rank {rank}] {input_file.name}",
    )

    n_dropped = _count_lines(failures_path)

    if len(ds) == 0:
        elapsed = time.monotonic() - t0
        logger.warning(
            "[rank %d] %s: all %d samples dropped, skipping shard",
            rank, input_file.name, n_lines,
        )
        stats: dict[str, Any] = {
            "input_file": str(input_file.relative_to(input_dir)),
            "shard_stem": shard_stem,
            "n_input_conversations": n_lines,
            "n_examples": 0,
            "n_dropped": n_dropped,
            "total_tokens": 0,
            "total_trained_tokens": 0,
            "elapsed_seconds": round(elapsed, 2),
            "examples_per_second": 0.0,
            "skipped": True,
        }
        _write_json_atomic(stats_path, stats)
        return

    ds.save_to_disk(str(final_path))
    elapsed = time.monotonic() - t0

    n_examples = len(ds)
    n_tokens_arr = np.array(ds["n_tokens"], dtype=np.int64)
    total_tokens = int(n_tokens_arr.sum())
    labels_flat = pc.list_flatten(ds.data.column("labels"))
    total_trained = int(pc.sum(pc.not_equal(labels_flat, -100)).as_py())

    stats: dict[str, Any] = {
        "input_file": str(input_file.relative_to(input_dir)),
        "shard_stem": shard_stem,
        "n_input_conversations": n_lines,
        "n_examples": n_examples,
        "n_dropped": n_dropped,
        "total_tokens": total_tokens,
        "total_trained_tokens": total_trained,
        "elapsed_seconds": round(elapsed, 2),
        "examples_per_second": round(n_examples / max(elapsed, 1e-6), 1),
    }
    _write_json_atomic(stats_path, stats)

    logger.info(
        "[rank %d] %s done: %d examples, %d tokens, %.1f ex/s",
        rank,
        input_file.name,
        n_examples,
        total_tokens,
        stats["examples_per_second"],
    )


def _write_manifest(
    output_dir: Path,
    input_dir: Path,
    input_files: list[Path],
    strategy_name: str,
    tokenizer_path: str,
    chat_template_kwargs: dict[str, Any],
) -> bool:
    """Aggregate per-shard stats into manifest.json. Returns True if this call wrote it."""
    tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)
    shards_dir = output_dir / "shards"
    all_stats: list[dict[str, Any]] = []
    skipped_files: list[str] = []
    for p in sorted(shards_dir.glob("*_stats.json")):
        with open(p) as f:
            s = json.load(f)
        if s.get("skipped", False):
            skipped_files.append(s["input_file"])
            continue
        all_stats.append(s)

    total_input_conversations = sum(s["n_input_conversations"] for s in all_stats)
    total_examples = sum(s["n_examples"] for s in all_stats)
    total_dropped = sum(s["n_dropped"] for s in all_stats)
    total_tokens = sum(s["total_tokens"] for s in all_stats)
    total_trained = sum(s["total_trained_tokens"] for s in all_stats)

    completed_shards = sorted(
        d.name
        for d in shards_dir.iterdir()
        if d.is_dir() and not d.name.endswith("_stats")
    )

    chat_template_sha256 = None
    jinja_path = Path(tokenizer_path) / "chat_template.jinja"
    if jinja_path.exists():
        chat_template_sha256 = _sha256_file(str(jinja_path))

    # Compute length distribution stats from the concatenated shards
    all_n_tokens = []
    for shard_name in completed_shards:
        shard_ds = load_from_disk(str(shards_dir / shard_name))
        all_n_tokens.extend(shard_ds["n_tokens"])

    if total_examples > 0:
        n_tokens_arr = np.array(all_n_tokens, dtype=np.int64)
        stats_block: dict[str, Any] = {
            "total_input_conversations": total_input_conversations,
            "total_examples": total_examples,
            "examples_dropped": total_dropped,
            "total_tokens": total_tokens,
            "total_trained_tokens": total_trained,
            "tokens_per_example": round(total_tokens / total_examples, 1),
            "trained_tokens_per_example": round(total_trained / total_examples, 1),
            "trained_to_total_tokens_ratio": total_trained / total_tokens,
        }
        length_stats_block: dict[str, Any] = {
            "min": int(n_tokens_arr.min()),
            "max": int(n_tokens_arr.max()),
            "mean": round(float(n_tokens_arr.mean()), 1),
            "median": int(np.median(n_tokens_arr)),
            "std": round(float(n_tokens_arr.std()), 1),
            "p95": int(np.percentile(n_tokens_arr, 95)),
        }
    else:
        stats_block = {
            "total_input_conversations": total_input_conversations,
            "total_examples": 0,
            "examples_dropped": total_dropped,
            "total_tokens": 0,
            "total_trained_tokens": 0,
            "tokens_per_example": 0.0,
            "trained_tokens_per_example": 0.0,
            "trained_to_total_tokens_ratio": 0.0,
        }
        length_stats_block = {}

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
        "skipped_files": skipped_files,
        "stats": stats_block,
        "length_stats": length_stats_block,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": str(input_dir),
        "input_files": [str(f) for f in sorted(input_files)],
    }

    manifest_path = output_dir / "manifest.json"
    with FileLock(str(manifest_path) + ".lock"):
        if manifest_path.exists():
            logger.info("Manifest already written by another rank, skipping")
            return False
        # Atomic: other ranks poll manifest_path.exists() and must not see partial content.
        tmp_manifest = manifest_path.with_suffix(".tmp")
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        tmp_manifest.rename(manifest_path)
    logger.info("Wrote manifest to %s", manifest_path)
    return True


_SHUFFLE_META_FILENAME = "shuffle_meta.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _finalize_shuffle(
    output_dir: Path,
    shards_dir: Path,
    shuffle_meta_path: Path,
    shuffle_seed: int,
    num_shards: int,
) -> None:
    """Rewrite manifest and write completion marker after shuffle swap."""
    manifest = _load_manifest(output_dir / "manifest.json")
    total_examples = manifest["stats"]["total_examples"]

    new_shard_names = [f"shard_{i:04d}" for i in range(num_shards)]
    new_manifest = {
        **manifest,
        "shards": {
            "completed": new_shard_names,
            "total_expected": num_shards,
        },
        "shuffle": {"seed": shuffle_seed, "num_shards": num_shards},
    }
    tmp_manifest = output_dir / "manifest_new.json"
    with open(tmp_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)
    tmp_manifest.rename(output_dir / "manifest.json")

    with open(shuffle_meta_path, "w") as f:
        json.dump(
            {"seed": shuffle_seed, "num_shards": num_shards, "total_examples": total_examples},
            f,
            indent=2,
        )
    logger.info("Global shuffle complete (seed=%d, %d shards)", shuffle_seed, num_shards)


def _shuffle_and_reshard(
    output_dir: Path, shuffle_seed: int, rank: int = 0, world_size: int = 1
) -> None:
    """Phase 2: Load all shards, globally shuffle, write back same shard count.

    When world_size > 1, shard writes are distributed across ranks. Each rank
    independently generates the identical permutation (deterministic seed), writes
    its assigned shards, then one rank performs the atomic directory swap.

    Resumable and atomic. Recovery state is inferred from directory presence:
    - shards_backup/ exists → crashed mid-swap, complete it
    - shuffle_meta.json exists → already done
    - shards_shuffled/ with partial stats → resume shard writes
    """
    shards_dir = output_dir / "shards"
    shuffled_dir = output_dir / "shards_shuffled"
    backup_dir = output_dir / "shards_backup"
    shuffle_meta_path = output_dir / _SHUFFLE_META_FILENAME

    # State recovery: if backup exists, a prior run crashed mid-swap.
    # FileLock ensures only one rank performs recovery.
    if backup_dir.exists():
        with FileLock(str(shuffle_meta_path) + ".lock"):
            if backup_dir.exists():
                if shuffled_dir.exists():
                    shuffled_dir.rename(shards_dir)
                logger.info("[rank %d] Recovered from interrupted swap", rank)
                perm_in_shards = shards_dir / "permutation.npy"
                if perm_in_shards.exists():
                    perm_in_shards.unlink()
                num_shards = len([
                    p for p in shards_dir.iterdir()
                    if p.is_dir() and not p.name.endswith("_stats")
                ])
                _finalize_shuffle(
                    output_dir, shards_dir, shuffle_meta_path, shuffle_seed, num_shards
                )
                shutil.rmtree(backup_dir)
        return

    if shuffle_meta_path.exists():
        with open(shuffle_meta_path) as f:
            meta = json.load(f)
        if meta["seed"] != shuffle_seed:
            raise ValueError(
                f"Shuffle already completed with seed={meta['seed']}, "
                f"but current invocation specifies seed={shuffle_seed}. "
                f"Remove {shuffle_meta_path} to re-shuffle."
            )
        logger.info("Shuffle already complete (seed=%d), skipping", shuffle_seed)
        return

    # HF datasets caches select/concatenate results keyed by file fingerprint; on
    # resume after a partial swap the fingerprints are stale, returning wrong data.
    disable_caching()

    manifest = _load_manifest(output_dir / "manifest.json")
    shard_names = sorted(manifest["shards"]["completed"])
    num_shards = len(shard_names)

    if num_shards == 0:
        logger.info("[rank %d] No shards to shuffle, skipping Phase 2", rank)
        return

    logger.info(
        "[rank %d] Loading %d shards for global shuffle (seed=%d)...",
        rank, num_shards, shuffle_seed,
    )
    full_dataset = concatenate_datasets(
        [load_from_disk(str(shards_dir / name)) for name in shard_names]
    )
    total_examples = len(full_dataset)

    rng = np.random.default_rng(shuffle_seed)
    permutation = rng.permutation(total_examples)

    shuffled_dir.mkdir(parents=True, exist_ok=True)

    examples_per_shard = total_examples // num_shards
    my_shards = list(range(rank, num_shards, world_size))
    t0 = time.monotonic()

    logger.info(
        "[rank %d] writing %d/%d shuffled shards", rank, len(my_shards), num_shards
    )

    for i in my_shards:
        shard_name = f"shard_{i:04d}"
        stats_path = shuffled_dir / f"{shard_name}_stats.json"
        if stats_path.exists():
            continue

        try:
            start = i * examples_per_shard
            end = (
                (start + examples_per_shard) if i < num_shards - 1 else total_examples
            )
            indices = permutation[start:end].tolist()
            shard = full_dataset.select(indices).flatten_indices()
            shard.save_to_disk(str(shuffled_dir / shard_name))

            n_tokens_arr = np.array(shard["n_tokens"], dtype=np.int64)
            labels_flat = shard.data.column("labels").combine_chunks().flatten()
            total_trained = int(pc.sum(pc.not_equal(labels_flat, -100)).as_py())
            stats: dict[str, Any] = {
                "shard_stem": shard_name,
                "n_examples": len(shard),
                "total_tokens": int(n_tokens_arr.sum()),
                "total_trained_tokens": total_trained,
            }
            _write_json_atomic(stats_path, stats)

            logger.info(
                "[rank %d] Wrote shuffled shard %d/%d: %s (%d examples)",
                rank, i + 1, num_shards, shard_name, len(shard),
            )
        except Exception as e:
            logger.error(
                "[rank %d] shuffle shard %s crashed: %s",
                rank, shard_name, e, exc_info=True,
            )
            error_stats: dict[str, Any] = {
                "shard_stem": shard_name,
                "n_examples": 0,
                "total_tokens": 0,
                "total_trained_tokens": 0,
                "error": f"{type(e).__name__}: {e}",
            }
            _write_json_atomic(stats_path, error_stats)

    elapsed = time.monotonic() - t0
    logger.info("[rank %d] Shuffle write complete in %.1fs", rank, elapsed)

    # All ranks poll until shuffle_meta.json signals completion.
    # The first rank to observe all shards done acquires the lock and performs the swap.
    while not shuffle_meta_path.exists():
        done = sum(
            1 for i in range(num_shards)
            if (shuffled_dir / f"shard_{i:04d}_stats.json").exists()
        )
        if done == num_shards:
            # Check for errors before committing the swap.
            all_errors: list[str] = []
            for i in range(num_shards):
                sp = shuffled_dir / f"shard_{i:04d}_stats.json"
                with open(sp) as f:
                    s = json.load(f)
                if "error" in s:
                    all_errors.append(f"shard_{i:04d}: {s['error']}")
            if all_errors:
                raise RuntimeError(
                    f"Shuffle failed — {len(all_errors)} shard(s) crashed:\n"
                    + "\n".join(all_errors)
                )

            with FileLock(str(shuffle_meta_path) + ".lock"):
                if not shuffle_meta_path.exists():
                    shards_dir.rename(backup_dir)
                    shuffled_dir.rename(shards_dir)
                    logger.info("[rank %d] Swapped shards_shuffled/ → shards/", rank)

                    perm_in_shards = shards_dir / "permutation.npy"
                    if perm_in_shards.exists():
                        perm_in_shards.unlink()

                    _finalize_shuffle(
                        output_dir, shards_dir, shuffle_meta_path, shuffle_seed, num_shards
                    )
                    shutil.rmtree(backup_dir)
            break
        logger.info(
            "[rank %d] shuffle: %d/%d shards done, waiting...", rank, done, num_shards
        )
        time.sleep(30)


def _assign_files_by_cost(
    input_files: list[Path], world_size: int
) -> list[list[Path]]:
    """Assign files to ranks via greedy bin-packing on file size."""
    # resolve() dereferences symlinks — without it we'd get the symlink's own size (~bytes)
    sized = sorted(
        ((os.path.getsize(f.resolve()), f) for f in input_files), key=lambda x: x[0], reverse=True
    )
    bins: list[list[Path]] = [[] for _ in range(world_size)]
    costs = [0] * world_size
    for size, f in sized:
        lightest = min(range(world_size), key=lambda r: costs[r])
        bins[lightest].append(f)
        costs[lightest] += size
    return bins


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
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for global cross-shard shuffle (Phase 2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s rank={args.rank}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Ensure we are re-tokenizing from scratch. Avoids potential issues like modifications to the
    # chat template without modifying the chat template path, which do not cause cache invalidation.
    disable_caching()

    input_dir = Path(args.input_dir)
    output_dir = Path(
        args.output_dir if args.output_dir else input_dir / f"pretok_{args.strategy}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    num_cpus: int = args.num_cpus or (multiprocessing.cpu_count() // 2)

    # Exclude output dirs from prior runs (identified by run_config.json) and
    # the current run's output_dir (which may not have run_config.json yet).
    pretok_dirs = {p.parent for p in input_dir.rglob(_RUN_CONFIG_FILENAME)}
    pretok_dirs.add(output_dir)
    input_files = sorted(
        f
        for f in input_dir.rglob("*.jsonl")
        if not any(f.is_relative_to(d) for d in pretok_dirs)
    )
    if not input_files:
        raise ValueError(f"No .jsonl files found in {input_dir} (searched recursively)")

    # Validate shard stem uniqueness before any work begins (all ranks).
    seen_stems: dict[str, Path] = {}
    for f in input_files:
        stem = _shard_stem(f, input_dir)
        if stem in seen_stems:
            raise ValueError(
                f"Shard stem collision: {f} and {seen_stems[stem]} both map to '{stem}'"
            )
        if len(stem) > 200:
            raise ValueError(
                f"Shard stem too long ({len(stem)} chars) for {f}. "
                "Flatten your directory structure or shorten filenames."
            )
        seen_stems[stem] = f

    strategy_cls = _STRATEGIES[args.strategy]
    strategy_for_config = strategy_cls(args.tokenizer_path)

    run_config_path = output_dir / _RUN_CONFIG_FILENAME
    with FileLock(str(run_config_path) + ".lock"):
        if not run_config_path.exists() and args.rank == 0:
            _save_run_config(
                input_dir,
                output_dir,
                args.strategy,
                strategy_for_config.chat_template_kwargs,
            )

    # All ranks validate config consistency (wait for rank 0 to write on fresh runs).
    while not run_config_path.exists():
        time.sleep(1)
    run_config = _load_run_config(output_dir)
    if run_config["strategy"] != args.strategy:
        raise ValueError(
            f"Resume mismatch: existing shards used strategy {run_config['strategy']!r} "
            f"but current invocation specifies {args.strategy!r}."
        )
    if run_config["chat_template_kwargs"] != strategy_for_config.chat_template_kwargs:
        raise ValueError(
            f"Resume mismatch: existing shards used chat_template_kwargs "
            f"{run_config['chat_template_kwargs']} but current invocation has "
            f"{strategy_for_config.chat_template_kwargs}."
        )

    shards_dir = output_dir / "shards"
    my_files = _assign_files_by_cost(input_files, args.world_size)[args.rank]

    completed = _completed_stems(shards_dir)
    num_skipped = sum(1 for f in my_files if _shard_stem(f, input_dir) in completed)
    if num_skipped:
        logger.info(
            "Resuming: skipping %d/%d already-completed files",
            num_skipped,
            len(my_files),
        )

    logger.info(
        "[rank %d/%d] processing %d files",
        args.rank,
        args.world_size,
        len(my_files) - num_skipped,
    )

    for input_file in my_files:
        shard_stem = _shard_stem(input_file, input_dir)
        if shard_stem in completed:
            continue
        _process_file(
            input_file,
            output_dir,
            strategy_cls,
            args.tokenizer_path,
            shard_stem=shard_stem,
            input_dir=input_dir,
            num_cpus=num_cpus,
            batch_size=args.batch_size,
            rank=args.rank,
        )

    # Barrier: wait for all tokenization to complete before entering Phase 2.
    manifest_path = output_dir / "manifest.json"
    while not manifest_path.exists():
        completed = _completed_stems(shards_dir)
        if len(completed) == len(input_files):
            _write_manifest(
                output_dir,
                input_dir,
                input_files,
                args.strategy,
                args.tokenizer_path,
                strategy_for_config.chat_template_kwargs,
            )
            break
        logger.info(
            "rank %d: %d/%d shards complete, waiting...",
            args.rank,
            len(completed),
            len(input_files),
        )
        time.sleep(30)

    # Phase 2: all ranks participate in distributed shuffle.
    _shuffle_and_reshard(output_dir, args.shuffle_seed, args.rank, args.world_size)


if __name__ == "__main__":
    main()
