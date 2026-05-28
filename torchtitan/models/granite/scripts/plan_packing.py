#!/usr/bin/env python3
"""Offline pack-plan generation for pretokenized SFT datasets.

Reads a pretokenized arrow dataset (produced by pretokenize_sft.py), runs FFD
(First Fit Decreasing) bin packing to assign examples to fixed-length packs,
sorts packs by attention cost, and serializes a lightweight lookup table that
the training dataloader can consume.

The plan enables cost-balanced batch formation: at training time, packs are
chunked into groups of dp_degree (consecutive in cost-sorted order), so all
FSDP ranks in a global batch process sequences with near-identical attention
cost — minimizing synchronization idle time.

Usage:
    python -m torchtitan.models.granite.scripts.plan_packing \
        --pretok_dir /path/to/pretok \
        --seq_len 131072
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

logger = logging.getLogger(__name__)


def _load_metadata_columns(
    manifest: dict, shards_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load n_tokens and attn_cost aligned to global row indices.

    Uses the same loading path as training (_load_shards) to guarantee
    identical row ordering.
    """
    from torchtitan.models.granite.pretokenized_dataset import _load_shards

    dataset = _load_shards(manifest, shards_dir)
    n_tokens = dataset.data.column("n_tokens").to_numpy().astype(np.int64)
    attn_cost = dataset.data.column("attn_cost").to_numpy().astype(np.int64)
    logger.info("Loaded %d examples from shards at %s", len(n_tokens), shards_dir)
    return n_tokens, attn_cost


def _pre_pack(
    n_tokens: np.ndarray,
    attn_cost: np.ndarray,
    seq_len: int,
    bucket_width: int = 128,
) -> tuple[list[list[int]], np.ndarray]:
    """First Fit Decreasing bin packing using a bucket structure.

    Returns:
        pack_contents: list of lists, each inner list contains global row
            indices of examples assigned to that pack.
        pack_attn_costs: 1-D array of total attn_cost per pack.
    """
    n_buckets = seq_len // bucket_width + 1

    # Pre-convert to Python lists to avoid numpy scalar overhead in the loop
    sorted_order = np.argsort(-n_tokens).tolist()
    lengths: list[int] = n_tokens.tolist()
    costs: list[int] = attn_cost.tolist()

    # Bucket structure: buckets[b] = list of pack_ids with remaining capacity
    # in [b*bucket_width, (b+1)*bucket_width)
    buckets: list[list[int]] = [[] for _ in range(n_buckets)]

    # Per-pack state
    pack_remaining: list[int] = []
    pack_contents: list[list[int]] = []
    pack_costs: list[int] = []

    for global_idx in sorted_order:
        length = lengths[global_idx]
        cost = costs[global_idx]

        if length <= 0 or length > seq_len:
            continue

        guaranteed_bucket = (length + bucket_width - 1) // bucket_width

        placed = False
        for b in range(guaranteed_bucket, n_buckets):
            if buckets[b]:
                pack_id = buckets[b].pop()
                new_remaining = pack_remaining[pack_id] - length
                pack_remaining[pack_id] = new_remaining
                pack_contents[pack_id].append(global_idx)
                pack_costs[pack_id] += cost
                if new_remaining > 0:
                    buckets[new_remaining // bucket_width].append(pack_id)
                placed = True
                break

        if not placed:
            pack_id = len(pack_remaining)
            pack_remaining.append(seq_len - length)
            pack_contents.append([global_idx])
            pack_costs.append(cost)
            remaining = seq_len - length
            if remaining > 0:
                buckets[remaining // bucket_width].append(pack_id)

    # Sort indices within each pack for read locality at training time
    for pack in pack_contents:
        pack.sort()

    return pack_contents, np.array(pack_costs, dtype=np.int64)


def _compute_manifest_sha256(manifest_path: Path) -> str:
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    return h.hexdigest()


def plan_packing(
    pretok_dir: Path,
    seq_len: int,
    output_dir: Path,
    bucket_width: int = 128,
) -> None:
    manifest_path = pretok_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {pretok_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    shards_dir = pretok_dir / "shards"

    logger.info("Loading metadata columns from %s ...", shards_dir)
    t0 = time.time()
    n_tokens, attn_cost = _load_metadata_columns(manifest, shards_dir)
    logger.info("Loaded %d examples in %.1fs", len(n_tokens), time.time() - t0)

    total_examples = len(n_tokens)
    overlong_count = int((n_tokens > seq_len).sum())
    if overlong_count > 0:
        logger.info(
            "Filtering %d examples exceeding seq_len=%d (%.2f%%)",
            overlong_count,
            seq_len,
            overlong_count / total_examples * 100,
        )

    logger.info("Running BFD packing (bucket_width=%d) ...", bucket_width)
    t0 = time.time()
    pack_contents, pack_costs = _pre_pack(
        n_tokens, attn_cost, seq_len, bucket_width=bucket_width
    )
    bfd_time = time.time() - t0
    logger.info(
        "BFD produced %d packs in %.1fs", len(pack_contents), bfd_time
    )

    # Sort packs by attention cost ascending
    cost_order = np.argsort(pack_costs)
    pack_contents = [pack_contents[i] for i in cost_order]
    pack_costs = pack_costs[cost_order]

    # Compute per-pack total_tokens
    pack_total_tokens = np.array(
        [sum(int(n_tokens[idx]) for idx in pack) for pack in pack_contents],
        dtype=np.int64,
    )

    total_packed_tokens = int(pack_total_tokens.sum())
    total_capacity = len(pack_contents) * seq_len
    padding_fraction = 1.0 - total_packed_tokens / total_capacity if total_capacity > 0 else 0.0

    examples_packed = total_examples - overlong_count
    logger.info(
        "Packing stats: %d packs, padding_fraction=%.6f, examples_packed=%d",
        len(pack_contents),
        padding_fraction,
        examples_packed,
    )

    # Serialize
    output_dir.mkdir(parents=True, exist_ok=True)

    # pack_plan.arrow
    pack_ids = pa.array(np.arange(len(pack_contents), dtype=np.int32))
    example_indices_col = pa.array(
        [pa.array(indices, type=pa.int32()) for indices in pack_contents],
        type=pa.list_(pa.int32()),
    )
    total_tokens_col = pa.array(pack_total_tokens)
    attn_cost_col = pa.array(pack_costs)

    table = pa.table(
        {
            "pack_id": pack_ids,
            "example_indices": example_indices_col,
            "total_tokens": total_tokens_col,
            "attn_cost": attn_cost_col,
        }
    )

    plan_path = output_dir / "pack_plan.arrow"
    with pa.OSFile(str(plan_path), "wb") as f:
        writer = ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()
    logger.info("Wrote %s (%.1f MB)", plan_path, plan_path.stat().st_size / 1e6)

    # metadata.json
    percentiles = [5, 25, 50, 75, 95]

    def _dist_stats(arr: np.ndarray) -> dict:
        pcts = np.percentile(arr, percentiles)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            **{f"p{p}": float(v) for p, v in zip(percentiles, pcts)},
        }

    metadata = {
        "seq_len": seq_len,
        "total_packs": len(pack_contents),
        "total_examples_packed": examples_packed,
        "overlong_examples_dropped": overlong_count,
        "padding_fraction": padding_fraction,
        "bucket_width": bucket_width,
        "example_attn_cost_stats": _dist_stats(attn_cost),
        "pack_attn_cost_stats": _dist_stats(pack_costs),
        "source_pretok_dir": str(pretok_dir),
        "source_manifest_sha256": _compute_manifest_sha256(manifest_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote %s", metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline pack plan for pretokenized SFT data."
    )
    parser.add_argument(
        "--pretok_dir",
        type=Path,
        required=True,
        help="Path to pretokenized directory (contains shards/ + manifest.json).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help="Target packed sequence length in tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: {pretok_dir}/pack_plans/seqlen_{seq_len}/",
    )
    parser.add_argument(
        "--bucket_width",
        type=int,
        default=128,
        help="Bucket width for BFD capacity discretization (default: 128).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output
    if output_dir is None:
        output_dir = args.pretok_dir / "pack_plans" / f"seqlen_{args.seq_len}"

    plan_packing(
        pretok_dir=args.pretok_dir,
        seq_len=args.seq_len,
        output_dir=output_dir,
        bucket_width=args.bucket_width,
    )


if __name__ == "__main__":
    main()
