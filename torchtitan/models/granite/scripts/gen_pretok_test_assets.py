"""Generate synthetic pre-tokenized test assets for StandardPackingDataset tests.

Writes Arrow shards + manifest.json to tests/assets/pretok_truncate_last/ using the
test tokenizer's vocabulary.  Does NOT call apply_chat_template; examples are
constructed directly so the script has no dependency on the Granite tokenizer.

Usage:
    python -m torchtitan.models.granite.scripts.gen_pretok_test_assets
"""

import json
from pathlib import Path

import numpy as np
from datasets import Dataset

EOS_ID = 2003  # tests/assets/tokenizer eos_id
IGNORE_INDEX = -100
VOCAB_SIZE = 2009

# fmt: off
_EXAMPLES = [
    # (input_ids, labels)  —  label[i] = next token if trained, else IGNORE_INDEX
    # Single-turn, with "reasoning" prefix masked
    ([1, 10, 20, 30, 40, EOS_ID],
     [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 40, EOS_ID, IGNORE_INDEX]),
    # Single-turn, no reasoning block
    ([1, 15, 25, 35, EOS_ID],
     [IGNORE_INDEX, IGNORE_INDEX, 35, EOS_ID, IGNORE_INDEX]),
    # Single-turn, longer
    ([1, 11, 22, 33, 44, 55, 66, EOS_ID],
     [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 44, 55, 66, EOS_ID, IGNORE_INDEX]),
    # Single-turn, short
    ([1, 7, 8, EOS_ID],
     [IGNORE_INDEX, 8, EOS_ID, IGNORE_INDEX]),
    # Multi-token response
    ([1, 5, 10, 15, 20, 25, EOS_ID],
     [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 25, EOS_ID, IGNORE_INDEX]),
    # Another short
    ([1, 9, 19, EOS_ID],
     [IGNORE_INDEX, 19, EOS_ID, IGNORE_INDEX]),
]
# fmt: on


def _fix_trailing_newlines(directory: Path) -> None:
    """Add trailing newlines to JSON files for pre-commit's end-of-file-fixer."""
    for json_path in directory.rglob("*.json"):
        text = json_path.read_text()
        if not text.endswith("\n"):
            json_path.write_text(text + "\n")


def _write_single_shard_asset() -> None:
    """Generate tests/assets/pretok_truncate_last/ — single shard, 6 examples."""
    out_dir = Path(__file__).parents[4] / "tests" / "assets" / "pretok_truncate_last"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for input_ids, labels in _EXAMPLES:
        assert len(input_ids) == len(labels), "input_ids / labels length mismatch"
        n = len(input_ids)
        train_tokens = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
        attn_cost = n * (n + 1) // 2
        examples.append({
            "input_ids": input_ids,
            "labels": labels,
            "n_tokens": n,
            "train_tokens": train_tokens,
            "attn_cost": attn_cost,
        })

    ds = Dataset.from_dict(
        {
            "input_ids": [ex["input_ids"] for ex in examples],
            "labels": [ex["labels"] for ex in examples],
            "n_tokens": [ex["n_tokens"] for ex in examples],
            "train_tokens": [ex["train_tokens"] for ex in examples],
            "attn_cost": [ex["attn_cost"] for ex in examples],
        }
    )
    shard_name = "shard_0000"
    ds.save_to_disk(str(shards_dir / shard_name))

    n_tokens_list = [ex["n_tokens"] for ex in examples]
    trained_tokens = sum(
        sum(1 for lbl in ex["labels"] if lbl != IGNORE_INDEX) for ex in examples
    )
    sorted_lengths = sorted(n_tokens_list)
    n = len(sorted_lengths)

    manifest = {
        "version": 1,
        "strategy": "truncate_last",
        "tokenizer": {
            "source_path": "tests/assets/tokenizer",
            "vocab_size": VOCAB_SIZE,
            "eos_token_id": EOS_ID,
            "chat_template_sha256": None,
        },
        "chat_template_kwargs": {"truncate_history_thinking": True},
        "input_files": {
            "total": 1,
            "paths": [],
            "skipped": [],
        },
        "shards": {
            "completed": [shard_name],
        },
        "stats": {
            "total_examples": n,
            "examples_dropped": 0,
            "total_tokens": sum(n_tokens_list),
            "total_trained_tokens": trained_tokens,
            "tokens_per_example": {
                "mean": round(sum(n_tokens_list) / n, 1),
                "median": sorted_lengths[n // 2],
                "p95": sorted_lengths[int(0.95 * n)],
                "max": sorted_lengths[-1],
            },
        },
        "created_at": "2026-01-01T00:00:00Z",
        "input_dir": "tests/assets/pretok_test_input",
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    _fix_trailing_newlines(shards_dir)

    print(f"Wrote {n} examples → {out_dir}")
    print(f"  token lengths: {n_tokens_list}")
    print(f"  total tokens:  {sum(n_tokens_list)}, trained: {trained_tokens}")


def _write_multishard_asset() -> None:
    """Generate tests/assets/pretok_multishard/ — 3 shards with distinct token ranges.

    Shard 0: tokens in [100, 199], Shard 1: [200, 299], Shard 2: [300, 399].
    4 examples per shard, 12 total. Token ranges make source-shard identification
    trivial after shuffling: input_ids[1] // 100 gives the source shard index + 1.
    """
    out_dir = Path(__file__).parents[4] / "tests" / "assets" / "pretok_multishard"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # fmt: off
    shard_examples = [
        # Shard 0: tokens 100-199
        [
            ([1, 101, 102, 103, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 103, EOS_ID]),
            ([1, 110, 120, 130, 140, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, 130, 140, EOS_ID, IGNORE_INDEX]),
            ([1, 150, 160, EOS_ID], [IGNORE_INDEX, 160, EOS_ID, IGNORE_INDEX]),
            ([1, 170, 180, 190, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, 190, EOS_ID, IGNORE_INDEX]),
        ],
        # Shard 1: tokens 200-299
        [
            ([1, 201, 202, 203, 204, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 204, EOS_ID, IGNORE_INDEX]),
            ([1, 210, 220, EOS_ID], [IGNORE_INDEX, 220, EOS_ID, IGNORE_INDEX]),
            ([1, 250, 260, 270, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, 270, EOS_ID, IGNORE_INDEX]),
            ([1, 280, 290, EOS_ID], [IGNORE_INDEX, 290, EOS_ID, IGNORE_INDEX]),
        ],
        # Shard 2: tokens 300-399
        [
            ([1, 301, 302, 303, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, 303, EOS_ID, IGNORE_INDEX]),
            ([1, 310, 320, 330, 340, 350, EOS_ID], [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 340, 350, EOS_ID, IGNORE_INDEX]),
            ([1, 360, 370, EOS_ID], [IGNORE_INDEX, 370, EOS_ID, IGNORE_INDEX]),
            ([1, 380, 390, EOS_ID], [IGNORE_INDEX, 390, EOS_ID, IGNORE_INDEX]),
        ],
    ]
    # fmt: on

    all_n_tokens: list[int] = []
    shard_names: list[str] = []

    for shard_idx, examples in enumerate(shard_examples):
        shard_name = f"shard_{shard_idx:04d}"
        shard_names.append(shard_name)

        input_ids_list = [inp for inp, _ in examples]
        labels_list = [lbl for _, lbl in examples]
        n_tokens_list = [len(inp) for inp in input_ids_list]
        train_tokens_list = [
            sum(1 for lbl in labels if lbl != IGNORE_INDEX)
            for _, labels in examples
        ]
        attn_cost_list = [n * (n + 1) // 2 for n in n_tokens_list]
        all_n_tokens.extend(n_tokens_list)

        for inp, lbl in examples:
            assert len(inp) == len(lbl), f"length mismatch in shard {shard_idx}"

        ds = Dataset.from_dict(
            {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "n_tokens": n_tokens_list,
                "train_tokens": train_tokens_list,
                "attn_cost": attn_cost_list,
            }
        )
        ds.save_to_disk(str(shards_dir / shard_name))

        n_tokens_arr = np.array(n_tokens_list, dtype=np.int64)
        stats = {
            "shard_stem": shard_name,
            "n_examples": len(examples),
            "n_dropped": 0,
            "total_tokens": int(n_tokens_arr.sum()),
            "total_trained_tokens": sum(train_tokens_list),
            "total_attn_cost": sum(attn_cost_list),
        }
        with open(shards_dir / f"{shard_name}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
            f.write("\n")

    total_examples = sum(len(s) for s in shard_examples)
    total_tokens = sum(all_n_tokens)
    trained_tokens = sum(
        sum(1 for lbl in labels if lbl != IGNORE_INDEX)
        for shard in shard_examples
        for _, labels in shard
    )
    total_attn_cost = sum(n * (n + 1) // 2 for n in all_n_tokens)

    manifest = {
        "version": 1,
        "strategy": "truncate_last",
        "tokenizer": {
            "source_path": "tests/assets/tokenizer",
            "vocab_size": VOCAB_SIZE,
            "eos_token_id": EOS_ID,
            "chat_template_sha256": None,
        },
        "chat_template_kwargs": {"truncate_history_thinking": True},
        "input_files": {
            "total": len(shard_names),
            "paths": [],
            "skipped": [],
        },
        "shards": {
            "completed": shard_names,
        },
        "stats": {
            "total_examples": total_examples,
            "examples_dropped": 0,
            "total_tokens": total_tokens,
            "total_trained_tokens": trained_tokens,
            "total_attn_cost": total_attn_cost,
            "tokens_per_example": round(total_tokens / total_examples, 1),
            "trained_tokens_per_example": round(trained_tokens / total_examples, 1),
        },
        "created_at": "2026-01-01T00:00:00Z",
        "input_dir": "tests/assets/pretok_test_input",
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    _fix_trailing_newlines(shards_dir)

    print(f"Wrote {total_examples} examples across {len(shard_names)} shards → {out_dir}")
    print(f"  total tokens: {total_tokens}, trained: {trained_tokens}")


def main() -> None:
    _write_single_shard_asset()
    _write_multishard_asset()


if __name__ == "__main__":
    main()
