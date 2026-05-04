"""Generate synthetic pre-tokenized test assets for TruncateLastDataset tests.

Writes Arrow shards + manifest.json to tests/assets/pretok_truncate_last/ using the
test tokenizer's vocabulary.  Does NOT call apply_chat_template; examples are
constructed directly so the script has no dependency on the Granite tokenizer.

Usage:
    python -m torchtitan.models.granite.scripts.gen_pretok_test_assets
"""

import json
from pathlib import Path

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


def main() -> None:
    out_dir = Path(__file__).parents[4] / "tests" / "assets" / "pretok_truncate_last"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for input_ids, labels in _EXAMPLES:
        assert len(input_ids) == len(labels), "input_ids / labels length mismatch"
        examples.append({"input_ids": input_ids, "labels": labels, "n_tokens": len(input_ids)})

    ds = Dataset.from_dict(
        {
            "input_ids": [ex["input_ids"] for ex in examples],
            "labels": [ex["labels"] for ex in examples],
            "n_tokens": [ex["n_tokens"] for ex in examples],
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
        "shards": {
            "completed": [shard_name],
            "total_expected": 1,
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
        "input_files_sha256": {},
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # HuggingFace save_to_disk writes dataset_info.json and state.json without a
    # trailing newline. Add them to keep pre-commit's end-of-file-fixer happy.
    for json_path in shards_dir.rglob("*.json"):
        text = json_path.read_text()
        if not text.endswith("\n"):
            json_path.write_text(text + "\n")

    print(f"Wrote {n} examples → {out_dir}")
    print(f"  token lengths: {n_tokens_list}")
    print(f"  total tokens:  {sum(n_tokens_list)}, trained: {trained_tokens}")


if __name__ == "__main__":
    main()
