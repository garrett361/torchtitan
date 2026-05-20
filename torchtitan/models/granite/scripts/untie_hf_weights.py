"""Convert a tied-weight HF Granite checkpoint to an untied one.

Reads a HuggingFace checkpoint directory where ``lm_head.weight`` is absent
(weight tying), copies ``model.embed_tokens.weight`` as an independent
``lm_head.weight`` tensor, and writes the result to a new directory with
``tie_word_embeddings: false`` in ``config.json``.

Usage:
    python -m torchtitan.models.granite.scripts.untie_hf_weights \\
        /path/to/tied/checkpoint /path/to/output/untied
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def untie_hf_weights(input_dir: Path, output_dir: Path) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir == input_dir:
        raise ValueError("output_dir must differ from input_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = input_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    if not config.get("tie_word_embeddings", True):
        raise ValueError(
            f"{config_path} already has tie_word_embeddings=false. "
            "Checkpoint is already untied."
        )

    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        weight_map = None

    if weight_map is not None and "lm_head.weight" in weight_map:
        raise ValueError(
            "Checkpoint index already contains lm_head.weight but "
            "tie_word_embeddings=true. Unexpected state."
        )

    # Find which shard contains model.embed_tokens.weight
    if weight_map is not None:
        embed_shard = weight_map["model.embed_tokens.weight"]
        shard_names = sorted(set(weight_map.values()))
    else:
        # Single-file checkpoint
        safetensor_files = sorted(input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files in {input_dir}")
        embed_shard = safetensor_files[0].name
        shard_names = [f.name for f in safetensor_files]

    # Copy all safetensor shards, adding lm_head.weight to the embed shard
    import torch

    for shard_name in shard_names:
        src = input_dir / shard_name
        tensors = load_file(str(src), device="cpu")

        if shard_name == embed_shard:
            if "model.embed_tokens.weight" not in tensors:
                raise ValueError(
                    f"Expected model.embed_tokens.weight in shard {embed_shard}, "
                    f"but found keys: {sorted(tensors.keys())}"
                )
            embed = tensors["model.embed_tokens.weight"]
            if embed.dtype != torch.bfloat16:
                raise ValueError(
                    f"Expected bfloat16 embedding weights, got {embed.dtype}. "
                    "This script only supports bfloat16 checkpoints."
                )
            tensors["lm_head.weight"] = embed.clone()

        save_file(tensors, str(output_dir / shard_name))

    # Update index
    if weight_map is not None:
        weight_map["lm_head.weight"] = embed_shard
        metadata = index.get("metadata", {})
        if "total_size" in metadata:
            metadata["total_size"] += (
                config["hidden_size"] * config["vocab_size"] * 2
            )
        index["metadata"] = metadata
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # Write modified config
    config["tie_word_embeddings"] = False
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy remaining files
    skip = {
        "config.json",
        "model.safetensors.index.json",
        *shard_names,
    }
    for item in input_dir.iterdir():
        if item.name in skip or item.name.startswith("."):
            continue
        dst = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a tied-weight HF Granite checkpoint to untied."
    )
    parser.add_argument("input_dir", type=Path, help="Input HF checkpoint directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for untied checkpoint")
    args = parser.parse_args()
    untie_hf_weights(args.input_dir, args.output_dir)
