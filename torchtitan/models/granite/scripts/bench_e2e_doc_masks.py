"""Granite 8B e2e benchmark with document masking: fwd and fwd+bwd timing.

Isolates the cost of document-packed attention masks (no CP) for flex and fa4
backends. Compare against bench_e2e.py (pure causal, no masks) to measure mask
overhead.

Usage:
    torchrun --nproc-per-node=1 torchtitan/models/granite/scripts/bench_e2e_doc_masks.py
    torchrun --nproc-per-node=1 torchtitan/models/granite/scripts/bench_e2e_doc_masks.py --attn fa4
    torchrun --nproc-per-node=1 torchtitan/models/granite/scripts/bench_e2e_doc_masks.py --attn fa4 --float8
    torchrun --nproc-per-node=1 torchtitan/models/granite/scripts/bench_e2e_doc_masks.py --attn fa4 --float8 rowwise --num-docs 8
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.attention import (
    build_fa4_mask,
    create_attention_mask,
    FA4Attention,
    get_causal_mask_mod,
    get_document_mask_mod_from_positions,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel

BATCH_SIZE = 1


def build_model(attn_backend: str, mesh, float8: str | None = None) -> GraniteModel:
    config = granite_configs["8B"](attn_backend=attn_backend)
    model = GraniteModel(config)
    model.init_states()
    model.cuda()

    if float8:
        config_kwargs = {"recipe_name": float8} if float8 != "tensorwise" else {}
        converter = Float8LinearConverter(
            Float8LinearConverter.Config(**config_kwargs),
            parallel_dims=ParallelDims(
                dp_shard=-1, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=1
            ),
            model_compile_enabled=True,
        )
        converter.convert(model)

    for block in model.layers.values():
        block.compile(fullgraph=True)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}

    fully_shard(
        [model.tok_embeddings, model.norm, model.output],
        **fsdp_config,
    )
    for block in model.layers.values():
        fully_shard(block, **fsdp_config)
    fully_shard(model, **fsdp_config)

    return config.vocab_size, model


def build_positions(seq_len: int, num_docs: int) -> torch.Tensor:
    base_len = seq_len // num_docs
    remainder = seq_len % num_docs
    doc_lengths = [base_len + (1 if i < remainder else 0) for i in range(num_docs)]
    positions = torch.cat([torch.arange(dl) for dl in doc_lengths])
    return positions.unsqueeze(0).cuda()


def build_attention_masks(
    attn_backend: str, positions: torch.Tensor, seq_len: int
):
    if attn_backend == "fa4":
        document_ids = Decoder._document_ids_from_positions(positions)
        return build_fa4_mask(document_ids=document_ids)

    mask_mod = and_masks(
        get_causal_mask_mod(),
        get_document_mask_mod_from_positions(positions),
    )
    return create_attention_mask(mask_mod, BATCH_SIZE, None, seq_len, seq_len)


def bench_fwd(model, tokens, attention_masks, positions):
    def fn():
        with torch.no_grad():
            model(tokens, attention_masks=attention_masks, positions=positions)

    return triton.testing.do_bench(fn)


def bench_fwd_bwd(model, tokens, vocab_size, attention_masks, positions):
    def fn():
        pred = model(tokens, attention_masks=attention_masks, positions=positions)
        loss = F.cross_entropy(
            pred[:, :-1].reshape(-1, vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        loss.backward()

    return triton.testing.do_bench(fn)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Granite 8B e2e benchmark with document masking"
    )
    parser.add_argument(
        "--attn",
        choices=["flex", "fa4"],
        default="flex",
    )
    parser.add_argument("--seq-len", type=int, default=16 * 1024)
    parser.add_argument("--num-docs", type=int, default=4)
    parser.add_argument(
        "--float8",
        nargs="?",
        const="tensorwise",
        choices=["tensorwise", "rowwise"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.distributed.init_process_group(backend="nccl")
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))

    vocab_size, model = build_model(args.attn, mesh, float8=args.float8)
    tokens = torch.randint(
        0, vocab_size, (BATCH_SIZE, args.seq_len), device="cuda"
    )
    positions = build_positions(args.seq_len, args.num_docs)
    attention_masks = build_attention_masks(args.attn, positions, args.seq_len)

    fwd_ms = bench_fwd(model, tokens, attention_masks, positions)
    fwd_bwd_ms = bench_fwd_bwd(model, tokens, vocab_size, attention_masks, positions)

    if int(os.environ.get("RANK", 0)) == 0:
        fp8_str = f", float8={args.float8}" if args.float8 else ""
        print(
            f"\nGranite 8B Doc Mask Benchmark "
            f"(B={BATCH_SIZE}, S={args.seq_len}, attn={args.attn}, "
            f"num_docs={args.num_docs}{fp8_str})"
        )
        print("-" * 60)
        print(f"  fwd:     {fwd_ms:8.2f} ms")
        print(f"  fwd+bwd: {fwd_bwd_ms:8.2f} ms")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
