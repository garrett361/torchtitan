"""Quick bench: FA4 fwd/bwd with block_sparse_tensors vs without, plus flex comparison.

Compares at the raw kernel level:
  1. FA4 causal=True, no mask (baseline)
  2. FA4 causal=False, mask_mod=doc_causal_mask (current slow path)
  3. FA4 causal=False, mask_mod + block_sparse_tensors (proposed fix)
  4. Flex attention with BlockMask doc_causal (for comparison)

Usage:
    python torchtitan/models/granite/scripts/bench_fa4_block_sparse.py
    python torchtitan/models/granite/scripts/bench_fa4_block_sparse.py --num-docs 8
"""

import argparse

import cutlass
import cutlass.cute as cute
import torch
import triton
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.utils import scalar_to_ssa
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)

B, H, D = 1, 32, 128
FWD_TILE_M, FWD_TILE_N = 256, 128
# BWD expects Q-direction (transposed) indexing: outer dim = KV blocks, inner dim = Q blocks.
# SM100 bwd: m_block_size=128, subtile_factor=2 → sparse_block_size_q=256, n_block_size=128.
BWD_TILE_M, BWD_TILE_N = 128, 256  # transposed: tile_m=n_block_size, tile_n=sparse_block_size_q


def build_doc_ids(seq_len: int, num_docs: int) -> torch.Tensor:
    base_len = seq_len // num_docs
    remainder = seq_len % num_docs
    doc_lengths = [base_len + (1 if i < remainder else 0) for i in range(num_docs)]
    ids = torch.cat([torch.full((dl,), i, dtype=torch.int32) for i, dl in enumerate(doc_lengths)])
    return ids.unsqueeze(0).unsqueeze(1).cuda()  # (1, 1, S)


@cute.jit
def doc_causal_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: list):
    batch_idx, q_idx, kv_idx = batch[0], m_idx[0], n_idx[0]
    doc_id_per_pos = aux_tensors[0]
    q_doc_id = scalar_to_ssa(doc_id_per_pos[batch_idx, 0, q_idx], cutlass.Int32)
    kv_doc_id = scalar_to_ssa(doc_id_per_pos[batch_idx, 0, kv_idx], cutlass.Int32)
    return (kv_idx <= q_idx) & (q_doc_id == kv_doc_id)


@cute.jit
def doc_causal_mask_transposed(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: list):
    """Transposed mask for bwd block sparsity: m_idx=KV, n_idx=Q."""
    batch_idx, kv_idx, q_idx = batch[0], m_idx[0], n_idx[0]
    doc_id_per_pos = aux_tensors[0]
    q_doc_id = scalar_to_ssa(doc_id_per_pos[batch_idx, 0, q_idx], cutlass.Int32)
    kv_doc_id = scalar_to_ssa(doc_id_per_pos[batch_idx, 0, kv_idx], cutlass.Int32)
    return (kv_idx <= q_idx) & (q_doc_id == kv_doc_id)


def build_flex_block_mask(seq_len: int, num_docs: int):
    base_len = seq_len // num_docs
    remainder = seq_len % num_docs
    doc_lengths = [base_len + (1 if i < remainder else 0) for i in range(num_docs)]
    positions = torch.cat([torch.arange(dl) for dl in doc_lengths]).unsqueeze(0).cuda()

    from torchtitan.models.common.attention import (
        get_causal_mask_mod,
        get_document_mask_mod_from_positions,
    )

    mask_mod = and_masks(get_causal_mask_mod(), get_document_mask_mod_from_positions(positions))
    return create_block_mask(mask_mod, B, None, seq_len, seq_len)


_compiled_flex = torch.compile(flex_attention)


def check_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        print(f"  PASS  {name}  (max_diff={max_diff:.2e})")
    except AssertionError:
        print(f"  FAIL  {name}  (max_diff={max_diff:.2e}, atol={atol}, rtol={rtol})")


def print_sparsity_stats(label: str, bst, tile_m: int, tile_n: int, S: int):
    total = (S // tile_m) * (S // tile_n) * H
    mask_b = bst.mask_block_cnt.sum().item()
    full_b = bst.full_block_cnt.sum().item() if bst.full_block_cnt is not None else 0
    skip = total - mask_b - full_b
    print(f"  {label}: total={total}, partial={mask_b}, full={full_b}, skip={skip} "
          f"({100 * skip / total:.0f}% skipped)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=16 * 1024)
    parser.add_argument("--num-docs", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()
    S = args.seq_len
    atol, rtol = args.atol, args.rtol

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: B={B}, S={S}, H={H}, D={D}, num_docs={args.num_docs}\n")

    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    doc_ids = build_doc_ids(S, args.num_docs)
    aux = [doc_ids]

    # --- FA4 block sparsity ---
    print("Computing FA4 block sparsity...")
    bst_fwd = compute_block_sparsity(
        FWD_TILE_M, FWD_TILE_N, B, H, S, S,
        mask_mod=doc_causal_mask, aux_tensors=aux, device="cuda",
    )
    bst_bwd = compute_block_sparsity(
        BWD_TILE_M, BWD_TILE_N, B, H, S, S,
        mask_mod=doc_causal_mask_transposed, aux_tensors=aux, device="cuda",
    )
    # normalize_block_sparse_config_bwd reads block_size as (sparse_q, sparse_kv)
    bst_bwd = bst_bwd._replace(block_size=(BWD_TILE_N, BWD_TILE_M))
    print_sparsity_stats("fwd", bst_fwd, FWD_TILE_M, FWD_TILE_N, S)
    print_sparsity_stats("bwd", bst_bwd, BWD_TILE_M, BWD_TILE_N, S)

    # --- Flex block mask ---
    print("\nBuilding flex BlockMask...")
    flex_block_mask = build_flex_block_mask(S, args.num_docs)

    # Flex needs (B, H, S, D) layout
    q_flex = q.transpose(1, 2).contiguous()
    k_flex = k.transpose(1, 2).contiguous()
    v_flex = v.transpose(1, 2).contiguous()

    # ==================== Correctness ====================
    print("\n--- Correctness checks ---")

    # FWD: run all paths
    out_causal, lse_causal = _flash_attn_fwd(q, k, v, causal=True, return_lse=True)
    out_mask, lse_mask = _flash_attn_fwd(q, k, v, causal=False, mask_mod=doc_causal_mask,
                                         aux_tensors=aux, return_lse=True)
    out_sparse, lse_sparse = _flash_attn_fwd(q, k, v, causal=False, mask_mod=doc_causal_mask,
                                             block_sparse_tensors=bst_fwd, aux_tensors=aux,
                                             return_lse=True)
    with torch.no_grad():
        out_flex_raw = _compiled_flex(q_flex, k_flex, v_flex, block_mask=flex_block_mask)
    out_flex_bshd = out_flex_raw.transpose(1, 2)  # (B,H,S,D) → (B,S,H,D)

    check_close("fwd: sparse vs mask_mod", out_sparse, out_mask, atol, rtol)
    check_close("fwd: FA4 mask_mod vs flex", out_mask, out_flex_bshd, atol, rtol)

    # BWD: compare sparse vs mask_mod gradients
    dout = torch.randn_like(out_mask)
    dq_mask, dk_mask, dv_mask = _flash_attn_bwd(
        q, k, v, out_mask, dout, lse_mask, causal=False,
        mask_mod=doc_causal_mask, aux_tensors=aux,
    )
    dq_sparse, dk_sparse, dv_sparse = _flash_attn_bwd(
        q, k, v, out_sparse, dout, lse_sparse, causal=False,
        mask_mod=doc_causal_mask, block_sparse_tensors=bst_bwd, aux_tensors=aux,
    )

    check_close("bwd dq: sparse vs mask_mod", dq_sparse, dq_mask, atol, rtol)
    check_close("bwd dk: sparse vs mask_mod", dk_sparse, dk_mask, atol, rtol)
    check_close("bwd dv: sparse vs mask_mod", dv_sparse, dv_mask, atol, rtol)

    # ==================== Benchmarks ====================
    print("\n--- Benchmarks ---")

    # --- FWD benchmarks ---
    def fa4_causal_fwd():
        _flash_attn_fwd(q, k, v, causal=True, return_lse=True)

    def fa4_mask_fwd():
        _flash_attn_fwd(q, k, v, causal=False, mask_mod=doc_causal_mask,
                        aux_tensors=aux, return_lse=True)

    def fa4_sparse_fwd():
        _flash_attn_fwd(q, k, v, causal=False, mask_mod=doc_causal_mask,
                        block_sparse_tensors=bst_fwd, aux_tensors=aux, return_lse=True)

    def flex_fwd():
        with torch.no_grad():
            _compiled_flex(q_flex, k_flex, v_flex, block_mask=flex_block_mask)

    for fn in [fa4_causal_fwd, fa4_mask_fwd, fa4_sparse_fwd, flex_fwd]:
        fn()
    torch.cuda.synchronize()

    t_causal_f = triton.testing.do_bench(fa4_causal_fwd)
    t_mask_f = triton.testing.do_bench(fa4_mask_fwd)
    t_sparse_f = triton.testing.do_bench(fa4_sparse_fwd)
    t_flex_f = triton.testing.do_bench(flex_fwd)

    # --- BWD benchmarks ---
    def fa4_causal_bwd():
        _flash_attn_bwd(q, k, v, out_causal, dout, lse_causal, causal=True)

    def fa4_mask_bwd():
        _flash_attn_bwd(q, k, v, out_mask, dout, lse_mask, causal=False,
                        mask_mod=doc_causal_mask, aux_tensors=aux)

    def fa4_sparse_bwd():
        _flash_attn_bwd(q, k, v, out_sparse, dout, lse_sparse, causal=False,
                        mask_mod=doc_causal_mask, block_sparse_tensors=bst_bwd,
                        aux_tensors=aux)

    # Flex bwd: pre-compute fwd graph, time only backward via retain_graph
    q_f = q_flex.detach().requires_grad_(True)
    k_f = k_flex.detach().requires_grad_(True)
    v_f = v_flex.detach().requires_grad_(True)
    out_flex_grad = _compiled_flex(q_f, k_f, v_f, block_mask=flex_block_mask)
    dout_flex = torch.randn_like(out_flex_grad)

    def flex_bwd():
        for t in [q_f, k_f, v_f]:
            if t.grad is not None:
                t.grad.zero_()
        out_flex_grad.backward(dout_flex, retain_graph=True)

    for fn in [fa4_causal_bwd, fa4_mask_bwd, fa4_sparse_bwd, flex_bwd]:
        fn()
    torch.cuda.synchronize()

    t_causal_b = triton.testing.do_bench(fa4_causal_bwd)
    t_mask_b = triton.testing.do_bench(fa4_mask_bwd)
    t_sparse_b = triton.testing.do_bench(fa4_sparse_bwd)
    t_flex_b = triton.testing.do_bench(flex_bwd)

    # --- Results ---
    print(f"\n{'':30s} {'fwd':>8s} {'bwd':>8s} {'fwd+bwd':>8s}")
    for label, tf, tb in [
        ("FA4 causal (no mask)", t_causal_f, t_causal_b),
        ("FA4 mask_mod only", t_mask_f, t_mask_b),
        ("FA4 mask + block_sparse", t_sparse_f, t_sparse_b),
        ("Flex BlockMask", t_flex_f, t_flex_b),
    ]:
        print(f"  {label:28s} {tf:7.2f}ms {tb:7.2f}ms {tf+tb:7.2f}ms")


if __name__ == "__main__":
    main()
