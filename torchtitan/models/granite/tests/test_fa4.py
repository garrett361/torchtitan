"""Phase 0 validation: FA4 capabilities on SM100 (Blackwell).

Tests confirm FA4 works correctly with CuTe DSL mask_mod, arbitrary masks,
Q/KV length mismatch (CP simulation), and composed masks before integrating
into torchtitan.

Run:
    python -m unittest torchtitan.models.granite.tests.test_fa4 -v
"""

import unittest
import warnings

import torch

from torchtitan.models.granite.tests.helpers import has_fa4, ref_attention

warnings.filterwarnings("ignore", category=DeprecationWarning)

B, H, HKV, D = 1, 32, 8, 128
SCALE = 1.0 / D
DTYPE = torch.bfloat16


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4Masks(unittest.TestCase):
    """Validate FA4 mask_mod correctness on SM100."""

    def setUp(self):
        torch.manual_seed(42)

    def _assert_close(self, a, b, name, atol=1e-2, rtol=1e-2):
        # min passing tol ~7e-3; 1e-2 gives headroom for non-determinism
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)

    def _ref_attention(self, q, k, v, mask):
        return ref_attention(q, k, v, mask, num_heads=H, scale=SCALE, dtype=DTYPE)

    def test_causal_mask_mod_matches_causal_flag(self):
        """FA4 causal mask_mod produces same output+grads as causal=True."""
        import cutlass.cute as cute
        from flash_attn.cute import flash_attn_func

        @cute.jit
        def causal_mask_mod(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: None):
            return m_idx >= n_idx

        seq_len = 2048
        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)

        # Reference: causal=True
        out_ref, _ = flash_attn_func(q, k, v, causal=True, softmax_scale=SCALE)
        out_ref.sum().backward()
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Test: mask_mod
        out_test, _ = flash_attn_func(
            q, k, v, causal=False, softmax_scale=SCALE, mask_mod=causal_mask_mod
        )
        out_test.sum().backward()

        self._assert_close(out_test, out_ref, "output")
        self._assert_close(q.grad, dq_ref, "dq")
        self._assert_close(k.grad, dk_ref, "dk")
        self._assert_close(v.grad, dv_ref, "dv")


    def test_document_mask_with_aux_tensors(self):
        """FA4 document+causal mask via aux_tensors matches dense reference."""
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import flash_attn_func
        from flash_attn.cute.utils import scalar_to_ssa

        @cute.jit
        def doc_causal_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: list):
            doc_ids = aux_tensors[0]  # (B, 1, S) int32
            m_doc = scalar_to_ssa(doc_ids[batch[0], 0, m_idx[0]], cutlass.Int32)
            n_doc = scalar_to_ssa(doc_ids[batch[0], 0, n_idx[0]], cutlass.Int32)
            return (m_doc == n_doc) & (m_idx >= n_idx)

        seq_len = 512
        n_docs = 4
        doc_len = seq_len // n_docs
        doc_ids = torch.arange(seq_len, device="cuda", dtype=torch.int32) // doc_len
        doc_ids = doc_ids.unsqueeze(0).unsqueeze(0)  # (1, 1, S)

        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)

        # FA4 with doc_causal_mask
        out_fa4, _ = flash_attn_func(
            q, k, v, causal=False, softmax_scale=SCALE,
            mask_mod=doc_causal_mask, aux_tensors=[doc_ids],
        )
        out_fa4.sum().backward()
        dq_fa4, dk_fa4, dv_fa4 = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Dense reference
        ids_flat = doc_ids.squeeze()  # (S,)
        idx = torch.arange(seq_len, device="cuda")
        same_doc = ids_flat.unsqueeze(0) == ids_flat.unsqueeze(1)  # (S, S)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # mask[i,j] = (i >= j)
        mask = same_doc & causal
        out_ref = self._ref_attention(q, k, v, mask)
        out_ref.sum().backward()

        self._assert_close(out_fa4, out_ref, "output")
        self._assert_close(q.grad, dq_fa4, "dq")
        self._assert_close(k.grad, dk_fa4, "dk")
        self._assert_close(v.grad, dv_fa4, "dv")


    def test_causal_offset_mask_cp_simulation(self):
        """FA4 causal offset mask (Q_len < KV_len) matches reference for CP last rank."""
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import flash_attn_func
        from flash_attn.cute.utils import scalar_to_ssa

        @cute.jit
        def causal_offset_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: None):
            offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
            offset_ssa = scalar_to_ssa(offset, cutlass.Int32)
            return n_idx <= (m_idx + offset_ssa)

        full_seq = 4096
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = cp_world - 1
        q_start = rank * local_seq

        q_full_data = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_data = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_data = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        # FA4: local Q slice, full K/V
        q_local = q_full_data[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_fa4 = k_data.clone().requires_grad_(True)
        v_fa4 = v_data.clone().requires_grad_(True)

        out_fa4, _ = flash_attn_func(
            q_local, k_fa4, v_fa4, causal=False, softmax_scale=SCALE,
            mask_mod=causal_offset_mask,
        )
        out_fa4.sum().backward()

        # Reference: full causal attention, extract rank's output rows
        q_ref = q_full_data.clone().requires_grad_(True)
        k_ref = k_data.clone().requires_grad_(True)
        v_ref = v_data.clone().requires_grad_(True)

        idx = torch.arange(full_seq, device="cuda")
        causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
        out_ref_full = self._ref_attention(q_ref, k_ref, v_ref, causal_mask)
        out_ref_full[:, q_start:q_start + local_seq].sum().backward()

        out_ref = out_ref_full[:, q_start:q_start + local_seq].detach()
        self._assert_close(out_fa4, out_ref, "output")
        self._assert_close(q_local.grad, q_ref.grad[:, q_start:q_start + local_seq], "dq")
        self._assert_close(k_fa4.grad, k_ref.grad, "dk")
        self._assert_close(v_fa4.grad, v_ref.grad, "dv")


    def test_compile_causal_fwd_bwd(self):
        """Production FA4 causal custom_ops compile with fullgraph=True (fwd+bwd)."""
        from flash_attn.cute import flash_attn_func

        import torchtitan.models.common.fa4_ops  # noqa: F401

        seq_len = 2048

        @torch.compile(fullgraph=True)
        def compiled_fwd(q, k, v, scale):
            out, _ = torch.ops.torchtitan.fa4_causal_fwd(q, k, v, scale)
            return out

        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)

        out = compiled_fwd(q, k, v, SCALE)
        out.sum().backward()
        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Eager reference
        out_ref, _ = flash_attn_func(q, k, v, causal=True, softmax_scale=SCALE)
        out_ref.sum().backward()

        self._assert_close(out, out_ref, "output")
        self._assert_close(dq, q.grad, "dq")
        self._assert_close(dk, k.grad, "dk")
        self._assert_close(dv, v.grad, "dv")

    def test_compile_doc_causal_fwd_bwd(self):
        """Production FA4 doc_causal custom_ops compile with fullgraph=True (fwd+bwd)."""
        from flash_attn.cute import flash_attn_func

        from torchtitan.models.common.attention import build_fa4_mask
        from torchtitan.models.common.fa4_ops import register_fa4_masked_ops

        seq_len = 512
        n_docs = 4
        doc_len = seq_len // n_docs
        document_ids = (
            torch.arange(seq_len, device="cuda", dtype=torch.int32) // doc_len
        ).unsqueeze(0)

        fa4_mask = build_fa4_mask(document_ids=document_ids)
        register_fa4_masked_ops(fa4_mask.mask_mod, fa4_mask.variant)

        aux = fa4_mask.aux_tensors
        fwd_op = getattr(torch.ops.torchtitan, f"fa4_{fa4_mask.variant}_fwd")

        @torch.compile(fullgraph=True)
        def compiled_fwd(q, k, v, scale, aux0, aux1, aux2):
            out, _ = fwd_op(q, k, v, scale, aux0, aux1, aux2)
            return out

        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)

        out = compiled_fwd(
            q, k, v, SCALE, aux[0],
            aux[1] if len(aux) > 1 else None,
            aux[2] if len(aux) > 2 else None,
        )
        out.sum().backward()
        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Eager reference
        out_ref, _ = flash_attn_func(
            q, k, v, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_ref.sum().backward()

        self._assert_close(out, out_ref, "output")
        self._assert_close(dq, q.grad, "dq")
        self._assert_close(dk, k.grad, "dk")
        self._assert_close(dv, v.grad, "dv")

    def test_compile_cp_causal_fwd_bwd(self):
        """Production FA4 cp_causal custom_ops compile with fullgraph=True (fwd+bwd)."""
        from flash_attn.cute import flash_attn_func
        from torch.distributed.tensor.experimental._attention import (
            _HeadTailLoadBalancer,
        )

        from torchtitan.models.common.attention import build_fa4_mask
        from torchtitan.models.common.fa4_ops import register_fa4_masked_ops

        full_seq = 2048
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 2

        lb = _HeadTailLoadBalancer(full_seq, cp_world, "cuda")
        shard_indices = lb._generate_indices(restore=False)

        fa4_mask = build_fa4_mask(
            shard_indices=shard_indices,
            cp_rank=rank,
            local_seq_len=local_seq,
        )
        register_fa4_masked_ops(fa4_mask.mask_mod, fa4_mask.variant)

        aux = fa4_mask.aux_tensors
        fwd_op = getattr(torch.ops.torchtitan, f"fa4_{fa4_mask.variant}_fwd")
        si_flat = shard_indices.squeeze(0)
        q_start = rank * local_seq

        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        q_perm = q_full[:, si_flat]
        k_perm = k_full[:, si_flat]
        v_perm = v_full[:, si_flat]

        q_local = q_perm[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_cp = k_perm.clone().requires_grad_(True)
        v_cp = v_perm.clone().requires_grad_(True)

        @torch.compile(fullgraph=True)
        def compiled_fwd(q, k, v, scale, aux0, aux1, aux2):
            out, _ = fwd_op(q, k, v, scale, aux0, aux1, aux2)
            return out

        out = compiled_fwd(
            q_local, k_cp, v_cp, SCALE, aux[0],
            aux[1] if len(aux) > 1 else None,
            aux[2] if len(aux) > 2 else None,
        )
        out.sum().backward()
        dq = q_local.grad.clone()
        dk, dv = k_cp.grad.clone(), v_cp.grad.clone()
        q_local.grad, k_cp.grad, v_cp.grad = None, None, None

        # Eager reference
        out_ref, _ = flash_attn_func(
            q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_ref.sum().backward()

        self._assert_close(out, out_ref, "output")
        self._assert_close(dq, q_local.grad, "dq")
        self._assert_close(dk, k_cp.grad, "dk")
        self._assert_close(dv, v_cp.grad, "dv")

    def test_compile_cp_doc_causal_fwd_bwd(self):
        """Production FA4 cp+doc causal custom_ops compile with fullgraph=True (fwd+bwd)."""
        from flash_attn.cute import flash_attn_func
        from torch.distributed.tensor.experimental._attention import (
            _HeadTailLoadBalancer,
        )

        from torchtitan.models.common.attention import build_fa4_mask
        from torchtitan.models.common.fa4_ops import register_fa4_masked_ops

        full_seq = 2048
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 1
        n_docs = 4
        doc_len = full_seq // n_docs

        lb = _HeadTailLoadBalancer(full_seq, cp_world, "cuda")
        shard_indices = lb._generate_indices(restore=False)

        document_ids = (
            torch.arange(full_seq, device="cuda", dtype=torch.int32) // doc_len
        ).unsqueeze(0)

        fa4_mask = build_fa4_mask(
            shard_indices=shard_indices,
            cp_rank=rank,
            local_seq_len=local_seq,
            document_ids=document_ids,
        )
        register_fa4_masked_ops(fa4_mask.mask_mod, fa4_mask.variant)

        aux = fa4_mask.aux_tensors
        fwd_op = getattr(torch.ops.torchtitan, f"fa4_{fa4_mask.variant}_fwd")
        si_flat = shard_indices.squeeze(0)
        q_start = rank * local_seq

        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        q_perm = q_full[:, si_flat]
        k_perm = k_full[:, si_flat]
        v_perm = v_full[:, si_flat]

        q_local = q_perm[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_cp = k_perm.clone().requires_grad_(True)
        v_cp = v_perm.clone().requires_grad_(True)

        @torch.compile(fullgraph=True)
        def compiled_fwd(q, k, v, scale, aux0, aux1, aux2):
            out, _ = fwd_op(q, k, v, scale, aux0, aux1, aux2)
            return out

        out = compiled_fwd(
            q_local, k_cp, v_cp, SCALE, aux[0],
            aux[1] if len(aux) > 1 else None,
            aux[2] if len(aux) > 2 else None,
        )
        out.sum().backward()
        dq = q_local.grad.clone()
        dk, dv = k_cp.grad.clone(), v_cp.grad.clone()
        q_local.grad, k_cp.grad, v_cp.grad = None, None, None

        # Eager reference
        out_ref, _ = flash_attn_func(
            q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_ref.sum().backward()

        self._assert_close(out, out_ref, "output")
        self._assert_close(dq, q_local.grad, "dq")
        self._assert_close(dk, k_cp.grad, "dk")
        self._assert_close(dv, v_cp.grad, "dv")


    def test_cp_document_causal_composed(self):
        """FA4 CP + document + causal mask composed via aux_tensors matches reference."""
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import flash_attn_func
        from flash_attn.cute.utils import scalar_to_ssa

        @cute.jit
        def cp_doc_causal_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: list):
            q_doc_ids = aux_tensors[0]   # (B, 1, local_seq)
            kv_doc_ids = aux_tensors[1]  # (B, 1, full_seq)
            q_positions = aux_tensors[2] # (B, 1, local_seq)

            m_doc = scalar_to_ssa(q_doc_ids[batch[0], 0, m_idx[0]], cutlass.Int32)
            n_doc = scalar_to_ssa(kv_doc_ids[batch[0], 0, n_idx[0]], cutlass.Int32)
            m_global_pos = scalar_to_ssa(q_positions[batch[0], 0, m_idx[0]], cutlass.Int32)

            return (m_doc == n_doc) & (n_idx <= m_global_pos)

        full_seq = 4096
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 2
        q_start = rank * local_seq

        # Document boundaries: 8 docs of 512 tokens each
        n_docs = 8
        doc_len = full_seq // n_docs
        all_doc_ids = torch.arange(full_seq, device="cuda", dtype=torch.int32) // doc_len

        q_doc_ids = all_doc_ids[q_start:q_start + local_seq].unsqueeze(0).unsqueeze(0)
        kv_doc_ids = all_doc_ids.unsqueeze(0).unsqueeze(0)
        q_positions = torch.arange(
            q_start, q_start + local_seq, device="cuda", dtype=torch.int32,
        ).unsqueeze(0).unsqueeze(0)

        q_full_data = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_data = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_data = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        # FA4: local Q, full K/V, composed mask
        q_local = q_full_data[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_fa4 = k_data.clone().requires_grad_(True)
        v_fa4 = v_data.clone().requires_grad_(True)

        out_fa4, _ = flash_attn_func(
            q_local, k_fa4, v_fa4, causal=False, softmax_scale=SCALE,
            mask_mod=cp_doc_causal_mask,
            aux_tensors=[q_doc_ids, kv_doc_ids, q_positions],
        )
        out_fa4.sum().backward()

        # Reference: full doc+causal attention, extract rank's rows
        q_ref = q_full_data.clone().requires_grad_(True)
        k_ref = k_data.clone().requires_grad_(True)
        v_ref = v_data.clone().requires_grad_(True)

        idx = torch.arange(full_seq, device="cuda")
        same_doc = all_doc_ids.unsqueeze(0) == all_doc_ids.unsqueeze(1)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        mask = same_doc & causal
        out_ref_full = self._ref_attention(q_ref, k_ref, v_ref, mask)
        out_ref_full[:, q_start:q_start + local_seq].sum().backward()

        out_ref = out_ref_full[:, q_start:q_start + local_seq].detach()
        self._assert_close(out_fa4, out_ref, "output")
        self._assert_close(q_local.grad, q_ref.grad[:, q_start:q_start + local_seq], "dq")
        self._assert_close(k_fa4.grad, k_ref.grad, "dk")
        self._assert_close(v_fa4.grad, v_ref.grad, "dv")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(has_fa4(), "flash_attn.cute (FA4) not installed")
class TestBuildFA4Mask(unittest.TestCase):
    """Validate build_fa4_mask produces correct masks for CP and document isolation."""

    def setUp(self):
        torch.manual_seed(42)

    def _ref_attention(self, q, k, v, mask):
        return ref_attention(q, k, v, mask, num_heads=H, scale=SCALE, dtype=DTYPE)

    def test_doc_causal_mask_matches_reference(self):
        """build_fa4_mask with document_ids matches dense doc+causal reference."""
        from flash_attn.cute import flash_attn_func

        from torchtitan.models.common.attention import build_fa4_mask

        seq_len = 512
        n_docs = 4
        doc_len = seq_len // n_docs
        # (B, S) document IDs — build_fa4_mask will unsqueeze to (B, 1, S)
        document_ids = (
            torch.arange(seq_len, device="cuda", dtype=torch.int32) // doc_len
        ).unsqueeze(0)

        fa4_mask = build_fa4_mask(document_ids=document_ids)

        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE, requires_grad=True)

        out_fa4, _ = flash_attn_func(
            q, k, v, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_fa4.sum().backward()
        dq_fa4 = q.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Dense reference
        ids_flat = document_ids.squeeze()
        idx = torch.arange(seq_len, device="cuda")
        same_doc = ids_flat.unsqueeze(0) == ids_flat.unsqueeze(1)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        mask = same_doc & causal
        out_ref = self._ref_attention(q, k, v, mask)
        out_ref.sum().backward()

        torch.testing.assert_close(out_fa4, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(dq_fa4, q.grad, atol=1e-2, rtol=1e-2)

    def test_cp_causal_headtail_matches_reference(self):
        """build_fa4_mask with headtail restore_indices matches causal reference."""
        from flash_attn.cute import flash_attn_func
        from torch.distributed.tensor.experimental._attention import (
            _HeadTailLoadBalancer,
        )

        from torchtitan.models.common.attention import build_fa4_mask

        full_seq = 2048
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 2

        lb = _HeadTailLoadBalancer(full_seq, cp_world, "cuda")
        shard_indices = lb._generate_indices(restore=False).squeeze(0)

        # Simulate CP: permute full sequence, split Q by rank, keep full K/V
        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        # Permute
        q_perm = q_full[:, shard_indices]
        k_perm = k_full[:, shard_indices]
        v_perm = v_full[:, shard_indices]

        # Rank's local Q slice, full permuted K/V (simulates all-gather)
        q_start = rank * local_seq
        q_local = q_perm[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_cp = k_perm.clone().requires_grad_(True)
        v_cp = v_perm.clone().requires_grad_(True)

        fa4_mask = build_fa4_mask(
            shard_indices=lb._generate_indices(restore=False),
            cp_rank=rank,
            local_seq_len=local_seq,
        )

        out_fa4, _ = flash_attn_func(
            q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_fa4.sum().backward()

        # Reference: full causal on original (unpermuted) sequence
        q_ref = q_full.clone().requires_grad_(True)
        k_ref = k_full.clone().requires_grad_(True)
        v_ref = v_full.clone().requires_grad_(True)

        idx = torch.arange(full_seq, device="cuda")
        causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
        out_ref_full = self._ref_attention(q_ref, k_ref, v_ref, causal_mask)

        # Extract the rows corresponding to this rank's Q positions
        # shard_indices[q_start:q_start+local_seq] are the ORIGINAL positions this rank handles
        rank_orig_positions = shard_indices[q_start:q_start + local_seq]
        out_ref_full[:, rank_orig_positions].sum().backward()
        out_ref = out_ref_full[:, rank_orig_positions].detach()

        torch.testing.assert_close(out_fa4, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            q_local.grad, q_ref.grad[:, rank_orig_positions], atol=1e-2, rtol=1e-2,
        )
        # K/V grads: permuted reference grads must match FA4's permuted grads
        torch.testing.assert_close(
            k_cp.grad, k_ref.grad[:, shard_indices], atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            v_cp.grad, v_ref.grad[:, shard_indices], atol=1e-2, rtol=1e-2,
        )

    def test_cp_doc_causal_headtail_matches_reference(self):
        """build_fa4_mask with restore_indices + document_ids matches composed reference."""
        from flash_attn.cute import flash_attn_func
        from torch.distributed.tensor.experimental._attention import (
            _HeadTailLoadBalancer,
        )

        from torchtitan.models.common.attention import build_fa4_mask

        full_seq = 2048
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 1
        n_docs = 4
        doc_len = full_seq // n_docs

        lb = _HeadTailLoadBalancer(full_seq, cp_world, "cuda")
        shard_indices = lb._generate_indices(restore=False).squeeze(0)

        # Document IDs indexed by original position (B, full_seq)
        document_ids = (
            torch.arange(full_seq, device="cuda", dtype=torch.int32) // doc_len
        ).unsqueeze(0)

        # Simulate CP
        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        q_perm = q_full[:, shard_indices]
        k_perm = k_full[:, shard_indices]
        v_perm = v_full[:, shard_indices]

        q_start = rank * local_seq
        q_local = q_perm[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_cp = k_perm.clone().requires_grad_(True)
        v_cp = v_perm.clone().requires_grad_(True)

        fa4_mask = build_fa4_mask(
            shard_indices=lb._generate_indices(restore=False),
            cp_rank=rank,
            local_seq_len=local_seq,
            document_ids=document_ids,
        )

        out_fa4, _ = flash_attn_func(
            q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_fa4.sum().backward()

        # Reference: full doc+causal on original sequence
        q_ref = q_full.clone().requires_grad_(True)
        k_ref = k_full.clone().requires_grad_(True)
        v_ref = v_full.clone().requires_grad_(True)

        idx = torch.arange(full_seq, device="cuda")
        doc_ids_flat = document_ids.squeeze()
        same_doc = doc_ids_flat.unsqueeze(0) == doc_ids_flat.unsqueeze(1)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        mask = same_doc & causal
        out_ref_full = self._ref_attention(q_ref, k_ref, v_ref, mask)

        rank_orig_positions = shard_indices[q_start:q_start + local_seq]
        out_ref_full[:, rank_orig_positions].sum().backward()
        out_ref = out_ref_full[:, rank_orig_positions].detach()

        torch.testing.assert_close(out_fa4, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            q_local.grad, q_ref.grad[:, rank_orig_positions], atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            k_cp.grad, k_ref.grad[:, shard_indices], atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            v_cp.grad, v_ref.grad[:, shard_indices], atol=1e-2, rtol=1e-2,
        )

    def test_cp_doc_causal_ptrr_matches_reference(self):
        """build_fa4_mask with PTRR-generated shard_indices matches composed reference."""
        from flash_attn.cute import flash_attn_func
        from torch.distributed.tensor.experimental._attention import (
            _PTRRLoadBalancer,
        )
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            build_fa4_mask,
            create_attention_mask,
            get_causal_mask_mod,
            get_document_mask_mod_from_positions,
        )

        full_seq = 2048
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 2
        n_docs = 4
        doc_len = full_seq // n_docs

        # Positions with doc resets (4 docs of equal length)
        single_doc_pos = torch.arange(doc_len, device="cuda")
        positions = single_doc_pos.repeat(n_docs).unsqueeze(0)  # (1, full_seq)

        # Build doc-aware BlockMask for PTRR scheduling (same as trainer.py)
        mask_mods = [get_causal_mask_mod()]
        mask_mods.append(get_document_mask_mod_from_positions(positions))
        block_mask = create_attention_mask(
            and_masks(*mask_mods), 1, None, full_seq, full_seq,
        )

        lb = _PTRRLoadBalancer(block_mask, cp_world)
        shard_indices = lb._generate_indices(restore=False).squeeze(0)

        # Document IDs indexed by original position (B, full_seq)
        document_ids = (
            torch.arange(full_seq, device="cuda", dtype=torch.int32) // doc_len
        ).unsqueeze(0)

        # Simulate CP
        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        q_perm = q_full[:, shard_indices]
        k_perm = k_full[:, shard_indices]
        v_perm = v_full[:, shard_indices]

        q_start = rank * local_seq
        q_local = q_perm[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_cp = k_perm.clone().requires_grad_(True)
        v_cp = v_perm.clone().requires_grad_(True)

        fa4_mask = build_fa4_mask(
            shard_indices=lb._generate_indices(restore=False),
            cp_rank=rank,
            local_seq_len=local_seq,
            document_ids=document_ids,
        )

        out_fa4, _ = flash_attn_func(
            q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_fa4.sum().backward()

        # Reference: full doc+causal on original sequence
        q_ref = q_full.clone().requires_grad_(True)
        k_ref = k_full.clone().requires_grad_(True)
        v_ref = v_full.clone().requires_grad_(True)

        idx = torch.arange(full_seq, device="cuda")
        doc_ids_flat = document_ids.squeeze()
        same_doc = doc_ids_flat.unsqueeze(0) == doc_ids_flat.unsqueeze(1)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        mask = same_doc & causal
        out_ref_full = self._ref_attention(q_ref, k_ref, v_ref, mask)

        rank_orig_positions = shard_indices[q_start:q_start + local_seq]
        out_ref_full[:, rank_orig_positions].sum().backward()
        out_ref = out_ref_full[:, rank_orig_positions].detach()

        torch.testing.assert_close(out_fa4, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            q_local.grad, q_ref.grad[:, rank_orig_positions], atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            k_cp.grad, k_ref.grad[:, shard_indices], atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            v_cp.grad, v_ref.grad[:, shard_indices], atol=1e-2, rtol=1e-2,
        )

    def test_cp_causal_headtail_rank0(self):
        """Boundary rank 0 (head-heavy) matches causal reference."""
        from flash_attn.cute import flash_attn_func
        from torch.distributed.tensor.experimental._attention import (
            _HeadTailLoadBalancer,
        )

        from torchtitan.models.common.attention import build_fa4_mask

        full_seq = 2048
        cp_world = 4
        local_seq = full_seq // cp_world
        rank = 0

        lb = _HeadTailLoadBalancer(full_seq, cp_world, "cuda")
        shard_indices = lb._generate_indices(restore=False)  # (1, full_seq)
        si_flat = shard_indices.squeeze(0)  # (full_seq,) for indexing

        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        q_perm = q_full[:, si_flat]
        k_perm = k_full[:, si_flat]
        v_perm = v_full[:, si_flat]

        q_start = rank * local_seq
        q_local = q_perm[:, q_start:q_start + local_seq].clone().requires_grad_(True)
        k_cp = k_perm.clone().requires_grad_(True)
        v_cp = v_perm.clone().requires_grad_(True)

        fa4_mask = build_fa4_mask(
            shard_indices=shard_indices,
            cp_rank=rank,
            local_seq_len=local_seq,
        )

        out_fa4, _ = flash_attn_func(
            q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
            mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
        )
        out_fa4.sum().backward()

        q_ref = q_full.clone().requires_grad_(True)
        k_ref = k_full.clone().requires_grad_(True)
        v_ref = v_full.clone().requires_grad_(True)

        idx = torch.arange(full_seq, device="cuda")
        causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
        out_ref_full = self._ref_attention(q_ref, k_ref, v_ref, causal_mask)

        rank_orig_positions = si_flat[q_start:q_start + local_seq]
        out_ref_full[:, rank_orig_positions].sum().backward()
        out_ref = out_ref_full[:, rank_orig_positions].detach()

        torch.testing.assert_close(out_fa4, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            q_local.grad, q_ref.grad[:, rank_orig_positions], atol=1e-2, rtol=1e-2,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(has_fa4(), "flash_attn.cute (FA4) not installed")
class TestDocumentIdsFromPositions(unittest.TestCase):
    """Unit tests for document_ids_from_positions."""

    def test_single_document(self):
        """Monotonically increasing positions → all doc_id 0."""
        from torchtitan.models.common.decoder import Decoder

        positions = torch.arange(64, device="cuda").unsqueeze(0)
        doc_ids = Decoder._document_ids_from_positions(positions)
        self.assertTrue((doc_ids == 0).all())

    def test_equal_length_docs(self):
        """Two equal-length docs produce [0,0,...,1,1,...]."""
        from torchtitan.models.common.decoder import Decoder

        pos = torch.cat([torch.arange(32), torch.arange(32)]).unsqueeze(0).cuda()
        doc_ids = Decoder._document_ids_from_positions(pos)
        expected = torch.cat([
            torch.zeros(32, dtype=torch.int32),
            torch.ones(32, dtype=torch.int32),
        ]).unsqueeze(0).cuda()
        self.assertTrue((doc_ids == expected).all())

    def test_unequal_length_docs(self):
        """Three docs of lengths 10, 30, 24."""
        from torchtitan.models.common.decoder import Decoder

        pos = torch.cat([
            torch.arange(10), torch.arange(30), torch.arange(24),
        ]).unsqueeze(0).cuda()
        doc_ids = Decoder._document_ids_from_positions(pos)
        expected = torch.cat([
            torch.full((10,), 0, dtype=torch.int32),
            torch.full((30,), 1, dtype=torch.int32),
            torch.full((24,), 2, dtype=torch.int32),
        ]).unsqueeze(0).cuda()
        self.assertTrue((doc_ids == expected).all())

    def test_batch_independent(self):
        """Each row is processed independently."""
        from torchtitan.models.common.decoder import Decoder

        row0 = torch.cat([torch.arange(32), torch.arange(32)])
        row1 = torch.cat([torch.arange(16), torch.arange(16), torch.arange(16), torch.arange(16)])
        pos = torch.stack([row0, row1]).cuda()
        doc_ids = Decoder._document_ids_from_positions(pos)
        self.assertEqual(doc_ids[0, -1].item(), 1)
        self.assertEqual(doc_ids[1, -1].item(), 3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4IndexDirectionCanary(unittest.TestCase):
    """Small known-permutation test to catch index direction regressions."""

    def setUp(self):
        torch.manual_seed(42)

    def _ref_attention(self, q, k, v, mask):
        return ref_attention(q, k, v, mask, num_heads=H, scale=SCALE, dtype=DTYPE)

    def test_known_permutation_4tokens(self):
        """Permutation [2,3,0,1]: rank 0 holds perm positions [0,1] = orig [2,3]."""
        from flash_attn.cute import flash_attn_func

        from torchtitan.models.common.attention import build_fa4_mask

        full_seq = 128
        cp_world = 2
        local_seq = full_seq // cp_world
        rank = 0

        # Known permutation: first half gets positions [64..127], second half [0..63]
        shard_indices = torch.cat([
            torch.arange(local_seq, full_seq),
            torch.arange(0, local_seq),
        ]).unsqueeze(0).cuda()  # (1, full_seq)

        si_flat = shard_indices.squeeze(0)  # (full_seq,) for indexing

        q_full = torch.randn(B, full_seq, H, D, device="cuda", dtype=DTYPE)
        k_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)
        v_full = torch.randn(B, full_seq, HKV, D, device="cuda", dtype=DTYPE)

        q_perm = q_full[:, si_flat]
        k_perm = k_full[:, si_flat]
        v_perm = v_full[:, si_flat]

        q_local = q_perm[:, :local_seq]
        k_cp = k_perm
        v_cp = v_perm

        fa4_mask = build_fa4_mask(
            shard_indices=shard_indices,
            cp_rank=rank,
            local_seq_len=local_seq,
        )

        with torch.no_grad():
            out_fa4, _ = flash_attn_func(
                q_local, k_cp, v_cp, causal=False, softmax_scale=SCALE,
                mask_mod=fa4_mask.mask_mod, aux_tensors=fa4_mask.aux_tensors,
            )

        # Reference: full causal on unpermuted sequence, extract rank's rows
        idx = torch.arange(full_seq, device="cuda")
        causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
        with torch.no_grad():
            out_ref_full = self._ref_attention(q_full, k_full, v_full, causal_mask)

        rank_orig_positions = si_flat[:local_seq]
        out_ref = out_ref_full[:, rank_orig_positions]

        torch.testing.assert_close(out_fa4, out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
