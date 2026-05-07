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
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

B, H, HKV, D = 1, 32, 8, 128
SCALE = 1.0 / D
DTYPE = torch.bfloat16


def _has_fa4():
    try:
        import cutlass.cute  # noqa: F401
        from flash_attn.cute import flash_attn_func  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4Masks(unittest.TestCase):
    """Validate FA4 mask_mod correctness on SM100."""

    def setUp(self):
        torch.manual_seed(42)

    def _assert_close(self, a, b, name, atol=1e-2, rtol=1e-2):
        # min passing tol ~7e-3; 1e-2 gives headroom for non-determinism
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)

    def _ref_attention(self, q, k, v, mask):
        """SDPA reference attention with explicit bool mask. Handles GQA."""
        # q: (B, S_q, H, D), k/v: (B, S_kv, HKV, D), mask: (S_q, S_kv) bool
        # SDPA expects (B, H, S, D)
        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        v_t = v.transpose(1, 2).float()
        attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask, scale=SCALE, enable_gqa=True,
        )
        return out.transpose(1, 2).to(DTYPE)

    def test_0a_causal_mask_mod_matches_causal_flag(self):
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


    def test_0b_document_mask_with_aux_tensors(self):
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


    def test_0c_causal_offset_mask_cp_simulation(self):
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


    def test_0d_compile_custom_op_no_graph_break(self):
        """FA4 registered as custom_op compiles with fullgraph=True (no graph breaks)."""
        from flash_attn.cute import flash_attn_func

        @torch.library.custom_op("test::fa4_fwd", mutates_args=())
        def fa4_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    softmax_scale: float, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
            return flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)

        @fa4_fwd.register_fake
        def fa4_fwd_fake(q, k, v, softmax_scale, causal):
            lse = torch.empty(q.shape[0], q.shape[2], q.shape[1],
                              dtype=torch.float32, device=q.device)
            return torch.empty_like(q), lse

        class FA4Module(torch.nn.Module):
            def forward(self, q, k, v):
                out, _ = torch.ops.test.fa4_fwd(q, k, v, SCALE, True)
                return out

        mod = torch.compile(FA4Module(), fullgraph=True)

        seq_len = 2048
        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE)

        with torch.no_grad():
            out_compiled = mod(q, k, v)
            out_eager, _ = flash_attn_func(q, k, v, causal=True, softmax_scale=SCALE)

        self._assert_close(out_compiled, out_eager, "compiled vs eager")


    def test_0e_compile_with_mask_mod_baked_in(self):
        """FA4 custom_op with mask_mod closed over compiles with fullgraph=True."""
        import cutlass.cute as cute
        from flash_attn.cute import flash_attn_func

        @cute.jit
        def causal_mask_mod(batch, head, m_idx, n_idx, seqlen_info, aux_tensors: None):
            return m_idx >= n_idx

        @torch.library.custom_op("test::fa4_causal", mutates_args=())
        def fa4_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       softmax_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
            return flash_attn_func(
                q, k, v, causal=False, softmax_scale=softmax_scale,
                mask_mod=causal_mask_mod,
            )

        @fa4_causal.register_fake
        def fa4_causal_fake(q, k, v, softmax_scale):
            lse = torch.empty(q.shape[0], q.shape[2], q.shape[1],
                              dtype=torch.float32, device=q.device)
            return torch.empty_like(q), lse

        class FA4CausalModule(torch.nn.Module):
            def forward(self, q, k, v):
                out, _ = torch.ops.test.fa4_causal(q, k, v, SCALE)
                return out

        mod = torch.compile(FA4CausalModule(), fullgraph=True)

        seq_len = 2048
        q = torch.randn(B, seq_len, H, D, device="cuda", dtype=DTYPE)
        k = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE)
        v = torch.randn(B, seq_len, HKV, D, device="cuda", dtype=DTYPE)

        with torch.no_grad():
            out_compiled = mod(q, k, v)
            out_eager, _ = flash_attn_func(q, k, v, causal=True, softmax_scale=SCALE)

        self._assert_close(out_compiled, out_eager, "compiled+mask_mod vs eager causal")


    def test_0f_cp_document_causal_composed(self):
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


if __name__ == "__main__":
    unittest.main()
