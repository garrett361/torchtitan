import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


class TestSPDA:
    def test_sdpa_causal_naive(self) -> None:
        bs, n_heads, seqlen, dim = 1, 4, 64, 128
        q = torch.randn(bs, n_heads, 1, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(bs, n_heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(bs, n_heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        o_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o_acausal = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        with pytest.raises(AssertionError):
            torch.testing.assert_close(o_causal, o_acausal)

    def test_sdpa_causal_mask(self) -> None:
        bs, n_heads, seqlen, dim = 1, 4, 64, 128
        q = torch.randn(bs, n_heads, 1, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(bs, n_heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(bs, n_heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        attn_mask = causal_lower_right(1, seqlen)
        o_masked = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o_acausal = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.testing.assert_close(o_masked, o_acausal)
