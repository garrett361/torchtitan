import pathlib

import pytest
import torch
import transformers
import vllm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from torchtitan.models.llama.model import (
    apply_rotary_emb,
    get_mscale,
    precompute_freqs_cis,
)

ORIG_SEQ_LEN = 8192
TRANSFORMERS_VERSION = "4.53.3"
VLLM_VERSION = "0.9.2"


def rotate_half(x):
    # From HF
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def account_for_rotate_half(x):
    """
    Transform to apply to titan q/k to account for the rotate_half usage in HF
    """
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.stack((x1, x2), dim=-1).flatten(-2, -1)


class TestYaRN:
    bsz = 1
    beta_fast_default = 32
    beta_slow_default = 1
    factor_default = 16.0
    num_attention_heads = 1  # Avoid OOMs
    device = "cuda"
    dtype = torch.bfloat16
    factory_kwargs = {"device": device, "dtype": dtype}
    config_dir = pathlib.Path(__file__).parent / "configs"

    def setup_method(self, method):
        torch.manual_seed(42)
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"Expected {transformers.__version__}={TRANSFORMERS_VERSION}, got {transformers.__version__=}"
            )
        if vllm.__version__ != VLLM_VERSION:
            raise RuntimeError(
                f"Expected {vllm.__version__}={VLLM_VERSION}, got {vllm.__version__=}"
            )

    def get_hf_cfg(self, seq_len: int) -> LlamaConfig:
        if seq_len <= ORIG_SEQ_LEN:
            return LlamaConfig.from_pretrained(self.config_dir / "llama_no_yarn")
        return LlamaConfig.from_pretrained(self.config_dir / "llama_yarn")

    def get_q_ks(self, seq_len: int, hf_cfg: LlamaConfig):
        q = torch.randn(
            self.bsz,
            seq_len,
            self.num_attention_heads,
            hf_cfg.head_dim,
            **self.factory_kwargs,
        )
        k = torch.randn(
            self.bsz,
            seq_len,
            self.num_attention_heads,
            hf_cfg.head_dim,
            **self.factory_kwargs,
        )

        # Account for fact that titan doesn't use rotate_half
        q_titan = account_for_rotate_half(q)
        k_titan = account_for_rotate_half(k)
        return q, k, q_titan, k_titan

    @pytest.mark.parametrize(
        "seq_len",
        [
            2 * ORIG_SEQ_LEN,
            4 * ORIG_SEQ_LEN,
            8 * ORIG_SEQ_LEN,
        ],
    )
    def test_hf_cos_sin(self, seq_len: int) -> None:
        hf_cfg = self.get_hf_cfg(seq_len=seq_len)
        position_ids = torch.arange(seq_len, device=self.device)[None]
        llama_rotary_emb = LlamaRotaryEmbedding(hf_cfg).cuda()
        q, k, q_titan, k_titan = self.get_q_ks(seq_len, hf_cfg)
        hf_cos, hf_sin = llama_rotary_emb(x=k, position_ids=position_ids)
        titan_freqs_cis = precompute_freqs_cis(
            dim=hf_cfg.head_dim,
            seq_len=seq_len,
            original_seq_len=ORIG_SEQ_LEN,
            rope_theta=hf_cfg.rope_theta,
            rope_factor=self.factor_default,
            beta_fast=self.beta_fast_default,
            beta_slow=self.beta_slow_default,
        ).cuda()

        titan_cos_sin = torch.view_as_real(titan_freqs_cis)
        titan_cos, titan_sin = titan_cos_sin[..., 0], titan_cos_sin[..., 1]
        # The HF cos/sin have redundant entries:
        torch.testing.assert_close(
            hf_cos[..., : hf_cfg.head_dim // 2], hf_cos[..., hf_cfg.head_dim // 2 :]
        )
        torch.testing.assert_close(
            hf_sin[..., : hf_cfg.head_dim // 2], hf_sin[..., hf_cfg.head_dim // 2 :]
        )
        # Check titan vs non-redundant hf entries
        torch.testing.assert_close(
            titan_cos[None].to(hf_cos.dtype),
            hf_cos[..., hf_cfg.head_dim // 2 :],
            atol=1e-1,
            rtol=1e-1,
        )
        torch.testing.assert_close(
            titan_sin[None].to(hf_cos.dtype),
            hf_sin[..., hf_cfg.head_dim // 2 :],
            atol=1e-1,
            rtol=1e-1,
        )

    @pytest.mark.parametrize(
        "seq_len",
        [
            2 * ORIG_SEQ_LEN,
            4 * ORIG_SEQ_LEN,
            8 * ORIG_SEQ_LEN,
        ],
    )
    def test_hf_zero_pos_sanity(self, seq_len: int) -> None:
        """
        Sanity check: verify the seq_idx=0 elements are unchanged, up to scaling by get_mscale
        """
        hf_cfg = self.get_hf_cfg(seq_len=seq_len)
        position_ids = torch.arange(seq_len, device=self.device)[None]
        llama_rotary_emb = LlamaRotaryEmbedding(self.get_hf_cfg(seq_len=seq_len)).cuda()
        q, k, q_titan, k_titan = self.get_q_ks(seq_len, hf_cfg)

        hf_cos, hf_sin = llama_rotary_emb(x=k, position_ids=position_ids)
        hf_q_rope, hf_k_rope = apply_rotary_pos_emb(
            q.transpose(1, 2), k.transpose(1, 2), hf_cos, hf_sin
        )
        hf_q_rope, hf_k_rope = hf_q_rope.transpose(1, 2), hf_k_rope.transpose(1, 2)

        titan_freqs_cis = precompute_freqs_cis(
            dim=hf_cfg.head_dim,
            seq_len=seq_len,
            original_seq_len=ORIG_SEQ_LEN,
            rope_theta=hf_cfg.rope_theta,
            rope_factor=self.factor_default,
            beta_fast=self.beta_fast_default,
            beta_slow=self.beta_slow_default,
        ).cuda()
        titan_q_rope, titan_k_rope = apply_rotary_emb(q_titan, k_titan, titan_freqs_cis)

        # Sanity checks: seq_idx=0 elements should not change
        mscale = get_mscale(self.factor_default)
        torch.testing.assert_close(hf_q_rope[0, 0], mscale * q[0, 0])
        torch.testing.assert_close(hf_k_rope[0, 0], mscale * k[0, 0])

        torch.testing.assert_close(titan_q_rope[0, 0], mscale * q_titan[0, 0])
        torch.testing.assert_close(titan_k_rope[0, 0], mscale * k_titan[0, 0])

    @pytest.mark.parametrize(
        "seq_len",
        [
            2 * ORIG_SEQ_LEN,
            4 * ORIG_SEQ_LEN,
            8 * ORIG_SEQ_LEN,
        ],
    )
    def test_hf_scores(self, seq_len: int) -> None:
        hf_cfg = self.get_hf_cfg(seq_len=seq_len)
        position_ids = torch.arange(seq_len, device=self.device)[None]
        llama_rotary_emb = LlamaRotaryEmbedding(self.get_hf_cfg(seq_len=seq_len)).cuda()
        q_hf, k_hf, q_titan, k_titan = self.get_q_ks(seq_len, hf_cfg)

        q_hf, k_hf = apply_rotary_pos_emb(
            q_hf.transpose(1, 2),
            k_hf.transpose(1, 2),
            *llama_rotary_emb(x=k_hf, position_ids=position_ids),
        )
        q_hf, k_hf = q_hf.transpose(1, 2), k_hf.transpose(1, 2)

        titan_freqs_cis = precompute_freqs_cis(
            dim=hf_cfg.head_dim,
            seq_len=seq_len,
            original_seq_len=ORIG_SEQ_LEN,
            rope_theta=hf_cfg.rope_theta,
            rope_factor=self.factor_default,
            beta_fast=self.beta_fast_default,
            beta_slow=self.beta_slow_default,
        ).cuda()

        q_titan, k_titan = apply_rotary_emb(q_titan, k_titan, titan_freqs_cis)
        del titan_freqs_cis

        # Check scores: hf_scores.shape = (bsz, n_heads, seq_len, seq_len)
        hf_scores = q_hf.permute(0, 2, 1, 3) @ k_hf.permute(0, 2, 3, 1)
        titan_scores = q_titan.permute(0, 2, 1, 3) @ k_titan.permute(0, 2, 3, 1)

        # torch.testing.assert_close is hard to pass with so many elements:
        mean_abs_diff = (
            hf_scores - titan_scores
        ).abs().mean() / titan_scores.abs().mean()
        assert mean_abs_diff < 1e-2, f"{mean_abs_diff=}"
