"""Integration tests for FA4 attention backend with the granite model.

Verifies correctness vs SDPA reference, training loop stability,
torch.compile compatibility, and block_causal masking.

Run:
    python -m unittest torchtitan.models.granite.tests.test_fa4_integration -v
"""

import math
import unittest
from types import SimpleNamespace

import torch

from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel


def _has_fa4():
    try:
        import cutlass.cute  # noqa: F401
        from flash_attn.cute import flash_attn_func  # noqa: F401

        return True
    except ImportError:
        return False


def _build_model(backend: str = "fa4") -> GraniteModel:
    config = granite_configs["debugmodel_fa4"](attn_backend=backend)
    model = GraniteModel(config)
    model.init_states()
    return model.cuda()


def _make_positions(batch_size: int, seq_len: int, n_docs: int = 2) -> torch.Tensor:
    """Build multi-document positions (each doc resets to 0)."""
    doc_len = seq_len // n_docs
    single = torch.cat([torch.arange(doc_len) for _ in range(n_docs)])
    return single.unsqueeze(0).expand(batch_size, -1).cuda()


def _train_loop(model, steps=20, lr=1e-3, seq_len=64):
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        tokens = torch.randint(0, config.vocab_size, (2, seq_len), device="cuda")
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4Model(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = _build_model("fa4")
        self.config = self.model.config

    def test_forward_shape(self):
        B, S = 2, 64
        tokens = torch.randint(0, self.config.vocab_size, (B, S), device="cuda")
        with torch.no_grad():
            out = self.model(tokens)
        self.assertEqual(out.shape, (B, S, self.config.vocab_size))

    def test_output_finite(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 64), device="cuda")
        with torch.no_grad():
            out = self.model(tokens)
        self.assertTrue(torch.isfinite(out).all())

    def test_backward_all_grads(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 32), device="cuda")
        out = self.model(tokens)
        out.sum().backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")

    def test_output_matches_sdpa(self):
        """FA4 and SDPA produce the same output on identical weights."""
        torch.manual_seed(0)
        model_fa4 = _build_model("fa4")
        torch.manual_seed(0)
        model_sdpa = _build_model("sdpa")

        tokens = torch.randint(0, 2048, (2, 64), device="cuda")
        with torch.no_grad():
            out_fa4 = model_fa4(tokens)
            out_sdpa = model_sdpa(tokens)

        torch.testing.assert_close(out_fa4, out_sdpa, atol=1e-2, rtol=1e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4Training(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_loss_decreases(self):
        model = _build_model("fa4")
        losses = _train_loop(model, steps=40, lr=3e-3)
        self.assertLess(
            losses[-1], losses[0],
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}",
        )

    def test_loss_finite(self):
        model = _build_model("fa4")
        losses = _train_loop(model)
        for i, l in enumerate(losses):
            self.assertTrue(math.isfinite(l), f"Non-finite loss at step {i}: {l}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4Compile(unittest.TestCase):
    def test_compile_forward_matches_eager(self):
        torch.manual_seed(0)
        model = _build_model("fa4")
        compiled = torch.compile(model)

        tokens = torch.randint(0, 2048, (2, 64), device="cuda")
        with torch.no_grad():
            out_eager = model(tokens)
            out_compiled = compiled(tokens)

        torch.testing.assert_close(out_compiled, out_eager, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_has_fa4(), "flash_attn.cute (FA4) not installed")
class TestFA4Masking(unittest.TestCase):
    """Verify block_causal masking path (get_attention_masks → forward)."""

    def test_raises_without_positions(self):
        model = _build_model("fa4")
        tokens = torch.randint(0, 2048, (2, 64), device="cuda")
        tokenizer = SimpleNamespace(eos_id=0)
        with self.assertRaises(ValueError, msg="block_causal.*positions"):
            model.get_attention_masks(
                input_batch=tokens, tokenizer=tokenizer, positions=None
            )

    def test_forward_with_doc_mask(self):
        """Forward with document-packed positions produces finite output."""
        torch.manual_seed(42)
        model = _build_model("fa4")
        B, S = 2, 64
        tokens = torch.randint(0, 2048, (B, S), device="cuda")
        positions = _make_positions(B, S, n_docs=2)
        tokenizer = SimpleNamespace(eos_id=0)

        masks = model.get_attention_masks(
            input_batch=tokens, tokenizer=tokenizer, positions=positions
        )
        with torch.no_grad():
            out = model(tokens, attention_masks=masks, positions=positions)

        self.assertEqual(out.shape, (B, S, model.config.vocab_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_doc_mask_differs_from_causal(self):
        """Document masking produces different output than plain causal."""
        torch.manual_seed(0)
        model = _build_model("fa4")
        B, S = 1, 64
        tokens = torch.randint(0, 2048, (B, S), device="cuda")
        positions = _make_positions(B, S, n_docs=2)
        tokenizer = SimpleNamespace(eos_id=0)

        masks = model.get_attention_masks(
            input_batch=tokens, tokenizer=tokenizer, positions=positions
        )
        with torch.no_grad():
            out_masked = model(tokens, attention_masks=masks, positions=positions)
            out_causal = model(tokens)

        self.assertFalse(
            torch.allclose(out_masked, out_causal, atol=1e-3),
            "Document-masked output should differ from plain causal",
        )

    def test_packed_matches_individual_docs(self):
        """Each doc in a packed sequence matches running that doc alone."""
        torch.manual_seed(0)
        model = _build_model("fa4")
        tokenizer = SimpleNamespace(eos_id=0)

        doc_len = 32
        n_docs = 2
        S = doc_len * n_docs

        # Two documents packed into one sequence
        tokens = torch.randint(0, 2048, (1, S), device="cuda")
        positions = _make_positions(1, S, n_docs=n_docs)

        masks = model.get_attention_masks(
            input_batch=tokens, tokenizer=tokenizer, positions=positions
        )
        with torch.no_grad():
            out_packed = model(tokens, attention_masks=masks, positions=positions)

        # Run each doc individually (plain causal, no packing)
        for doc_idx in range(n_docs):
            start = doc_idx * doc_len
            doc_tokens = tokens[:, start:start + doc_len]
            with torch.no_grad():
                out_single = model(doc_tokens)
            torch.testing.assert_close(
                out_packed[:, start:start + doc_len],
                out_single,
                atol=1e-2,
                rtol=1e-2,
                msg=f"Doc {doc_idx} packed output doesn't match individual",
            )


if __name__ == "__main__":
    unittest.main()
