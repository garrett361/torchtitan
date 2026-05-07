"""Integration tests for FA4 attention backend with the granite model.

Verifies correctness vs SDPA reference, training loop stability, and
torch.compile compatibility.

Run:
    python -m unittest torchtitan.models.granite.tests.test_fa4_integration -v
"""

import math
import unittest

import torch

from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel


def _build_model(backend: str = "fa4") -> GraniteModel:
    config = granite_configs["debugmodel_fa4"](attn_backend=backend)
    model = GraniteModel(config)
    model.init_states()
    return model.cuda()


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


if __name__ == "__main__":
    unittest.main()
