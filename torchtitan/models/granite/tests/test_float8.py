import math
import unittest

import torch
import torch.nn as nn

from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.distributed import ParallelDims
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel


def _build_converter(
    *,
    recipe_name: str | None = None,
    filter_fqns: list[str] | None = None,
) -> Float8LinearConverter:
    """Build a Float8LinearConverter for single-GPU testing.

    When recipe_name is None, uses tensorwise defaults (no FSDP all-gather since
    these tests are single-process). When recipe_name is set (e.g. "rowwise"),
    uses recipe-based config which is mutually exclusive with the all-gather flags.
    """
    if recipe_name is not None:
        kwargs = {"recipe_name": recipe_name}
    else:
        kwargs = {}
    if filter_fqns is not None:
        kwargs["filter_fqns"] = filter_fqns

    config = Float8LinearConverter.Config(**kwargs)
    parallel_dims = ParallelDims(
        dp_shard=-1, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=1
    )
    return Float8LinearConverter(
        config, parallel_dims=parallel_dims, model_compile_enabled=False
    )


def _build_model() -> GraniteModel:
    config = granite_configs["debugmodel"]()
    model = GraniteModel(config)
    model.init_states()
    return model


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFloat8Conversion(unittest.TestCase):
    def setUp(self):
        self.model = _build_model().cuda()

    def test_float8_converter_builds(self):
        converter = _build_converter()
        self.assertTrue(converter.enabled)

    def test_float8_converts_linear_layers(self):
        from torchao.float8.float8_linear import Float8Linear

        converter = _build_converter()
        converter.convert(self.model)

        converted = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, Float8Linear):
                converted.append(name)

        self.assertGreater(len(converted), 0, "No layers converted to Float8Linear")
        # Expect attention projections and FFN layers to be converted
        has_ffn = any("feed_forward" in n for n in converted)
        has_attn = any("attention" in n for n in converted)
        self.assertTrue(has_ffn, f"FFN layers not converted. Converted: {converted}")
        self.assertTrue(
            has_attn, f"Attention layers not converted. Converted: {converted}"
        )

    def test_float8_weight_tying_preserved(self):
        converter = _build_converter()
        converter.convert(self.model)
        self.assertIs(
            self.model.tok_embeddings.weight,
            self.model.output.weight,
            "Weight tying broken after float8 conversion",
        )

    def test_float8_forward_produces_finite_output(self):
        converter = _build_converter()
        converter.convert(self.model)
        config = granite_configs["debugmodel"]()
        tokens = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
        with torch.no_grad():
            out = self.model(tokens)
        self.assertFalse(torch.any(torch.isnan(out)), "NaN in float8 output")
        self.assertFalse(torch.any(torch.isinf(out)), "Inf in float8 output")

    def test_float8_forward_shape_unchanged(self):
        converter = _build_converter()
        converter.convert(self.model)
        config = granite_configs["debugmodel"]()
        B, S = 2, 64
        tokens = torch.randint(0, config.vocab_size, (B, S), device="cuda")
        with torch.no_grad():
            out = self.model(tokens)
        self.assertEqual(out.shape, (B, S, config.vocab_size))

    def test_float8_backward_runs(self):
        converter = _build_converter()
        converter.convert(self.model)
        config = granite_configs["debugmodel"]()
        tokens = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
        out = self.model(tokens)
        loss = out.sum()
        loss.backward()
        # Verify at least one grad was computed
        grads_found = any(
            p.grad is not None for p in self.model.parameters() if p.requires_grad
        )
        self.assertTrue(grads_found, "No gradients computed after backward")

    def test_float8_output_filter_fqns(self):
        from torchao.float8.float8_linear import Float8Linear

        converter = _build_converter(filter_fqns=["output"])
        converter.convert(self.model)
        self.assertNotIsInstance(
            self.model.output,
            Float8Linear,
            "output layer should NOT be converted when filtered",
        )

    def test_float8_rowwise_recipe_converts(self):
        converter = _build_converter(recipe_name="rowwise")
        converter.convert(self.model)
        config = granite_configs["debugmodel"]()
        tokens = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
        with torch.no_grad():
            out = self.model(tokens)
        self.assertFalse(torch.any(torch.isnan(out)), "NaN in rowwise float8 output")
        self.assertFalse(torch.any(torch.isinf(out)), "Inf in rowwise float8 output")


def _build_float8_model(recipe: str | None = None) -> GraniteModel:
    """Build a debugmodel with float8 conversion applied via the standard config path."""
    from torchtitan.config import ConfigManager
    from torchtitan.protocols.model_converter import ModelConvertersContainer

    config_name = "granite_debugmodel_float8"
    cm = ConfigManager()
    trainer_config = cm.parse_args(
        ["--module", "granite", "--config", config_name]
    )

    if recipe is not None:
        from torchtitan.components.quantization.float8 import Float8LinearConverter

        trainer_config.model_converters = ModelConvertersContainer.Config(
            converters=[Float8LinearConverter.Config(recipe_name=recipe)],
        )

    parallel_dims = ParallelDims(
        dp_shard=-1, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=1
    )
    model_compile_enabled = (
        trainer_config.compile.enable
        and "model" in trainer_config.compile.components
    )
    model_converters = trainer_config.model_converters.build(
        parallel_dims=parallel_dims,
        model_compile_enabled=model_compile_enabled,
    )

    model = _build_model().cuda()
    model_converters.convert(model)
    return model


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFloat8E2E(unittest.TestCase):
    """End-to-end training loop tests with float8 debugmodel."""

    def _train_loop(self, model, steps=20, lr=1e-3, seq_len=64):
        config = granite_configs["debugmodel"]()
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

    def test_float8_tensorwise_training_loop(self):
        model = _build_float8_model()
        losses = self._train_loop(model)
        for i, l in enumerate(losses):
            self.assertTrue(math.isfinite(l), f"Non-finite loss at step {i}: {l}")
        self.assertLess(
            losses[-1], losses[0],
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}",
        )

    def test_float8_rowwise_training_loop(self):
        model = _build_float8_model(recipe="rowwise")
        losses = self._train_loop(model)
        for i, l in enumerate(losses):
            self.assertTrue(math.isfinite(l), f"Non-finite loss at step {i}: {l}")
        self.assertLess(
            losses[-1], losses[0],
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}",
        )

    def test_float8_loss_close_to_bf16(self):
        """Float8 first-step loss should be close to bf16 on same weights/data."""
        config = granite_configs["debugmodel"]()
        torch.manual_seed(42)
        tokens = torch.randint(0, config.vocab_size, (2, 64), device="cuda")

        # bf16 baseline
        torch.manual_seed(0)
        model_bf16 = _build_model().cuda()
        with torch.no_grad():
            logits_bf16 = model_bf16(tokens)
        loss_bf16 = torch.nn.functional.cross_entropy(
            logits_bf16[:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        ).item()

        # float8 with same initial weights
        torch.manual_seed(0)
        model_fp8 = _build_float8_model()
        with torch.no_grad():
            logits_fp8 = model_fp8(tokens)
        loss_fp8 = torch.nn.functional.cross_entropy(
            logits_fp8[:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        ).item()

        rel_diff = abs(loss_fp8 - loss_bf16) / loss_bf16
        self.assertLess(
            rel_diff, 0.05,
            f"Float8 loss too far from bf16: fp8={loss_fp8:.4f}, bf16={loss_bf16:.4f}, "
            f"rel_diff={rel_diff:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
