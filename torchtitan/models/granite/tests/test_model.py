import dataclasses
import glob
import os
import unittest
from unittest.mock import MagicMock

import torch

from torchtitan.models.common.rope import RoPE, apply_rotary_emb_complex
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel, GraniteTransformerBlock
from torchtitan.models.granite.state_dict_adapter import GraniteStateDictAdapter


def _load_hf_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    shards = sorted(
        glob.glob(f"{ckpt_path}/model*.safetensors")
    ) or sorted(glob.glob(f"{ckpt_path}/*.safetensors"))
    assert shards, f"No safetensors found in {ckpt_path}"
    hf_sd = {}
    for shard in shards:
        hf_sd.update(load_file(shard, device="cpu"))
    return hf_sd


def _assert_roundtrip_state_dict(test_case, ckpt_path: str, config_key: str):
    """Verify HF→TT→HF state dict round-trip is bitwise exact."""
    config = granite_configs[config_key]()
    adapter = GraniteStateDictAdapter(config, hf_assets_path=ckpt_path)
    hf_sd = _load_hf_state_dict(ckpt_path)
    tt_sd = adapter.from_hf(hf_sd)
    roundtripped_hf_sd = adapter.to_hf(tt_sd)
    del tt_sd

    missing = set(hf_sd.keys()) - set(roundtripped_hf_sd.keys())
    test_case.assertTrue(
        missing == {"lm_head.weight"} or missing == set(),
        f"Unexpected missing keys: {missing}",
    )

    for key in roundtripped_hf_sd:
        test_case.assertTrue(
            torch.equal(hf_sd[key], roundtripped_hf_sd[key]),
            f"Mismatch at {key}",
        )


def _make_config(**overrides) -> GraniteModel.Config:
    """Start from the standard debugmodel config and apply overrides."""
    config = granite_configs["debugmodel"]()
    return dataclasses.replace(config, **overrides)


def _trainer_config_mock(
    seq_len: int = 512,
    tp: int = 1,
    cp: int = 1,
    pp: int = 1,
) -> MagicMock:
    m = MagicMock()
    m.training.seq_len = seq_len
    m.parallelism.tensor_parallel_degree = tp
    m.parallelism.context_parallel_degree = cp
    m.parallelism.pipeline_parallel_degree = pp
    return m


class TestGraniteInstantiation(unittest.TestCase):
    def test_debugmodel_builds(self):
        config = granite_configs["debugmodel"]()
        model = GraniteModel(config)
        self.assertIsInstance(model, GraniteModel)

    def test_3b_config_builds_on_meta(self):
        config = granite_configs["3B"]()
        with torch.device("meta"):
            model = GraniteModel(config)
        self.assertIsInstance(model, GraniteModel)
        self.assertEqual(len(model.layers), 40)

    def test_8b_config_builds_on_meta(self):
        config = granite_configs["8B"]()
        with torch.device("meta"):
            model = GraniteModel(config)
        self.assertIsInstance(model, GraniteModel)

    def test_30b_config_builds_on_meta(self):
        config = granite_configs["30B"]()
        with torch.device("meta"):
            model = GraniteModel(config)
        self.assertIsInstance(model, GraniteModel)
        self.assertEqual(len(model.layers), 64)


class TestGraniteForwardShape(unittest.TestCase):
    def setUp(self):
        self.config = granite_configs["debugmodel"]()
        self.model = GraniteModel(self.config)
        self.model.init_states()

    def test_output_shape(self):
        B, S = 2, 64
        tokens = torch.randint(0, self.config.vocab_size, (B, S))
        out = self.model(tokens)
        self.assertEqual(out.shape, (B, S, self.config.vocab_size))


class TestGraniteWeightTying(unittest.TestCase):
    def test_weights_shared_after_init(self):
        model = GraniteModel(_make_config())
        self.assertIs(
            model.tok_embeddings.weight,
            model.output.weight,
            "tok_embeddings.weight and output.weight must be the same tensor",
        )

    def test_weights_remain_tied_after_init_states(self):
        model = GraniteModel(_make_config())
        model.init_states()
        self.assertIs(
            model.tok_embeddings.weight,
            model.output.weight,
            "tok_embeddings.weight and output.weight must remain tied after init_states",
        )


class TestGraniteMultipliers(unittest.TestCase):
    def setUp(self):
        self.config = granite_configs["debugmodel"]()
        self.model = GraniteModel(self.config)
        self.model.init_states()
        self.tokens = torch.randint(0, self.config.vocab_size, (1, 16))

    def test_embedding_multiplier_zero_gives_zero_output(self):
        # Zero embedding_multiplier means the hidden states entering all layers
        # are zero, which produces zero output regardless of weights.
        config = _make_config(embedding_multiplier=0.0)
        model = GraniteModel(config)
        model.init_states()
        with torch.no_grad():
            out = model(self.tokens)
        self.assertTrue(torch.all(out == 0))

    def test_logits_scaling_halves_logits_when_doubled(self):
        # logits_scaling is a divisor: output / logits_scaling.
        # Doubling logits_scaling halves the logits.
        config1 = _make_config(logits_scaling=1.0)
        config2 = _make_config(logits_scaling=2.0)
        m1 = GraniteModel(config1)
        m1.init_states()
        # Share all weights so the only difference is the scaling constant.
        m2 = GraniteModel(config2)
        m2.load_state_dict(m1.state_dict(), strict=True)
        m2.tok_embeddings.weight = m1.tok_embeddings.weight
        m2.output.weight = m1.output.weight
        with torch.no_grad():
            out1 = m1(self.tokens)
            out2 = m2(self.tokens)
        self.assertTrue(torch.allclose(out2, 0.5 * out1))

    def test_residual_multiplier_zero_propagates_only_norm(self):
        # residual_multiplier=0 means every block is the identity map: the
        # attention and ffn contributions are multiplied by zero and discarded.
        config = dataclasses.replace(
            _make_config(),
            layers=[
                dataclasses.replace(layer, residual_multiplier=0.0)
                for layer in _make_config().layers
            ],
        )
        model = GraniteModel(config)
        model.init_states()
        tokens = torch.randint(0, self.config.vocab_size, (1, 16))
        with torch.no_grad():
            out1 = model(tokens)
        # Perturb every parameter inside the blocks. With multiplier=0 all block
        # computations are zeroed out, so the output must be unchanged.
        for name, param in model.named_parameters():
            if name.startswith("layers."):
                param.data.uniform_(-10.0, 10.0)
        with torch.no_grad():
            out2 = model(tokens)
        self.assertTrue(torch.allclose(out1, out2))


class TestGraniteAttnScale(unittest.TestCase):
    def test_attn_scale_is_inverse_head_dim(self):
        config = granite_configs["debugmodel"]()
        block_config = config.layers[0]
        head_dim = config.dim // block_config.attention.n_heads
        expected_scale = 1.0 / head_dim
        self.assertAlmostEqual(
            block_config.attention.attn_scale,
            expected_scale,
            places=7,
            msg="attn_scale must be 1/head_dim for Granite",
        )

    def test_attn_scale_not_inverse_sqrt_head_dim(self):
        config = granite_configs["debugmodel"]()
        block_config = config.layers[0]
        head_dim = config.dim // block_config.attention.n_heads
        sqrt_scale = head_dim**-0.5
        self.assertNotAlmostEqual(
            block_config.attention.attn_scale,
            sqrt_scale,
            places=5,
            msg="attn_scale must differ from 1/sqrt(head_dim) for Granite",
        )

    def test_gqattention_uses_attn_scale(self):
        config = granite_configs["debugmodel"]()
        block_config = config.layers[0]
        block = GraniteTransformerBlock(block_config)
        expected = block_config.attention.attn_scale
        self.assertAlmostEqual(block.attention.scaling, expected, places=7)

    def test_attn_scale_3b(self):
        config = granite_configs["3B"]()
        block_config = config.layers[0]
        head_dim = config.dim // block_config.attention.n_heads
        self.assertEqual(head_dim, 64)
        self.assertAlmostEqual(
            block_config.attention.attn_scale, 1.0 / 64, places=7
        )

    def test_attn_scale_30b(self):
        config = granite_configs["30B"]()
        block_config = config.layers[0]
        head_dim = config.dim // block_config.attention.n_heads
        self.assertEqual(head_dim, 128)
        self.assertAlmostEqual(
            block_config.attention.attn_scale, 1.0 / 128, places=7
        )


class TestGraniteConfigValues(unittest.TestCase):
    def test_rope_theta_3b(self):
        config = granite_configs["3B"]()
        self.assertEqual(config.rope.theta, 10_000_000)

    def test_rope_theta_8b(self):
        config = granite_configs["8B"]()
        self.assertEqual(config.rope.theta, 10_000_000)

    def test_rope_theta_30b(self):
        config = granite_configs["30B"]()
        self.assertEqual(config.rope.theta, 50_000_000)


class TestGraniteUpdateFromConfig(unittest.TestCase):
    def test_rope_max_seq_len_synced(self):
        config = granite_configs["debugmodel"]()
        config.update_from_config(trainer_config=_trainer_config_mock(seq_len=512))
        self.assertEqual(config.rope.max_seq_len, 512)

    def test_tp_raises_on_bad_n_heads(self):
        config = granite_configs["debugmodel"]()
        # debugmodel has n_heads=16; tp=3 doesn't divide 16
        with self.assertRaises(ValueError):
            config.update_from_config(trainer_config=_trainer_config_mock(tp=3))

    def test_pp_always_raises(self):
        config = granite_configs["debugmodel"]()
        with self.assertRaises(NotImplementedError):
            config.update_from_config(trainer_config=_trainer_config_mock(pp=2))


class TestGraniteStateDictAdapter(unittest.TestCase):
    def _make_adapter(self) -> tuple[GraniteModel, GraniteStateDictAdapter]:
        config = granite_configs["debugmodel"]()
        model = GraniteModel(config)
        model.init_states()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=None)
        return model, adapter

    def test_to_hf_skips_output_weight(self):
        model, adapter = self._make_adapter()
        hf_sd = adapter.to_hf(model.state_dict())
        self.assertNotIn("lm_head.weight", hf_sd)
        self.assertIn("model.embed_tokens.weight", hf_sd)

    def test_from_hf_roundtrip(self):
        model, adapter = self._make_adapter()
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        hf_sd = adapter.to_hf(original_sd)
        recovered_sd = adapter.from_hf(hf_sd)
        # All keys must survive and values must be bit-exact: the permute/reverse-permute
        # operations on Q/K are pure index reshuffles with no arithmetic.
        self.assertEqual(set(original_sd.keys()), set(recovered_sd.keys()))
        for key in original_sd:
            self.assertTrue(
                torch.equal(original_sd[key], recovered_sd[key]),
                f"Round-trip mismatch for {key!r}",
            )
        # Weight tying: lm_head is absent from the HF checkpoint and synthesized
        # from embed_tokens on load; both must remain equal after the round-trip.
        self.assertTrue(
            torch.equal(
                recovered_sd["tok_embeddings.weight"], recovered_sd["output.weight"]
            ),
            "tok_embeddings.weight and output.weight must be equal after round-trip",
        )


class TestGraniteUntied(unittest.TestCase):
    """Tests for untied weight model variants."""

    def test_debugmodel_untied_builds(self):
        config = granite_configs["debugmodel_untied"]()
        model = GraniteModel(config)
        self.assertIsInstance(model, GraniteModel)
        self.assertFalse(model.enable_weight_tying)

    def test_weights_are_independent(self):
        config = granite_configs["debugmodel_untied"]()
        model = GraniteModel(config)
        model.init_states()
        self.assertIsNot(
            model.tok_embeddings.weight,
            model.output.weight,
            "Untied: tok_embeddings.weight and output.weight must be different tensors",
        )

    def test_forward_shape(self):
        config = granite_configs["debugmodel_untied"]()
        model = GraniteModel(config)
        model.init_states()
        B, S = 2, 64
        tokens = torch.randint(0, config.vocab_size, (B, S))
        out = model(tokens)
        self.assertEqual(out.shape, (B, S, config.vocab_size))

    def test_untied_checkpoint_produces_identical_logits_to_tied(self):
        """Simulate the conversion script workflow: tied model → HF export → add
        lm_head.weight (what untie_hf_weights.py does) → load into untied model."""
        tied_config = granite_configs["debugmodel"]()
        untied_config = granite_configs["debugmodel_untied"]()

        tied_model = GraniteModel(tied_config)
        tied_model.init_states()

        # Export tied model to HF format (drops lm_head.weight)
        tied_adapter = GraniteStateDictAdapter(tied_config, hf_assets_path=None)
        hf_sd = tied_adapter.to_hf(tied_model.state_dict())
        # Simulate untie_hf_weights.py: copy embed_tokens as independent lm_head
        hf_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"].clone()

        untied_adapter = GraniteStateDictAdapter(untied_config, hf_assets_path=None)
        untied_sd = untied_adapter.from_hf(hf_sd)

        untied_model = GraniteModel(untied_config)
        untied_model.init_states()
        untied_model.load_state_dict(untied_sd, strict=True)

        tokens = torch.randint(0, tied_config.vocab_size, (1, 16))
        with torch.no_grad():
            tied_logits = tied_model(tokens)
            untied_logits = untied_model(tokens)
        torch.testing.assert_close(tied_logits, untied_logits, atol=0.0, rtol=0.0)

    def test_state_dict_roundtrip_exports_both_weights(self):
        config = granite_configs["debugmodel_untied"]()
        model = GraniteModel(config)
        model.init_states()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=None)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        hf_sd = adapter.to_hf(original_sd)

        self.assertIn("model.embed_tokens.weight", hf_sd)
        self.assertIn("lm_head.weight", hf_sd)

        recovered_sd = adapter.from_hf(hf_sd)
        self.assertEqual(set(original_sd.keys()), set(recovered_sd.keys()))
        for key in original_sd:
            self.assertTrue(
                torch.equal(original_sd[key], recovered_sd[key]),
                f"Round-trip mismatch for {key!r}",
            )

    def test_tp_validation_passes_for_untied(self):
        config = granite_configs["debugmodel_untied"]()
        config.update_from_config(trainer_config=_trainer_config_mock(tp=2))

    def test_pp_validation_passes_for_untied(self):
        config = granite_configs["debugmodel_untied"]()
        config.update_from_config(trainer_config=_trainer_config_mock(pp=2))

    def test_untied_to_hf_logits_match_hf_model(self):
        from transformers import GraniteConfig, GraniteForCausalLM

        tt_config = granite_configs["debugmodel_untied"]()
        tt_model = GraniteModel(tt_config)
        tt_model.init_states()

        adapter = GraniteStateDictAdapter(tt_config, hf_assets_path=None)
        hf_sd = adapter.to_hf(tt_model.state_dict())

        hf_config = GraniteConfig(
            vocab_size=tt_config.vocab_size,
            hidden_size=tt_config.dim,
            intermediate_size=512,
            num_hidden_layers=len(tt_config.layers),
            num_attention_heads=tt_config.layers[0].attention.n_heads,
            num_key_value_heads=tt_config.layers[0].attention.n_heads,
            tie_word_embeddings=False,
            embedding_multiplier=tt_config.embedding_multiplier,
            logits_scaling=tt_config.logits_scaling,
            residual_multiplier=tt_config.layers[0].residual_multiplier,
            attention_multiplier=tt_config.layers[0].attention.attn_scale,
            max_position_embeddings=tt_config.rope.max_seq_len,
            rope_theta=tt_config.rope.theta,
        )
        hf_model = GraniteForCausalLM(hf_config)
        hf_model.load_state_dict(hf_sd, strict=True)
        hf_model.eval()
        tt_model.eval()

        tokens = torch.randint(0, tt_config.vocab_size, (1, 16))
        with torch.no_grad():
            tt_logits = tt_model(tokens)
            hf_logits = hf_model(tokens).logits

        torch.testing.assert_close(tt_logits, hf_logits, atol=1e-4, rtol=0.0)


class TestGraniteRealCheckpoint(unittest.TestCase):
    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.ckpt_path = os.getenv("HF_ASSETS_PATH_8B")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "HF_ASSETS_PATH_8B not set. Add it to .env or export it before "
                "running real-checkpoint tests."
            )

    def test_hf_checkpoint_loads_finite_loss(self):
        from safetensors.torch import load_file

        config = granite_configs["8B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)

        shards = sorted(glob.glob(f"{self.ckpt_path}/model*.safetensors")) or sorted(
            glob.glob(f"{self.ckpt_path}/*.safetensors")
        )
        self.assertTrue(shards, "No safetensors found in HF_ASSETS_PATH_8B")
        hf_sd = {}
        for shard in shards:
            hf_sd.update(load_file(shard, device="cpu"))

        tt_sd = adapter.from_hf(hf_sd)

        with torch.device("cpu"):
            model = GraniteModel(config)
        model.to_empty(device="cpu")
        model.init_states()
        model.load_state_dict(tt_sd, strict=True)
        model.eval()

        tokens = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            out = model(tokens)
        self.assertEqual(out.shape, (1, 16, config.vocab_size))
        self.assertFalse(torch.any(torch.isnan(out)))
        self.assertFalse(torch.any(torch.isinf(out)))

    def test_logits_match_hf(self):
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0).to(device)

        # Load HF model, run forward, then free — keeps peak VRAM at one float32 8B.
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32
        )
        hf_sd = hf_model.state_dict()  # CPU tensors; used below for TT conversion
        hf_model.to(device).eval()
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits.cpu()
        del hf_model
        if device == "cuda":
            torch.cuda.empty_cache()

        config = granite_configs["8B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)
        tt_sd = adapter.from_hf(hf_sd)

        tt_model = GraniteModel(config)
        tt_model.init_states()
        tt_model.load_state_dict(tt_sd, strict=True)
        tt_model.to(device=device).eval()
        with torch.no_grad():
            tt_logits = tt_model(tokens).cpu()

        torch.testing.assert_close(tt_logits, hf_logits, atol=1e-4, rtol=0.0)

    def test_to_hf_roundtrip_logits(self):
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0).to(device)

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32
        )
        hf_model.to(device).eval()
        with torch.no_grad():
            original_logits = hf_model(tokens).logits.cpu()
        hf_sd = hf_model.state_dict()
        del hf_model
        if device == "cuda":
            torch.cuda.empty_cache()

        config = granite_configs["8B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)
        tt_sd = adapter.from_hf(hf_sd)
        del hf_sd
        roundtripped_hf_sd = adapter.to_hf(tt_sd)
        del tt_sd

        hf_model2 = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        hf_model2.load_state_dict(roundtripped_hf_sd, strict=False)
        del roundtripped_hf_sd
        hf_model2.to(device).eval()
        with torch.no_grad():
            roundtripped_logits = hf_model2(tokens).logits.cpu()

        torch.testing.assert_close(
            roundtripped_logits, original_logits, atol=1e-4, rtol=0.0
        )

    def test_to_hf_roundtrip_state_dict(self):
        _assert_roundtrip_state_dict(self, self.ckpt_path, "8B")


class TestGranite3BRealCheckpoint(unittest.TestCase):
    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.ckpt_path = os.getenv("HF_ASSETS_PATH_3B")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "HF_ASSETS_PATH_3B not set. Add it to .env or export it before "
                "running 3B real-checkpoint tests."
            )

    def test_hf_checkpoint_loads_finite_loss(self):
        from safetensors.torch import load_file

        config = granite_configs["3B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)

        shards = sorted(glob.glob(f"{self.ckpt_path}/model*.safetensors")) or sorted(
            glob.glob(f"{self.ckpt_path}/*.safetensors")
        )
        self.assertTrue(shards, "No safetensors found in HF_ASSETS_PATH_3B")
        hf_sd = {}
        for shard in shards:
            hf_sd.update(load_file(shard, device="cpu"))

        tt_sd = adapter.from_hf(hf_sd)

        with torch.device("cpu"):
            model = GraniteModel(config)
        model.to_empty(device="cpu")
        model.init_states()
        model.load_state_dict(tt_sd, strict=True)
        model.eval()

        tokens = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            out = model(tokens)
        self.assertEqual(out.shape, (1, 16, config.vocab_size))
        self.assertFalse(torch.any(torch.isnan(out)))
        self.assertFalse(torch.any(torch.isinf(out)))

    def test_logits_match_hf(self):
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0).to(device)

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32
        )
        hf_sd = hf_model.state_dict()
        hf_model.to(device).eval()
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits.cpu()
        del hf_model
        if device == "cuda":
            torch.cuda.empty_cache()

        config = granite_configs["3B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)
        tt_sd = adapter.from_hf(hf_sd)

        tt_model = GraniteModel(config)
        tt_model.init_states()
        tt_model.load_state_dict(tt_sd, strict=True)
        tt_model.to(device=device).eval()
        with torch.no_grad():
            tt_logits = tt_model(tokens).cpu()

        torch.testing.assert_close(tt_logits, hf_logits, atol=1e-4, rtol=0.0)

    def test_to_hf_roundtrip_logits(self):
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0).to(device)

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        hf_model.to(device).eval()
        with torch.no_grad():
            original_logits = hf_model(tokens).logits.cpu()
        hf_sd = hf_model.state_dict()
        del hf_model
        if device == "cuda":
            torch.cuda.empty_cache()

        config = granite_configs["3B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)
        tt_sd = adapter.from_hf(hf_sd)
        del hf_sd
        roundtripped_hf_sd = adapter.to_hf(tt_sd)
        del tt_sd

        hf_model2 = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        hf_model2.load_state_dict(roundtripped_hf_sd, strict=False)
        del roundtripped_hf_sd
        hf_model2.to(device).eval()
        with torch.no_grad():
            roundtripped_logits = hf_model2(tokens).logits.cpu()

        torch.testing.assert_close(
            roundtripped_logits, original_logits, atol=1e-4, rtol=0.0
        )

    def test_to_hf_roundtrip_state_dict(self):
        _assert_roundtrip_state_dict(self, self.ckpt_path, "3B")


class TestGranite30BRealCheckpoint(unittest.TestCase):
    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.ckpt_path = os.getenv("HF_ASSETS_PATH_30B")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "HF_ASSETS_PATH_30B not set. Add it to .env or export it before "
                "running 30B real-checkpoint tests."
            )

    def test_to_hf_roundtrip_state_dict(self):
        _assert_roundtrip_state_dict(self, self.ckpt_path, "30B")


class TestGraniteRoPEBuffer(unittest.TestCase):
    """Verify RoPE buffer integrity (complex64 dtype, shape, device safety)."""

    def setUp(self):
        self.config = granite_configs["debugmodel"]()
        self.model = GraniteModel(self.config)
        self.model.init_states()

    def test_freqs_cis_is_complex64(self):
        self.assertTrue(self.model.freqs_cis.is_complex())
        self.assertEqual(self.model.freqs_cis.dtype, torch.complex64)

    def test_freqs_cis_survives_device_move(self):
        original = self.model.freqs_cis.clone()
        self.model.to(device="cpu")
        self.assertTrue(self.model.freqs_cis.is_complex())
        self.assertEqual(self.model.freqs_cis.dtype, torch.complex64)
        torch.testing.assert_close(self.model.freqs_cis, original)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_freqs_cis_survives_cuda_move(self):
        original = self.model.freqs_cis.clone()
        self.model.to(device="cuda")
        self.assertTrue(self.model.freqs_cis.is_complex())
        self.assertEqual(self.model.freqs_cis.dtype, torch.complex64)
        torch.testing.assert_close(self.model.freqs_cis.cpu(), original)

    def test_freqs_cis_max_seq_len_coverage(self):
        self.assertEqual(
            self.model.freqs_cis.shape[0], self.config.rope.max_seq_len
        )

    def test_rope_complex_matches_manual_rotation(self):
        """Verify complex-backend RoPE matches manual complex rotation at high positions."""
        config = self.config
        head_dim = config.rope.dim
        theta = config.rope.theta
        max_seq_len = config.rope.max_seq_len

        positions_to_test = [0, 1, 1000, 50_000, max_seq_len - 1]

        freqs_cis = self.model.freqs_cis

        torch.manual_seed(42)
        n_heads = 4
        x = torch.randn(1, len(positions_to_test), n_heads, head_dim)

        # Apply complex-backend RoPE using the actual function
        freqs_for_positions = freqs_cis[positions_to_test]
        xq_complex, _ = apply_rotary_emb_complex(
            x, x, freqs_for_positions, positions=None
        )

        # Manually compute the same rotation: pair adjacent elements as complex,
        # multiply by cis(pos * freq), then view as real.
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim)
        )
        pos_tensor = torch.tensor(positions_to_test, dtype=torch.float32)
        angles = torch.outer(pos_tensor, inv_freq)  # (S, head_dim//2)
        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # (1, S, 1, head_dim//2)
        sin = angles.sin().unsqueeze(0).unsqueeze(2)

        # Complex backend pairs (x[2i], x[2i+1]) as real/imag
        x_pairs = x.float().reshape(1, len(positions_to_test), n_heads, head_dim // 2, 2)
        x_real = x_pairs[..., 0]  # (1, S, n_heads, head_dim//2)
        x_imag = x_pairs[..., 1]

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        xq_manual = torch.stack([out_real, out_imag], dim=-1).flatten(-2)

        torch.testing.assert_close(xq_complex, xq_manual, atol=1e-5, rtol=1e-5)


class TestGraniteRoPEAgreement(unittest.TestCase):
    """Long-sequence logit agreement between torchtitan and HF (requires checkpoint)."""

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.ckpt_path = os.getenv("HF_ASSETS_PATH_8B")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "HF_ASSETS_PATH_8B not set. Add it to .env or export it before "
                "running RoPE agreement tests."
            )

    def _load_models(self, device: str, seq_len: int):
        from transformers import AutoModelForCausalLM

        tokens = torch.randint(1, 1000, (1, seq_len), dtype=torch.long).to(device)

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_path, torch_dtype=torch.float32
        )
        hf_sd = hf_model.state_dict()
        hf_model.to(device=device).eval()
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits.cpu()
        del hf_model
        if device == "cuda":
            torch.cuda.empty_cache()

        config = granite_configs["8B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)
        tt_sd = adapter.from_hf(hf_sd)

        tt_model = GraniteModel(config)
        tt_model.init_states()
        tt_model.load_state_dict(tt_sd, strict=True)
        tt_model.to(device=device).eval()
        with torch.no_grad():
            tt_logits = tt_model(tokens).cpu()

        return tt_logits, hf_logits

    def test_long_sequence_logits_match_hf(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_len = 4096
        tt_logits, hf_logits = self._load_models(device, seq_len)
        # Slightly looser than the 8-token test (1e-4) to account for float32
        # accumulation over 4096 positions through 40 layers.
        torch.testing.assert_close(tt_logits, hf_logits, atol=2e-4, rtol=0.0)
        # Specifically check late positions where RoPE rotations are largest —
        # a rope bug would show up here first.
        torch.testing.assert_close(
            tt_logits[:, -256:, :], hf_logits[:, -256:, :], atol=2e-4, rtol=0.0
        )


if __name__ == "__main__":
    unittest.main()
