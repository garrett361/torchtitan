import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from torchtitan.models.granite.scripts.untie_hf_weights import untie_hf_weights


def _make_tied_checkpoint(tmp_dir: Path, *, multi_shard: bool = True) -> Path:
    """Create a minimal tied HF checkpoint for testing."""
    ckpt_dir = tmp_dir / "tied_ckpt"
    ckpt_dir.mkdir()

    vocab_size, dim = 64, 32
    embed_weight = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    layer_weight = torch.randn(dim, dim, dtype=torch.bfloat16)
    norm_weight = torch.randn(dim, dtype=torch.bfloat16)

    if multi_shard:
        shard1 = {
            "model.embed_tokens.weight": embed_weight,
            "model.layers.0.self_attn.o_proj.weight": layer_weight,
        }
        shard2 = {
            "model.norm.weight": norm_weight,
        }
        save_file(shard1, str(ckpt_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(ckpt_dir / "model-00002-of-00002.safetensors"))

        index = {
            "metadata": {"total_size": embed_weight.numel() * 2 + layer_weight.numel() * 2 + norm_weight.numel() * 2},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
            },
        }
        with open(ckpt_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)
    else:
        tensors = {
            "model.embed_tokens.weight": embed_weight,
            "model.layers.0.self_attn.o_proj.weight": layer_weight,
            "model.norm.weight": norm_weight,
        }
        save_file(tensors, str(ckpt_dir / "model.safetensors"))

    config = {
        "hidden_size": dim,
        "vocab_size": vocab_size,
        "tie_word_embeddings": True,
        "model_type": "granite",
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Add a dummy tokenizer file to verify copying
    (ckpt_dir / "tokenizer.json").write_text('{"version": "1.0"}')

    return ckpt_dir


class TestUntieHfWeights(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_multi_shard_produces_lm_head(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        shard1 = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
        self.assertIn("lm_head.weight", shard1)
        self.assertIn("model.embed_tokens.weight", shard1)
        torch.testing.assert_close(
            shard1["lm_head.weight"],
            shard1["model.embed_tokens.weight"],
            atol=0.0,
            rtol=0.0,
        )

    def test_lm_head_is_independent_tensor(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        shard1 = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
        self.assertFalse(
            shard1["lm_head.weight"].data_ptr()
            == shard1["model.embed_tokens.weight"].data_ptr(),
            "lm_head.weight must not share storage with embed_tokens",
        )

    def test_other_weights_unchanged(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        original_shard2 = load_file(
            str(ckpt_dir / "model-00002-of-00002.safetensors")
        )

        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        new_shard2 = load_file(str(out_dir / "model-00002-of-00002.safetensors"))
        for key in original_shard2:
            torch.testing.assert_close(
                original_shard2[key], new_shard2[key], atol=0.0, rtol=0.0
            )

    def test_config_json_updated(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        with open(out_dir / "config.json") as f:
            config = json.load(f)
        self.assertFalse(config["tie_word_embeddings"])

    def test_index_json_includes_lm_head(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        with open(out_dir / "model.safetensors.index.json") as f:
            index = json.load(f)
        self.assertIn("lm_head.weight", index["weight_map"])
        self.assertEqual(
            index["weight_map"]["lm_head.weight"],
            "model-00001-of-00002.safetensors",
        )

    def test_extra_files_copied(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        self.assertTrue((out_dir / "tokenizer.json").exists())
        with open(out_dir / "tokenizer.json") as f:
            self.assertEqual(json.load(f), {"version": "1.0"})

    def test_single_shard_checkpoint(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=False)
        out_dir = self.tmp_dir / "untied"
        untie_hf_weights(ckpt_dir, out_dir)

        tensors = load_file(str(out_dir / "model.safetensors"))
        self.assertIn("lm_head.weight", tensors)
        self.assertIn("model.embed_tokens.weight", tensors)
        torch.testing.assert_close(
            tensors["lm_head.weight"],
            tensors["model.embed_tokens.weight"],
            atol=0.0,
            rtol=0.0,
        )

    def test_already_untied_raises(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        # Modify config to be already untied
        config_path = ckpt_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        config["tie_word_embeddings"] = False
        with open(config_path, "w") as f:
            json.dump(config, f)

        out_dir = self.tmp_dir / "untied"
        with self.assertRaises(ValueError, msg="already.*untied"):
            untie_hf_weights(ckpt_dir, out_dir)

    def test_non_bf16_raises(self):
        ckpt_dir = self.tmp_dir / "fp32_ckpt"
        ckpt_dir.mkdir()
        tensors = {
            "model.embed_tokens.weight": torch.randn(64, 32, dtype=torch.float32),
            "model.norm.weight": torch.randn(32, dtype=torch.float32),
        }
        save_file(tensors, str(ckpt_dir / "model.safetensors"))
        config = {"hidden_size": 32, "vocab_size": 64, "tie_word_embeddings": True}
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config, f)

        out_dir = self.tmp_dir / "untied"
        with self.assertRaises(ValueError, msg="bfloat16"):
            untie_hf_weights(ckpt_dir, out_dir)

    def test_same_dir_raises(self):
        ckpt_dir = _make_tied_checkpoint(self.tmp_dir, multi_shard=True)
        with self.assertRaises(ValueError):
            untie_hf_weights(ckpt_dir, ckpt_dir)


class TestUntieHfWeightsRealCheckpoint(unittest.TestCase):
    """Integration test using a real 3B checkpoint (gated on env var)."""

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.ckpt_path = os.getenv("HF_ASSETS_PATH_3B")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "HF_ASSETS_PATH_3B not set. Add it to .env or export it before "
                "running real-checkpoint tests."
            )
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name) / "untied_3b"

    def tearDown(self):
        self.tmp.cleanup()

    def test_conversion_produces_loadable_untied_checkpoint(self):
        from transformers import GraniteConfig, GraniteForCausalLM

        from torchtitan.models.granite import granite_configs
        from torchtitan.models.granite.model import GraniteModel
        from torchtitan.models.granite.state_dict_adapter import (
            GraniteStateDictAdapter,
        )

        untie_hf_weights(Path(self.ckpt_path), self.out_dir)

        # Load into untied TT model
        tt_config = granite_configs["3B_untied"]()
        adapter = GraniteStateDictAdapter(tt_config, hf_assets_path=str(self.out_dir))

        from safetensors.torch import load_file as st_load

        hf_sd = {}
        import glob

        for shard in sorted(glob.glob(str(self.out_dir / "*.safetensors"))):
            hf_sd.update(st_load(shard, device="cpu"))

        tt_sd = adapter.from_hf(hf_sd)
        tt_model = GraniteModel(tt_config)
        tt_model.init_states()
        tt_model.load_state_dict(tt_sd, strict=True)
        tt_model.eval()

        # Load into HF model
        hf_model = GraniteForCausalLM.from_pretrained(
            str(self.out_dir), torch_dtype=torch.float32
        )
        hf_model.eval()

        # Compare logits
        tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            tt_logits = tt_model(tokens)
            hf_logits = hf_model(tokens).logits

        torch.testing.assert_close(tt_logits, hf_logits, atol=1e-4, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
