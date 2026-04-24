# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import glob
import json
import os
import unittest
from unittest.mock import MagicMock

import torch

from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel, GraniteTransformerBlock
from torchtitan.models.granite.sft_dataset import GraniteSFTDataset
from torchtitan.models.granite.state_dict_adapter import GraniteStateDictAdapter


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

    def test_8b_config_builds_on_meta(self):
        config = granite_configs["8B"]()
        with torch.device("meta"):
            model = GraniteModel(config)
        self.assertIsInstance(model, GraniteModel)


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
            torch.equal(recovered_sd["tok_embeddings.weight"], recovered_sd["output.weight"]),
            "tok_embeddings.weight and output.weight must be equal after round-trip",
        )


class TestGraniteRealCheckpoint(unittest.TestCase):
    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.ckpt_path = os.getenv("GRANITE_CKPT_PATH")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "GRANITE_CKPT_PATH not set. Add it to .env or export it before "
                "running real-checkpoint tests."
            )

    def test_hf_checkpoint_loads_finite_loss(self):
        from safetensors.torch import load_file

        config = granite_configs["8B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)

        shards = sorted(glob.glob(f"{self.ckpt_path}/model*.safetensors")) or sorted(
            glob.glob(f"{self.ckpt_path}/*.safetensors")
        )
        self.assertTrue(shards, "No safetensors found in GRANITE_CKPT_PATH")
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
            self.ckpt_path, dtype=torch.float32
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


class TestGraniteSFTDatasetUnit(unittest.TestCase):
    """Unit tests for GraniteSFTDataset validation and prompt-messages logic.

    No environment variables or real checkpoints required.
    """

    _user = {"role": "user", "content": "hello"}
    _asst = {"role": "assistant", "content": "hi"}
    _sys = {"role": "system", "content": "be helpful"}

    def test_validate_accepts_user_assistant(self):
        GraniteSFTDataset._validate_messages([self._user, self._asst])

    def test_validate_accepts_system_user_assistant(self):
        GraniteSFTDataset._validate_messages([self._sys, self._user, self._asst])

    def test_validate_rejects_multi_turn(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages(
                [self._sys, self._user, self._asst, self._user, self._asst]
            )

    def test_validate_rejects_wrong_first_role_in_3msg(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages(
                [self._user, self._user, self._asst]
            )

    def test_validate_rejects_missing_assistant(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages([self._user, self._user])

    def test_prompt_messages_two_turn(self):
        msgs = [self._user, self._asst]
        result = GraniteSFTDataset._prompt_messages(None, msgs)  # type: ignore[arg-type]
        self.assertEqual(result, [self._user])

    def test_prompt_messages_three_turn(self):
        msgs = [self._sys, self._user, self._asst]
        result = GraniteSFTDataset._prompt_messages(None, msgs)  # type: ignore[arg-type]
        self.assertEqual(result, [self._sys, self._user])


class TestGraniteSFTDataFormat(unittest.TestCase):
    """Structural checks on the raw GLM-5.1 Reasoning dataset.

    Requires GRANITE_DATA1_PATH set in the environment or a .env file.
    Skips if the variable is absent.
    """

    _SAMPLE_COUNT = 200

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        data_path = os.getenv("GRANITE_DATA1_PATH")
        if data_path is None:
            self.skipTest("GRANITE_DATA1_PATH not set")
        self.data_path = data_path

    def test_all_examples_single_turn(self):
        jsonl_files = sorted(glob.glob(os.path.join(self.data_path, "*.jsonl")))
        if not jsonl_files:
            self.skipTest("No .jsonl files found in GRANITE_DATA1_PATH")
        for fpath in jsonl_files:
            fname = os.path.basename(fpath)
            with open(fpath) as f:
                for i, line in enumerate(f):
                    if i >= self._SAMPLE_COUNT:
                        break
                    record = json.loads(line)
                    msgs = record["messages"]
                    self.assertEqual(
                        len(msgs),
                        3,
                        f"{fname} record {i}: expected 3 messages, got {len(msgs)}",
                    )
                    self.assertEqual(msgs[0]["role"], "system", f"{fname} record {i}")
                    self.assertEqual(msgs[1]["role"], "user", f"{fname} record {i}")
                    self.assertEqual(msgs[2]["role"], "assistant", f"{fname} record {i}")
                    rc = msgs[2].get("reasoning_content", "")
                    self.assertTrue(
                        rc,
                        f"{fname} record {i}: reasoning_content missing or empty",
                    )


class TestGraniteSFTData(unittest.TestCase):
    """End-to-end tokenization and masking tests for Granite SFT with thinking template.

    Requires GRANITE_CKPT_PATH and GRANITE_DATA1_PATH.
    Skips if any variable is absent.
    """

    _tokenizer = None
    _input_ids = None
    _labels = None
    _sample_msgs = None

    @classmethod
    def setUpClass(cls):
        from datasets import Dataset
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        load_dotenv()
        ckpt_path = os.getenv("GRANITE_CKPT_PATH")
        data_path = os.getenv("GRANITE_DATA1_PATH")
        if any(v is None for v in (ckpt_path, data_path)):
            return

        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)
        cls._IGNORE_INDEX = IGNORE_INDEX

        # Load a handful of samples from the first JSONL file found.
        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not jsonl_files:
            return
        fpath = jsonl_files[0]
        records = []
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= 8:
                    break

        cls._sample_msgs = records[0]["messages"]

        # Build a minimal GraniteSFTDataset to get one packed batch.
        dataset = Dataset.from_list(records)
        ds = GraniteSFTDataset(
            dataset=dataset,
            tokenizer=cls._tokenizer,
            sample_processor=lambda s: s["messages"],
            seq_len=8192,
            infinite=False,
        )
        batch, labels = next(iter(ds))
        cls._input_ids = batch["input"].tolist()
        cls._labels = labels.tolist()

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if any(
            os.getenv(v) is None
            for v in ("GRANITE_CKPT_PATH", "GRANITE_DATA1_PATH")
        ):
            self.skipTest("GRANITE_CKPT_PATH and GRANITE_DATA1_PATH must both be set")

    def test_chat_template_renders_system_and_thinking(self):
        rendered = self._tokenizer.apply_chat_template(self._sample_msgs)
        self.assertIn("<|im_start|>system", rendered)
        self.assertIn("<|im_end|>", rendered)
        self.assertIn("<think>", rendered)
        self.assertIn("</think>", rendered)
        rc_prefix = self._sample_msgs[2]["reasoning_content"][:20]
        self.assertIn(rc_prefix, rendered)

    def test_prompt_masked_thinking_trained(self):
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        labels = self._labels
        self.assertTrue(
            any(l == IGNORE_INDEX for l in labels),
            "Expected some prompt tokens to be masked",
        )
        trained = sum(1 for l in labels if l != IGNORE_INDEX)
        self.assertGreater(trained, 10, "Expected many trained tokens from reasoning_content")

    def test_think_token_is_last_masked_position(self):
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        think_id = self._tokenizer.token_to_id("<think>")
        self.assertIsNotNone(think_id, "<think> must be registered as a token")

        think_pos = self._input_ids.index(think_id)
        # labels[think_pos] predicts the token after <think>, which is \n —
        # still part of the masked generation prefix.
        self.assertEqual(
            self._labels[think_pos],
            IGNORE_INDEX,
            "Token immediately after <think> must be masked (part of generation prefix)",
        )
        # labels[think_pos + 1] = first reasoning_content token = trained.
        self.assertNotEqual(
            self._labels[think_pos + 1],
            IGNORE_INDEX,
            "First reasoning_content token must be trained",
        )

    def test_eos_present(self):
        eos_id = self._tokenizer.eos_id
        self.assertIn(eos_id, self._input_ids, "EOS token must appear in input_ids")


if __name__ == "__main__":
    unittest.main()
