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
        self.ckpt_path = os.getenv("HF_ASSETS_PATH")
        if self.ckpt_path is None:
            raise EnvironmentError(
                "HF_ASSETS_PATH not set. Add it to .env or export it before "
                "running real-checkpoint tests."
            )

    def test_hf_checkpoint_loads_finite_loss(self):
        from safetensors.torch import load_file

        config = granite_configs["8B"]()
        adapter = GraniteStateDictAdapter(config, hf_assets_path=self.ckpt_path)

        shards = sorted(glob.glob(f"{self.ckpt_path}/model*.safetensors")) or sorted(
            glob.glob(f"{self.ckpt_path}/*.safetensors")
        )
        self.assertTrue(shards, "No safetensors found in HF_ASSETS_PATH")
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
    """Unit tests for GraniteSFTDataset / ChatDataset validation.

    No environment variables or real checkpoints required.
    """

    _user = {"role": "user", "content": "hello"}
    _asst = {"role": "assistant", "content": "hi"}
    _sys = {"role": "system", "content": "be helpful"}

    def test_validate_accepts_user_assistant(self):
        GraniteSFTDataset._validate_messages([self._user, self._asst])

    def test_validate_accepts_system_user_assistant(self):
        GraniteSFTDataset._validate_messages([self._sys, self._user, self._asst])

    def test_validate_accepts_multi_turn(self):
        # General multi-turn is now valid.
        GraniteSFTDataset._validate_messages(
            [self._sys, self._user, self._asst, self._user, self._asst]
        )

    def test_validate_rejects_missing_assistant(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages([self._user, self._user])

    def test_validate_rejects_system_not_first(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages(
                [self._user, self._sys, self._asst]
            )


class TestGraniteSFTDataFormat(unittest.TestCase):
    """Structural checks on the raw GLM-5.1 Reasoning dataset.

    Requires DATA_PATH set in the environment or a .env file.
    Skips if the variable is absent.
    """

    _SAMPLE_COUNT = 200

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        data_path = os.getenv("DATA_PATH")
        if data_path is None:
            self.skipTest("DATA_PATH not set")
        self.data_path = data_path

    def test_all_examples_single_turn(self):
        jsonl_files = sorted(glob.glob(os.path.join(self.data_path, "*.jsonl")))
        if not jsonl_files:
            self.skipTest("No .jsonl files found in DATA_PATH")
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

    Requires HF_ASSETS_PATH and DATA_PATH.
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
        ckpt_path = os.getenv("HF_ASSETS_PATH")
        data_path = os.getenv("DATA_PATH")
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
            for v in ("HF_ASSETS_PATH", "DATA_PATH")
        ):
            self.skipTest("HF_ASSETS_PATH and DATA_PATH must both be set")

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


class TestGraniteMultiTurnMasking(unittest.TestCase):
    """Rigorous boundary tests for multi-turn label masking with the real
    Granite tokenizer and thinking template.

    Requires GRANITE_41_8B_HF_ASSETS_PATH and GRANITE_DATA1_PATH.
    Skips if any variable is absent.

    These tests guard against off-by-one errors by independently computing
    the expected assistant token range for each turn and asserting:
      - label at start is a real token (first assistant token is trained)
      - label at start-1 is IGNORE_INDEX (fence before the turn)
      - label at end-1 is a real token (last token of the turn is trained)
      - label at end is IGNORE_INDEX (fence after, for non-final turns)
    """

    _tokenizer = None
    _IGNORE_INDEX = None

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        load_dotenv()
        ckpt_path = os.getenv("GRANITE_41_8B_HF_ASSETS_PATH")
        if ckpt_path is None:
            return
        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)
        cls._IGNORE_INDEX = IGNORE_INDEX

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if os.getenv("GRANITE_41_8B_HF_ASSETS_PATH") is None:
            self.skipTest("GRANITE_41_8B_HF_ASSETS_PATH not set")

    def _tokenize(self, messages):
        """Return (full_tokens, label_ids) via GraniteSFTDataset._tokenize_sample."""
        from datasets import Dataset

        ds_obj = GraniteSFTDataset(
            dataset=Dataset.from_list([{"messages": messages}]),
            tokenizer=self._tokenizer,
            sample_processor=lambda s: s["messages"],
            seq_len=8192,
            infinite=False,
        )
        result = ds_obj._tokenize_sample({"messages": messages})
        self.assertIsNotNone(result, "Sample was dropped (exceeds seq_len?)")
        _, label_ids = result
        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)
        return full_tokens, label_ids

    def _asst_range(self, messages, turn_idx):
        """Independently compute the (start, end) label_ids range for
        the assistant turn at turn_idx using the same formula as _tokenize_sample."""
        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        prefix_text = self._tokenizer.apply_chat_template(
            messages[:turn_idx], add_generation_prompt=True
        )
        prefix_tokens = self._tokenizer.encode(
            prefix_text, add_bos=True, add_eos=False
        )
        start = len(prefix_tokens) - 1
        if turn_idx == last_asst_idx:
            full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
            full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
            if full_tokens[-1] != self._tokenizer.eos_id:
                full_tokens.append(self._tokenizer.eos_id)
            end = len(full_tokens) - 1
        else:
            suffix_text = self._tokenizer.apply_chat_template(
                messages[: turn_idx + 1]
            )
            # rstrip matches _tokenize_sample: exclude the between-turn \n.
            suffix_tokens = self._tokenizer.encode(
                suffix_text.rstrip("\n"), add_bos=True, add_eos=False
            )
            end = len(suffix_tokens) - 1
        return start, end

    def _assert_fence(self, label_ids, start, end, *, is_last):
        IGN = self._IGNORE_INDEX
        self.assertGreater(start, 0, "start must be > 0 to have a fence before it")
        self.assertEqual(
            label_ids[start - 1], IGN, "label just before assistant start must be masked"
        )
        self.assertNotEqual(
            label_ids[start], IGN, "first assistant token must be trained"
        )
        self.assertNotEqual(
            label_ids[end - 1], IGN, "last assistant token must be trained"
        )
        if not is_last:
            self.assertEqual(
                label_ids[end], IGN, "label just after intermediate assistant end must be masked"
            )

    def test_single_turn_system_user_assistant_fences(self):
        """3-turn [system, user, assistant]: verify fence positions around the single assistant turn."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        full_tokens, label_ids = self._tokenize(messages)
        start, end = self._asst_range(messages, 2)
        self._assert_fence(label_ids, start, end, is_last=True)
        # System and user fully masked.
        self.assertTrue(
            all(l == self._IGNORE_INDEX for l in label_ids[:start]),
            "System and user tokens must all be masked",
        )

    def test_regression_matches_old_single_boundary_for_three_turn(self):
        """The new per-turn masking must produce bit-identical labels to the
        old _prompt_messages approach for a 3-turn [system, user, assistant]
        sample.  The old approach set mask_end = len(prefix_tokens) - 1 where
        prefix = apply_chat_template([system, user], add_generation_prompt=True).
        """
        messages = [
            {"role": "system", "content": "You are a math assistant."},
            {"role": "user", "content": "Compute 7 * 6."},
            {"role": "assistant", "content": "42"},
        ]
        _, label_ids = self._tokenize(messages)

        # Reproduce the old single-boundary logic directly.
        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        prompt_len = len(prompt_tokens)
        expected = list(full_tokens[1:])
        mask_end = min(max(prompt_len - 1, 0), len(expected))
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        expected[:mask_end] = [IGNORE_INDEX] * mask_end

        self.assertEqual(label_ids, expected, "Multi-turn result must be bit-identical to old single-boundary approach")

    def test_think_token_boundary_unchanged(self):
        """The <think> boundary invariant must hold under the new masking path:
        label at <think> position is masked; label one position later is trained."""
        from datasets import Dataset

        load_dotenv = __import__("dotenv").load_dotenv
        load_dotenv()
        data_path = os.getenv("GRANITE_DATA1_PATH")
        if data_path is None:
            self.skipTest("GRANITE_DATA1_PATH not set")

        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not jsonl_files:
            self.skipTest("No .jsonl files found in GRANITE_DATA1_PATH")

        with open(jsonl_files[0]) as f:
            record = json.loads(f.readline())
        messages = record["messages"]

        _, label_ids = self._tokenize(messages)
        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)

        think_id = self._tokenizer.token_to_id("<think>")
        self.assertIsNotNone(think_id)
        think_pos = full_tokens.index(think_id)

        self.assertEqual(
            label_ids[think_pos],
            self._IGNORE_INDEX,
            "label at <think> position must be masked (generation prefix)",
        )
        self.assertNotEqual(
            label_ids[think_pos + 1],
            self._IGNORE_INDEX,
            "label one after <think> must be trained (first reasoning token)",
        )

    def test_multi_turn_with_tool_messages(self):
        """Construct a synthetic multi-turn conversation with a tool result
        and verify that only the two assistant spans are trained on."""
        messages = [
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": "Find info about Python."},
            {"role": "assistant", "content": "Searching now."},
            {"role": "tool", "content": "Python is a programming language."},
            {"role": "assistant", "content": "Python is a high-level language."},
        ]
        full_tokens, label_ids = self._tokenize(messages)
        IGN = self._IGNORE_INDEX

        start1, end1 = self._asst_range(messages, 2)
        start2, end2 = self._asst_range(messages, 4)

        # Pre-first-assistant: masked.
        self.assertTrue(all(l == IGN for l in label_ids[:start1]))
        # First assistant span fences.
        self._assert_fence(label_ids, start1, end1, is_last=False)
        # First assistant span is all trained.
        self.assertTrue(all(l != IGN for l in label_ids[start1:end1]))
        # Tool turn between the two assistant turns: fully masked.
        self.assertTrue(
            all(l == IGN for l in label_ids[end1:start2]),
            "Tool message tokens must be fully masked",
        )
        # Second assistant span fences.
        self._assert_fence(label_ids, start2, end2, is_last=True)
        # Second assistant span is all trained.
        self.assertTrue(all(l != IGN for l in label_ids[start2:end2]))

        # Scaffolding masking verified in test_inter_turn_scaffolding_masked.

    def test_system_and_user_fully_masked_in_multi_turn(self):
        """In a multi-turn conversation, no system or user token should appear
        as a trained label."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        full_tokens, label_ids = self._tokenize(messages)
        IGN = self._IGNORE_INDEX

        start1, end1 = self._asst_range(messages, 2)
        start2, _ = self._asst_range(messages, 4)

        # System/user before first assistant turn: all masked.
        self.assertTrue(all(l == IGN for l in label_ids[:start1]))
        # User turn between assistant turns: all masked.
        self.assertTrue(all(l == IGN for l in label_ids[end1:start2]))


    def test_inter_turn_scaffolding_masked(self):
        """The inter-turn separator emitted by apply_chat_template after each
        assistant turn (a trailing \\n for the Granite / ChatML template) must
        be masked, not trained on.

        Rationale: during inference the model generates up to and including the
        turn-end delimiter (<|im_end|>) and then stops.  The \\n that follows is
        injected by the inference framework as structural scaffolding; the model
        never produces it.  _tokenize_sample uses rstrip("\\n") on the suffix
        text before measuring the boundary, which is correct for all practical
        templates (ChatML, Llama, Mistral all use trailing newlines as
        inter-turn separators).  This test pins that contract: the last trained
        label for an intermediate assistant turn is <|im_end|>, and the very
        next label is IGNORE_INDEX.
        """
        im_end_id = self._tokenizer.token_to_id("<|im_end|>")
        self.assertIsNotNone(im_end_id, "<|im_end|> must be a known token")

        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        _, label_ids = self._tokenize(messages)
        start1, end1 = self._asst_range(messages, 1)

        # Last trained label of the intermediate assistant turn must be <|im_end|>.
        self.assertEqual(
            label_ids[end1 - 1],
            im_end_id,
            f"label_ids[end1-1] should be im_end_id={im_end_id}, "
            f"got {label_ids[end1-1]}",
        )
        # The inter-turn separator immediately after must be masked.
        self.assertEqual(
            label_ids[end1],
            self._IGNORE_INDEX,
            "Inter-turn separator after <|im_end|> must be IGNORE_INDEX",
        )


if __name__ == "__main__":
    unittest.main()
