"""RED-GREEN tests for decoder backbone_suffix integration.

These tests exercise get_attention_masks and the full forward path with
backbone+suffix data via the Granite debug model.

RED phase: tests FAIL because _get_flex_attention_masks doesn't process the
suffix_ids sentinel — it falls through to block_causal which isolates suffixes
from the backbone (position resets look like document boundaries).

GREEN phase: after implementing the sentinel dispatch, suffix tokens attend to
their backbone prefix → logits match isolated forwards.

Requires CUDA.
"""

import unittest

import torch

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel

_TOKENIZER_PATH = "tests/assets/tokenizer"
_VOCAB_SIZE = 2048
_SEED = 42


def _build_single_suffix_batch(seq_len=28):
    """backbone(16) + suffix_0(6) + suffix_1(6), one conversation.

    Layout:
        [backbone: 16 tokens][suffix_0: 6 tokens][suffix_1: 6 tokens]

    suffix_0: insertion_limit=4, positions start at 5
    suffix_1: insertion_limit=8, positions start at 9
    """
    torch.manual_seed(_SEED)
    tokens = torch.randint(1, _VOCAB_SIZE, (1, seq_len), dtype=torch.long)

    backbone_len = 16
    suffix_0_len = 6
    suffix_1_len = 6
    insertion_limit_0 = 4
    insertion_limit_1 = 8

    # Positions
    positions = torch.zeros(1, seq_len, dtype=torch.long)
    positions[0, :backbone_len] = torch.arange(backbone_len)
    positions[0, backbone_len : backbone_len + suffix_0_len] = torch.arange(
        insertion_limit_0 + 1, insertion_limit_0 + 1 + suffix_0_len
    )
    positions[0, backbone_len + suffix_0_len :] = torch.arange(
        insertion_limit_1 + 1, insertion_limit_1 + 1 + suffix_1_len
    )

    # conv_ids: all same conversation (1)
    conv_ids = torch.ones(1, seq_len, dtype=torch.long)

    # suffix_ids: 0=backbone, 1=suffix_0, 2=suffix_1
    suffix_ids = torch.zeros(1, seq_len, dtype=torch.long)
    suffix_ids[0, backbone_len : backbone_len + suffix_0_len] = 1
    suffix_ids[0, backbone_len + suffix_0_len :] = 2

    # insertion_limits: backbone gets backbone_len-1, each suffix gets its limit
    insertion_limits = torch.full((1, seq_len), backbone_len - 1, dtype=torch.long)
    insertion_limits[0, backbone_len : backbone_len + suffix_0_len] = insertion_limit_0
    insertion_limits[0, backbone_len + suffix_0_len :] = insertion_limit_1

    return {
        "tokens": tokens,
        "positions": positions,
        "conv_ids": conv_ids,
        "suffix_ids": suffix_ids,
        "insertion_limits": insertion_limits,
        "backbone_len": backbone_len,
        "suffix_0_len": suffix_0_len,
        "suffix_1_len": suffix_1_len,
        "insertion_limit_0": insertion_limit_0,
        "insertion_limit_1": insertion_limit_1,
    }


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBackboneSuffixDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(_SEED)
        config = granite_configs["debugmodel"](attn_backend="flex")
        cls.model = GraniteModel(config)
        cls.model.to(device="cuda")
        cls.model.init_states()
        cls.model.eval()
        cls.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)

    def test_sentinel_pops_keys(self):
        """After get_attention_masks, suffix_ids/conv_ids/insertion_limits are removed."""
        batch = _build_single_suffix_batch()
        extra_inputs = {
            "suffix_ids": batch["suffix_ids"].cuda(),
            "conv_ids": batch["conv_ids"].cuda(),
            "insertion_limits": batch["insertion_limits"].cuda(),
        }
        self.model.get_attention_masks(
            batch["tokens"].cuda(),
            self.tokenizer,
            extra_inputs=extra_inputs,
            positions=batch["positions"].cuda(),
        )
        self.assertNotIn("suffix_ids", extra_inputs)
        self.assertNotIn("conv_ids", extra_inputs)
        self.assertNotIn("insertion_limits", extra_inputs)

    def test_fallthrough_no_suffix_ids(self):
        """Without suffix_ids, existing block_causal mask logic runs unchanged."""
        batch = _build_single_suffix_batch()
        # Only backbone tokens, sequential positions → single document
        tokens = batch["tokens"][:, : batch["backbone_len"]].cuda()
        positions = batch["positions"][:, : batch["backbone_len"]].cuda()

        # Should work with no extra_inputs
        mask_none = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=None, positions=positions
        )
        self.assertIsNotNone(mask_none)

        # Should work with empty extra_inputs
        mask_empty = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs={}, positions=positions
        )
        self.assertIsNotNone(mask_empty)

    def test_suffix_logits_match_isolated_forward(self):
        """Suffix logits in packed forward match running suffix with its backbone prefix.

        For suffix_0 (insertion_limit=4):
          - Packed: suffix_0 attends to backbone[0:5] via backbone_suffix mask
          - Isolated: [backbone[0:5] + suffix_0] with causal attention

        Logits for suffix_0 positions must match.
        """
        batch = _build_single_suffix_batch()
        tokens = batch["tokens"].cuda()
        positions = batch["positions"].cuda()
        backbone_len = batch["backbone_len"]
        suffix_0_len = batch["suffix_0_len"]
        insertion_limit_0 = batch["insertion_limit_0"]

        extra_inputs = {
            "suffix_ids": batch["suffix_ids"].cuda(),
            "conv_ids": batch["conv_ids"].cuda(),
            "insertion_limits": batch["insertion_limits"].cuda(),
        }
        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra_inputs, positions=positions
        )
        with torch.no_grad():
            logits_packed = self.model(tokens, positions=positions, attention_masks=attn_masks)

        # Suffix_0 logits from packed forward
        suffix_0_start = backbone_len
        suffix_0_end = backbone_len + suffix_0_len
        packed_suffix_0_logits = logits_packed[:, suffix_0_start:suffix_0_end, :]

        # Non-degeneracy: logits must be non-trivial
        self.assertFalse(
            torch.allclose(packed_suffix_0_logits, torch.zeros_like(packed_suffix_0_logits)),
            "Suffix logits are all zeros — model may be in degenerate state",
        )

        # Isolated forward: backbone[0:insertion_limit_0+1] + suffix_0
        # Use actual packed positions to match RoPE embeddings exactly
        prefix_len = insertion_limit_0 + 1
        iso_tokens = torch.cat(
            [tokens[:, :prefix_len], tokens[:, suffix_0_start:suffix_0_end]], dim=1
        )
        iso_positions = torch.cat(
            [positions[:, :prefix_len], positions[:, suffix_0_start:suffix_0_end]], dim=1
        )
        iso_mask = self.model.get_attention_masks(
            iso_tokens, self.tokenizer, positions=iso_positions
        )
        with torch.no_grad():
            iso_logits = self.model(iso_tokens, positions=iso_positions, attention_masks=iso_mask)

        # Compare suffix_0 portion
        iso_suffix_0_logits = iso_logits[:, prefix_len:, :]
        torch.testing.assert_close(
            packed_suffix_0_logits,
            iso_suffix_0_logits,
            atol=1e-4,
            rtol=0.0,
            msg="Suffix_0 logits must match isolated forward",
        )

    def test_multi_suffix_logits_match_isolated(self):
        """Both suffix_0 and suffix_1 logits match their respective isolated forwards."""
        batch = _build_single_suffix_batch()
        tokens = batch["tokens"].cuda()
        positions = batch["positions"].cuda()
        backbone_len = batch["backbone_len"]
        suffix_0_len = batch["suffix_0_len"]
        suffix_1_len = batch["suffix_1_len"]
        insertion_limit_0 = batch["insertion_limit_0"]
        insertion_limit_1 = batch["insertion_limit_1"]

        extra_inputs = {
            "suffix_ids": batch["suffix_ids"].cuda(),
            "conv_ids": batch["conv_ids"].cuda(),
            "insertion_limits": batch["insertion_limits"].cuda(),
        }
        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra_inputs, positions=positions
        )
        with torch.no_grad():
            logits_packed = self.model(tokens, positions=positions, attention_masks=attn_masks)

        # --- Suffix 0 ---
        suffix_0_start = backbone_len
        suffix_0_end = backbone_len + suffix_0_len
        packed_suffix_0_logits = logits_packed[:, suffix_0_start:suffix_0_end, :]

        prefix_len_0 = insertion_limit_0 + 1
        iso_tokens_0 = torch.cat(
            [tokens[:, :prefix_len_0], tokens[:, suffix_0_start:suffix_0_end]], dim=1
        )
        iso_positions_0 = torch.cat(
            [positions[:, :prefix_len_0], positions[:, suffix_0_start:suffix_0_end]], dim=1
        )
        iso_mask_0 = self.model.get_attention_masks(
            iso_tokens_0, self.tokenizer, positions=iso_positions_0
        )
        with torch.no_grad():
            iso_logits_0 = self.model(
                iso_tokens_0, positions=iso_positions_0, attention_masks=iso_mask_0
            )

        iso_suffix_0_logits = iso_logits_0[:, prefix_len_0:, :]
        torch.testing.assert_close(
            packed_suffix_0_logits,
            iso_suffix_0_logits,
            atol=1e-4,
            rtol=0.0,
            msg="Suffix_0 logits must match isolated forward",
        )

        # --- Suffix 1 ---
        suffix_1_start = backbone_len + suffix_0_len
        suffix_1_end = suffix_1_start + suffix_1_len
        packed_suffix_1_logits = logits_packed[:, suffix_1_start:suffix_1_end, :]

        self.assertFalse(
            torch.allclose(packed_suffix_1_logits, torch.zeros_like(packed_suffix_1_logits)),
            "Suffix_1 logits are all zeros — model may be in degenerate state",
        )

        prefix_len_1 = insertion_limit_1 + 1
        iso_tokens_1 = torch.cat(
            [tokens[:, :prefix_len_1], tokens[:, suffix_1_start:suffix_1_end]], dim=1
        )
        iso_positions_1 = torch.cat(
            [positions[:, :prefix_len_1], positions[:, suffix_1_start:suffix_1_end]], dim=1
        )
        iso_mask_1 = self.model.get_attention_masks(
            iso_tokens_1, self.tokenizer, positions=iso_positions_1
        )
        with torch.no_grad():
            iso_logits_1 = self.model(
                iso_tokens_1, positions=iso_positions_1, attention_masks=iso_mask_1
            )

        iso_suffix_1_logits = iso_logits_1[:, prefix_len_1:, :]
        torch.testing.assert_close(
            packed_suffix_1_logits,
            iso_suffix_1_logits,
            atol=1e-4,
            rtol=0.0,
            msg="Suffix_1 logits must match isolated forward",
        )

    def test_backbone_logits_unaffected_by_suffix(self):
        """Backbone logits are identical whether suffixes are present or not.

        This passes in both RED and GREEN (block_causal also isolates backbone
        from suffix via position resets). Serves as a regression guard.
        """
        batch = _build_single_suffix_batch()
        tokens = batch["tokens"].cuda()
        positions = batch["positions"].cuda()
        backbone_len = batch["backbone_len"]

        # Packed forward with backbone_suffix mask
        extra_inputs = {
            "suffix_ids": batch["suffix_ids"].cuda(),
            "conv_ids": batch["conv_ids"].cuda(),
            "insertion_limits": batch["insertion_limits"].cuda(),
        }
        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra_inputs, positions=positions
        )
        with torch.no_grad():
            logits_packed = self.model(tokens, positions=positions, attention_masks=attn_masks)
        backbone_logits_packed = logits_packed[:, :backbone_len, :]

        # Backbone-only forward with causal mask
        backbone_tokens = tokens[:, :backbone_len]
        backbone_positions = positions[:, :backbone_len]
        backbone_mask = self.model.get_attention_masks(
            backbone_tokens, self.tokenizer, positions=backbone_positions
        )
        with torch.no_grad():
            backbone_logits_alone = self.model(
                backbone_tokens, positions=backbone_positions, attention_masks=backbone_mask
            )

        self.assertFalse(
            torch.allclose(backbone_logits_packed, torch.zeros_like(backbone_logits_packed)),
            "Backbone logits are all zeros — model may be in degenerate state",
        )
        torch.testing.assert_close(
            backbone_logits_packed,
            backbone_logits_alone,
            atol=1e-4,
            rtol=0.0,
            msg="Backbone logits must be unaffected by suffix presence",
        )

    def test_suffixes_do_not_attend_to_each_other(self):
        """Suffix_0 logits are identical whether suffix_1 is present or not.

        Verifies cross-suffix isolation: suffix_0 can't attend to suffix_1.
        """
        batch = _build_single_suffix_batch()
        tokens = batch["tokens"].cuda()
        positions = batch["positions"].cuda()
        backbone_len = batch["backbone_len"]
        suffix_0_len = batch["suffix_0_len"]

        # Full packed forward (both suffixes)
        extra_inputs_full = {
            "suffix_ids": batch["suffix_ids"].cuda(),
            "conv_ids": batch["conv_ids"].cuda(),
            "insertion_limits": batch["insertion_limits"].cuda(),
        }
        attn_masks_full = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra_inputs_full, positions=positions
        )
        with torch.no_grad():
            logits_full = self.model(tokens, positions=positions, attention_masks=attn_masks_full)
        suffix_0_logits_full = logits_full[:, backbone_len : backbone_len + suffix_0_len, :]

        # Partial: only backbone + suffix_0 (no suffix_1)
        partial_len = backbone_len + suffix_0_len
        partial_tokens = tokens[:, :partial_len]
        partial_positions = positions[:, :partial_len]
        partial_suffix_ids = batch["suffix_ids"][:, :partial_len].cuda()
        partial_conv_ids = batch["conv_ids"][:, :partial_len].cuda()
        partial_insertion_limits = batch["insertion_limits"][:, :partial_len].cuda()

        extra_inputs_partial = {
            "suffix_ids": partial_suffix_ids,
            "conv_ids": partial_conv_ids,
            "insertion_limits": partial_insertion_limits,
        }
        attn_masks_partial = self.model.get_attention_masks(
            partial_tokens,
            self.tokenizer,
            extra_inputs=extra_inputs_partial,
            positions=partial_positions,
        )
        with torch.no_grad():
            logits_partial = self.model(
                partial_tokens, positions=partial_positions, attention_masks=attn_masks_partial
            )
        suffix_0_logits_partial = logits_partial[:, backbone_len:, :]

        torch.testing.assert_close(
            suffix_0_logits_full,
            suffix_0_logits_partial,
            atol=1e-4,
            rtol=0.0,
            msg="Suffix_0 logits must be identical with or without suffix_1",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBackboneSuffixMultiConv(unittest.TestCase):
    """Multi-conversation packed batch: two conversations co-packed."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(_SEED)
        config = granite_configs["debugmodel"](attn_backend="flex")
        cls.model = GraniteModel(config)
        cls.model.to(device="cuda")
        cls.model.init_states()
        cls.model.eval()
        cls.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)

    def test_cross_conv_isolation(self):
        """Two co-packed conversations produce same logits as running independently.

        Layout: [conv1_backbone(8)][conv1_suffix(4)][conv2_backbone(10)][conv2_suffix(6)]
        """
        torch.manual_seed(_SEED + 1)
        seq_len = 28

        conv1_bb_len = 8
        conv1_suf_len = 4
        conv1_ins_limit = 3
        conv2_bb_len = 10
        conv2_suf_len = 6
        conv2_ins_limit = 5

        tokens = torch.randint(1, _VOCAB_SIZE, (1, seq_len), dtype=torch.long)

        # Positions
        positions = torch.zeros(1, seq_len, dtype=torch.long)
        off = 0
        positions[0, off : off + conv1_bb_len] = torch.arange(conv1_bb_len)
        off += conv1_bb_len
        positions[0, off : off + conv1_suf_len] = torch.arange(
            conv1_ins_limit + 1, conv1_ins_limit + 1 + conv1_suf_len
        )
        off += conv1_suf_len
        positions[0, off : off + conv2_bb_len] = torch.arange(conv2_bb_len)
        off += conv2_bb_len
        positions[0, off : off + conv2_suf_len] = torch.arange(
            conv2_ins_limit + 1, conv2_ins_limit + 1 + conv2_suf_len
        )

        # conv_ids: conv1=1, conv2=2
        conv_ids = torch.zeros(1, seq_len, dtype=torch.long)
        conv_ids[0, : conv1_bb_len + conv1_suf_len] = 1
        conv_ids[0, conv1_bb_len + conv1_suf_len :] = 2

        # suffix_ids: unique per suffix across the row
        suffix_ids = torch.zeros(1, seq_len, dtype=torch.long)
        suffix_ids[0, conv1_bb_len : conv1_bb_len + conv1_suf_len] = 1
        off2 = conv1_bb_len + conv1_suf_len + conv2_bb_len
        suffix_ids[0, off2:] = 2

        # insertion_limits
        insertion_limits = torch.zeros(1, seq_len, dtype=torch.long)
        # conv1 backbone: can attend to all of conv1 backbone
        insertion_limits[0, :conv1_bb_len] = conv1_bb_len - 1
        # conv1 suffix: insertion_limit relative to row start
        insertion_limits[0, conv1_bb_len : conv1_bb_len + conv1_suf_len] = conv1_ins_limit
        # conv2 backbone: offset = conv1_bb_len + conv1_suf_len
        conv2_off = conv1_bb_len + conv1_suf_len
        insertion_limits[0, conv2_off : conv2_off + conv2_bb_len] = (
            conv2_off + conv2_bb_len - 1
        )
        # conv2 suffix: offset + insertion_limit
        insertion_limits[0, conv2_off + conv2_bb_len :] = conv2_off + conv2_ins_limit

        # Packed forward
        extra_inputs = {
            "suffix_ids": suffix_ids.cuda(),
            "conv_ids": conv_ids.cuda(),
            "insertion_limits": insertion_limits.cuda(),
        }
        attn_masks = self.model.get_attention_masks(
            tokens.cuda(),
            self.tokenizer,
            extra_inputs=extra_inputs,
            positions=positions.cuda(),
        )
        with torch.no_grad():
            logits_packed = self.model(
                tokens.cuda(), positions=positions.cuda(), attention_masks=attn_masks
            )

        # --- Conv2 suffix isolated ---
        # conv2 suffix sees backbone[conv2_off : conv2_off + conv2_ins_limit + 1] + suffix
        conv2_suf_start = conv2_off + conv2_bb_len
        prefix_len = conv2_ins_limit + 1
        iso_tokens = torch.cat(
            [
                tokens[:, conv2_off : conv2_off + prefix_len],
                tokens[:, conv2_suf_start:],
            ],
            dim=1,
        ).cuda()
        iso_positions = torch.cat(
            [
                positions[:, conv2_off : conv2_off + prefix_len],
                positions[:, conv2_suf_start:],
            ],
            dim=1,
        ).cuda()
        iso_mask = self.model.get_attention_masks(
            iso_tokens, self.tokenizer, positions=iso_positions
        )
        with torch.no_grad():
            iso_logits = self.model(iso_tokens, positions=iso_positions, attention_masks=iso_mask)

        packed_conv2_suffix_logits = logits_packed[:, conv2_suf_start:, :]
        iso_conv2_suffix_logits = iso_logits[:, prefix_len:, :]

        torch.testing.assert_close(
            packed_conv2_suffix_logits,
            iso_conv2_suffix_logits,
            atol=1e-4,
            rtol=0.0,
            msg="Conv2 suffix logits must match isolated forward (cross-conv isolation)",
        )


if __name__ == "__main__":
    unittest.main()
