"""GPU end-to-end tests: pre-tokenized datasets → GraniteModel → correct masking."""

import unittest

import torch

from torchtitan.components.loss import IGNORE_INDEX, cross_entropy_loss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel
from torchtitan.models.granite.pretokenized_dataset import TruncateLastDataset

_MANIFEST = "tests/assets/pretok_truncate_last/manifest.json"
_TOKENIZER_PATH = "tests/assets/tokenizer"
_SEQ_LEN = 32


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTruncateLastE2E(unittest.TestCase):
    def setUp(self):
        config = granite_configs["debugmodel"](attn_backend="flex")
        self.model = GraniteModel(config)
        self.model.to_empty(device="cuda")
        self.model.init_states()
        self.model.train()

    def _first_batch(self):
        ds = TruncateLastDataset(_MANIFEST, seq_len=_SEQ_LEN, infinite=False)
        batch_dict, labels, _stats = next(iter(ds))
        tokens = batch_dict["input"].unsqueeze(0).cuda()
        positions = batch_dict["positions"].unsqueeze(0).cuda()
        labels = labels.unsqueeze(0).cuda()
        return tokens, positions, labels

    def test_loss_is_finite(self):
        tokens, positions, labels = self._first_batch()
        with torch.no_grad():
            logits = self.model(tokens, positions=positions)
        loss = cross_entropy_loss(logits, labels)
        self.assertTrue(torch.isfinite(loss), f"loss is not finite: {loss.item()}")

    def test_backward_flows(self):
        tokens, positions, labels = self._first_batch()
        logits = self.model(tokens, positions=positions)
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads, "no gradients after backward")
        has_nonzero = any(g.abs().sum().item() > 0 for g in grads)
        self.assertTrue(has_nonzero, "all gradients are zero")

    def test_packed_logits_match_unpacked(self):
        """Flex attention document masking isolates per-document attention.

        Pulls a real packed batch from TruncateLastDataset, recovers document
        boundaries from position resets (positions[t] < positions[t-1]), runs
        each segment through the model independently, and verifies the logits
        agree with the corresponding slice of the packed forward pass.

        RED step (what would fail with atol=0): FP rounding in flex_attention
        differs between a packed seq of length SEQ_LEN and an unpacked seg of
        length L, so exact equality never holds.

        GREEN step (atol=1e-4): logits agree within FP tolerance because
        document masking prevents cross-document attention.  Cross-doc leakage
        would shift logits by >> 1e-4.
        """
        self.model.eval()
        tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
        tokens_packed, positions_packed, _ = self._first_batch()

        # Recover document boundaries: any position that resets to a lower value
        # than its predecessor marks the start of a new document.
        pos = positions_packed[0]  # [SEQ_LEN]
        resets = (pos[1:] < pos[:-1]).nonzero(as_tuple=True)[0] + 1
        boundaries = [0] + resets.tolist() + [_SEQ_LEN]
        segments = list(zip(boundaries[:-1], boundaries[1:]))
        self.assertGreater(len(segments), 1, "need at least 2 segments to test masking")

        attn_masks_packed = self.model.get_attention_masks(
            tokens_packed, tokenizer, positions=positions_packed
        )
        with torch.no_grad():
            logits_packed = self.model(
                tokens_packed, positions=positions_packed, attention_masks=attn_masks_packed
            )

        for start, end in segments:
            seg_tokens = tokens_packed[:, start:end]
            seg_pos = torch.arange(end - start, dtype=torch.long).unsqueeze(0).cuda()
            seg_attn = self.model.get_attention_masks(seg_tokens, tokenizer, positions=seg_pos)
            with torch.no_grad():
                seg_logits = self.model(seg_tokens, positions=seg_pos, attention_masks=seg_attn)

            packed_slice = logits_packed[:, start:end, :]

            # RED: not bit-identical (FP rounding differs across sequence lengths).
            self.assertFalse(torch.equal(packed_slice, seg_logits))

            # GREEN: agree within atol=1e-4; cross-doc leakage would produce >> 1e-4.
            torch.testing.assert_close(packed_slice, seg_logits, atol=1e-4, rtol=0.0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBackboneSuffixE2E(unittest.TestCase):
    """Verify backbone+suffix masking produces correct logit isolation.

    Uses synthetic token sequences (no real tokenizer needed) fed through the
    GraniteModel with backbone_suffix flex attention masks.
    """

    @classmethod
    def setUpClass(cls):
        config = granite_configs["debugmodel"](attn_backend="flex")
        cls.model = GraniteModel(config)
        cls.model.to_empty(device="cuda")
        cls.model.init_states()
        cls.model.eval()
        cls.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
        cls.seq_len = 32

    def _build_backbone_suffix_batch(self):
        """Build a synthetic packed batch with one backbone + one suffix.

        Layout (seq_len=32):
          [backbone: 10 tokens][suffix: 8 tokens][padding: 14 tokens]

        The suffix has insertion_limit=4 (attends to backbone[0:5]).
        """
        S = self.seq_len
        backbone_len = 10
        suffix_len = 8
        insertion_limit = 4  # suffix attends to backbone positions 0..4

        # Random token IDs in valid vocab range
        torch.manual_seed(42)
        vocab_size = self.model.config.vocab_size
        all_tokens = torch.randint(1, vocab_size, (backbone_len + suffix_len,))

        # Build tensors
        input_ids = torch.zeros(S, dtype=torch.long)
        input_ids[: backbone_len + suffix_len] = all_tokens
        # Padding uses token 0 (doesn't matter for logit comparison)

        positions = torch.zeros(S, dtype=torch.long)
        # Backbone: 0, 1, ..., 9
        positions[:backbone_len] = torch.arange(backbone_len)
        # Suffix: insertion_limit+1, ..., insertion_limit+suffix_len
        positions[backbone_len: backbone_len + suffix_len] = torch.arange(
            insertion_limit + 1, insertion_limit + 1 + suffix_len
        )
        # Padding: 0, 1, ...
        pad_len = S - backbone_len - suffix_len
        positions[backbone_len + suffix_len:] = torch.arange(pad_len)

        conv_ids = torch.zeros(S, dtype=torch.long)
        conv_ids[: backbone_len + suffix_len] = 1  # same conversation

        suffix_ids = torch.zeros(S, dtype=torch.long)
        suffix_ids[backbone_len: backbone_len + suffix_len] = 1  # suffix region

        insertion_limits = torch.full((S,), -1, dtype=torch.long)
        insertion_limits[:backbone_len] = backbone_len - 1  # backbone sees full backbone
        insertion_limits[backbone_len: backbone_len + suffix_len] = insertion_limit

        return (
            input_ids.unsqueeze(0).cuda(),
            positions.unsqueeze(0).cuda(),
            conv_ids.unsqueeze(0).cuda(),
            suffix_ids.unsqueeze(0).cuda(),
            insertion_limits.unsqueeze(0).cuda(),
            backbone_len,
            suffix_len,
            insertion_limit,
        )

    def test_loss_is_finite(self):
        """Backbone+suffix forward pass produces finite loss."""
        (tokens, positions, conv_ids, suffix_ids,
         ins_limits, bb_len, sf_len, _) = self._build_backbone_suffix_batch()

        extra = {"conv_ids": conv_ids, "suffix_ids": suffix_ids, "insertion_limits": ins_limits}
        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra, positions=positions
        )

        with torch.no_grad():
            logits = self.model(tokens, positions=positions, attention_masks=attn_masks)

        # Label all non-padding positions
        labels = torch.full_like(tokens, IGNORE_INDEX)
        labels[0, 1:bb_len + sf_len] = tokens[0, 1:bb_len + sf_len]
        loss = cross_entropy_loss(logits, labels)
        self.assertTrue(torch.isfinite(loss), f"loss not finite: {loss.item()}")

    def test_backbone_logits_isolated_from_suffix(self):
        """Backbone logits are identical whether suffix is present or not.

        The backbone never attends to suffix tokens (suffix_ids > 0 fails the
        to_backbone check, same_suffix fails because backbone has suffix_ids=0).
        So backbone logits should match running the backbone segment alone.
        """
        (tokens, positions, conv_ids, suffix_ids,
         ins_limits, bb_len, sf_len, _) = self._build_backbone_suffix_batch()

        # Packed forward with suffix
        extra = {"conv_ids": conv_ids.clone(), "suffix_ids": suffix_ids.clone(),
                 "insertion_limits": ins_limits.clone()}
        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra, positions=positions
        )
        with torch.no_grad():
            logits_packed = self.model(tokens, positions=positions, attention_masks=attn_masks)

        # Unpacked: just the backbone segment with standard causal attention
        bb_tokens = tokens[:, :bb_len]
        bb_pos = torch.arange(bb_len, dtype=torch.long).unsqueeze(0).cuda()
        bb_attn = self.model.get_attention_masks(bb_tokens, self.tokenizer, positions=bb_pos)
        with torch.no_grad():
            logits_unpacked = self.model(bb_tokens, positions=bb_pos, attention_masks=bb_attn)

        packed_bb = logits_packed[:, :bb_len, :]
        torch.testing.assert_close(packed_bb, logits_unpacked, atol=1e-4, rtol=0.0)

    def test_suffix_logits_match_prefix_plus_suffix(self):
        """Suffix logits match running [backbone_prefix + suffix] independently.

        The suffix attends to backbone[0:insertion_limit+1] and itself (causal).
        Running [prefix + suffix] as a single causal sequence should produce
        identical logits for the suffix portion.
        """
        (tokens, positions, conv_ids, suffix_ids,
         ins_limits, bb_len, sf_len, ins_limit) = self._build_backbone_suffix_batch()

        # Packed forward
        extra = {"conv_ids": conv_ids.clone(), "suffix_ids": suffix_ids.clone(),
                 "insertion_limits": ins_limits.clone()}
        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra, positions=positions
        )
        with torch.no_grad():
            logits_packed = self.model(tokens, positions=positions, attention_masks=attn_masks)

        # Independent: [backbone_prefix (0..ins_limit)] + [suffix tokens]
        prefix_len = ins_limit + 1
        prefix_tokens = tokens[:, :prefix_len]
        suffix_tokens = tokens[:, bb_len: bb_len + sf_len]
        combined = torch.cat([prefix_tokens, suffix_tokens], dim=1)
        combined_pos = torch.arange(prefix_len + sf_len, dtype=torch.long).unsqueeze(0).cuda()
        combined_attn = self.model.get_attention_masks(
            combined, self.tokenizer, positions=combined_pos
        )
        with torch.no_grad():
            logits_combined = self.model(
                combined, positions=combined_pos, attention_masks=combined_attn
            )

        # Compare suffix portion: packed[bb_len:bb_len+sf_len] vs combined[prefix_len:]
        packed_suffix = logits_packed[:, bb_len: bb_len + sf_len, :]
        independent_suffix = logits_combined[:, prefix_len:, :]
        torch.testing.assert_close(packed_suffix, independent_suffix, atol=1e-4, rtol=0.0)

    def test_no_suffix_matches_truncate_last(self):
        """Zero-suffix backbone logits match block_causal masking.

        Padding differs: backbone_suffix blocks all padding attention (conv_ids=0,
        insertion_limits=-1), while block_causal allows padding-to-padding via
        position-reset document detection. So we compare only backbone positions.
        """
        S = self.seq_len
        backbone_len = 12

        torch.manual_seed(99)
        vocab_size = self.model.config.vocab_size
        input_ids = torch.zeros(S, dtype=torch.long)
        input_ids[:backbone_len] = torch.randint(1, vocab_size, (backbone_len,))

        positions = torch.zeros(S, dtype=torch.long)
        positions[:backbone_len] = torch.arange(backbone_len)
        positions[backbone_len:] = torch.arange(S - backbone_len)

        conv_ids = torch.zeros(S, dtype=torch.long)
        conv_ids[:backbone_len] = 1

        suffix_ids = torch.zeros(S, dtype=torch.long)

        insertion_limits = torch.full((S,), -1, dtype=torch.long)
        insertion_limits[:backbone_len] = backbone_len - 1

        tokens = input_ids.unsqueeze(0).cuda()
        pos = positions.unsqueeze(0).cuda()

        # Backbone+suffix path (no actual suffixes)
        extra = {
            "conv_ids": conv_ids.unsqueeze(0).cuda(),
            "suffix_ids": suffix_ids.unsqueeze(0).cuda(),
            "insertion_limits": insertion_limits.unsqueeze(0).cuda(),
        }
        attn_bs = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra, positions=pos
        )
        with torch.no_grad():
            logits_bs = self.model(tokens, positions=pos, attention_masks=attn_bs)

        # Standard block_causal path
        attn_bc = self.model.get_attention_masks(tokens, self.tokenizer, positions=pos)
        with torch.no_grad():
            logits_bc = self.model(tokens, positions=pos, attention_masks=attn_bc)

        # Compare only backbone positions (padding attention semantics differ)
        torch.testing.assert_close(
            logits_bs[:, :backbone_len, :],
            logits_bc[:, :backbone_len, :],
            atol=1e-5,
            rtol=0.0,
        )

    def test_cross_conv_isolation(self):
        """Two co-packed conversations produce identical logits to running each alone."""
        S = self.seq_len
        len_a, len_b = 8, 10

        torch.manual_seed(77)
        vocab_size = self.model.config.vocab_size
        tokens_a = torch.randint(1, vocab_size, (len_a,))
        tokens_b = torch.randint(1, vocab_size, (len_b,))

        input_ids = torch.zeros(S, dtype=torch.long)
        input_ids[:len_a] = tokens_a
        input_ids[len_a: len_a + len_b] = tokens_b

        positions = torch.zeros(S, dtype=torch.long)
        positions[:len_a] = torch.arange(len_a)
        positions[len_a: len_a + len_b] = torch.arange(len_b)
        positions[len_a + len_b:] = torch.arange(S - len_a - len_b)

        conv_ids = torch.zeros(S, dtype=torch.long)
        conv_ids[:len_a] = 1
        conv_ids[len_a: len_a + len_b] = 2

        suffix_ids = torch.zeros(S, dtype=torch.long)

        insertion_limits = torch.full((S,), -1, dtype=torch.long)
        insertion_limits[:len_a] = len_a - 1
        insertion_limits[len_a: len_a + len_b] = len_a + len_b - 1

        tokens = input_ids.unsqueeze(0).cuda()
        pos = positions.unsqueeze(0).cuda()

        extra = {
            "conv_ids": conv_ids.unsqueeze(0).cuda(),
            "suffix_ids": suffix_ids.unsqueeze(0).cuda(),
            "insertion_limits": insertion_limits.unsqueeze(0).cuda(),
        }
        attn = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra, positions=pos
        )
        with torch.no_grad():
            logits_packed = self.model(tokens, positions=pos, attention_masks=attn)

        # Run each conversation independently
        for start, length, tok_data in [(0, len_a, tokens_a), (len_a, len_b, tokens_b)]:
            seg = tok_data.unsqueeze(0).cuda()
            seg_pos = torch.arange(length, dtype=torch.long).unsqueeze(0).cuda()
            seg_attn = self.model.get_attention_masks(seg, self.tokenizer, positions=seg_pos)
            with torch.no_grad():
                seg_logits = self.model(seg, positions=seg_pos, attention_masks=seg_attn)
            torch.testing.assert_close(
                logits_packed[:, start: start + length, :],
                seg_logits,
                atol=1e-4,
                rtol=0.0,
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBackboneSuffixMaskInvariants(unittest.TestCase):
    """Property-based tests on the materialized BlockMask.

    These don't need a model forward pass — they verify the mask itself
    satisfies the expected invariants for random inputs.
    """

    @classmethod
    def setUpClass(cls):
        config = granite_configs["debugmodel"](attn_backend="flex")
        cls.model = GraniteModel(config)
        cls.model.to_empty(device="cuda")
        cls.model.init_states()
        cls.tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)

    def _random_batch(self, seed=0):
        """Generate a random backbone+suffix batch and its materialized mask."""
        S = 32
        torch.manual_seed(seed)

        # Layout: conv1(backbone=8, suffix=4), conv2(backbone=6), padding=14
        conv_ids = torch.zeros(S, dtype=torch.long)
        conv_ids[:12] = 1  # conv1: backbone(8) + suffix(4)
        conv_ids[12:18] = 2  # conv2: backbone only

        suffix_ids = torch.zeros(S, dtype=torch.long)
        suffix_ids[8:12] = 1  # conv1's suffix

        insertion_limits = torch.full((S,), -1, dtype=torch.long)
        insertion_limits[:8] = 7  # conv1 backbone sees itself
        insertion_limits[8:12] = 3  # conv1 suffix sees backbone[0:4]
        insertion_limits[12:18] = 17  # conv2 backbone sees itself

        positions = torch.zeros(S, dtype=torch.long)
        positions[:8] = torch.arange(8)
        positions[8:12] = torch.arange(4, 8)  # suffix continues from ins_limit+1=4
        positions[12:18] = torch.arange(6)
        positions[18:] = torch.arange(14)

        vocab_size = self.model.config.vocab_size
        tokens = torch.randint(1, vocab_size, (1, S)).cuda()
        pos = positions.unsqueeze(0).cuda()

        extra = {
            "conv_ids": conv_ids.unsqueeze(0).cuda(),
            "suffix_ids": suffix_ids.unsqueeze(0).cuda(),
            "insertion_limits": insertion_limits.unsqueeze(0).cuda(),
        }

        attn_masks = self.model.get_attention_masks(
            tokens, self.tokenizer, extra_inputs=extra, positions=pos
        )
        return attn_masks, conv_ids, suffix_ids, insertion_limits

    def _materialize_mask(self, block_mask):
        """Evaluate the mask_mod into a dense [S, S] boolean tensor."""
        S = block_mask.shape[-1]
        q_idx = torch.arange(S, device="cuda").unsqueeze(1).expand(S, S)
        kv_idx = torch.arange(S, device="cuda").unsqueeze(0).expand(S, S)
        b = torch.zeros(S, S, dtype=torch.long, device="cuda")
        h = torch.zeros(S, S, dtype=torch.long, device="cuda")
        return block_mask.mask_mod(b, h, q_idx, kv_idx)

    def test_no_cross_conv_attention(self):
        """Positions in different conversations never attend to each other."""
        attn_masks, conv_ids, _, _ = self._random_batch()
        mask = self._materialize_mask(attn_masks)
        S = conv_ids.shape[0]
        for q in range(S):
            for kv in range(S):
                if conv_ids[q] != conv_ids[kv]:
                    self.assertFalse(
                        mask[q, kv].item(),
                        f"cross-conv attention at q={q} (conv={conv_ids[q]}) "
                        f"→ kv={kv} (conv={conv_ids[kv]})",
                    )

    def test_backbone_never_attends_to_suffix(self):
        """Backbone positions (suffix_ids=0) never attend to suffix (suffix_ids>0)."""
        attn_masks, _, suffix_ids, _ = self._random_batch()
        mask = self._materialize_mask(attn_masks)
        S = suffix_ids.shape[0]
        for q in range(S):
            if suffix_ids[q] != 0:
                continue
            for kv in range(S):
                if suffix_ids[kv] > 0:
                    self.assertFalse(
                        mask[q, kv].item(),
                        f"backbone q={q} attends to suffix kv={kv}",
                    )

    def test_suffix_respects_insertion_limit(self):
        """Suffix positions only attend to backbone up to their insertion_limit."""
        attn_masks, conv_ids, suffix_ids, ins_limits = self._random_batch()
        mask = self._materialize_mask(attn_masks)
        S = suffix_ids.shape[0]
        for q in range(S):
            if suffix_ids[q] == 0:
                continue
            for kv in range(S):
                if suffix_ids[kv] != 0:
                    continue
                if conv_ids[q] != conv_ids[kv]:
                    continue
                if kv > ins_limits[q]:
                    self.assertFalse(
                        mask[q, kv].item(),
                        f"suffix q={q} (limit={ins_limits[q]}) attends to "
                        f"backbone kv={kv} beyond limit",
                    )

    def test_different_suffixes_dont_attend(self):
        """Positions in different suffixes never attend to each other."""
        attn_masks, _, suffix_ids, _ = self._random_batch()
        mask = self._materialize_mask(attn_masks)
        S = suffix_ids.shape[0]
        for q in range(S):
            if suffix_ids[q] == 0:
                continue
            for kv in range(S):
                if suffix_ids[kv] == 0:
                    continue
                if suffix_ids[q] != suffix_ids[kv]:
                    self.assertFalse(
                        mask[q, kv].item(),
                        f"cross-suffix attention: q={q} (sid={suffix_ids[q]}) "
                        f"→ kv={kv} (sid={suffix_ids[kv]})",
                    )

    def test_padding_never_attended_to(self):
        """Padding positions (conv_ids=0) are never attended to."""
        attn_masks, conv_ids, _, _ = self._random_batch()
        mask = self._materialize_mask(attn_masks)
        S = conv_ids.shape[0]
        pad_positions = [i for i in range(S) if conv_ids[i] == 0]
        for kv in pad_positions:
            for q in range(S):
                self.assertFalse(
                    mask[q, kv].item(),
                    f"q={q} attends to padding kv={kv}",
                )

    def test_causality(self):
        """No position attends to a later position (q_idx < kv_idx → blocked)."""
        attn_masks, _, _, _ = self._random_batch()
        mask = self._materialize_mask(attn_masks)
        S = mask.shape[0]
        for q in range(S):
            for kv in range(q + 1, S):
                self.assertFalse(
                    mask[q, kv].item(),
                    f"non-causal attention: q={q} → kv={kv}",
                )


if __name__ == "__main__":
    unittest.main()
