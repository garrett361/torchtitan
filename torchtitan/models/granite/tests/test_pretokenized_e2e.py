"""GPU end-to-end tests: TruncateLastDataset → GraniteModel → finite loss."""

import unittest

import torch

from torchtitan.components.loss import cross_entropy_loss
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
        batch_dict, labels = next(iter(ds))
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


if __name__ == "__main__":
    unittest.main()
