# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.attention import (
    get_document_mask_mod,
    get_document_mask_mod_from_positions,
)


def _evaluate_mask(mask_fn, batch_size, seq_len):
    """Evaluate a mask_mod function into a [B, S, S] boolean tensor."""
    b = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, seq_len)
    h = torch.zeros_like(b)
    q = torch.arange(seq_len).unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, seq_len)
    kv = torch.arange(seq_len).unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, seq_len)
    return mask_fn(b, h, q, kv)


class TestGetDocumentMaskModFromPositions(unittest.TestCase):

    def test_single_document_allows_full_attention(self):
        positions = torch.arange(8).unsqueeze(0)  # [1, 8]: 0,1,2,...,7
        mask_fn = get_document_mask_mod_from_positions(positions)
        mask = _evaluate_mask(mask_fn, 1, 8)
        self.assertTrue(mask.all())

    def test_two_packed_documents(self):
        # doc0: positions 0,1,2  doc1: positions 0,1,2,3,4
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
        mask_fn = get_document_mask_mod_from_positions(positions)
        mask = _evaluate_mask(mask_fn, 1, 8)

        # Within doc0 (indices 0-2): all attend to each other
        self.assertTrue(mask[0, :3, :3].all())
        # Within doc1 (indices 3-7): all attend to each other
        self.assertTrue(mask[0, 3:, 3:].all())
        # Cross-document: blocked
        self.assertFalse(mask[0, :3, 3:].any())
        self.assertFalse(mask[0, 3:, :3].any())

    def test_three_packed_documents(self):
        # doc0: [0,1], doc1: [0,1,2], doc2: [0,1]
        positions = torch.tensor([[0, 1, 0, 1, 2, 0, 1]])
        mask_fn = get_document_mask_mod_from_positions(positions)
        mask = _evaluate_mask(mask_fn, 1, 7)

        # Each doc attends within itself
        self.assertTrue(mask[0, :2, :2].all())
        self.assertTrue(mask[0, 2:5, 2:5].all())
        self.assertTrue(mask[0, 5:, 5:].all())
        # Cross-document: all blocked
        self.assertFalse(mask[0, :2, 2:].any())
        self.assertFalse(mask[0, 2:5, :2].any())
        self.assertFalse(mask[0, 2:5, 5:].any())
        self.assertFalse(mask[0, 5:, :5].any())

    def test_batched(self):
        # batch=0: single doc, batch=1: two docs
        positions = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 0, 1],
        ])
        mask_fn = get_document_mask_mod_from_positions(positions)
        mask = _evaluate_mask(mask_fn, 2, 5)

        # batch 0: single doc — full attention
        self.assertTrue(mask[0].all())
        # batch 1: two docs
        self.assertTrue(mask[1, :3, :3].all())
        self.assertTrue(mask[1, 3:, 3:].all())
        self.assertFalse(mask[1, :3, 3:].any())
        self.assertFalse(mask[1, 3:, :3].any())

    def test_matches_eos_based_when_eos_only_at_boundaries(self):
        """When eos appears exactly once per document (at the end), both
        methods should produce identical document IDs."""
        eos_id = 99
        # Two docs: tokens [10, 20, 99, 30, 40, 50, 99]
        input_batch = torch.tensor([[10, 20, eos_id, 30, 40, 50, eos_id]])
        # Corresponding positions: doc0=[0,1,2], doc1=[0,1,2,3]
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])

        mask_eos = _evaluate_mask(get_document_mask_mod(input_batch, eos_id), 1, 7)
        mask_pos = _evaluate_mask(get_document_mask_mod_from_positions(positions), 1, 7)
        self.assertTrue(torch.equal(mask_eos, mask_pos))

    def test_diverges_from_eos_when_eos_inside_document(self):
        """When eos appears mid-document (chat template role-closing tags),
        the eos-based mask incorrectly splits the document. The positions-based
        mask should keep it as one document."""
        eos_id = 99
        # One chat example: [start, system_text, eos, start, user_text, eos, start, assistant_text, eos]
        # eos appears 3 times but this is a single packed document
        input_batch = torch.tensor([[1, 2, eos_id, 3, 4, eos_id, 5, 6, eos_id]])
        # Single document — positions are monotonically increasing
        positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

        mask_eos = _evaluate_mask(get_document_mask_mod(input_batch, eos_id), 1, 9)
        mask_pos = _evaluate_mask(get_document_mask_mod_from_positions(positions), 1, 9)

        # eos-based mask incorrectly blocks cross-"role" attention
        self.assertFalse(mask_eos[0, 0, 3].item())  # system can't see user
        # positions-based mask correctly allows full intra-document attention
        self.assertTrue(mask_pos.all())


if __name__ == "__main__":
    unittest.main()
