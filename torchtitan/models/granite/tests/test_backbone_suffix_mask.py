# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.attention import (
    get_backbone_suffix_mask_mod,
    get_causal_mask_mod,
)


def _evaluate_mask(mask_fn, batch_size, seq_len):
    """Evaluate a mask_mod function into a [B, S, S] boolean tensor."""
    b = (
        torch.arange(batch_size)
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(batch_size, seq_len, seq_len)
    )
    h = torch.zeros_like(b)
    q = (
        torch.arange(seq_len)
        .unsqueeze(0)
        .unsqueeze(2)
        .expand(batch_size, seq_len, seq_len)
    )
    kv = (
        torch.arange(seq_len)
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(batch_size, seq_len, seq_len)
    )
    return mask_fn(b, h, q, kv)


def _compose_with_causal(mask_fn, batch_size, seq_len):
    """Evaluate backbone_suffix mask AND causal mask."""
    causal = _evaluate_mask(get_causal_mask_mod(), batch_size, seq_len)
    backbone_suffix = _evaluate_mask(mask_fn, batch_size, seq_len)
    return causal & backbone_suffix


class TestBackboneSuffixMaskMod(unittest.TestCase):
    """Test get_backbone_suffix_mask_mod with synthetic tensors.

    Layout used in most tests (seq_len=10, single batch):
      positions 0-4: backbone (conv=1, suffix_id=0, insertion_limit=4)
      positions 5-7: suffix_1 (conv=1, suffix_id=1, insertion_limit=2)
      positions 8-9: suffix_2 (conv=1, suffix_id=2, insertion_limit=4)
    """

    def _make_single_conv(self):
        """Single conversation: backbone[0:5], suffix_1[5:8], suffix_2[8:10]."""
        conv_ids = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        suffix_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2]])
        insertion_limits = torch.tensor([[4, 4, 4, 4, 4, 2, 2, 2, 4, 4]])
        return conv_ids, suffix_ids, insertion_limits

    def test_backbone_to_backbone_causal(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # Backbone positions 0-4 can attend to all earlier backbone positions
        for q in range(5):
            for kv in range(q + 1):
                self.assertTrue(
                    mask[0, q, kv],
                    f"backbone[{q}] should attend to backbone[{kv}]",
                )

    def test_backbone_cannot_attend_to_suffix(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        for q in range(5):
            for kv in range(5, 10):
                self.assertFalse(
                    mask[0, q, kv],
                    f"backbone[{q}] must not attend to suffix position [{kv}]",
                )

    def test_suffix_to_backbone_within_limit(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # suffix_1 (positions 5-7) has insertion_limit=2 → attends to backbone[0:3]
        for q in range(5, 8):
            for kv in range(3):
                self.assertTrue(
                    mask[0, q, kv],
                    f"suffix_1[{q}] should attend to backbone[{kv}] (limit=2)",
                )

    def test_suffix_to_backbone_beyond_limit(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # suffix_1 (positions 5-7) has insertion_limit=2 → cannot attend to backbone[3:5]
        for q in range(5, 8):
            for kv in range(3, 5):
                self.assertFalse(
                    mask[0, q, kv],
                    f"suffix_1[{q}] must not attend to backbone[{kv}] (limit=2)",
                )

    def test_suffix_self_attention_causal(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # Within suffix_1: causal self-attention
        for q in range(5, 8):
            for kv in range(5, 8):
                expected = q >= kv
                self.assertEqual(
                    mask[0, q, kv].item(),
                    expected,
                    f"suffix_1 self-attention [{q}]->[{kv}] should be {expected}",
                )

    def test_different_suffixes_blocked(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # suffix_2 (positions 8-9) cannot attend to suffix_1 (positions 5-7)
        for q in range(8, 10):
            for kv in range(5, 8):
                self.assertFalse(
                    mask[0, q, kv],
                    f"suffix_2[{q}] must not attend to suffix_1[{kv}]",
                )

        # suffix_1 cannot attend to suffix_2 (also blocked by causality)
        for q in range(5, 8):
            for kv in range(8, 10):
                self.assertFalse(
                    mask[0, q, kv],
                    f"suffix_1[{q}] must not attend to suffix_2[{kv}]",
                )

    def test_suffix2_full_backbone_access(self):
        conv_ids, suffix_ids, insertion_limits = self._make_single_conv()
        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # suffix_2 (positions 8-9) has insertion_limit=4 → attends to all backbone[0:5]
        for q in range(8, 10):
            for kv in range(5):
                self.assertTrue(
                    mask[0, q, kv],
                    f"suffix_2[{q}] should attend to backbone[{kv}] (limit=4)",
                )

    def test_cross_conversation_blocked(self):
        """Two conversations packed in one row must not attend to each other."""
        # conv1: backbone[0:3] + suffix[3:5]
        # conv2: backbone[5:8] + suffix[8:10]
        conv_ids = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]])
        suffix_ids = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 1, 1]])
        insertion_limits = torch.tensor([[2, 2, 2, 1, 1, 7, 7, 7, 6, 6]])

        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # conv2 positions cannot attend to conv1 positions
        for q in range(5, 10):
            for kv in range(5):
                self.assertFalse(
                    mask[0, q, kv],
                    f"conv2[{q}] must not attend to conv1[{kv}]",
                )

        # conv1 positions cannot attend to conv2 positions (also blocked by causality
        # for forward positions, but explicitly verify the mask logic)
        for q in range(5):
            for kv in range(5, 10):
                self.assertFalse(
                    mask[0, q, kv],
                    f"conv1[{q}] must not attend to conv2[{kv}]",
                )

    def test_padding_isolation(self):
        """Padding (conv_ids=0, suffix_ids=0, insertion_limits=-1) is fully isolated."""
        # Real content: positions 0-5 (conv=1, backbone)
        # Padding: positions 6-9
        conv_ids = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        suffix_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        insertion_limits = torch.tensor([[5, 5, 5, 5, 5, 5, -1, -1, -1, -1]])

        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 10)

        # Real content cannot attend to padding
        for q in range(6):
            for kv in range(6, 10):
                self.assertFalse(
                    mask[0, q, kv],
                    f"real[{q}] must not attend to padding[{kv}]",
                )

        # Padding cannot attend to real content (insertion_limits=-1 blocks to_backbone,
        # same_suffix requires >0)
        for q in range(6, 10):
            for kv in range(6):
                self.assertFalse(
                    mask[0, q, kv],
                    f"padding[{q}] must not attend to real[{kv}]",
                )

        # Padding cannot attend to itself (conv_ids=0 == conv_ids=0 is true,
        # but suffix_ids=0 fails >0 check, and insertion_limits=-1 blocks backbone)
        for q in range(6, 10):
            for kv in range(6, 10):
                self.assertFalse(
                    mask[0, q, kv],
                    f"padding[{q}] must not attend to padding[{kv}]",
                )

    def test_batched(self):
        """Multiple batch entries are handled independently."""
        # batch=0: simple backbone-only
        # batch=1: backbone + suffix
        conv_ids = torch.tensor([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ])
        suffix_ids = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
        ])
        insertion_limits = torch.tensor([
            [3, 3, 3, 3, -1],
            [2, 2, 2, 1, 1],
        ])

        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 2, 5)

        # batch=0: backbone[0:4] causal, padding[4] isolated
        for q in range(4):
            for kv in range(q + 1):
                self.assertTrue(mask[0, q, kv])
        self.assertFalse(mask[0, 0, 4])
        self.assertFalse(mask[0, 4, 0])

        # batch=1: suffix[3:5] attends to backbone[0:2] (limit=1)
        self.assertTrue(mask[1, 3, 0])
        self.assertTrue(mask[1, 3, 1])
        self.assertFalse(mask[1, 3, 2])  # beyond limit

    def test_insertion_limit_is_inclusive(self):
        """insertion_limit is the position of <think> — suffix CAN attend to it."""
        conv_ids = torch.tensor([[1, 1, 1, 1, 1]])
        suffix_ids = torch.tensor([[0, 0, 0, 1, 1]])
        insertion_limits = torch.tensor([[2, 2, 2, 2, 2]])

        mask_fn = get_backbone_suffix_mask_mod(conv_ids, suffix_ids, insertion_limits)
        mask = _compose_with_causal(mask_fn, 1, 5)

        # Suffix position 3 can attend to backbone position 2 (the limit itself)
        self.assertTrue(mask[0, 3, 2])
        # But not beyond
        # (there's nothing beyond in this example, but backbone position is at limit)


class TestBackboneSuffixMaskModProperties(unittest.TestCase):
    """Property-based tests with random tensor configurations."""

    def _random_config(self, batch_size=2, seq_len=16, seed=42):
        """Generate a random but valid backbone+suffix configuration."""
        rng = torch.Generator().manual_seed(seed)

        conv_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        suffix_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        insertion_limits = torch.full(
            (batch_size, seq_len), -1, dtype=torch.long
        )

        for b in range(batch_size):
            pos = 0
            conv_counter = 1
            suffix_counter = 1

            while pos < seq_len:
                remaining = seq_len - pos
                if remaining < 3:
                    break

                # Random backbone length
                backbone_len = torch.randint(
                    2, min(remaining, 8) + 1, (1,), generator=rng
                ).item()
                backbone_end = pos + backbone_len

                conv_ids[b, pos:backbone_end] = conv_counter
                suffix_ids[b, pos:backbone_end] = 0
                insertion_limits[b, pos:backbone_end] = backbone_end - 1

                suffix_pos = backbone_end
                while suffix_pos < seq_len:
                    suffix_remaining = seq_len - suffix_pos
                    if suffix_remaining < 2 or torch.rand(1, generator=rng) > 0.6:
                        break
                    suffix_len = torch.randint(
                        1, min(suffix_remaining, 4) + 1, (1,), generator=rng
                    ).item()
                    # insertion_limit is within backbone
                    ins_lim = torch.randint(
                        pos, backbone_end, (1,), generator=rng
                    ).item()

                    conv_ids[b, suffix_pos : suffix_pos + suffix_len] = conv_counter
                    suffix_ids[b, suffix_pos : suffix_pos + suffix_len] = (
                        suffix_counter
                    )
                    insertion_limits[b, suffix_pos : suffix_pos + suffix_len] = (
                        ins_lim
                    )
                    suffix_counter += 1
                    suffix_pos += suffix_len

                pos = suffix_pos
                conv_counter += 1
                suffix_counter += 1

        return conv_ids, suffix_ids, insertion_limits

    def _assert_has_suffix(self, suffix_ids, seed):
        self.assertTrue(
            (suffix_ids > 0).any().item(),
            f"seed={seed}: _random_config produced no suffix positions",
        )

    def test_no_cross_conv_attention(self):
        for seed in range(5):
            conv_ids, suffix_ids, insertion_limits = self._random_config(seed=seed)
            mask_fn = get_backbone_suffix_mask_mod(
                conv_ids, suffix_ids, insertion_limits
            )
            mask = _compose_with_causal(mask_fn, 2, 16)

            for b in range(2):
                for q in range(16):
                    for kv in range(16):
                        if conv_ids[b, q] != conv_ids[b, kv]:
                            self.assertFalse(
                                mask[b, q, kv],
                                f"seed={seed} cross-conv attention at b={b} q={q} kv={kv}",
                            )

    def test_suffix_never_attends_past_insertion_limit(self):
        for seed in range(5):
            conv_ids, suffix_ids, insertion_limits = self._random_config(seed=seed)
            self._assert_has_suffix(suffix_ids, seed)
            mask_fn = get_backbone_suffix_mask_mod(
                conv_ids, suffix_ids, insertion_limits
            )
            mask = _compose_with_causal(mask_fn, 2, 16)

            for b in range(2):
                for q in range(16):
                    if suffix_ids[b, q] > 0:
                        limit = insertion_limits[b, q].item()
                        for kv in range(16):
                            if suffix_ids[b, kv] == 0 and kv > limit:
                                self.assertFalse(
                                    mask[b, q, kv],
                                    f"seed={seed} suffix past limit b={b} q={q} kv={kv} limit={limit}",
                                )

    def test_backbone_never_attends_to_suffix(self):
        for seed in range(5):
            conv_ids, suffix_ids, insertion_limits = self._random_config(seed=seed)
            self._assert_has_suffix(suffix_ids, seed)
            mask_fn = get_backbone_suffix_mask_mod(
                conv_ids, suffix_ids, insertion_limits
            )
            mask = _compose_with_causal(mask_fn, 2, 16)

            for b in range(2):
                for q in range(16):
                    if suffix_ids[b, q] == 0 and conv_ids[b, q] > 0:
                        for kv in range(16):
                            if suffix_ids[b, kv] > 0:
                                self.assertFalse(
                                    mask[b, q, kv],
                                    f"seed={seed} backbone->suffix b={b} q={q} kv={kv}",
                                )

    def test_different_suffixes_never_attend(self):
        for seed in range(5):
            conv_ids, suffix_ids, insertion_limits = self._random_config(seed=seed)
            self._assert_has_suffix(suffix_ids, seed)
            mask_fn = get_backbone_suffix_mask_mod(
                conv_ids, suffix_ids, insertion_limits
            )
            mask = _compose_with_causal(mask_fn, 2, 16)

            for b in range(2):
                for q in range(16):
                    if suffix_ids[b, q] > 0:
                        for kv in range(16):
                            if (
                                suffix_ids[b, kv] > 0
                                and suffix_ids[b, kv] != suffix_ids[b, q]
                            ):
                                self.assertFalse(
                                    mask[b, q, kv],
                                    f"seed={seed} cross-suffix b={b} q={q} kv={kv}",
                                )

    def test_padding_never_attends_or_is_attended(self):
        for seed in range(5):
            conv_ids, suffix_ids, insertion_limits = self._random_config(seed=seed)
            mask_fn = get_backbone_suffix_mask_mod(
                conv_ids, suffix_ids, insertion_limits
            )
            mask = _compose_with_causal(mask_fn, 2, 16)

            for b in range(2):
                for pos in range(16):
                    if conv_ids[b, pos] == 0:
                        # Padding cannot attend to anything
                        self.assertFalse(
                            mask[b, pos, :].any(),
                            f"seed={seed} padding attends at b={b} pos={pos}",
                        )
                        # Nothing can attend to padding
                        self.assertFalse(
                            mask[b, :, pos].any(),
                            f"seed={seed} attended to padding at b={b} pos={pos}",
                        )


if __name__ == "__main__":
    unittest.main()
