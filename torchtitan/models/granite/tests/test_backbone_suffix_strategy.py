"""RED-GREEN tests for BackboneSuffixStrategy.

These tests verify the output of BackboneSuffixStrategy._tokenize_one against
the template-contract assumptions validated in test_template_contracts.py.

RED: BackboneSuffixStrategy doesn't exist yet → ImportError.
GREEN: After implementation, all tests pass.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template.
"""

import os
import unittest

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")

try:
    from torchtitan.models.granite.tokenization_strategies import (
        BackboneSuffixStrategy,
    )

    _STRATEGY_AVAILABLE = True
except ImportError:
    _STRATEGY_AVAILABLE = False


def _skip_unless_ready(fn):
    fn = unittest.skipUnless(
        _HF_ASSETS_PATH,
        "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
    )(fn)
    fn = unittest.skipUnless(
        _STRATEGY_AVAILABLE,
        "BackboneSuffixStrategy not yet implemented",
    )(fn)
    return fn


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
@unittest.skipUnless(_STRATEGY_AVAILABLE, "BackboneSuffixStrategy not yet implemented")
class TestBackboneSuffixOutputStructure(unittest.TestCase):
    """Verify output dict has correct keys and consistent lengths."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = BackboneSuffixStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")

    def _tokenize(self, messages):
        return self.strategy._tokenize_one(messages)

    def test_output_keys(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        expected_keys = {"input_ids", "labels", "positions", "suffix_starts", "insertion_limits", "n_tokens"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_lengths_consistent(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        n = result["n_tokens"]
        self.assertEqual(len(result["input_ids"]), n)
        self.assertEqual(len(result["labels"]), n)
        self.assertEqual(len(result["positions"]), n)

    def test_suffix_starts_and_insertion_limits_parallel(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        self.assertEqual(len(result["suffix_starts"]), len(result["insertion_limits"]))

    def test_suffix_starts_within_bounds(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        n = result["n_tokens"]
        for i, start in enumerate(result["suffix_starts"]):
            self.assertGreater(start, 0, f"suffix_starts[{i}] must be > 0")
            self.assertLess(start, n, f"suffix_starts[{i}] must be < n_tokens")

    def test_suffix_starts_ascending(self):
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        starts = result["suffix_starts"]
        for i in range(1, len(starts)):
            self.assertGreater(starts[i], starts[i - 1])

    def test_suffixes_contiguous(self):
        """Suffixes are packed contiguously: each starts where the prior ends."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        starts = result["suffix_starts"]
        n = result["n_tokens"]
        self.assertEqual(len(starts), 2)
        # First suffix starts at backbone end
        backbone_end = starts[0]
        self.assertGreater(backbone_end, 0)
        # Second suffix starts immediately after first ends
        # (last suffix ends at n_tokens)
        suffix_0_len = starts[1] - starts[0]
        suffix_1_len = n - starts[1]
        self.assertGreater(suffix_0_len, 0)
        self.assertGreater(suffix_1_len, 0)
        self.assertEqual(backbone_end + suffix_0_len + suffix_1_len, n)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
@unittest.skipUnless(_STRATEGY_AVAILABLE, "BackboneSuffixStrategy not yet implemented")
class TestBackboneSuffixEquivalence(unittest.TestCase):
    """Cases where BackboneSuffixStrategy must match TruncateLastStrategy."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = BackboneSuffixStrategy(_HF_ASSETS_PATH)
        from torchtitan.models.granite.tokenization_strategies import TruncateLastStrategy
        cls.truncate_strategy = TruncateLastStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def test_single_turn_no_suffixes(self):
        """Single turn: zero suffixes, backbone = TruncateLastStrategy output."""
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Simple math."},
        ]
        bs_result = self.strategy._tokenize_one(msgs)
        tl_result = self.truncate_strategy._tokenize_one(msgs)

        self.assertEqual(bs_result["suffix_starts"], [])
        self.assertEqual(bs_result["insertion_limits"], [])
        self.assertEqual(bs_result["input_ids"], tl_result["input_ids"])
        self.assertEqual(bs_result["labels"], tl_result["labels"])

    def test_single_user_tool_chain_no_suffixes(self):
        """Tool chain with single user: no truncation fires → no suffixes."""
        msgs = [
            {"role": "user", "content": "Search for X."},
            {"role": "assistant", "content": "Calling search.", "reasoning_content": "Need to search."},
            {"role": "tool", "content": "Found X at location Y."},
            {"role": "assistant", "content": "X is at Y.", "reasoning_content": "Got result."},
        ]
        bs_result = self.strategy._tokenize_one(msgs)
        tl_result = self.truncate_strategy._tokenize_one(msgs)

        self.assertEqual(bs_result["suffix_starts"], [])
        self.assertEqual(bs_result["insertion_limits"], [])
        self.assertEqual(bs_result["input_ids"], tl_result["input_ids"])
        self.assertEqual(bs_result["labels"], tl_result["labels"])

    def test_backbone_matches_truncate_last(self):
        """Multi-turn: backbone portion (before first suffix) matches TruncateLastStrategy exactly."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        bs_result = self.strategy._tokenize_one(msgs)
        tl_result = self.truncate_strategy._tokenize_one(msgs)

        backbone_end = bs_result["suffix_starts"][0]
        self.assertEqual(backbone_end, len(tl_result["input_ids"]))
        self.assertEqual(bs_result["input_ids"][:backbone_end], tl_result["input_ids"])
        self.assertEqual(bs_result["labels"][:backbone_end], tl_result["labels"])


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
@unittest.skipUnless(_STRATEGY_AVAILABLE, "BackboneSuffixStrategy not yet implemented")
class TestBackboneSuffixMultiTurn(unittest.TestCase):
    """Multi-turn conversations with suffixes."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = BackboneSuffixStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")

    def test_clean_multi_turn_suffix_count(self):
        """u→a→u→a→u→a: two suffixes (one per historical assistant turn)."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(len(result["suffix_starts"]), 2)

    def test_tool_chain_with_followup_one_suffix(self):
        """u→a→tool→a→u→a: one suffix covering both assistant turns in the group."""
        msgs = [
            {"role": "user", "content": "Do search."},
            {"role": "assistant", "content": "Calling.", "reasoning_content": "Need search."},
            {"role": "tool", "content": "Found result."},
            {"role": "assistant", "content": "Here it is.", "reasoning_content": "Got it."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "Welcome.", "reasoning_content": "Done."},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(len(result["suffix_starts"]), 1)

    def test_insertion_limit_points_to_think_in_backbone(self):
        """Each insertion_limit indexes <think> in the backbone tokens."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self.strategy._tokenize_one(msgs)
        backbone_end = result["suffix_starts"][0]
        # insertion_limits are relative to sequence start (= backbone positions)
        for i, ins_lim in enumerate(result["insertion_limits"]):
            self.assertLess(ins_lim, backbone_end,
                            f"insertion_limits[{i}] must be within backbone")
            self.assertEqual(
                result["input_ids"][ins_lim],
                self.think_id,
                f"insertion_limits[{i}]={ins_lim} must point to <think>",
            )

    def test_suffix_positions_sequential(self):
        """Suffix positions start at insertion_limit+1 and increment sequentially."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(len(result["suffix_starts"]), 1)

        suffix_start = result["suffix_starts"][0]
        ins_lim = result["insertion_limits"][0]
        suffix_positions = result["positions"][suffix_start:]

        expected_start = ins_lim + 1
        for i, pos in enumerate(suffix_positions):
            self.assertEqual(
                pos, expected_start + i,
                f"suffix position[{i}] should be {expected_start + i}, got {pos}",
            )

    def test_backbone_positions_sequential_from_zero(self):
        """Backbone positions are 0, 1, 2, ..., backbone_len-1."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        backbone_end = result["suffix_starts"][0]
        backbone_positions = result["positions"][:backbone_end]
        expected = list(range(backbone_end))
        self.assertEqual(backbone_positions, expected)

    def test_system_message_does_not_break_suffix_arithmetic(self):
        """System message shifts token positions but suffix arithmetic still holds."""
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(len(result["suffix_starts"]), 1)

        # insertion_limit points to <think> in backbone
        ins_lim = result["insertion_limits"][0]
        self.assertEqual(result["input_ids"][ins_lim], self.think_id)

        # Suffix positions start at insertion_limit + 1
        suffix_start = result["suffix_starts"][0]
        self.assertEqual(result["positions"][suffix_start], ins_lim + 1)

        # Suffix has trained labels
        suffix_labels = result["labels"][suffix_start:]
        trained = [lbl for lbl in suffix_labels if lbl != IGNORE_INDEX]
        self.assertGreater(len(trained), 0)

    def test_suffix_labels_have_trained_tokens(self):
        """Suffix must contain at least some trained labels (non-IGNORE_INDEX)."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        suffix_start = result["suffix_starts"][0]
        suffix_labels = result["labels"][suffix_start:]
        trained = [lbl for lbl in suffix_labels if lbl != IGNORE_INDEX]
        self.assertGreater(len(trained), 0, "Suffix must have trained tokens")

    def test_suffix_content_matches_thinking_preserved_rendering(self):
        """Suffix tokens match the thinking-preserved rendering of the suffix group."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        suffix_start = result["suffix_starts"][0]
        ins_lim = result["insertion_limits"][0]

        # Suffix source: msgs[:2] rendered with truncate=True (thinking preserved
        # because no user after it in the slice)
        suffix_source_text = self.tok.apply_chat_template(
            msgs[:2], truncate_history_thinking=True
        )
        suffix_source_tokens = self.tok.encode(suffix_source_text, add_bos=True, add_eos=False)

        # Suffix = suffix_source_tokens[ins_lim+1 : end]
        # where end is determined by the strategy (up to last <|im_end|>)
        suffix_tokens = result["input_ids"][suffix_start:]
        expected_suffix = suffix_source_tokens[ins_lim + 1:]

        # The suffix input_ids should be a prefix of expected_suffix
        # (strategy may add eos or trim slightly)
        self.assertEqual(
            suffix_tokens[:len(expected_suffix)],
            expected_suffix[:len(suffix_tokens)],
            "Suffix tokens must match thinking-preserved rendering",
        )


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
@unittest.skipUnless(_STRATEGY_AVAILABLE, "BackboneSuffixStrategy not yet implemented")
class TestBackboneSuffixMixedReasoning(unittest.TestCase):
    """Conversations where some turns have reasoning and others don't."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = BackboneSuffixStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")

    def test_no_reasoning_in_historical_turns_no_suffix(self):
        """If no historical turn has reasoning_content, no suffix is created."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},  # no reasoning
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(result["suffix_starts"], [])
        self.assertEqual(result["insertion_limits"], [])

    def test_whitespace_only_reasoning_no_suffix(self):
        """Whitespace-only reasoning_content is treated as no reasoning."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "   \n  "},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(result["suffix_starts"], [])
        self.assertEqual(result["insertion_limits"], [])

    def test_mixed_reasoning_suffix_covers_all_turns_in_group(self):
        """Suffix group with mixed reasoning/no-reasoning still creates one suffix.

        The group has [asst_with_rc, tool, asst_no_rc]. Since at least one turn
        has reasoning, the suffix is created covering all turns.
        """
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "tool", "content": "T0"},
            {"role": "assistant", "content": "A1"},  # no reasoning
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "Final", "reasoning_content": "RF"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(len(result["suffix_starts"]), 1)

        # Suffix should contain content from both assistant turns
        suffix_start = result["suffix_starts"][0]
        suffix_tokens = result["input_ids"][suffix_start:]
        decoded = self.tok.decode(suffix_tokens, skip_special_tokens=False)
        self.assertIn("R0", decoded, "Suffix must contain reasoning from first turn")
        self.assertIn("A1", decoded, "Suffix must contain response from second turn")

    def test_reasoning_in_one_of_multiple_groups(self):
        """Multiple suffix groups, only some have reasoning → only those create suffixes."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},  # no reasoning → no suffix
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},  # has reasoning → suffix
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self.strategy._tokenize_one(msgs)
        # Only one suffix (for the group between Q1 and Q2 that has reasoning)
        self.assertEqual(len(result["suffix_starts"]), 1)

    def test_no_reasoning_turn_label_starts_at_end_think(self):
        """For no-reasoning turn in suffix, label starts at </think> (label = first response token)."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A0"},  # no reasoning — but won't create suffix alone
            {"role": "tool", "content": "T"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},  # has reasoning
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "Final", "reasoning_content": "RF"},
        ]
        result = self.strategy._tokenize_one(msgs)
        self.assertEqual(len(result["suffix_starts"]), 1)

        suffix_start = result["suffix_starts"][0]
        suffix_input = result["input_ids"][suffix_start:]
        suffix_labels = result["labels"][suffix_start:]

        # First trained position in suffix should be </think> (no-reasoning first turn)
        first_trained_idx = next(
            i for i, lbl in enumerate(suffix_labels) if lbl != IGNORE_INDEX
        )
        self.assertEqual(
            suffix_input[first_trained_idx],
            self.end_think_id,
            "First trained position in suffix should be </think> for no-reasoning turn",
        )


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
@unittest.skipUnless(_STRATEGY_AVAILABLE, "BackboneSuffixStrategy not yet implemented")
class TestBackboneSuffixLabelBoundaries(unittest.TestCase):
    """Verify label boundaries within suffixes follow the rendering-based formula."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = BackboneSuffixStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")
        cls.newline_id = cls.tok.encode("\n", add_bos=False, add_eos=False)[0]
        cls.im_end_id = cls.tok.token_to_id("<|im_end|>")

    def test_reasoning_turn_label_starts_after_think_newline(self):
        """For reasoning turn in suffix, first label is at \\n position (label = first reasoning token)."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        suffix_start = result["suffix_starts"][0]
        suffix_input = result["input_ids"][suffix_start:]
        suffix_labels = result["labels"][suffix_start:]

        # First trained position: \n after <think>
        first_trained_idx = next(
            i for i, lbl in enumerate(suffix_labels) if lbl != IGNORE_INDEX
        )
        self.assertEqual(
            suffix_input[first_trained_idx],
            self.newline_id,
            "First trained position should be \\n (after <think>) for reasoning turn",
        )

    def test_im_end_position_is_ignore_index(self):
        """<|im_end|> positions within suffix have IGNORE_INDEX labels.

        Uses a tool chain so the suffix has intermediate <|im_end|> tokens
        (single-turn suffixes have no <|im_end|> in suffix_input because the
        final eos is shifted off).
        """
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "tool", "content": "T0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "Final", "reasoning_content": "RF"},
        ]
        result = self.strategy._tokenize_one(msgs)
        suffix_start = result["suffix_starts"][0]
        suffix_input = result["input_ids"][suffix_start:]
        suffix_labels = result["labels"][suffix_start:]

        im_end_positions = [i for i, tok in enumerate(suffix_input) if tok == self.im_end_id]
        self.assertGreater(len(im_end_positions), 0, "Suffix must contain <|im_end|>")
        for pos in im_end_positions:
            self.assertEqual(
                suffix_labels[pos],
                IGNORE_INDEX,
                f"<|im_end|> at suffix position {pos} must have IGNORE_INDEX label",
            )

    def test_multi_turn_suffix_has_multiple_label_regions(self):
        """Tool chain suffix (2 assistant turns) has two separate labeled regions."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "tool", "content": "T0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "Final", "reasoning_content": "RF"},
        ]
        result = self.strategy._tokenize_one(msgs)
        suffix_start = result["suffix_starts"][0]
        suffix_labels = result["labels"][suffix_start:]

        # Count transitions from IGNORE_INDEX to non-IGNORE_INDEX
        transitions = 0
        prev_is_ignore = True
        for lbl in suffix_labels:
            curr_is_ignore = (lbl == IGNORE_INDEX)
            if prev_is_ignore and not curr_is_ignore:
                transitions += 1
            prev_is_ignore = curr_is_ignore

        self.assertEqual(
            transitions, 2,
            "Tool chain suffix should have 2 labeled regions (one per assistant turn)",
        )

    def test_last_trained_label_is_eos(self):
        """Last non-IGNORE_INDEX label in the suffix is eos_id."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result = self.strategy._tokenize_one(msgs)
        suffix_start = result["suffix_starts"][0]
        suffix_labels = result["labels"][suffix_start:]

        trained_labels = [lbl for lbl in suffix_labels if lbl != IGNORE_INDEX]
        self.assertGreater(len(trained_labels), 0)
        self.assertEqual(
            trained_labels[-1],
            self.tok.eos_id,
            "Last trained label in suffix must be eos_id",
        )

    def test_all_label_starts_nonnegative(self):
        """label_start_in_suffix must be >= 0 for all turns in all suffix groups."""
        conversations = [
            # Simple multi-turn
            [
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            ],
            # Tool chain
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
                {"role": "tool", "content": "T0"},
                {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
                {"role": "user", "content": "Follow"},
                {"role": "assistant", "content": "Final", "reasoning_content": "RF"},
            ],
            # Three groups
            [
                {"role": "user", "content": "Q0"},
                {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            ],
        ]
        for msgs in conversations:
            result = self.strategy._tokenize_one(msgs)
            for k, suffix_start in enumerate(result["suffix_starts"]):
                suffix_labels = result["labels"][suffix_start:]
                # Find first trained position — must exist and be at index >= 0
                first_trained = next(
                    (i for i, lbl in enumerate(suffix_labels) if lbl != IGNORE_INDEX),
                    None,
                )
                self.assertIsNotNone(
                    first_trained,
                    f"Suffix {k} has no trained tokens (label_start likely negative)",
                )
                self.assertGreaterEqual(first_trained, 0)


if __name__ == "__main__":
    unittest.main()
