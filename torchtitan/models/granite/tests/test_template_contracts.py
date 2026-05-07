"""Template-contract tests for Backbone+Suffix assumptions.

These tests verify claims about the Granite thinking chat template that the
BackboneSuffixStrategy label logic depends on. Each test corresponds to a
numbered claim in the plan's Verification section.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template. All tests skip if not set.
"""

import os
import unittest

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")


def _skip_unless_granite(fn):
    return unittest.skipUnless(
        _HF_ASSETS_PATH,
        "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
    )(fn)


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestGenerationPromptEnding(unittest.TestCase):
    """#1: Generation prompt ends with [<think>][\\n] as the last two tokens."""

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.newline_id = cls.tok.encode("\n", add_bos=False, add_eos=False)[0]

    def _gen_prompt_tokens(self, messages):
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, truncate_history_thinking=True
        )
        return self.tok.encode(text, add_bos=True, add_eos=False)

    def test_single_user_generation_prompt_ends_think_newline(self):
        msgs = [{"role": "user", "content": "Hello."}]
        tokens = self._gen_prompt_tokens(msgs)
        self.assertEqual(tokens[-2], self.think_id)
        self.assertEqual(tokens[-1], self.newline_id)

    def test_system_user_generation_prompt_ends_think_newline(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        tokens = self._gen_prompt_tokens(msgs)
        self.assertEqual(tokens[-2], self.think_id)
        self.assertEqual(tokens[-1], self.newline_id)

    def test_multi_turn_generation_prompt_ends_think_newline(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
        ]
        tokens = self._gen_prompt_tokens(msgs)
        self.assertEqual(tokens[-2], self.think_id)
        self.assertEqual(tokens[-1], self.newline_id)

    def test_tool_chain_generation_prompt_ends_think_newline(self):
        msgs = [
            {"role": "user", "content": "Do a search."},
            {"role": "assistant", "content": "Calling tool.", "reasoning_content": "R"},
            {"role": "tool", "content": "Result."},
        ]
        tokens = self._gen_prompt_tokens(msgs)
        self.assertEqual(tokens[-2], self.think_id)
        self.assertEqual(tokens[-1], self.newline_id)


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestLabelStartToken(unittest.TestCase):
    """#2/#3: Verify which token `start = len(prefix_tokens) - 1` lands on."""

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")
        cls.newline_id = cls.tok.encode("\n", add_bos=False, add_eos=False)[0]

    def _full_and_prefix(self, messages, turn_idx):
        """Get full tokens and prefix tokens for a turn."""
        kwargs = {"truncate_history_thinking": True}
        full_text = self.tok.apply_chat_template(messages, **kwargs).rstrip("\n")
        full_tokens = self.tok.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self.tok.eos_id:
            full_tokens.append(self.tok.eos_id)

        prefix_text = self.tok.apply_chat_template(
            messages[:turn_idx], add_generation_prompt=True, **kwargs
        )
        prefix_tokens = self.tok.encode(prefix_text, add_bos=True, add_eos=False)
        return full_tokens, prefix_tokens

    def test_reasoning_turn_start_is_newline(self):
        """#3: For a reasoning last turn, full_tokens[start] == \\n token."""
        msgs = [
            {"role": "user", "content": "Question."},
            {
                "role": "assistant",
                "content": "Answer.",
                "reasoning_content": "Thinking hard.",
            },
        ]
        full_tokens, prefix_tokens = self._full_and_prefix(msgs, 1)
        start = len(prefix_tokens) - 1
        self.assertEqual(
            full_tokens[start],
            self.newline_id,
            f"Expected \\n at start={start}, got token id {full_tokens[start]}",
        )

    def test_no_reasoning_turn_start_is_end_think(self):
        """#2: For a no-reasoning last turn, full_tokens[start] == </think>."""
        msgs = [
            {"role": "user", "content": "Question."},
            {"role": "assistant", "content": "Answer."},
        ]
        full_tokens, prefix_tokens = self._full_and_prefix(msgs, 1)
        start = len(prefix_tokens) - 1
        self.assertEqual(
            full_tokens[start],
            self.end_think_id,
            f"Expected </think> at start={start}, got token id {full_tokens[start]}",
        )

    def test_reasoning_label_target_is_first_reasoning_token(self):
        """The label at `start` predicts the first reasoning token."""
        msgs = [
            {"role": "user", "content": "Question."},
            {
                "role": "assistant",
                "content": "Answer.",
                "reasoning_content": "Alpha beta.",
            },
        ]
        full_tokens, prefix_tokens = self._full_and_prefix(msgs, 1)
        start = len(prefix_tokens) - 1
        # label at position `start` is full_tokens[start + 1]
        label_target = full_tokens[start + 1]
        # The first reasoning token: encode "Alpha beta." and take the first token
        reasoning_tokens = self.tok.encode("Alpha beta.", add_bos=False, add_eos=False)
        self.assertEqual(label_target, reasoning_tokens[0])

    def test_no_reasoning_label_target_is_first_response_token(self):
        """For no-reasoning, label at `start` predicts first response token."""
        msgs = [
            {"role": "user", "content": "Question."},
            {"role": "assistant", "content": "Gamma delta."},
        ]
        full_tokens, prefix_tokens = self._full_and_prefix(msgs, 1)
        start = len(prefix_tokens) - 1
        label_target = full_tokens[start + 1]
        response_tokens = self.tok.encode("Gamma delta.", add_bos=False, add_eos=False)
        self.assertEqual(label_target, response_tokens[0])


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestEndThinkLabelBehavior(unittest.TestCase):
    """#4: Verify </think> label behavior.

    For reasoning turns: </think> IS a label target (predicted by the \\n before it).
    For no-reasoning turns: </think> is an INPUT token at `start`; its label is the
    first response token (not </think> itself).
    """

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.end_think_id = cls.tok.token_to_id("</think>")
        from torchtitan.models.granite.tokenization_strategies import (
            TruncateLastStrategy,
        )

        cls.strategy = TruncateLastStrategy(_HF_ASSETS_PATH)

    def test_reasoning_end_think_is_predicted(self):
        """For reasoning turns, </think> appears as a label target (predicted token)."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        result = self.strategy._tokenize_one(msgs)
        targets = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        self.assertIn(self.end_think_id, targets)

    def test_no_reasoning_end_think_not_predicted(self):
        """For no-reasoning turns, </think> is NOT a label target.

        It sits at input_ids[start] with label = first response token.
        """
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = self.strategy._tokenize_one(msgs)
        targets = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        self.assertGreater(len(targets), 0, "No trained labels — label mask may be broken")
        self.assertNotIn(self.end_think_id, targets)

    def test_no_reasoning_start_input_is_end_think(self):
        """For no-reasoning, the INPUT at start position is </think>."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Answer text."},
        ]
        result = self.strategy._tokenize_one(msgs)
        # Find the transition point
        first_trained_idx = next(
            i for i, lbl in enumerate(result["labels"]) if lbl != IGNORE_INDEX
        )
        self.assertEqual(result["input_ids"][first_trained_idx], self.end_think_id)


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestInterTurnInvariance(unittest.TestCase):
    """#9: Inter-turn content (tool messages, headers) is identical under truncation True/False."""

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def test_tool_message_tokens_identical(self):
        """Inter-turn tokens (from first asst <|im_end|> through second asst <think>)
        are identical under truncation=True vs False.

        Requires a follow-up user message so truncation actually fires on asst0.
        """
        msgs = [
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": "Calling.", "reasoning_content": "Think."},
            {"role": "tool", "content": "Found X."},
            {"role": "assistant", "content": "Done.", "reasoning_content": "Wrap up."},
            {"role": "user", "content": "Follow up."},
            {"role": "assistant", "content": "Final.", "reasoning_content": "Last."},
        ]
        text_true = self.tok.apply_chat_template(msgs, truncate_history_thinking=True)
        text_false = self.tok.apply_chat_template(msgs, truncate_history_thinking=False)

        tokens_true = self.tok.encode(text_true, add_bos=True, add_eos=False)
        tokens_false = self.tok.encode(text_false, add_bos=True, add_eos=False)

        # Verify truncation actually fired
        decoded_true = self.tok.decode(tokens_true, skip_special_tokens=False)
        decoded_false = self.tok.decode(tokens_false, skip_special_tokens=False)
        self.assertNotIn("Think.", decoded_true)
        self.assertIn("Think.", decoded_false)

        # Find inter-turn region using structural markers:
        # from first <|im_end|> after first assistant through second <think> (inclusive)
        im_end_id = self.tok.token_to_id("<|im_end|>")
        think_id = self.tok.token_to_id("<think>")

        # First <|im_end|> after position 5 (past system+user) = end of first asst
        def find_nth(seq, val, n):
            count = 0
            for i, v in enumerate(seq):
                if v == val:
                    count += 1
                    if count == n:
                        return i
            return -1

        # In both: system<im_end>, user<im_end>, asst<im_end> → 3rd im_end
        end_asst0_true = find_nth(tokens_true, im_end_id, 3)
        end_asst0_false = find_nth(tokens_false, im_end_id, 3)
        self.assertGreater(end_asst0_true, 0)
        self.assertGreater(end_asst0_false, 0)

        # Find the <think> that starts the second assistant (asst1)
        # It's the second <think> in the sequence (first is asst0's)
        think_asst1_true = find_nth(tokens_true, think_id, 2)
        think_asst1_false = find_nth(tokens_false, think_id, 2)
        self.assertGreater(think_asst1_true, end_asst0_true)
        self.assertGreater(think_asst1_false, end_asst0_false)

        # The inter-turn region must be identical
        region_true = tokens_true[end_asst0_true : think_asst1_true + 1]
        region_false = tokens_false[end_asst0_false : think_asst1_false + 1]
        self.assertEqual(
            region_true,
            region_false,
            "Inter-turn tokens (tool message + headers) must be identical under truncation",
        )

    def test_no_truncation_single_user_identical(self):
        """When truncation does NOT fire (single user), both renderings are identical.

        Pure tool chain: only one user message → last_user_idx = 0 → all assistant turns
        come after it → thinking preserved in both True and False.
        """
        msgs = [
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": "Calling.", "reasoning_content": "Think."},
            {"role": "tool", "content": "Found X."},
            {"role": "assistant", "content": "Done.", "reasoning_content": "Wrap up."},
        ]
        text_true = self.tok.apply_chat_template(msgs, truncate_history_thinking=True)
        text_false = self.tok.apply_chat_template(msgs, truncate_history_thinking=False)

        tokens_true = self.tok.encode(text_true, add_bos=True, add_eos=False)
        tokens_false = self.tok.encode(text_false, add_bos=True, add_eos=False)

        # Entire token sequences should be identical (no truncation fired)
        self.assertEqual(tokens_true, tokens_false)

        # Verify thinking IS present (sanity check)
        decoded = self.tok.decode(tokens_true, skip_special_tokens=False)
        self.assertIn("Think.", decoded)
        self.assertIn("Wrap up.", decoded)

    def test_user_message_tokens_identical(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        text_true = self.tok.apply_chat_template(msgs, truncate_history_thinking=True)
        text_false = self.tok.apply_chat_template(msgs, truncate_history_thinking=False)

        tokens_true = self.tok.encode(text_true, add_bos=True, add_eos=False)
        tokens_false = self.tok.encode(text_false, add_bos=True, add_eos=False)

        # Locate "Q2" in both sequences
        q2_tokens = self.tok.encode("Q2", add_bos=False, add_eos=False)

        def find_subseq(seq, subseq):
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i : i + len(subseq)] == subseq:
                    return i
            return -1

        pos_true = find_subseq(tokens_true, q2_tokens)
        pos_false = find_subseq(tokens_false, q2_tokens)
        self.assertGreater(pos_true, 0)
        self.assertGreater(pos_false, 0)

        # From Q2 through next <think> should be identical
        think_id = self.tok.token_to_id("<think>")
        next_think_true = tokens_true.index(think_id, pos_true)
        next_think_false = tokens_false.index(think_id, pos_false)

        region_true = tokens_true[pos_true : next_think_true + 1]
        region_false = tokens_false[pos_false : next_think_false + 1]
        self.assertEqual(region_true, region_false)


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestPartialRenderingIsPrefix(unittest.TestCase):
    """#10: Partial rendering prefix guarantees.

    Two properties that insertion_limit and suffix extraction depend on:
    1. backbone_tokens[: len(prefix) - 1] == prefix[:-1] — always true (up to <think>)
    2. suffix_source_tokens[: len(prefix)] == prefix — always true (thinking preserved)

    The FULL prefix match (backbone[: len(prefix)] == prefix) only holds when the
    next turn has reasoning. For no-reasoning turns, backbone has <think></think>
    where prefix has <think>\\n — but insertion_limit uses len(prefix)-2 which is
    the <think> position, so this doesn't matter.
    """

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")

    def _backbone_tokens(self, messages):
        text = self.tok.apply_chat_template(
            messages, truncate_history_thinking=True
        ).rstrip("\n")
        tokens = self.tok.encode(text, add_bos=True, add_eos=False)
        if tokens[-1] != self.tok.eos_id:
            tokens.append(self.tok.eos_id)
        return tokens

    def _prefix_tokens(self, messages):
        text = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            truncate_history_thinking=True,
        )
        return self.tok.encode(text, add_bos=True, add_eos=False)

    def _suffix_source_tokens(self, messages):
        """Tokens for suffix source: thinking preserved (no later user in slice)."""
        text = self.tok.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        return self.tok.encode(text, add_bos=True, add_eos=False)

    def test_backbone_prefix_up_to_think_always_matches(self):
        """backbone[: len(prefix) - 1] == prefix[:-1] for all turn types.

        The <think> at len(prefix)-2 always matches. Only the last byte (\n vs </think>)
        may differ for no-reasoning turns.
        """
        msgs_reasoning = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        msgs_no_reasoning = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        for msgs in [msgs_reasoning, msgs_no_reasoning]:
            backbone = self._backbone_tokens(msgs)
            prefix = self._prefix_tokens(msgs[:1])
            # Up to and including <think> (len-2) always matches
            self.assertEqual(
                backbone[: len(prefix) - 1],
                prefix[:-1],
                f"Prefix[:-1] must match backbone start",
            )
            # <think> specifically at len(prefix)-2
            self.assertEqual(backbone[len(prefix) - 2], self.think_id)

    def test_reasoning_turn_full_prefix_matches_backbone(self):
        """When next turn has reasoning, full prefix matches backbone."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        backbone = self._backbone_tokens(msgs)
        prefix = self._prefix_tokens(msgs[:1])
        self.assertEqual(backbone[: len(prefix)], prefix)

    def test_no_reasoning_turn_prefix_diverges_at_last_byte(self):
        """When next turn has no reasoning, prefix[-1] != backbone[len(prefix)-1].

        Prefix ends with \\n (from generation prompt <think>\\n).
        Backbone has </think> at that position (no-reasoning form: <think></think>).
        """
        end_think_id = self.tok.token_to_id("</think>")
        newline_id = self.tok.encode("\n", add_bos=False, add_eos=False)[0]

        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        backbone = self._backbone_tokens(msgs)
        prefix = self._prefix_tokens(msgs[:1])

        self.assertEqual(prefix[-1], newline_id)
        self.assertEqual(backbone[len(prefix) - 1], end_think_id)

    def test_suffix_source_full_prefix_always_matches(self):
        """prefix IS always a prefix of suffix_source (thinking preserved in both).

        This is the guarantee suffix extraction depends on.
        """
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        # Suffix_0 covers asst0 (between user0 and user1)
        # Prefix: messages[:1] with gen prompt
        prefix = self._prefix_tokens(msgs[:1])
        # Suffix source: messages[:2] (up to but not including user1)
        suffix_source = self._suffix_source_tokens(msgs[:2])
        self.assertEqual(
            suffix_source[: len(prefix)],
            prefix,
            "Prefix must match start of suffix_source (thinking preserved)",
        )

    def test_suffix_source_prefix_matches_tool_chain(self):
        """Prefix matches suffix_source for tool chain suffix groups."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "tool", "content": "T1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        # Suffix_0 covers [asst1, tool1, asst2] between user0 and user1
        prefix = self._prefix_tokens(msgs[:1])
        suffix_source = self._suffix_source_tokens(msgs[:4])
        self.assertEqual(suffix_source[: len(prefix)], prefix)

    def test_suffix_source_no_reasoning_first_turn(self):
        """Suffix source prefix match when first turn has NO reasoning.

        The suffix group has [asst_no_rc, tool, asst_with_rc]. The prefix
        (gen prompt ending <think>\\n) does NOT match suffix_source at the
        last byte (suffix_source has <think></think>). But insertion_limit
        (len(prefix)-2) still correctly points to <think>.
        """
        end_think_id = self.tok.token_to_id("</think>")
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},  # no reasoning
            {"role": "tool", "content": "T1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        prefix = self._prefix_tokens(msgs[:1])
        suffix_source = self._suffix_source_tokens(msgs[:4])

        # Up to <think> (len-2) matches
        self.assertEqual(suffix_source[: len(prefix) - 1], prefix[:-1])
        # <think> at insertion_limit position
        insertion_limit = len(prefix) - 2
        self.assertEqual(suffix_source[insertion_limit], self.think_id)
        # Last byte diverges: prefix has \n, suffix_source has </think>
        # (because first turn has no reasoning → template renders <think></think>)
        self.assertEqual(suffix_source[len(prefix) - 1], end_think_id)

    def test_suffix_source_reasoning_first_turn_full_match(self):
        """When first turn HAS reasoning, full prefix matches suffix_source."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "tool", "content": "T1"},
            {"role": "assistant", "content": "A2"},  # no reasoning in later turn
            {"role": "user", "content": "Follow"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        prefix = self._prefix_tokens(msgs[:1])
        suffix_source = self._suffix_source_tokens(msgs[:4])
        # Full prefix matches because first turn has reasoning → <think>\n in both
        self.assertEqual(suffix_source[: len(prefix)], prefix)

    def test_multi_turn_insertion_limit_into_backbone(self):
        """insertion_limit = len(prefix)-2 always points to <think> in backbone."""
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        backbone = self._backbone_tokens(msgs)
        for turn_idx in [2, 4, 6]:
            prefix = self._prefix_tokens(msgs[:turn_idx])
            insertion_limit = len(prefix) - 2
            self.assertEqual(
                backbone[insertion_limit],
                self.think_id,
                f"insertion_limit for turn {turn_idx} must point to <think> in backbone",
            )


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestSliceDependentThinkingPreservation(unittest.TestCase):
    """#11: Same `truncate_history_thinking=True` produces different results based on slice.

    Within a suffix group (no intervening user messages), all assistant turns come
    AFTER the last user in the slice → thinking preserved. This is the foundation
    of coordinate consistency.
    """

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")

    def _encode(self, messages):
        text = self.tok.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        return self.tok.encode(text, add_bos=True, add_eos=False)

    def test_thinking_preserved_in_shorter_slice(self):
        """Render messages[:5] (before user1): asst0 and asst1 thinking preserved."""
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "Think0"},
            {"role": "tool", "content": "Tool0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Think1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Think2"},
        ]
        # Render up to asst1 (exclusive of user1)
        tokens_short = self._encode(msgs[:5])
        decoded_short = self.tok.decode(tokens_short, skip_special_tokens=False)
        # Both thinking traces preserved (no user after them in the slice)
        self.assertIn("Think0", decoded_short)
        self.assertIn("Think1", decoded_short)

    def test_thinking_stripped_in_full_render(self):
        """Render all messages: asst0 and asst1 thinking stripped (user1 is after them)."""
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "Think0"},
            {"role": "tool", "content": "Tool0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Think1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Think2"},
        ]
        tokens_full = self._encode(msgs)
        decoded_full = self.tok.decode(tokens_full, skip_special_tokens=False)
        # Think0 and Think1 are stripped (user1 at index 5 is after them)
        self.assertNotIn("Think0", decoded_full)
        self.assertNotIn("Think1", decoded_full)
        # Think2 preserved (last assistant)
        self.assertIn("Think2", decoded_full)

    def test_tool_chain_no_user_all_preserved(self):
        """Pure tool chain (single user): all thinking preserved regardless of slice."""
        msgs = [
            {"role": "user", "content": "Do work."},
            {"role": "assistant", "content": "Step1.", "reasoning_content": "R1"},
            {"role": "tool", "content": "T1"},
            {"role": "assistant", "content": "Step2.", "reasoning_content": "R2"},
            {"role": "tool", "content": "T2"},
            {"role": "assistant", "content": "Done.", "reasoning_content": "R3"},
        ]
        tokens = self._encode(msgs)
        decoded = self.tok.decode(tokens, skip_special_tokens=False)
        self.assertIn("R1", decoded)
        self.assertIn("R2", decoded)
        self.assertIn("R3", decoded)

    def test_slice_boundary_produces_different_tokens(self):
        """Same messages, different slice → same kwargs produce different token sequences."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "Reasoning0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Reasoning1"},
        ]
        # Full render: Reasoning0 stripped, Reasoning1 preserved
        full = self._encode(msgs)
        # Slice [:2]: only user0+asst0, Reasoning0 preserved
        short = self._encode(msgs[:2])

        # The short slice has "Reasoning0" in it
        decoded_short = self.tok.decode(short, skip_special_tokens=False)
        self.assertIn("Reasoning0", decoded_short)

        # The full render does NOT have "Reasoning0"
        decoded_full = self.tok.decode(full, skip_special_tokens=False)
        self.assertNotIn("Reasoning0", decoded_full)


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestCoordinateSystemConsistency(unittest.TestCase):
    """#12: insertion_limit computed from shorter slice indexes correctly into longer slice.

    Both use truncate_history_thinking=True. Since partial renderings are prefixes
    of full renderings (same kwargs), insertion_limit computed from messages[:k]
    correctly indexes into tokens from messages[:m] where m > k.
    """

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.newline_id = cls.tok.encode("\n", add_bos=False, add_eos=False)[0]

    def _prefix_tokens(self, messages):
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, truncate_history_thinking=True
        )
        return self.tok.encode(text, add_bos=True, add_eos=False)

    def _source_tokens(self, messages):
        """Tokens for suffix source (rendered with True, no generation prompt)."""
        text = self.tok.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        return self.tok.encode(text, add_bos=True, add_eos=False)

    def test_insertion_limit_indexes_into_suffix_source(self):
        """insertion_limit from messages[:2] correctly indexes into messages[:5] tokens."""
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "Think0"},
            {"role": "tool", "content": "Tool0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Think1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Think2"},
        ]
        # insertion_limit for suffix_0 (asst0): render messages[:2] with gen prompt
        prefix = self._prefix_tokens(msgs[:2])
        insertion_limit = len(prefix) - 2  # position of <think>

        # Suffix source: messages[:5] (everything before user1)
        suffix_source = self._source_tokens(msgs[:5])

        # insertion_limit should point to <think> in the suffix source too
        self.assertEqual(
            suffix_source[insertion_limit],
            self.think_id,
            f"insertion_limit={insertion_limit} should point to <think> in suffix source, "
            f"got token id {suffix_source[insertion_limit]}",
        )

    def test_label_start_indexes_into_suffix(self):
        """label_start for asst1 from messages[:4] indexes correctly into suffix."""
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "Think0"},
            {"role": "tool", "content": "Tool0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Think1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Think2"},
        ]
        # insertion_limit for suffix_0 (first asst)
        prefix_first = self._prefix_tokens(msgs[:2])
        insertion_limit = len(prefix_first) - 2

        # label_start for asst1 (turn at index 4)
        prefix_second = self._prefix_tokens(msgs[:4])
        label_start_global = len(prefix_second) - 1

        # suffix_source covers messages[:5]
        suffix_source = self._source_tokens(msgs[:5])

        # Convert to suffix-local coordinate
        suffix_start_global = insertion_limit + 1
        label_start_in_suffix = label_start_global - suffix_start_global

        # Verify it lands on the expected token (\n after <think> for reasoning turn)
        suffix_tokens = suffix_source[suffix_start_global:]
        self.assertGreater(
            label_start_in_suffix,
            0,
            "label_start_in_suffix must be > 0",
        )
        self.assertLess(
            label_start_in_suffix,
            len(suffix_tokens),
            "label_start_in_suffix must be within suffix bounds",
        )
        # For reasoning turn: token at label_start should be \n (after <think>)
        self.assertEqual(
            suffix_tokens[label_start_in_suffix],
            self.newline_id,
            f"Expected \\n at suffix-local label_start={label_start_in_suffix}, "
            f"got token id {suffix_tokens[label_start_in_suffix]}",
        )

    def test_multi_suffix_coordinate_consistency(self):
        """Multiple suffix groups: each insertion_limit indexes correctly."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        # Full render (backbone)
        full_text = self.tok.apply_chat_template(
            msgs, truncate_history_thinking=True
        ).rstrip("\n")
        backbone_tokens = self.tok.encode(full_text, add_bos=True, add_eos=False)
        if backbone_tokens[-1] != self.tok.eos_id:
            backbone_tokens.append(self.tok.eos_id)

        # Suffix_0: covers asst0 (between user0 and user1)
        # insertion_limit from messages[:1] gen prompt
        prefix_0 = self._prefix_tokens(msgs[:1])
        insertion_limit_0 = len(prefix_0) - 2
        self.assertEqual(
            backbone_tokens[insertion_limit_0],
            self.think_id,
            "insertion_limit_0 must point to <think> in backbone",
        )

        # Suffix_1: covers asst1 (between user1 and user2)
        # insertion_limit from messages[:3] gen prompt
        prefix_1 = self._prefix_tokens(msgs[:3])
        insertion_limit_1 = len(prefix_1) - 2
        self.assertEqual(
            backbone_tokens[insertion_limit_1],
            self.think_id,
            "insertion_limit_1 must point to <think> in backbone",
        )

        # Verify both limits are different and in ascending order
        self.assertLess(insertion_limit_0, insertion_limit_1)

    def test_label_start_no_reasoning_turn_in_suffix(self):
        """label_start for a no-reasoning turn lands on </think> (suffix-local)."""
        end_think_id = self.tok.token_to_id("</think>")
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},  # no reasoning
            {"role": "tool", "content": "Tool0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Think1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Think2"},
        ]
        # insertion_limit for suffix_0 (first asst at index 2)
        prefix_first = self._prefix_tokens(msgs[:2])
        insertion_limit = len(prefix_first) - 2

        # label_start for asst0 (turn at index 2 — the no-reasoning one)
        # Since it's the first turn in the suffix, label_start = insertion_limit + 1
        # which is the position right after <think> in the suffix source.
        # For no-reasoning: suffix_source has <think></think>A0... so the token at
        # insertion_limit+1 is </think>.
        suffix_source = self._source_tokens(msgs[:5])
        suffix_start_global = insertion_limit + 1
        suffix_tokens = suffix_source[suffix_start_global:]

        # label_start_in_suffix = 0 for the first turn in the suffix
        # (gen prompt for messages[:2] gives insertion_limit; label_start from the
        # same prefix is len(prefix_first)-1 = insertion_limit+1 = suffix_start_global)
        label_start_global = len(prefix_first) - 1
        label_start_in_suffix = label_start_global - suffix_start_global
        self.assertEqual(label_start_in_suffix, 0)

        # The token at suffix[0] should be </think> (no reasoning → <think></think>...)
        self.assertEqual(
            suffix_tokens[0],
            end_think_id,
            f"Expected </think> at suffix start for no-reasoning turn, "
            f"got token id {suffix_tokens[0]}",
        )

    def test_prefix_is_prefix_of_suffix_source(self):
        """messages[:first_asst_idx] prefix is a prefix of messages[:next_user_idx] tokens."""
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "Think0"},
            {"role": "tool", "content": "Tool0"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Think1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Think2"},
        ]
        # Prefix for insertion_limit: messages[:2] with gen prompt
        prefix = self._prefix_tokens(msgs[:2])

        # Suffix source: messages[:5] (up to but not including user1)
        suffix_source = self._source_tokens(msgs[:5])

        # The prefix (without gen prompt's final tokens) must be a prefix of suffix source
        # Actually the full prefix (including <think>\n) should match the suffix source start
        self.assertEqual(
            suffix_source[: len(prefix)],
            prefix,
            "Prefix tokens must match start of suffix source tokens",
        )


@unittest.skipUnless(
    _HF_ASSETS_PATH,
    "HF_ASSETS_PATH not set — skipping Granite tokenizer tests",
)
class TestTruncateLastStrategyConsistency(unittest.TestCase):
    """Verify TruncateLastStrategy uses the same rendering mechanism as BackboneSuffix will.

    These tests validate that TruncateLastStrategy's backbone (input_ids, labels for
    last turn) is derivable via the same two-rendering method BackboneSuffix will use.
    """

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        from torchtitan.models.granite.tokenization_strategies import (
            TruncateLastStrategy,
        )

        cls.strategy = TruncateLastStrategy(_HF_ASSETS_PATH)

    def test_backbone_tokens_match_full_rendering(self):
        """input_ids = full_tokens[:-1] from apply_chat_template with truncate=True."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self.strategy._tokenize_one(msgs)

        # Reproduce the rendering
        full_text = self.tok.apply_chat_template(
            msgs, truncate_history_thinking=True
        ).rstrip("\n")
        full_tokens = self.tok.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self.tok.eos_id:
            full_tokens.append(self.tok.eos_id)

        self.assertEqual(result["input_ids"], full_tokens[:-1])

    def test_label_boundary_from_prefix_len(self):
        """start = len(prefix_tokens) - 1 yields correct label boundary."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self.strategy._tokenize_one(msgs)

        # Compute start independently
        prefix_text = self.tok.apply_chat_template(
            msgs[:3],  # everything before last assistant
            add_generation_prompt=True,
            truncate_history_thinking=True,
        )
        prefix_tokens = self.tok.encode(prefix_text, add_bos=True, add_eos=False)
        start = len(prefix_tokens) - 1

        # All labels before start should be IGNORE_INDEX
        self.assertTrue(all(lbl == IGNORE_INDEX for lbl in result["labels"][:start]))
        # Label at start should NOT be IGNORE_INDEX
        self.assertNotEqual(result["labels"][start], IGNORE_INDEX)

    def test_tool_chain_backbone_identical_with_and_without_suffix(self):
        """For tool chain with single user: TruncateLastStrategy produces same backbone
        that BackboneSuffix would (since no suffixes are created — no thinking stripped)."""
        msgs = [
            {"role": "user", "content": "Do search."},
            {"role": "assistant", "content": "Calling.", "reasoning_content": "R1"},
            {"role": "tool", "content": "Found."},
            {"role": "assistant", "content": "Answer.", "reasoning_content": "R2"},
        ]
        result = self.strategy._tokenize_one(msgs)

        # Since there's only one user message, no thinking is stripped.
        # Verify the full sequence contains all reasoning.
        decoded = self.tok.decode(result["input_ids"], skip_special_tokens=False)
        self.assertIn("R1", decoded)
        self.assertIn("R2", decoded)


if __name__ == "__main__":
    unittest.main()
