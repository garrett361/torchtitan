"""Cross-strategy trained-token equivalence tests.

Verifies that BackboneSuffix, TruncateEveryTurn, and FullThinking produce
equivalent aggregate trained tokens for any conversation, differing only by
the well-understood +2 per no-reasoning turn for FullThinking.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template.
"""

import os
import unittest

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")

try:
    from torchtitan.models.granite.tokenization_strategies import (
        BackboneSuffixStrategy,
        FullThinkingStrategy,
        TruncateEveryTurnStrategy,
    )

    _STRATEGY_AVAILABLE = True
except ImportError:
    _STRATEGY_AVAILABLE = False


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
@unittest.skipUnless(_STRATEGY_AVAILABLE, "Strategy imports not available")
class TestStrategyTrainedTokenEquivalence(unittest.TestCase):
    """All strategies produce equivalent aggregate trained tokens per conversation.

    BackboneSuffix and TruncateEveryTurn (both truncate_history_thinking=True) must
    match exactly.

    FullThinking (truncate_history_thinking=False) differs by +2 tokens per
    no-reasoning assistant turn. At inference with truncate_history_thinking=True,
    no-reasoning turns appear as <think></think>{content} — the model's label span
    starts at the first content token. At inference with truncate_history_thinking=False,
    no-reasoning turns appear as they were generated: the model received <think>\\n
    from the generation prompt and produced </think>\\n{content}. The label span
    starts after <think>\\n, so FullThinking labels the </think> and \\n tokens that
    the model actually generated — two extra trained tokens per no-reasoning turn.
    """

    @classmethod
    def setUpClass(cls):
        cls.backbone_suffix = BackboneSuffixStrategy(_HF_ASSETS_PATH)
        cls.truncate_every = TruncateEveryTurnStrategy(_HF_ASSETS_PATH)
        cls.full_thinking = FullThinkingStrategy(_HF_ASSETS_PATH)

    def _trained_count_backbone_suffix(self, msgs):
        result = self.backbone_suffix._tokenize_one(msgs)
        return sum(1 for lbl in result["labels"] if lbl != IGNORE_INDEX)

    def _trained_count_truncate_every(self, msgs):
        batch = self.truncate_every({"messages": [msgs]})
        return sum(
            sum(1 for lbl in labels if lbl != IGNORE_INDEX)
            for labels in batch["labels"]
        )

    def _trained_count_full_thinking(self, msgs):
        result = self.full_thinking._tokenize_one(msgs)
        return sum(1 for lbl in result["labels"] if lbl != IGNORE_INDEX)

    def _no_reasoning_turn_count(self, msgs):
        return sum(
            1 for m in msgs
            if m["role"] == "assistant" and not m.get("reasoning_content", "").strip()
        )

    def _assert_same_trained_tokens(self, msgs, desc):
        bs_count = self._trained_count_backbone_suffix(msgs)
        te_count = self._trained_count_truncate_every(msgs)
        ft_count = self._trained_count_full_thinking(msgs)
        self.assertEqual(
            bs_count, te_count,
            f"{desc}: backbone_suffix={bs_count}, truncate_every_turn={te_count}",
        )
        expected_ft_extra = 2 * self._no_reasoning_turn_count(msgs)
        self.assertEqual(
            ft_count - bs_count, expected_ft_extra,
            f"{desc}: full_thinking={ft_count}, backbone_suffix={bs_count}, "
            f"expected +{expected_ft_extra} (2 × {expected_ft_extra // 2} no-reasoning turns)",
        )

    def test_single_turn_with_reasoning(self):
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Simple addition."},
        ]
        self._assert_same_trained_tokens(msgs, "single turn with reasoning")

    def test_single_turn_no_reasoning(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        self._assert_same_trained_tokens(msgs, "single turn no reasoning")

    def test_multi_turn_all_reasoning(self):
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        self._assert_same_trained_tokens(msgs, "multi-turn all reasoning")

    def test_multi_turn_no_reasoning(self):
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        self._assert_same_trained_tokens(msgs, "multi-turn no reasoning")

    def test_multi_turn_mixed_reasoning(self):
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        self._assert_same_trained_tokens(msgs, "multi-turn mixed reasoning")

    def test_tool_chain_with_reasoning(self):
        msgs = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "call_tool", "reasoning_content": "Need to search"},
            {"role": "tool", "content": "result of search"},
            {"role": "assistant", "content": "Found it", "reasoning_content": "Got result"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Welcome", "reasoning_content": "Done"},
        ]
        self._assert_same_trained_tokens(msgs, "tool chain with reasoning")

    def test_tool_chain_no_reasoning(self):
        msgs = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "call_tool"},
            {"role": "tool", "content": "result of search"},
            {"role": "assistant", "content": "Found it"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Welcome", "reasoning_content": "Done"},
        ]
        self._assert_same_trained_tokens(msgs, "tool chain no reasoning")

    def test_many_turns(self):
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Q4"},
            {"role": "assistant", "content": "A4", "reasoning_content": "R4"},
        ]
        self._assert_same_trained_tokens(msgs, "many turns mixed")

    def test_system_message(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        self._assert_same_trained_tokens(msgs, "system message")

    def test_tool_chain_no_trailing_user(self):
        """Unclosed group: no user message after the tool chain."""
        msgs = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "call_tool", "reasoning_content": "Need search"},
            {"role": "tool", "content": "result of search"},
            {"role": "assistant", "content": "Here's what I found", "reasoning_content": "Got it"},
        ]
        self._assert_same_trained_tokens(msgs, "tool chain no trailing user")

    def test_multi_step_agent_no_trailing_user(self):
        """Multiple tool rounds with no trailing user message."""
        msgs = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "read file", "reasoning_content": "R0"},
            {"role": "tool", "content": "file content"},
            {"role": "assistant", "content": "edit file", "reasoning_content": "R1"},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Done", "reasoning_content": "R2"},
        ]
        self._assert_same_trained_tokens(msgs, "multi-step agent no trailing user")

    def test_single_user_no_reasoning_tool_chain(self):
        """No-reasoning tool chain with no trailing user."""
        msgs = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "calc(2+2)"},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "The answer is 4"},
        ]
        self._assert_same_trained_tokens(msgs, "no-reasoning tool chain no trailing user")

    def test_mixed_reasoning_no_trailing_user(self):
        """Some turns have reasoning, some don't, no trailing user."""
        msgs = [
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "step1"},
            {"role": "tool", "content": "result1"},
            {"role": "assistant", "content": "step2", "reasoning_content": "thinking"},
            {"role": "tool", "content": "result2"},
            {"role": "assistant", "content": "final answer", "reasoning_content": "done"},
        ]
        self._assert_same_trained_tokens(msgs, "mixed reasoning no trailing user")


if __name__ == "__main__":
    unittest.main()
