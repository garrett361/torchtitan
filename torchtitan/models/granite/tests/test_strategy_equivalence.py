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
from torchtitan.models.granite.tests.conversations import (
    MANY_TURNS_MIXED,
    MIXED_REASONING_NO_TRAILING_USER,
    MULTI_STEP_AGENT_NO_TRAILING_USER,
    MULTI_TURN_ALL_REASONING,
    MULTI_TURN_MIXED_REASONING,
    MULTI_TURN_NO_REASONING,
    NO_REASONING_TOOL_CHAIN_NO_TRAILING_USER,
    SINGLE_TURN_NO_REASONING,
    SINGLE_TURN_REASONING,
    TOOL_CHAIN_NO_REASONING,
    TOOL_CHAIN_NO_TRAILING_USER,
    TOOL_CHAIN_REASONING,
    WITH_SYSTEM_MESSAGE,
)

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
        self._assert_same_trained_tokens(SINGLE_TURN_REASONING, "single turn with reasoning")

    def test_single_turn_no_reasoning(self):
        self._assert_same_trained_tokens(SINGLE_TURN_NO_REASONING, "single turn no reasoning")

    def test_multi_turn_all_reasoning(self):
        self._assert_same_trained_tokens(MULTI_TURN_ALL_REASONING, "multi-turn all reasoning")

    def test_multi_turn_no_reasoning(self):
        self._assert_same_trained_tokens(MULTI_TURN_NO_REASONING, "multi-turn no reasoning")

    def test_multi_turn_mixed_reasoning(self):
        self._assert_same_trained_tokens(MULTI_TURN_MIXED_REASONING, "multi-turn mixed reasoning")

    def test_tool_chain_with_reasoning(self):
        self._assert_same_trained_tokens(TOOL_CHAIN_REASONING, "tool chain with reasoning")

    def test_tool_chain_no_reasoning(self):
        self._assert_same_trained_tokens(TOOL_CHAIN_NO_REASONING, "tool chain no reasoning")

    def test_many_turns(self):
        self._assert_same_trained_tokens(MANY_TURNS_MIXED, "many turns mixed")

    def test_system_message(self):
        self._assert_same_trained_tokens(WITH_SYSTEM_MESSAGE, "system message")

    def test_tool_chain_no_trailing_user(self):
        self._assert_same_trained_tokens(TOOL_CHAIN_NO_TRAILING_USER, "tool chain no trailing user")

    def test_multi_step_agent_no_trailing_user(self):
        self._assert_same_trained_tokens(MULTI_STEP_AGENT_NO_TRAILING_USER, "multi-step agent no trailing user")

    def test_single_user_no_reasoning_tool_chain(self):
        self._assert_same_trained_tokens(NO_REASONING_TOOL_CHAIN_NO_TRAILING_USER, "no-reasoning tool chain no trailing user")

    def test_mixed_reasoning_no_trailing_user(self):
        self._assert_same_trained_tokens(MIXED_REASONING_NO_TRAILING_USER, "mixed reasoning no trailing user")


if __name__ == "__main__":
    unittest.main()
