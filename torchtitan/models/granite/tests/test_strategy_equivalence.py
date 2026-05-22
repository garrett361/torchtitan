"""Cross-strategy trained-token equivalence tests.

Verifies pairwise relationships between BackboneSuffix, TruncateEveryTurn,
and FullThinking for every conversation fixture:

  BS == TE:  Both truncate_history_thinking=True strategies must match exactly.
  FT == BS + 2*K:  FullThinking exceeds BackboneSuffix by +2 per no-reasoning turn.
  FT == TE + 2*K:  Same relationship holds against TruncateEveryTurn.

Also verifies a structural property of the chat template that the offset-based
boundary code relies on: sub-renders must be exact prefixes of the full render.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template.
"""

import os

import pytest
from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.granite.tests.conversations import ALL_CONVERSATIONS

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

pytestmark = [
    pytest.mark.skipif(not _HF_ASSETS_PATH, reason="HF_ASSETS_PATH not set"),
    pytest.mark.skipif(not _STRATEGY_AVAILABLE, reason="Strategy imports not available"),
]


@pytest.fixture(scope="module")
def backbone_suffix():
    return BackboneSuffixStrategy(_HF_ASSETS_PATH)


@pytest.fixture(scope="module")
def truncate_every():
    return TruncateEveryTurnStrategy(_HF_ASSETS_PATH)


@pytest.fixture(scope="module")
def full_thinking():
    return FullThinkingStrategy(_HF_ASSETS_PATH)


def _trained_count(strategy, msgs):
    result = strategy._tokenize_one(msgs)
    return sum(1 for lbl in result["labels"] if lbl != IGNORE_INDEX)


def _trained_count_te(strategy, msgs):
    batch = strategy({"messages": [msgs]})
    return sum(
        sum(1 for lbl in labels if lbl != IGNORE_INDEX)
        for labels in batch["labels"]
    )


def _no_reasoning_turn_count(msgs):
    return sum(
        1 for m in msgs
        if m["role"] == "assistant" and not m.get("reasoning_content", "").strip()
    )


@pytest.mark.parametrize("name,msgs", ALL_CONVERSATIONS, ids=[c[0] for c in ALL_CONVERSATIONS])
class TestStrategyPairwise:
    """Each pairwise invariant tested independently per conversation."""

    def test_bs_equals_te(self, name, msgs, backbone_suffix, truncate_every):
        bs = _trained_count(backbone_suffix, msgs)
        te = _trained_count_te(truncate_every, msgs)
        assert bs == te, (
            f"backbone_suffix={bs}, truncate_every_turn={te}, diff={bs - te}"
        )

    def test_ft_minus_bs(self, name, msgs, backbone_suffix, full_thinking):
        bs = _trained_count(backbone_suffix, msgs)
        ft = _trained_count(full_thinking, msgs)
        expected_extra = 2 * _no_reasoning_turn_count(msgs)
        assert ft - bs == expected_extra, (
            f"full_thinking={ft}, backbone_suffix={bs}, "
            f"actual_diff={ft - bs}, expected_diff={expected_extra} "
            f"(2 × {expected_extra // 2} no-reasoning turns)"
        )

    def test_ft_minus_te(self, name, msgs, truncate_every, full_thinking):
        te = _trained_count_te(truncate_every, msgs)
        ft = _trained_count(full_thinking, msgs)
        expected_extra = 2 * _no_reasoning_turn_count(msgs)
        assert ft - te == expected_extra, (
            f"full_thinking={ft}, truncate_every_turn={te}, "
            f"actual_diff={ft - te}, expected_diff={expected_extra} "
            f"(2 × {expected_extra // 2} no-reasoning turns)"
        )


@pytest.mark.parametrize("name,msgs", ALL_CONVERSATIONS, ids=[c[0] for c in ALL_CONVERSATIONS])
class TestChatTemplateStructure:
    """The offset-based code assumes sub-renders are prefixes of the full render.

    For truncate_history_thinking=True: apply_chat_template(msgs[:k]) should
    produce text that, after rstrip('\\n'), equals full_text[:len(sub_render)]
    for every split point k.
    """

    def test_subrender_prefix_equivalence(self, name, msgs, backbone_suffix):
        tok = backbone_suffix.tokenizer
        kwargs = backbone_suffix.chat_template_kwargs

        asst_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if not asst_indices:
            pytest.skip("No assistant turns")
        effective = msgs[:asst_indices[-1] + 1]

        full_text = tok.apply_chat_template(effective, **kwargs).rstrip("\n")

        for k in range(1, len(effective)):
            sub_render = tok.apply_chat_template(
                effective[:k], **kwargs
            ).rstrip("\n")
            prefix = full_text[:len(sub_render)]
            assert sub_render == prefix, (
                f"Sub-render diverges at k={k} (msg role={effective[k-1]['role']})\n"
                f"  sub_render[-40:] = {sub_render[-40:]!r}\n"
                f"  full_text[...] =   {prefix[-40:]!r}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
