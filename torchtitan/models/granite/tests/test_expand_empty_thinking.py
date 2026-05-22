"""Unit tests for _expand_empty_thinking_tokens.

Verifies token-level expansion of no-reasoning turns (<think></think> →
<think>\\n</think>\\n) with position-awareness to avoid expanding content-level
occurrences of the same pattern.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")

try:
    from torchtitan.models.granite.tokenization_strategies import (
        _expand_empty_thinking_tokens,
    )

    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

pytestmark = [
    pytest.mark.skipif(not _HF_ASSETS_PATH, reason="HF_ASSETS_PATH not set"),
    pytest.mark.skipif(not _IMPORT_OK, reason="Strategy imports not available"),
]


@pytest.fixture(scope="module")
def tokenizer():
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    return HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)


@pytest.fixture(scope="module")
def token_ids(tokenizer):
    """Lookup commonly-used token IDs once."""
    return {
        "think": tokenizer.tokenizer.token_to_id("<think>"),
        "end_think": tokenizer.tokenizer.token_to_id("</think>"),
        "newline": tokenizer.encode("\n", add_bos=False, add_eos=False)[0],
        "im_start": tokenizer.tokenizer.token_to_id("<|im_start|>"),
        "im_end": tokenizer.tokenizer.token_to_id("<|im_end|>"),
        "bos": tokenizer.bos_id,
    }


class TestNoExpansion:
    """Cases where no expansion should occur."""

    def test_empty_sequence(self, tokenizer, token_ids):
        tokens = []
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [], tokenizer)
        assert expanded == []
        assert pos_map == []

    def test_no_think_positions(self, tokenizer, token_ids):
        tokens = [token_ids["bos"], 100, 200, 300]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [], tokenizer)
        assert expanded == tokens
        assert pos_map == [0, 1, 2, 3]

    def test_reasoning_turn_not_expanded(self, tokenizer, token_ids):
        """<think> followed by reasoning tokens (not </think>) is untouched."""
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        tokens = [token_ids["bos"], T, NL, 999, 888, NL, ET, NL, 777]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        assert expanded == tokens
        assert pos_map == list(range(len(tokens)))

    def test_content_level_think_not_expanded(self, tokenizer, token_ids):
        """<think></think> in content (not at a turn position) is ignored."""
        T, ET = token_ids["think"], token_ids["end_think"]
        tokens = [token_ids["bos"], T, token_ids["newline"], ET, token_ids["newline"],
                  500, T, ET, 600]
        # Only position 1 is a turn-level <think>; position 6 is content-level
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        assert expanded == tokens  # position 1 has reasoning (not adjacent to </think>)


class TestSingleExpansion:
    """Single no-reasoning turn expansion."""

    def test_basic_expansion(self, tokenizer, token_ids):
        """Adjacent <think></think> at a turn position expands to <think>\\n</think>\\n."""
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        tokens = [token_ids["bos"], T, ET, 777, 888]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        assert expanded == [token_ids["bos"], T, NL, ET, NL, 777, 888]
        assert pos_map[0] == 0  # bos unchanged
        assert pos_map[1] == 1  # <think> stays at 1
        assert pos_map[2] == 3  # </think> moved from 2→3
        assert pos_map[3] == 5  # content moved from 3→5
        assert pos_map[4] == 6  # content moved from 4→6

    def test_pos_map_length_matches_original(self, tokenizer, token_ids):
        T, ET = token_ids["think"], token_ids["end_think"]
        tokens = [token_ids["bos"], T, ET, 100, 200, 300]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        assert len(pos_map) == len(tokens)

    def test_expanded_length_increases_by_two(self, tokenizer, token_ids):
        T, ET = token_ids["think"], token_ids["end_think"]
        tokens = [token_ids["bos"], T, ET, 100]
        expanded, _ = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        assert len(expanded) == len(tokens) + 2

    def test_pos_map_monotonically_increasing(self, tokenizer, token_ids):
        T, ET = token_ids["think"], token_ids["end_think"]
        tokens = [token_ids["bos"], T, ET, 100, 200, 300]
        _, pos_map = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        for i in range(1, len(pos_map)):
            assert pos_map[i] > pos_map[i - 1]


class TestMultipleExpansions:
    """Multiple no-reasoning turns in one sequence."""

    def test_two_no_reasoning_turns(self, tokenizer, token_ids):
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        IE = token_ids["im_end"]
        # Two turns: both no-reasoning
        tokens = [token_ids["bos"], T, ET, 100, IE, T, ET, 200, IE]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1, 5], tokenizer)
        expected = [token_ids["bos"], T, NL, ET, NL, 100, IE, T, NL, ET, NL, 200, IE]
        assert expanded == expected
        assert len(expanded) == len(tokens) + 4
        # pos_map: tokens after first expansion shift +2, after second shift +4
        assert pos_map[0] == 0   # bos
        assert pos_map[1] == 1   # first <think>
        assert pos_map[2] == 3   # first </think> (shifted by 1 inserted \n)
        assert pos_map[3] == 5   # content 100 (shifted by 2)
        assert pos_map[4] == 6   # <|im_end|> (shifted by 2)
        assert pos_map[5] == 7   # second <think> (shifted by 2)
        assert pos_map[6] == 9   # second </think> (shifted by 3)
        assert pos_map[7] == 11  # content 200 (shifted by 4)
        assert pos_map[8] == 12  # <|im_end|> (shifted by 4)

    def test_mixed_reasoning_and_no_reasoning(self, tokenizer, token_ids):
        """First turn has reasoning (no expansion), second has none (expands)."""
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        IE = token_ids["im_end"]
        # Turn 1: <think> reasoning </think> content
        # Turn 2: <think></think> content
        tokens = [token_ids["bos"], T, NL, 999, NL, ET, NL, 100, IE, T, ET, 200, IE]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1, 9], tokenizer)
        # Only turn at position 9 should expand
        expected = [token_ids["bos"], T, NL, 999, NL, ET, NL, 100, IE, T, NL, ET, NL, 200, IE]
        assert expanded == expected
        assert len(expanded) == len(tokens) + 2

    def test_three_consecutive_no_reasoning(self, tokenizer, token_ids):
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        tokens = [token_ids["bos"], T, ET, 10, T, ET, 20, T, ET, 30]
        expanded, _ = _expand_empty_thinking_tokens(tokens, [1, 4, 7], tokenizer)
        assert len(expanded) == len(tokens) + 6  # +2 per expansion
        # Verify structure: each <think> followed by \n, each </think> followed by \n
        assert expanded == [token_ids["bos"],
                           T, NL, ET, NL, 10,
                           T, NL, ET, NL, 20,
                           T, NL, ET, NL, 30]


class TestContentContamination:
    """Content containing <think></think> must not be expanded."""

    def test_content_think_after_turn_think(self, tokenizer, token_ids):
        """Turn-level expands; identical pattern in content does not."""
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        # Turn <think></think> then content with literal <think></think>
        tokens = [token_ids["bos"], T, ET, 500, T, ET, 600]
        # Only position 1 is turn-level
        expanded, _ = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        expected = [token_ids["bos"], T, NL, ET, NL, 500, T, ET, 600]
        assert expanded == expected

    def test_content_think_in_reasoning_turn(self, tokenizer, token_ids):
        """Reasoning turn with <think></think> in content — neither expands."""
        T, ET, NL = token_ids["think"], token_ids["end_think"], token_ids["newline"]
        # Turn has reasoning: <think> \n reasoning \n </think> \n content<think></think>more
        tokens = [token_ids["bos"], T, NL, 999, NL, ET, NL, 100, T, ET, 200]
        expanded, _ = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        # Position 1: <think> followed by \n (not </think>), so no expansion
        # Position 8: not in turn_think_positions, so no expansion
        assert expanded == tokens


class TestEdgeCases:
    """Boundary and edge cases."""

    def test_think_at_end_of_sequence(self, tokenizer, token_ids):
        """<think> at sequence end with no following token — no expansion."""
        T = token_ids["think"]
        tokens = [token_ids["bos"], T]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [1], tokenizer)
        assert expanded == tokens

    def test_pos_map_identity_when_no_expansion(self, tokenizer, token_ids):
        tokens = [token_ids["bos"], 1, 2, 3, 4, 5]
        _, pos_map = _expand_empty_thinking_tokens(tokens, [], tokenizer)
        assert pos_map == list(range(len(tokens)))

    def test_single_token_sequence(self, tokenizer, token_ids):
        tokens = [token_ids["bos"]]
        expanded, pos_map = _expand_empty_thinking_tokens(tokens, [], tokenizer)
        assert expanded == [token_ids["bos"]]
        assert pos_map == [0]
