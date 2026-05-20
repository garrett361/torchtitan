"""Tests for _tokenize_all_turns helper.

Verifies that _tokenize_all_turns correctly labels every assistant turn in a
multi-turn conversation, producing the right output structure, label regions,
content, and boundary positions.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template.
"""

import os
import unittest

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.tokenization_strategies import (
    TruncateLastStrategy,
    _tokenize_all_turns,
)

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")

_CHAT_TEMPLATE_KWARGS: dict = {"truncate_history_thinking": True}


def _count_labeled_regions(labels: list[int]) -> int:
    """Count contiguous non-IGNORE_INDEX regions (IGNORE→non-IGNORE transitions)."""
    count = 0
    in_region = False
    for lbl in labels:
        if lbl != IGNORE_INDEX:
            if not in_region:
                count += 1
                in_region = True
        else:
            in_region = False
    return count


def _get_labeled_spans(labels: list[int]) -> list[tuple[int, int]]:
    """Return (start, end) for each contiguous non-IGNORE_INDEX region."""
    spans = []
    start = None
    for i, lbl in enumerate(labels):
        if lbl != IGNORE_INDEX:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def _num_assistant_turns(messages: list[dict]) -> int:
    """Count assistant turns in effective messages (up to last assistant)."""
    last_asst_idx = max(
        i for i, m in enumerate(messages) if m["role"] == "assistant"
    )
    return sum(
        1 for m in messages[: last_asst_idx + 1] if m["role"] == "assistant"
    )



SINGLE_TURN_REASONING = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4", "reasoning_content": "Simple addition."},
]

SINGLE_TURN_NO_REASONING = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"},
]

MULTI_TURN_ALL_REASONING = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
]

MULTI_TURN_NO_REASONING = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
]

MULTI_TURN_MIXED_REASONING = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
]

TOOL_CHAIN_REASONING = [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "call_tool", "reasoning_content": "Need to search"},
    {"role": "tool", "content": "result of search"},
    {"role": "assistant", "content": "Found it", "reasoning_content": "Got result"},
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "Welcome", "reasoning_content": "Done"},
]

TOOL_CHAIN_NO_REASONING = [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "call_tool"},
    {"role": "tool", "content": "result of search"},
    {"role": "assistant", "content": "Found it"},
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "Welcome", "reasoning_content": "Done"},
]

MANY_TURNS_MIXED = [
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

WITH_SYSTEM_MESSAGE = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
]

TOOL_CHAIN_NO_TRAILING_USER = [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "call_tool", "reasoning_content": "Need search"},
    {"role": "tool", "content": "result of search"},
    {"role": "assistant", "content": "Here's what I found", "reasoning_content": "Got it"},
]

MULTI_STEP_AGENT_NO_TRAILING_USER = [
    {"role": "user", "content": "Fix the bug"},
    {"role": "assistant", "content": "read file", "reasoning_content": "R0"},
    {"role": "tool", "content": "file content"},
    {"role": "assistant", "content": "edit file", "reasoning_content": "R1"},
    {"role": "tool", "content": "ok"},
    {"role": "assistant", "content": "Done", "reasoning_content": "R2"},
]

NO_REASONING_TOOL_CHAIN_NO_TRAILING_USER = [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "calc(2+2)"},
    {"role": "tool", "content": "4"},
    {"role": "assistant", "content": "The answer is 4"},
]

MIXED_REASONING_NO_TRAILING_USER = [
    {"role": "user", "content": "Help me"},
    {"role": "assistant", "content": "step1"},
    {"role": "tool", "content": "result1"},
    {"role": "assistant", "content": "step2", "reasoning_content": "thinking"},
    {"role": "tool", "content": "result2"},
    {"role": "assistant", "content": "final answer", "reasoning_content": "done"},
]

ALL_CONVERSATIONS = [
    ("single_turn_reasoning", SINGLE_TURN_REASONING),
    ("single_turn_no_reasoning", SINGLE_TURN_NO_REASONING),
    ("multi_turn_all_reasoning", MULTI_TURN_ALL_REASONING),
    ("multi_turn_no_reasoning", MULTI_TURN_NO_REASONING),
    ("multi_turn_mixed_reasoning", MULTI_TURN_MIXED_REASONING),
    ("tool_chain_reasoning", TOOL_CHAIN_REASONING),
    ("tool_chain_no_reasoning", TOOL_CHAIN_NO_REASONING),
    ("many_turns_mixed", MANY_TURNS_MIXED),
    ("with_system_message", WITH_SYSTEM_MESSAGE),
    ("tool_chain_no_trailing_user", TOOL_CHAIN_NO_TRAILING_USER),
    ("multi_step_agent_no_trailing_user", MULTI_STEP_AGENT_NO_TRAILING_USER),
    ("no_reasoning_tool_chain_no_trailing_user", NO_REASONING_TOOL_CHAIN_NO_TRAILING_USER),
    ("mixed_reasoning_no_trailing_user", MIXED_REASONING_NO_TRAILING_USER),
]


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def test_output_keys(self):
        result = self._call(SINGLE_TURN_NO_REASONING)
        self.assertEqual(set(result.keys()), {"input_ids", "labels", "n_tokens"})

    def test_n_tokens_matches_input_ids_length(self):
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                self.assertEqual(result["n_tokens"], len(result["input_ids"]))

    def test_labels_same_length_as_input_ids(self):
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                self.assertEqual(len(result["labels"]), len(result["input_ids"]))

    def test_input_ids_starts_with_bos(self):
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                self.assertEqual(result["input_ids"][0], self.tok.bos_id)

    def test_last_non_ignore_label_is_eos(self):
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                non_ignore = [l for l in result["labels"] if l != IGNORE_INDEX]
                self.assertEqual(non_ignore[-1], self.tok.eos_id)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsLabelRegions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.im_end_id = cls.tok.token_to_id("<|im_end|>")

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def test_region_count_matches_assistant_turns(self):
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                expected = _num_assistant_turns(msgs)
                actual = _count_labeled_regions(result["labels"])
                self.assertEqual(actual, expected)

    def test_regions_non_overlapping_and_ordered(self):
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                spans = _get_labeled_spans(result["labels"])
                for i in range(1, len(spans)):
                    self.assertGreaterEqual(
                        spans[i][0], spans[i - 1][1],
                        f"Regions overlap or not ordered: {spans}",
                    )

    def test_labels_are_valid_shifted_tokens(self):
        """Non-IGNORE labels equal the next token in the full sequence."""
        for name, msgs in ALL_CONVERSATIONS:
            with self.subTest(name):
                result = self._call(msgs)
                full_tokens = result["input_ids"] + [self.tok.eos_id]
                for i, lbl in enumerate(result["labels"]):
                    if lbl != IGNORE_INDEX:
                        self.assertEqual(
                            lbl, full_tokens[i + 1],
                            f"Position {i}: label={lbl}, expected={full_tokens[i+1]}",
                        )


    def test_label_start_boundary_pinned(self):
        """Each assistant turn's label starts exactly at the prefix boundary."""
        multi_turn_cases = [
            ("multi_turn_no_reasoning", MULTI_TURN_NO_REASONING),
            ("multi_turn_all_reasoning", MULTI_TURN_ALL_REASONING),
            ("many_turns_mixed", MANY_TURNS_MIXED),
            ("with_system_message", WITH_SYSTEM_MESSAGE),
            ("tool_chain_no_trailing_user", TOOL_CHAIN_NO_TRAILING_USER),
        ]
        for name, msgs in multi_turn_cases:
            with self.subTest(name):
                result = self._call(msgs)
                last_asst_idx = max(
                    i for i, m in enumerate(msgs) if m["role"] == "assistant"
                )
                effective = msgs[: last_asst_idx + 1]
                for i, msg in enumerate(effective):
                    if msg["role"] != "assistant":
                        continue
                    prefix_text = self.tok.apply_chat_template(
                        effective[:i],
                        add_generation_prompt=True,
                        **_CHAT_TEMPLATE_KWARGS,
                    )
                    prefix_tokens = self.tok.encode(
                        prefix_text, add_bos=True, add_eos=False
                    )
                    start = len(prefix_tokens) - 1
                    self.assertNotEqual(
                        result["labels"][start], IGNORE_INDEX,
                        f"{name}: label at start={start} should be non-IGNORE",
                    )
                    if start > 0:
                        self.assertEqual(
                            result["labels"][start - 1], IGNORE_INDEX,
                            f"{name}: label at start-1={start-1} should be IGNORE",
                        )


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsLabelContent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.im_end_id = cls.tok.token_to_id("<|im_end|>")

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def test_decoded_labels_contain_assistant_content(self):
        msgs = MULTI_TURN_NO_REASONING
        result = self._call(msgs)
        labeled_ids = [l for l in result["labels"] if l != IGNORE_INDEX]
        decoded = self.tok.decode(labeled_ids)
        for m in msgs:
            if m["role"] == "assistant":
                self.assertIn(m["content"], decoded)

    def test_intermediate_labels_exclude_next_turn_header(self):
        msgs = MULTI_TURN_NO_REASONING
        result = self._call(msgs)
        spans = _get_labeled_spans(result["labels"])
        # First span is intermediate (not the last assistant turn)
        first_span = spans[0]
        first_labels = result["labels"][first_span[0] : first_span[1]]
        decoded = self.tok.decode([l for l in first_labels if l != IGNORE_INDEX])
        self.assertNotIn("<|im_start|>user", decoded)

    def test_intermediate_turn_predicts_im_end(self):
        """Intermediate turns include im_end as a prediction target."""
        msgs = MULTI_TURN_NO_REASONING
        result = self._call(msgs)
        spans = _get_labeled_spans(result["labels"])
        # First span is intermediate
        first_span = spans[0]
        first_labels = result["labels"][first_span[0] : first_span[1]]
        self.assertIn(self.im_end_id, first_labels)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsSingleTurnEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.truncate_last = TruncateLastStrategy(_HF_ASSETS_PATH)

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def test_with_reasoning(self):
        result = self._call(SINGLE_TURN_REASONING)
        ref = self.truncate_last._tokenize_one(SINGLE_TURN_REASONING)
        self.assertEqual(result["input_ids"], ref["input_ids"])
        self.assertEqual(result["labels"], ref["labels"])

    def test_no_reasoning(self):
        result = self._call(SINGLE_TURN_NO_REASONING)
        ref = self.truncate_last._tokenize_one(SINGLE_TURN_NO_REASONING)
        self.assertEqual(result["input_ids"], ref["input_ids"])
        self.assertEqual(result["labels"], ref["labels"])

    def test_empty_reasoning(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": ""},
        ]
        result = self._call(msgs)
        ref = self.truncate_last._tokenize_one(msgs)
        self.assertEqual(result["input_ids"], ref["input_ids"])
        self.assertEqual(result["labels"], ref["labels"])


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsMultiTurnComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.truncate_last = TruncateLastStrategy(_HF_ASSETS_PATH)

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def _assert_same_input_ids_more_labels(self, msgs, desc):
        result = self._call(msgs)
        ref = self.truncate_last._tokenize_one(msgs)
        self.assertEqual(
            result["input_ids"], ref["input_ids"],
            f"{desc}: input_ids differ",
        )
        all_trained = sum(1 for l in result["labels"] if l != IGNORE_INDEX)
        last_trained = sum(1 for l in ref["labels"] if l != IGNORE_INDEX)
        self.assertGreater(
            all_trained, last_trained,
            f"{desc}: all-turns ({all_trained}) should exceed last-only ({last_trained})",
        )

    def test_two_turn_no_reasoning(self):
        self._assert_same_input_ids_more_labels(
            MULTI_TURN_NO_REASONING, "two turn no reasoning"
        )

    def test_two_turn_all_reasoning(self):
        self._assert_same_input_ids_more_labels(
            MULTI_TURN_ALL_REASONING, "two turn all reasoning"
        )

    def test_three_turn_mixed(self):
        self._assert_same_input_ids_more_labels(
            MULTI_TURN_MIXED_REASONING, "three turn mixed"
        )

    def test_tool_chain(self):
        self._assert_same_input_ids_more_labels(
            TOOL_CHAIN_REASONING, "tool chain"
        )

    def test_many_turns(self):
        self._assert_same_input_ids_more_labels(
            MANY_TURNS_MIXED, "many turns"
        )


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsTrailingMessages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def test_trailing_user_message_dropped(self):
        base = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        with_trailing = base + [{"role": "user", "content": "TRAILING"}]
        result_base = self._call(base)
        result_trailing = self._call(with_trailing)
        self.assertEqual(result_base["input_ids"], result_trailing["input_ids"])
        self.assertEqual(result_base["labels"], result_trailing["labels"])

    def test_trailing_tool_message_dropped(self):
        base = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        with_trailing = base + [{"role": "tool", "content": "RESULT"}]
        result_base = self._call(base)
        result_trailing = self._call(with_trailing)
        self.assertEqual(result_base["input_ids"], result_trailing["input_ids"])
        self.assertEqual(result_base["labels"], result_trailing["labels"])


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTokenizeAllTurnsValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def _call(self, messages):
        return _tokenize_all_turns(messages, self.tok, _CHAT_TEMPLATE_KWARGS)

    def test_empty_messages_raises(self):
        with self.assertRaises(ValueError):
            self._call([])

    def test_no_assistant_turn_raises(self):
        with self.assertRaises(ValueError):
            self._call([{"role": "user", "content": "Q"}])

    def test_first_role_assistant_raises(self):
        with self.assertRaises(ValueError):
            self._call([
                {"role": "assistant", "content": "A"},
            ])

    def test_first_role_tool_raises(self):
        with self.assertRaises(ValueError):
            self._call([
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "A"},
            ])

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            self._call([
                {"role": "unknown", "content": "X"},
                {"role": "assistant", "content": "A"},
            ])

    def test_system_not_first_raises(self):
        with self.assertRaises(ValueError):
            self._call([
                {"role": "user", "content": "Q"},
                {"role": "system", "content": "S"},
                {"role": "assistant", "content": "A"},
            ])


if __name__ == "__main__":
    unittest.main()
