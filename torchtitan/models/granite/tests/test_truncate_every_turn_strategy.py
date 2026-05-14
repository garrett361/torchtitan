"""Tests for TruncateEveryTurnStrategy.

Verifies decomposition of multi-turn conversations into one example per
assistant turn, with labels only on the last turn of each sub-example.

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template.
"""

import os
import unittest

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.tokenization_strategies import (
    TruncateEveryTurnStrategy,
    TruncateLastStrategy,
)

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTruncateEveryTurnOutputStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.strategy = TruncateEveryTurnStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def _call(self, messages_list):
        return self.strategy({"messages": messages_list})

    def test_output_keys(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        result = self._call([msgs])
        self.assertEqual(set(result.keys()), {"input_ids", "labels", "n_tokens"})

    def test_n_tokens_consistent(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._call([msgs])
        for i in range(len(result["input_ids"])):
            n = result["n_tokens"][i]
            self.assertEqual(len(result["input_ids"][i]), n)
            self.assertEqual(len(result["labels"][i]), n)

    def test_single_turn_produces_one_example(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        result = self._call([msgs])
        self.assertEqual(len(result["input_ids"]), 1)

    def test_multi_turn_produces_n_examples(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        result = self._call([msgs])
        self.assertEqual(len(result["input_ids"]), 3)

    def test_system_message_in_every_sub_example(self):
        system_content = "You are a helpful assistant."
        msgs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._call([msgs])
        self.assertEqual(len(result["input_ids"]), 2)
        for i in range(2):
            decoded = self.tok.decode(result["input_ids"][i])
            self.assertIn(system_content, decoded)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTruncateEveryTurnEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.strategy = TruncateEveryTurnStrategy(_HF_ASSETS_PATH)
        cls.truncate_last = TruncateLastStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def _call(self, messages_list):
        return self.strategy({"messages": messages_list})

    def test_last_example_matches_truncate_last(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        every_result = self._call([msgs])
        last_result = self.truncate_last._tokenize_one(msgs)
        self.assertEqual(every_result["input_ids"][-1], last_result["input_ids"])
        self.assertEqual(every_result["labels"][-1], last_result["labels"])

    def test_single_turn_identical_to_truncate_last(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        every_result = self._call([msgs])
        last_result = self.truncate_last._tokenize_one(msgs)
        self.assertEqual(every_result["input_ids"][0], last_result["input_ids"])
        self.assertEqual(every_result["labels"][0], last_result["labels"])

    def test_each_example_labels_only_last_turn(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        every_result = self._call([msgs])
        for i in range(len(every_result["input_ids"])):
            truncated = msgs[: self._asst_indices(msgs)[i] + 1]
            tl_result = self.truncate_last._tokenize_one(truncated)
            self.assertEqual(every_result["labels"][i], tl_result["labels"])

    def test_context_is_prefix_of_next_example(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        every_result = self._call([msgs])
        for i in range(len(every_result["input_ids"]) - 1):
            labels_i = every_result["labels"][i]
            label_start = next(
                j for j, lbl in enumerate(labels_i) if lbl != IGNORE_INDEX
            )
            prefix = every_result["input_ids"][i][:label_start]
            next_ids = every_result["input_ids"][i + 1][:label_start]
            self.assertEqual(prefix, next_ids)

    @staticmethod
    def _asst_indices(messages):
        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        effective = messages[: last_asst_idx + 1]
        return [i for i, m in enumerate(effective) if m["role"] == "assistant"]


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTruncateEveryTurnMultiTurn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.strategy = TruncateEveryTurnStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")

    def _call(self, messages_list):
        return self.strategy({"messages": messages_list})

    def test_progressive_context_growth(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        result = self._call([msgs])
        lengths = [result["n_tokens"][i] for i in range(3)]
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])

    def test_tool_chain_groups_included_as_context(self):
        msgs = [
            {"role": "user", "content": "Do a search."},
            {"role": "assistant", "content": "Calling.", "reasoning_content": "Need."},
            {"role": "tool", "content": "Found result."},
            {"role": "assistant", "content": "Here.", "reasoning_content": "Got it."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "Welcome.", "reasoning_content": "Done."},
        ]
        result = self._call([msgs])
        # 3 assistant turns → 3 examples
        self.assertEqual(len(result["input_ids"]), 3)
        # Third example should contain tool content as context
        decoded_last = self.tok.decode(result["input_ids"][2])
        self.assertIn("Found result.", decoded_last)

    def test_consecutive_assistant_turns(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        result = self._call([msgs])
        self.assertEqual(len(result["input_ids"]), 3)

    def test_history_thinking_truncated(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "Historical reasoning trace"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._call([msgs])
        # Second example: historical thinking from turn 1 should be truncated
        ids = result["input_ids"][1]
        decoded = self.tok.decode(ids)
        self.assertNotIn("Historical reasoning trace", decoded)

    def test_last_turn_thinking_preserved(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "My deep thinking"},
        ]
        result = self._call([msgs])
        ids = result["input_ids"][0]
        decoded = self.tok.decode(ids)
        self.assertIn("My deep thinking", decoded)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTruncateEveryTurnBatchInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.strategy = TruncateEveryTurnStrategy(_HF_ASSETS_PATH)

    def _call(self, messages_list):
        return self.strategy({"messages": messages_list})

    def test_batch_emits_n_rows_per_conversation(self):
        conv_3_turn = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        conv_2_turn = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._call([conv_3_turn, conv_2_turn])
        self.assertEqual(len(result["input_ids"]), 5)

    def test_partial_failure_preserves_good_conversations(self):
        good_conv = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        bad_conv = []  # empty messages list will fail validation
        result = self._call([good_conv, bad_conv, good_conv])
        self.assertEqual(len(result["input_ids"]), 2)

    def test_empty_batch(self):
        result = self._call([])
        self.assertEqual(result["input_ids"], [])
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["n_tokens"], [])


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestTruncateEveryTurnEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.strategy = TruncateEveryTurnStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)

    def _call(self, messages_list):
        return self.strategy({"messages": messages_list})

    def test_no_reasoning_still_produces_example(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = self._call([msgs])
        self.assertEqual(len(result["input_ids"]), 2)
        for i in range(2):
            trained = [lbl for lbl in result["labels"][i] if lbl != IGNORE_INDEX]
            self.assertGreater(len(trained), 0)

    def test_trailing_user_messages_dropped(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": "user", "content": "Trailing user message"},
        ]
        result = self._call([msgs])
        # Only 1 assistant turn → 1 example
        self.assertEqual(len(result["input_ids"]), 1)
        decoded = self.tok.decode(result["input_ids"][0])
        self.assertNotIn("Trailing user message", decoded)

    def test_malformed_sample_dropped_gracefully(self):
        bad_msgs = [{"role": "user", "content": "Only user, no assistant"}]
        good_msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        result = self._call([bad_msgs, good_msgs])
        self.assertEqual(len(result["input_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
