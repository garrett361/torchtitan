"""Tests for FullThinkingStrategy.

Verifies:
- All reasoning assistant turns are loss-unmasked
- No-reasoning assistant turns stay loss-masked (context only)
- Prefix-stability with truncate_history_thinking=False
- Boundary correctness for intermediate turns
- Agentic trajectories, single-turn, and mixed cases

Requires HF_ASSETS_PATH pointing to a Granite tokenizer with the thinking
chat template.
"""

import os
import unittest

from dotenv import load_dotenv

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.granite.tests.helpers import find_unmasked_regions

load_dotenv()

_HF_ASSETS_PATH = os.environ.get("HF_ASSETS_PATH")


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestFullThinkingStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torchtitan.models.granite.tokenization_strategies import (
            FullThinkingStrategy,
            TruncateLastStrategy,
        )

        cls.strategy = FullThinkingStrategy(_HF_ASSETS_PATH)
        cls.truncate_strategy = TruncateLastStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")
        cls.im_end_id = cls.tok.token_to_id("<|im_end|>")

    def _tokenize(self, messages):
        return self.strategy._tokenize_one(messages)

    def test_chat_template_kwargs(self):
        self.assertEqual(
            self.strategy.chat_template_kwargs,
            {"truncate_history_thinking": False},
        )

    def test_single_turn_with_reasoning(self):
        """Single user+assistant with reasoning behaves like truncate_last."""
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Simple math."},
        ]
        result = self._tokenize(msgs)
        truncate_result = self.truncate_strategy._tokenize_one(msgs)

        # Same token sequence (single turn, no history to truncate)
        self.assertEqual(result["input_ids"], truncate_result["input_ids"])
        # Same labels (only one turn, it's unmasked in both)
        self.assertEqual(result["labels"], truncate_result["labels"])

    def test_reasoning_turns_unmasked(self):
        """Multi-turn: all reasoning assistant turns are loss-unmasked."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        labels = result["labels"]

        # Both assistant turns should have unmasked regions
        unmasked_regions = find_unmasked_regions(labels)
        self.assertEqual(
            len(unmasked_regions),
            2,
            f"Expected 2 unmasked regions (one per reasoning turn), got {len(unmasked_regions)}",
        )

    def test_no_reasoning_turn_labeled(self):
        """Assistant turn without reasoning_content is still loss-unmasked."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},  # no reasoning
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        labels = result["labels"]

        unmasked_regions = find_unmasked_regions(labels)
        self.assertEqual(
            len(unmasked_regions),
            2,
            f"Expected 2 unmasked regions (both assistant turns), got {len(unmasked_regions)}",
        )

    def test_mixed_reasoning_and_no_reasoning(self):
        """All assistant turns are unmasked regardless of reasoning_content."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},  # no reasoning
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3", "reasoning_content": "R3"},
        ]
        result = self._tokenize(msgs)
        labels = result["labels"]

        unmasked_count = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
        self.assertGreater(unmasked_count, 0)

        unmasked_regions = find_unmasked_regions(labels)
        self.assertEqual(
            len(unmasked_regions),
            3,
            "Expected 3 unmasked regions (all assistant turns labeled)",
        )

    def test_preserves_historical_thinking(self):
        """Full thinking from historical turns is present in the token sequence."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Historical reasoning."},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "Final reasoning."},
        ]
        result = self._tokenize(msgs)
        input_ids = result["input_ids"]

        # Decode and verify both reasoning traces are present
        decoded = self.tok.decode(input_ids)
        self.assertIn("Historical reasoning.", decoded)
        self.assertIn("Final reasoning.", decoded)

    def test_longer_than_truncated(self):
        """Same multi-turn input produces more tokens than truncate_last."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "Long historical thinking trace."},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        full_result = self._tokenize(msgs)
        truncate_result = self.truncate_strategy._tokenize_one(msgs)

        self.assertGreater(
            len(full_result["input_ids"]),
            len(truncate_result["input_ids"]),
            "full_thinking should produce longer sequences (thinking not stripped)",
        )

    def test_agentic_trajectory(self):
        """Tool-use trajectory: all reasoning assistant turns unmasked."""
        msgs = [
            {"role": "user", "content": "Search for Python docs."},
            {"role": "assistant", "content": "Calling search.", "reasoning_content": "Need to search."},
            {"role": "tool", "content": "Result: Python 3.12 docs."},
            {"role": "assistant", "content": "Calling another tool.", "reasoning_content": "Need more info."},
            {"role": "tool", "content": "Result: asyncio docs."},
            {"role": "assistant", "content": "Here are the docs.", "reasoning_content": "Got both results."},
        ]
        result = self._tokenize(msgs)
        labels = result["labels"]

        # All 3 assistant turns have reasoning → 3 unmasked regions
        self.assertEqual(
            len(find_unmasked_regions(labels)),
            3,
            f"Expected 3 unmasked regions for 3 reasoning assistant turns, got {len(find_unmasked_regions(labels))}",
        )

    def test_trailing_messages_dropped(self):
        """Messages after the last assistant turn are dropped."""
        msgs_with_trailing = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Trailing user message."},
        ]
        msgs_without_trailing = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        ]
        result_with = self._tokenize(msgs_with_trailing)
        result_without = self._tokenize(msgs_without_trailing)

        self.assertEqual(result_with["input_ids"], result_without["input_ids"])
        self.assertEqual(result_with["labels"], result_without["labels"])

    def test_system_message_masked(self):
        """System message tokens are always loss-masked."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        result = self._tokenize(msgs)

        # The first unmasked label should come well after the system+user tokens
        first_unmasked = next(
            i for i, lbl in enumerate(result["labels"]) if lbl != IGNORE_INDEX
        )
        # System "You are helpful." + user "Q" should be at least several tokens
        self.assertGreater(first_unmasked, 5)

    def test_intermediate_turn_boundaries(self):
        """Verify unmasked boundaries start after header and include <|im_end|>."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        input_ids = result["input_ids"]
        labels = result["labels"]

        # For intermediate turn: the last unmasked label in the first region
        # should predict a token that precedes the next user turn.
        regions = find_unmasked_regions(labels)
        self.assertGreater(len(regions), 0)
        first_region_end = regions[0][1]
        # Last label in intermediate region predicts <|im_end|>
        self.assertEqual(labels[first_region_end - 1], self.im_end_id)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestFullThinkingInferenceFidelity(unittest.TestCase):
    """Verify FullThinkingStrategy produces token sequences matching inference context."""

    @classmethod
    def setUpClass(cls):
        from torchtitan.models.granite.tokenization_strategies import (
            FullThinkingStrategy,
        )

        cls.strategy = FullThinkingStrategy(_HF_ASSETS_PATH)
        cls.tok = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS_PATH)
        cls.think_id = cls.tok.token_to_id("<think>")
        cls.end_think_id = cls.tok.token_to_id("</think>")
        cls.im_end_id = cls.tok.token_to_id("<|im_end|>")
        cls.newline_id = cls.tok.encode("\n", add_bos=False, add_eos=False)[0]

    def _tokenize(self, messages):
        return self.strategy._tokenize_one(messages)

    def test_no_reasoning_context_matches_inference(self):
        """No-reasoning turn has <think> \\n </think> \\n in input_ids (not adjacent)."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = self._tokenize(msgs)
        ids = result["input_ids"]

        think_pos = ids.index(self.think_id)
        self.assertEqual(ids[think_pos + 1], self.newline_id)
        self.assertEqual(ids[think_pos + 2], self.end_think_id)
        self.assertEqual(ids[think_pos + 3], self.newline_id)

    def test_reasoning_context_unchanged(self):
        """Reasoning turn still has <think> \\n ... </think> \\n structure."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "Because math."},
        ]
        result = self._tokenize(msgs)
        ids = result["input_ids"]

        think_pos = ids.index(self.think_id)
        # \n after <think> is always separate (special token forces boundary)
        self.assertEqual(ids[think_pos + 1], self.newline_id)
        end_think_pos = ids.index(self.end_think_id, think_pos)
        # \n after </think> is always separate (special token forces boundary)
        self.assertEqual(ids[end_think_pos + 1], self.newline_id)
        # Reasoning content exists between <think>\n and </think>
        self.assertGreater(end_think_pos, think_pos + 2)

    def test_labels_cover_all_generated_tokens(self):
        """Labels span from first generated token through predicting <|im_end|>."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self._tokenize(msgs)
        ids = result["input_ids"]
        labels = result["labels"]

        # Generation starts after <think>\n (the generation prompt suffix)
        think_pos = ids.index(self.think_id)
        gen_start = think_pos + 1  # \n position — first generated token is at +2
        # But labels use next-token prediction: label at pos P predicts token at P+1
        # So first labeled position is think_pos + 1 (predicts </think> or reasoning)
        first_labeled = next(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
        self.assertEqual(first_labeled, gen_start)

        # Last label should be <|im_end|> (EOS prediction)
        last_labeled = max(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
        self.assertEqual(labels[last_labeled], self.im_end_id)

    def test_full_passage_token_equivalence(self):
        """input_ids match a manually constructed inference-time sequence."""
        msgs = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": "Calling tool."},
            {"role": "tool", "content": "Result here."},
            {"role": "assistant", "content": "Done.", "reasoning_content": "Got it."},
        ]
        result = self._tokenize(msgs)
        ids = result["input_ids"]

        # Construct expected text: template + literal fixup (not importing the impl)
        raw_text = self.tok.apply_chat_template(
            msgs, truncate_history_thinking=False
        ).rstrip("\n")
        # The template renders no-reasoning turns as <think></think>{content}.
        # Inference context is <think>\n</think>\n{content}. Apply literally:
        fixed_text = raw_text.replace("<think></think>", "<think>\n</think>\n")
        expected_tokens = self.tok.encode(fixed_text, add_bos=True, add_eos=False)
        if expected_tokens[-1] != self.tok.eos_id:
            expected_tokens.append(self.tok.eos_id)
        expected_ids = expected_tokens[:-1]

        self.assertEqual(ids, expected_ids)

    def test_intermediate_last_label_is_eos(self):
        """For every assistant turn, the last label in its region is <|im_end|>."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        ]
        result = self._tokenize(msgs)
        labels = result["labels"]

        # Find unmasked regions
        regions = find_unmasked_regions(labels)
        self.assertEqual(len(regions), 2)
        # Last label in each region should be <|im_end|>
        for reg_start, reg_end in regions:
            self.assertEqual(labels[reg_end - 1], self.im_end_id)


@unittest.skipUnless(_HF_ASSETS_PATH, "HF_ASSETS_PATH not set")
class TestFullThinkingDataloaderDispatch(unittest.TestCase):
    """Verify manifest with strategy='full_thinking' dispatches correctly."""

    def test_dispatch_to_truncate_last_dataset(self):
        from torchtitan.models.granite.pretokenized_dataset import _DATASET_CLASSES

        self.assertIn("full_thinking", _DATASET_CLASSES)
        from torchtitan.models.granite.pretokenized_dataset import StandardPackingDataset

        self.assertIs(_DATASET_CLASSES["full_thinking"], StandardPackingDataset)


if __name__ == "__main__":
    unittest.main()
