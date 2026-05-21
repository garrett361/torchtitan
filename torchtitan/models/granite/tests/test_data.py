import glob
import json
import os
import unittest

from torchtitan.models.granite.sft_dataset import GraniteSFTDataset


class TestGraniteSFTDatasetUnit(unittest.TestCase):
    """Unit tests for GraniteSFTDataset / ChatDataset validation.

    No environment variables or real checkpoints required.
    """

    _user = {"role": "user", "content": "hello"}
    _asst = {"role": "assistant", "content": "hi"}
    _sys = {"role": "system", "content": "be helpful"}

    def test_validate_accepts_user_assistant(self):
        GraniteSFTDataset._validate_messages([self._user, self._asst])

    def test_validate_accepts_system_user_assistant(self):
        GraniteSFTDataset._validate_messages([self._sys, self._user, self._asst])

    def test_validate_accepts_multi_turn(self):
        # General multi-turn is now valid.
        GraniteSFTDataset._validate_messages(
            [self._sys, self._user, self._asst, self._user, self._asst]
        )

    def test_validate_rejects_missing_assistant(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages([self._user, self._user])

    def test_validate_rejects_system_not_first(self):
        with self.assertRaises(ValueError):
            GraniteSFTDataset._validate_messages([self._user, self._sys, self._asst])


class TestChatTemplate(unittest.TestCase):
    """Verifies chat template rendering behavior for thinking, tool calls, and BPE boundaries.

    Three distinct assistant-turn forms with truncate_history_thinking=True:
      - Full thinking:   <think>\\n{reasoning}\\n</think>\\n{response}
      - Truncated:       <think></think>\\n{response}   (\\n survives stripping)
      - No reasoning:    <think></think>{response}      (no \\n at all)

    Requires HF_ASSETS_PATH. Skips if absent.
    """

    _tokenizer = None

    # Two reasoning turns + one no-reasoning turn.  With truncate_history_thinking=True,
    # last_user_idx=5 (third user message), so the assistants at indices 2 and 4 are
    # historical: idx-2 has rc → truncated form; idx-4 has no rc → no-reasoning form.
    # The assistant at idx-6 is current → full-thinking form.
    _MESSAGES = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "First question."},
        {
            "role": "assistant",
            "content": "First response.",
            "reasoning_content": "First reasoning.",
        },
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "content": "Second response."},
        {"role": "user", "content": "Third question."},
        {
            "role": "assistant",
            "content": "Third response.",
            "reasoning_content": "Third reasoning.",
        },
    ]

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        load_dotenv()
        ckpt_path = os.getenv("HF_ASSETS_PATH")
        if ckpt_path is None:
            return
        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if os.getenv("HF_ASSETS_PATH") is None:
            self.skipTest("HF_ASSETS_PATH not set")

    def _render(self):
        return self._tokenizer.apply_chat_template(
            self._MESSAGES, truncate_history_thinking=True
        )

    def test_full_thinking_form(self):
        """Last assistant turn: reasoning flanked by newlines, \\n follows </think>."""
        self.assertIn("<think>\nThird reasoning.\n</think>\nThird response.", self._render())

    def test_truncated_thinking_preserves_trailing_newline(self):
        """Historical turn with reasoning_content: thinking stripped, trailing \\n survives."""
        self.assertIn("<think></think>\nFirst response.", self._render())

    def test_no_reasoning_form_has_no_newline(self):
        """Assistant turn without reasoning_content: <think></think> with no newline."""
        rendered = self._render()
        self.assertIn("<think></think>Second response.", rendered)
        self.assertNotIn("<think></think>\nSecond response.", rendered)

    def test_tool_chain_single_user_preserves_all_thinking(self):
        """Single initial user message: last_user_idx=0, so no assistant is historical.

        All reasoning_content survives intact — truncation never fires regardless of how
        many tool round-trips occur between the user message and the final response.
        """
        messages = [
            {"role": "user", "content": "Search for info."},
            {
                "role": "assistant",
                "content": "Calling search.",
                "reasoning_content": "I should search.",
            },
            {"role": "tool", "content": "Search results here."},
            {
                "role": "assistant",
                "content": "Processing results.",
                "reasoning_content": "Results indicate.",
            },
            {"role": "tool", "content": "More results here."},
            {
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "Putting it together.",
            },
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        self.assertIn("<think>\nI should search.\n</think>", rendered)
        self.assertIn("<think>\nResults indicate.\n</think>", rendered)
        self.assertIn("<think>\nPutting it together.\n</think>", rendered)
        self.assertNotIn("<think></think>", rendered)

    def test_tool_chain_followed_by_user_strips_intermediate_thinking(self):
        """Tool-use block before a follow-up user message: those assistant turns are historical.

        last_user_idx is determined by the last role=user in the original message list.
        Assistants before that index — including those inside a prior tool-call loop — have
        their thinking stripped.  Only the final assistant (after the last user) keeps it.
        """
        messages = [
            {"role": "user", "content": "Look something up."},
            {
                "role": "assistant",
                "content": "Calling tool.",
                "reasoning_content": "Tool reasoning.",
            },
            {"role": "tool", "content": "Tool result."},
            {
                "role": "assistant",
                "content": "Here is the result.",
                "reasoning_content": "Result reasoning.",
            },
            {"role": "user", "content": "Follow-up question."},
            {
                "role": "assistant",
                "content": "Follow-up answer.",
                "reasoning_content": "Follow-up reasoning.",
            },
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        # Assistants before the follow-up user turn: thinking stripped.
        self.assertIn("<think></think>\nCalling tool.", rendered)
        self.assertIn("<think></think>\nHere is the result.", rendered)
        # Final assistant: thinking preserved in full.
        self.assertIn(
            "<think>\nFollow-up reasoning.\n</think>\nFollow-up answer.", rendered
        )

    def test_tool_messages_rendered_as_user_turns(self):
        """Tool messages appear inside <|im_start|>user blocks, not as a distinct role."""
        messages = [
            {"role": "user", "content": "Do a lookup."},
            {"role": "assistant", "content": "Looking up."},
            {"role": "tool", "content": "Lookup result."},
            {"role": "assistant", "content": "Done."},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        self.assertNotIn("<|im_start|>tool", rendered)
        self.assertIn("Lookup result.", rendered)
        # The <|im_start|> immediately before the tool content must open a user turn.
        pos = rendered.index("Lookup result.")
        last_start = rendered.rfind("<|im_start|>", 0, pos)
        self.assertIn("user", rendered[last_start : last_start + 25])

    def test_consecutive_tool_messages_share_one_user_block(self):
        """Consecutive tool messages are grouped under a single <|im_start|>user block.

        Verified by absence of any <|im_end|> between the two tool result strings.
        """
        messages = [
            {"role": "user", "content": "Do two lookups."},
            {"role": "assistant", "content": "Calling both."},
            {"role": "tool", "content": "Result one."},
            {"role": "tool", "content": "Result two."},
            {"role": "assistant", "content": "Both done."},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, truncate_history_thinking=True
        )
        self.assertIn("Result one.", rendered)
        self.assertIn("Result two.", rendered)
        pos1 = rendered.index("Result one.")
        pos2 = rendered.index("Result two.")
        self.assertNotIn(
            "<|im_end|>",
            rendered[pos1:pos2],
            "Consecutive tool messages must not be separated by a turn boundary",
        )

    def test_bpe_boundary_response_tokens_unchanged_by_preceding_newline(self):
        """Response tokens are identical whether they follow </think>\\n or </think> alone.

        The pre-tokenizer splits on newlines as their own word, and <think>/<\\/think> are
        registered add tokens (always atomic).  So the \\n between </think> and the response
        is an independent token and does not merge with adjacent response text — the response
        token sequence is identical in both contexts, enabling token-level splicing of
        truncated sequences from full tokenizations without re-tokenization.
        """
        end_think_id = self._tokenizer.token_to_id("</think>")
        self.assertIsNotNone(end_think_id, "</think> must be a registered token")

        response_text = "Alpha beta gamma delta epsilon."
        tokens_with_newline = self._tokenizer.encode(
            f"</think>\n{response_text}", add_bos=False, add_eos=False
        )
        tokens_without_newline = self._tokenizer.encode(
            f"</think>{response_text}", add_bos=False, add_eos=False
        )
        self.assertEqual(tokens_with_newline[0], end_think_id)
        self.assertEqual(tokens_without_newline[0], end_think_id)
        # Skip </think> and the \n in the first sequence; response tokens must be equal.
        self.assertEqual(
            tokens_with_newline[2:],
            tokens_without_newline[1:],
            "Response tokens after </think> must be identical regardless of preceding \\n",
        )

    # --- truncate_history_thinking=False behavior ---

    def test_no_reasoning_form_preserved_when_thinking_not_truncated(self):
        """With truncate_history_thinking=False, no-reasoning turns still render <think></think>{content}."""
        rendered = self._tokenizer.apply_chat_template(
            self._MESSAGES, truncate_history_thinking=False
        )
        self.assertIn("<think></think>Second response.", rendered)
        self.assertNotIn("<think></think>\nSecond response.", rendered)

    def test_generation_prompt_ends_with_think_newline(self):
        """Generation prompt ends with <think>\\n — what the model receives at inference."""
        rendered = self._tokenizer.apply_chat_template(
            self._MESSAGES[:5],
            add_generation_prompt=True,
            truncate_history_thinking=False,
        )
        self.assertTrue(
            rendered.endswith("<think>\n"),
            f"Expected generation prompt to end with '<think>\\n', got: ...{rendered[-30:]!r}",
        )

    def test_no_reasoning_inference_context_diverges(self):
        """Generation prompt has \\n where full render has </think> — the fixup target.

        At inference the model receives <think>\\n from the generation prompt.
        In the full render (template normalization), the same position has </think>
        (from <think></think>{content}). This token-level divergence is what
        _fix_empty_thinking corrects for FullThinkingStrategy.
        """
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        prefix = self._tokenizer.apply_chat_template(
            msgs[:1],
            add_generation_prompt=True,
            truncate_history_thinking=False,
        )
        backbone = self._tokenizer.apply_chat_template(
            msgs, truncate_history_thinking=False
        )
        prefix_tokens = self._tokenizer.encode(prefix, add_bos=True, add_eos=False)
        backbone_tokens = self._tokenizer.encode(backbone, add_bos=True, add_eos=False)

        newline_id = self._tokenizer.encode("\n", add_bos=False, add_eos=False)[0]
        think_id = self._tokenizer.token_to_id("<think>")
        end_think_id = self._tokenizer.token_to_id("</think>")

        # Prefix and backbone share tokens up to the divergence point
        self.assertEqual(
            backbone_tokens[: len(prefix_tokens) - 1],
            prefix_tokens[:-1],
        )
        # Both have <think> immediately before the divergence
        self.assertEqual(prefix_tokens[-2], think_id)
        self.assertEqual(backbone_tokens[len(prefix_tokens) - 2], think_id)
        # Divergence: prefix has \n, backbone has </think>
        self.assertEqual(prefix_tokens[-1], newline_id)
        self.assertEqual(backbone_tokens[len(prefix_tokens) - 1], end_think_id)

    def test_special_tokens_are_atomic_in_offset_table(self):
        """Special tokens produce exactly one entry in the Rust encoder's offset table.

        The offset-based _tokenize_one relies on bisect over character offsets. If
        <think>, </think>, <|im_start|>, or <|im_end|> were split into multiple
        sub-tokens, char_to_token_idx would return the wrong index.
        """
        for token in ("<think>", "</think>", "<|im_start|>", "<|im_end|>"):
            token_id = self._tokenizer.tokenizer.token_to_id(token)
            self.assertIsNotNone(token_id, f"{token} must be a registered token")

            text = f"hello{token}world"
            encoding = self._tokenizer.tokenizer.encode(text)
            token_ids = encoding.ids
            offsets = encoding.offsets

            positions = [i for i, tid in enumerate(token_ids) if tid == token_id]
            self.assertEqual(
                len(positions), 1,
                f"{token} must appear exactly once in encoding of {text!r}, got {positions}",
            )
            idx = positions[0]
            start, end = offsets[idx]
            self.assertEqual(
                end - start, len(token),
                f"{token} offset span must equal its character length "
                f"(atomic tokenization); got span ({start}, {end})",
            )


class TestPrefixInvariant(unittest.TestCase):
    """Structural guarantees relied upon by the offset-based _tokenize_one.

    The optimized tokenizer derives label boundaries via bisect on a character offset
    table. This requires that render(msgs[:k]).rstrip("\\n") is a character prefix of
    the target text for the cases used by BackboneSuffixStrategy.

    Requires HF_ASSETS_PATH. Skips if absent.
    """

    _tokenizer = None

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        load_dotenv()
        ckpt_path = os.getenv("HF_ASSETS_PATH")
        if ckpt_path is None:
            return
        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if os.getenv("HF_ASSETS_PATH") is None:
            self.skipTest("HF_ASSETS_PATH not set")

    def _render(self, msgs, **kwargs):
        return self._tokenizer.apply_chat_template(
            msgs, truncate_history_thinking=True, **kwargs
        )

    def test_prefix_invariant_no_reasoning(self):
        """render(msgs[:k]).rstrip is a prefix of render(msgs) when no assistant has reasoning."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
        ]
        full = self._render(msgs).rstrip("\n")
        for k in range(1, len(msgs)):
            prefix = self._render(msgs[:k]).rstrip("\n")
            self.assertTrue(
                full.startswith(prefix),
                f"Prefix invariant failed at k={k}: render(msgs[:{k}]) is not a prefix of full render",
            )

    def test_prefix_invariant_no_reasoning_with_system(self):
        """Same as above but with a system message and longer content to stress BPE boundaries."""
        msgs = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain how binary search works in Python."},
            {"role": "assistant", "content": "Binary search repeatedly halves the search interval."},
            {"role": "user", "content": "Can you show me an implementation?"},
            {"role": "assistant", "content": "Here is an iterative implementation of binary search."},
        ]
        full = self._render(msgs).rstrip("\n")
        for k in range(1, len(msgs)):
            prefix = self._render(msgs[:k]).rstrip("\n")
            self.assertTrue(
                full.startswith(prefix),
                f"Prefix invariant failed at k={k} (with system message)",
            )

    def test_prefix_invariant_reasoning_suffix(self):
        """render(msgs[:k]).rstrip is a prefix of render(msgs[:group_end]) for turns within a reasoning group."""
        msgs = [
            {"role": "user", "content": "Search for X."},
            {"role": "assistant", "content": "Calling tool.", "reasoning_content": "I should search."},
            {"role": "tool", "content": "Result from search."},
            {"role": "assistant", "content": "Final answer.", "reasoning_content": "Analyzing results."},
        ]
        group_end = len(msgs)
        target = self._render(msgs[:group_end]).rstrip("\n")
        for k in range(1, group_end):
            prefix = self._render(msgs[:k]).rstrip("\n")
            self.assertTrue(
                target.startswith(prefix),
                f"Prefix invariant failed at k={k} for reasoning group",
            )

    def test_prefix_invariant_fails_for_last_reasoning_turn(self):
        """Prefix invariant does NOT hold when subset ends with a reasoning assistant that has a subsequent user.

        The mechanism: in the subset render, the reasoning assistant is the last turn
        so thinking is preserved. In the full render, a subsequent user makes it historical
        so thinking is stripped. If the template changes to no longer strip thinking,
        this test would fail — that's intentional: the offset-based algorithm's safety
        argument depends on this divergence being known and handled separately.
        """
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        full = self._render(msgs).rstrip("\n")
        subset = self._render(msgs[:2]).rstrip("\n")
        self.assertFalse(
            full.startswith(subset),
            "Expected invariant to FAIL when subset ends with reasoning assistant followed by user",
        )

    def test_close_think_follows_turn_header(self):
        """</think> appears immediately after <think> in truncated no-reasoning turns."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        rendered = self._render(msgs).rstrip("\n")
        for asst_idx in [1, 3]:
            prefix = self._render(msgs[:asst_idx]).rstrip("\n")
            after_prefix = rendered[len(prefix):]
            think_pos = after_prefix.index("<think>")
            close_think_pos = after_prefix.index("</think>")
            self.assertEqual(
                close_think_pos, think_pos + len("<think>"),
                f"</think> must immediately follow <think> in no-reasoning turn at index {asst_idx}",
            )

    def test_open_think_newline_in_preserved_turn(self):
        """<think>\\n appears in preserved reasoning turns (suffix search anchor)."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
        ]
        rendered = self._render(msgs).rstrip("\n")
        prefix = self._render(msgs[:1]).rstrip("\n")
        after_prefix = rendered[len(prefix):]
        self.assertIn(
            "<think>\n",
            after_prefix,
            "Preserved reasoning turn must contain <think>\\n",
        )


class TestGraniteSFTDataFormat(unittest.TestCase):
    """Structural checks on the raw GLM-5.1 Reasoning dataset.

    Requires DATA_PATH set in the environment or a .env file.
    Skips if the variable is absent.
    """

    _SAMPLE_COUNT = 200

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        data_path = os.getenv("DATA_PATH")
        if data_path is None:
            self.skipTest("DATA_PATH not set")
        self.data_path = data_path

    def test_all_examples_single_turn(self):
        jsonl_files = sorted(glob.glob(os.path.join(self.data_path, "*.jsonl")))
        if not jsonl_files:
            self.skipTest("No .jsonl files found in DATA_PATH")
        for fpath in jsonl_files:
            fname = os.path.basename(fpath)
            with open(fpath) as f:
                for i, line in enumerate(f):
                    if i >= self._SAMPLE_COUNT:
                        break
                    record = json.loads(line)
                    msgs = record["messages"]
                    self.assertEqual(
                        len(msgs),
                        3,
                        f"{fname} record {i}: expected 3 messages, got {len(msgs)}",
                    )
                    self.assertEqual(msgs[0]["role"], "system", f"{fname} record {i}")
                    self.assertEqual(msgs[1]["role"], "user", f"{fname} record {i}")
                    self.assertEqual(
                        msgs[2]["role"], "assistant", f"{fname} record {i}"
                    )
                    rc = msgs[2].get("reasoning_content", "")
                    self.assertTrue(
                        rc,
                        f"{fname} record {i}: reasoning_content missing or empty",
                    )


class TestGraniteSFTData(unittest.TestCase):
    """End-to-end tokenization and masking tests for Granite SFT with thinking template.

    Requires HF_ASSETS_PATH and DATA_PATH.
    Skips if any variable is absent.
    """

    _tokenizer = None
    _input_ids = None
    _labels = None
    _sample_msgs = None

    @classmethod
    def setUpClass(cls):
        from datasets import Dataset
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        load_dotenv()
        ckpt_path = os.getenv("HF_ASSETS_PATH")
        data_path = os.getenv("DATA_PATH")
        if any(v is None for v in (ckpt_path, data_path)):
            return

        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)
        cls._IGNORE_INDEX = IGNORE_INDEX

        # Load a handful of samples from the first JSONL file found.
        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not jsonl_files:
            return
        fpath = jsonl_files[0]
        records = []
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= 8:
                    break

        cls._sample_msgs = records[0]["messages"]

        # Build a minimal GraniteSFTDataset to get one packed batch.
        dataset = Dataset.from_list(records)
        ds = GraniteSFTDataset(
            dataset=dataset,
            tokenizer=cls._tokenizer,
            sample_processor=lambda s: s["messages"],
            seq_len=8192,
            infinite=False,
        )
        batch, labels = next(iter(ds))
        cls._input_ids = batch["input"].tolist()
        cls._labels = labels.tolist()

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if any(os.getenv(v) is None for v in ("HF_ASSETS_PATH", "DATA_PATH")):
            self.skipTest("HF_ASSETS_PATH and DATA_PATH must both be set")

    def test_chat_template_renders_system_and_thinking(self):
        rendered = self._tokenizer.apply_chat_template(self._sample_msgs)
        self.assertIn("<|im_start|>system", rendered)
        self.assertIn("<|im_end|>", rendered)
        self.assertIn("<think>", rendered)
        self.assertIn("</think>", rendered)
        rc_prefix = self._sample_msgs[2]["reasoning_content"][:20]
        self.assertIn(rc_prefix, rendered)

    def test_prompt_masked_thinking_trained(self):
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        labels = self._labels
        self.assertTrue(
            any(l == IGNORE_INDEX for l in labels),
            "Expected some prompt tokens to be masked",
        )
        trained = sum(1 for l in labels if l != IGNORE_INDEX)
        self.assertGreater(
            trained, 10, "Expected many trained tokens from reasoning_content"
        )

    def test_think_token_is_last_masked_position(self):
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        think_id = self._tokenizer.token_to_id("<think>")
        self.assertIsNotNone(think_id, "<think> must be registered as a token")

        think_pos = self._input_ids.index(think_id)
        # labels[think_pos] predicts the token after <think>, which is \n —
        # still part of the masked generation prefix.
        self.assertEqual(
            self._labels[think_pos],
            IGNORE_INDEX,
            "Token immediately after <think> must be masked (part of generation prefix)",
        )
        # labels[think_pos + 1] = first reasoning_content token = trained.
        self.assertNotEqual(
            self._labels[think_pos + 1],
            IGNORE_INDEX,
            "First reasoning_content token must be trained",
        )

    def test_eos_present(self):
        eos_id = self._tokenizer.eos_id
        self.assertIn(eos_id, self._input_ids, "EOS token must appear in input_ids")


class TestGraniteMultiTurnMasking(unittest.TestCase):
    """Rigorous boundary tests for multi-turn label masking with the real
    Granite tokenizer and thinking template.

    Requires HF_ASSETS_PATH and DATA_PATH.
    Skips if any variable is absent.

    These tests guard against off-by-one errors by independently computing
    the expected assistant token range for each turn and asserting:
      - label at start is a real token (first assistant token is trained)
      - label at start-1 is IGNORE_INDEX (fence before the turn)
      - label at end-1 is a real token (last token of the turn is trained)
      - label at end is IGNORE_INDEX (fence after, for non-final turns)
    """

    _tokenizer = None
    _IGNORE_INDEX = None

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        load_dotenv()
        ckpt_path = os.getenv("HF_ASSETS_PATH")
        if ckpt_path is None:
            return
        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)
        cls._IGNORE_INDEX = IGNORE_INDEX

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if os.getenv("HF_ASSETS_PATH") is None:
            self.skipTest("HF_ASSETS_PATH not set")

    def _tokenize(self, messages):
        """Return (full_tokens, label_ids) via GraniteSFTDataset._tokenize_sample."""
        from datasets import Dataset

        ds_obj = GraniteSFTDataset(
            dataset=Dataset.from_list([{"messages": messages}]),
            tokenizer=self._tokenizer,
            sample_processor=lambda s: s["messages"],
            seq_len=8192,
            infinite=False,
        )
        result = ds_obj._tokenize_sample({"messages": messages})
        self.assertIsNotNone(result, "Sample was dropped (exceeds seq_len?)")
        _, label_ids = result
        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)
        return full_tokens, label_ids

    def _asst_range(self, messages, turn_idx):
        """Independently compute the (start, end) label_ids range for
        the assistant turn at turn_idx using the same formula as _tokenize_sample."""
        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        prefix_text = self._tokenizer.apply_chat_template(
            messages[:turn_idx], add_generation_prompt=True
        )
        prefix_tokens = self._tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        start = len(prefix_tokens) - 1
        if turn_idx == last_asst_idx:
            full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
            full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
            if full_tokens[-1] != self._tokenizer.eos_id:
                full_tokens.append(self._tokenizer.eos_id)
            end = len(full_tokens) - 1
        else:
            suffix_text = self._tokenizer.apply_chat_template(messages[: turn_idx + 1])
            # rstrip matches _tokenize_sample: exclude the between-turn \n.
            suffix_tokens = self._tokenizer.encode(
                suffix_text.rstrip("\n"), add_bos=True, add_eos=False
            )
            end = len(suffix_tokens) - 1
        return start, end

    def _assert_fence(self, label_ids, start, end, *, is_last):
        IGN = self._IGNORE_INDEX
        self.assertGreater(start, 0, "start must be > 0 to have a fence before it")
        self.assertEqual(
            label_ids[start - 1],
            IGN,
            "label just before assistant start must be masked",
        )
        self.assertNotEqual(
            label_ids[start], IGN, "first assistant token must be trained"
        )
        self.assertNotEqual(
            label_ids[end - 1], IGN, "last assistant token must be trained"
        )
        if not is_last:
            self.assertEqual(
                label_ids[end],
                IGN,
                "label just after intermediate assistant end must be masked",
            )

    def test_single_turn_system_user_assistant_fences(self):
        """3-turn [system, user, assistant]: verify fence positions around the single assistant turn."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        full_tokens, label_ids = self._tokenize(messages)
        start, end = self._asst_range(messages, 2)
        self._assert_fence(label_ids, start, end, is_last=True)
        # System and user fully masked.
        self.assertTrue(
            all(l == self._IGNORE_INDEX for l in label_ids[:start]),
            "System and user tokens must all be masked",
        )

    def test_regression_matches_old_single_boundary_for_three_turn(self):
        """The new per-turn masking must produce bit-identical labels to the
        old _prompt_messages approach for a 3-turn [system, user, assistant]
        sample.  The old approach set mask_end = len(prefix_tokens) - 1 where
        prefix = apply_chat_template([system, user], add_generation_prompt=True).
        """
        messages = [
            {"role": "system", "content": "You are a math assistant."},
            {"role": "user", "content": "Compute 7 * 6."},
            {"role": "assistant", "content": "42"},
        ]
        _, label_ids = self._tokenize(messages)

        # Reproduce the old single-boundary logic directly.
        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        prompt_len = len(prompt_tokens)
        expected = list(full_tokens[1:])
        mask_end = min(max(prompt_len - 1, 0), len(expected))
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        expected[:mask_end] = [IGNORE_INDEX] * mask_end

        self.assertEqual(
            label_ids,
            expected,
            "Multi-turn result must be bit-identical to old single-boundary approach",
        )

    def test_think_token_boundary_unchanged(self):
        """The <think> boundary invariant must hold under the new masking path:
        label at <think> position is masked; label one position later is trained."""

        load_dotenv = __import__("dotenv").load_dotenv
        load_dotenv()
        data_path = os.getenv("DATA_PATH")
        if data_path is None:
            self.skipTest("DATA_PATH not set")

        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not jsonl_files:
            self.skipTest("No .jsonl files found in DATA_PATH")

        with open(jsonl_files[0]) as f:
            record = json.loads(f.readline())
        messages = record["messages"]

        _, label_ids = self._tokenize(messages)
        full_text = self._tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)

        think_id = self._tokenizer.token_to_id("<think>")
        self.assertIsNotNone(think_id)
        think_pos = full_tokens.index(think_id)

        self.assertEqual(
            label_ids[think_pos],
            self._IGNORE_INDEX,
            "label at <think> position must be masked (generation prefix)",
        )
        self.assertNotEqual(
            label_ids[think_pos + 1],
            self._IGNORE_INDEX,
            "label one after <think> must be trained (first reasoning token)",
        )

    def test_multi_turn_with_tool_messages(self):
        """Construct a synthetic multi-turn conversation with a tool result
        and verify that only the two assistant spans are trained on."""
        messages = [
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": "Find info about Python."},
            {"role": "assistant", "content": "Searching now."},
            {"role": "tool", "content": "Python is a programming language."},
            {"role": "assistant", "content": "Python is a high-level language."},
        ]
        full_tokens, label_ids = self._tokenize(messages)
        IGN = self._IGNORE_INDEX

        start1, end1 = self._asst_range(messages, 2)
        start2, end2 = self._asst_range(messages, 4)

        # Pre-first-assistant: masked.
        self.assertTrue(all(l == IGN for l in label_ids[:start1]))
        # First assistant span fences.
        self._assert_fence(label_ids, start1, end1, is_last=False)
        # First assistant span is all trained.
        self.assertTrue(all(l != IGN for l in label_ids[start1:end1]))
        # Tool turn between the two assistant turns: fully masked.
        self.assertTrue(
            all(l == IGN for l in label_ids[end1:start2]),
            "Tool message tokens must be fully masked",
        )
        # Second assistant span fences.
        self._assert_fence(label_ids, start2, end2, is_last=True)
        # Second assistant span is all trained.
        self.assertTrue(all(l != IGN for l in label_ids[start2:end2]))

        # Scaffolding masking verified in test_inter_turn_scaffolding_masked.

    def test_system_and_user_fully_masked_in_multi_turn(self):
        """In a multi-turn conversation, no system or user token should appear
        as a trained label."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        full_tokens, label_ids = self._tokenize(messages)
        IGN = self._IGNORE_INDEX

        start1, end1 = self._asst_range(messages, 2)
        start2, _ = self._asst_range(messages, 4)

        # System/user before first assistant turn: all masked.
        self.assertTrue(all(l == IGN for l in label_ids[:start1]))
        # User turn between assistant turns: all masked.
        self.assertTrue(all(l == IGN for l in label_ids[end1:start2]))

    def test_inter_turn_scaffolding_masked(self):
        """The inter-turn separator emitted by apply_chat_template after each
        assistant turn (a trailing \\n for the Granite / ChatML template) must
        be masked, not trained on.

        Rationale: during inference the model generates up to and including the
        turn-end delimiter (<|im_end|>) and then stops.  The \\n that follows is
        injected by the inference framework as structural scaffolding; the model
        never produces it.  _tokenize_sample uses rstrip("\\n") on the suffix
        text before measuring the boundary, which is correct for all practical
        templates (ChatML, Llama, Mistral all use trailing newlines as
        inter-turn separators).  This test pins that contract: the last trained
        label for an intermediate assistant turn is <|im_end|>, and the very
        next label is IGNORE_INDEX.
        """
        im_end_id = self._tokenizer.token_to_id("<|im_end|>")
        self.assertIsNotNone(im_end_id, "<|im_end|> must be a known token")

        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        _, label_ids = self._tokenize(messages)
        start1, end1 = self._asst_range(messages, 1)

        # Last trained label of the intermediate assistant turn must be <|im_end|>.
        self.assertEqual(
            label_ids[end1 - 1],
            im_end_id,
            f"label_ids[end1-1] should be im_end_id={im_end_id}, "
            f"got {label_ids[end1 - 1]}",
        )
        # The inter-turn separator immediately after must be masked.
        self.assertEqual(
            label_ids[end1],
            self._IGNORE_INDEX,
            "Inter-turn separator after <|im_end|> must be IGNORE_INDEX",
        )


class TestGraniteSFT7MBalanced(unittest.TestCase):
    """Tokenization correctness tests for the 7M-Balanced SFT dataset.

    This dataset is heterogeneous in ways the existing tests do not cover:
    - Multi-turn records with 2–8+ assistant turns
    - Mixed thinking/no-thinking within a single conversation
    - Tool-message patterns ([user, asst, tool, asst, ...])

    The most novel coverage is the no-thinking boundary invariant: when an
    assistant turn has no ``reasoning_content`` the template renders
    ``<think></think>{content}``, and the masking boundary ``start =
    len(prefix_tokens) - 1`` lands such that ``input_ids[start] == </think>``
    and ``label_ids[start]`` is the first content token (trained).

    Requires HF_ASSETS_PATH and DATA_PATH_7M_BALANCED_TEST_SAMPLE.  Skips if
    either is absent.  Generate the test sample with::

        python -m torchtitan.models.granite.scripts.gen_test_data \\
            --source /path/to/train_v1_7m_balanced
    """

    _tokenizer = None
    _IGNORE_INDEX = None
    _records = None
    _single_turn = None
    _multi_turn = None
    _mixed_rc = None
    _no_rc = None
    _tool_recs = None

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX

        load_dotenv()
        ckpt_path = os.getenv("HF_ASSETS_PATH")
        data_path = os.getenv("DATA_PATH_7M_BALANCED_TEST_SAMPLE")
        if any(v is None for v in (ckpt_path, data_path)):
            return

        cls._tokenizer = HuggingFaceTokenizer(tokenizer_path=ckpt_path)
        cls._IGNORE_INDEX = IGNORE_INDEX

        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not jsonl_files:
            return

        records = []
        with open(jsonl_files[0]) as f:
            for line in f:
                records.append(json.loads(line))
        cls._records = records

        def _n_asst(r):
            return sum(1 for m in r["messages"] if m["role"] == "assistant")

        def _asst_msgs(r):
            return [m for m in r["messages"] if m["role"] == "assistant"]

        cls._single_turn = [r for r in records if _n_asst(r) == 1]
        cls._multi_turn = [r for r in records if _n_asst(r) > 1]
        cls._mixed_rc = [
            r
            for r in cls._multi_turn
            if any("reasoning_content" in m for m in _asst_msgs(r))
            and any("reasoning_content" not in m for m in _asst_msgs(r))
        ]
        cls._no_rc = [
            r
            for r in records
            if not any("reasoning_content" in m for m in r["messages"])
        ]
        cls._tool_recs = [
            r for r in records if any(m["role"] == "tool" for m in r["messages"])
        ]

    def setUp(self):
        from dotenv import load_dotenv

        load_dotenv()
        if any(
            os.getenv(v) is None
            for v in ("HF_ASSETS_PATH", "DATA_PATH_7M_BALANCED_TEST_SAMPLE")
        ):
            self.skipTest(
                "HF_ASSETS_PATH and DATA_PATH_7M_BALANCED_TEST_SAMPLE must both be set"
            )

    # ------------------------------------------------------------------ helpers

    def _tokenize(self, messages):
        """Return (full_tokens, label_ids), or None if the sample is too long."""
        from datasets import Dataset

        ds_obj = GraniteSFTDataset(
            dataset=Dataset.from_list([{"messages": messages}]),
            tokenizer=self._tokenizer,
            sample_processor=lambda s: s["messages"],
            seq_len=32768,
            infinite=False,
        )
        result = ds_obj._tokenize_sample({"messages": messages})
        if result is None:
            return None
        # Reconstruct full_tokens from input_ids rather than re-running
        # apply_chat_template — guarantees the oracle uses exactly the same
        # token sequence that _tokenize_sample produced.
        input_ids, label_ids = result
        full_tokens = list(input_ids) + [self._tokenizer.eos_id]
        return full_tokens, label_ids

    def _asst_range(self, messages, turn_idx):
        """Independently recompute (start, end) for the assistant turn at turn_idx."""
        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        _kwargs = {"truncate_history_thinking": True}
        prefix_text = self._tokenizer.apply_chat_template(
            messages[:turn_idx], add_generation_prompt=True, **_kwargs
        )
        prefix_tokens = self._tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        start = len(prefix_tokens) - 1
        full_text = self._tokenizer.apply_chat_template(messages, **_kwargs).rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)
        if turn_idx == last_asst_idx:
            end = len(full_tokens) - 1
        else:
            im_end_id = self._tokenizer.token_to_id("<|im_end|>")
            end = full_tokens.index(im_end_id, start + 1)
        return start, end

    def _assert_fence(self, label_ids, start, end, *, is_last):
        IGN = self._IGNORE_INDEX
        self.assertGreater(start, 0, "start must be > 0 to have a fence before it")
        self.assertEqual(
            label_ids[start - 1],
            IGN,
            "label just before assistant start must be masked",
        )
        self.assertNotEqual(
            label_ids[start], IGN, "first assistant token must be trained"
        )
        self.assertNotEqual(
            label_ids[end - 1], IGN, "last assistant token must be trained"
        )
        if not is_last:
            self.assertEqual(
                label_ids[end],
                IGN,
                "label just after intermediate assistant end must be masked",
            )

    @staticmethod
    def _asst_indices(messages):
        return [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    # ------------------------------------------------------------------ tests

    def test_validate_all_records(self):
        from torchtitan.hf_datasets.text_datasets import ChatDataset

        for i, r in enumerate(self._records):
            try:
                ChatDataset._validate_messages(r["messages"])
            except Exception as e:
                self.fail(f"Record {i} failed _validate_messages: {e}")

    def test_last_message_always_assistant(self):
        for i, r in enumerate(self._records):
            self.assertEqual(
                r["messages"][-1]["role"],
                "assistant",
                f"Record {i}: last message role is {r['messages'][-1]['role']!r}",
            )

    def test_dataset_has_expected_heterogeneity(self):
        self.assertGreater(
            len(self._multi_turn), 0, "No multi-turn records in test sample"
        )
        self.assertGreater(len(self._mixed_rc), 0, "No mixed-rc records in test sample")
        self.assertGreater(
            len(self._tool_recs), 0, "No tool-message records in test sample"
        )
        self.assertGreater(len(self._no_rc), 0, "No all-no-rc records in test sample")

    def test_no_thinking_turn_boundary(self):
        """No-thinking turns: </think> sits at input_ids[start]; first content token trained.

        When an assistant turn has no ``reasoning_content`` the template renders
        ``<|im_start|>assistant\\n<think></think>{content}``.  The generation
        prompt ends with ``<think>\\n``, so ``start = len(prefix_tokens) - 1``
        points to the position where the full sequence has ``</think>`` rather
        than ``\\n``.  This pins that the mismatch is resolved correctly.
        """
        end_think_id = self._tokenizer.token_to_id("</think>")
        self.assertIsNotNone(end_think_id, "</think> must be a registered token")
        IGN = self._IGNORE_INDEX

        tested = 0
        for r in self._no_rc:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            full_tokens, label_ids = result
            for turn_idx in self._asst_indices(messages):
                start, _ = self._asst_range(messages, turn_idx)
                self.assertEqual(
                    full_tokens[start],
                    end_think_id,
                    f"No-thinking turn {turn_idx}: expected </think> at input_ids[{start}], "
                    f"got token id {full_tokens[start]}",
                )
                self.assertEqual(
                    label_ids[start - 1],
                    IGN,
                    "Fence before no-thinking turn must be masked",
                )
                self.assertNotEqual(
                    label_ids[start], IGN, "First content token must be trained"
                )
                tested += 1
        self.assertGreater(tested, 0, "No no-thinking assistant turns were verified")

    def test_thinking_turn_boundary(self):
        """Thinking/no-thinking boundary token invariant.

        The full sequence is built with ``truncate_history_thinking=True``, so
        only the *last* assistant turn preserves its thinking trace.  For that
        turn, ``input_ids[start]`` is the ``\\n`` that follows ``<think>``.

        All other turns — whether they have ``reasoning_content`` or not — have
        thinking stripped (or absent), so the full sequence has
        ``<think></think>`` there and ``input_ids[start] == </think>``.
        """
        end_think_id = self._tokenizer.token_to_id("</think>")
        IGN = self._IGNORE_INDEX

        thinking_recs = [
            r
            for r in self._records
            if any(
                "reasoning_content" in m
                for m in r["messages"]
                if m["role"] == "assistant"
            )
        ]
        tested_last = 0
        tested_intermediate = 0
        for r in thinking_recs:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            full_tokens, label_ids = result
            asst_idxs = self._asst_indices(messages)
            last_asst_idx = max(asst_idxs)
            for turn_idx in asst_idxs:
                start, _ = self._asst_range(messages, turn_idx)
                is_last = turn_idx == last_asst_idx
                has_rc = "reasoning_content" in messages[turn_idx]
                if is_last and has_rc:
                    # Last turn with thinking: <think>\n preserved → start points at \n
                    self.assertNotEqual(
                        full_tokens[start],
                        end_think_id,
                        f"Last thinking turn {turn_idx}: expected \\n, not </think>, "
                        f"at input_ids[{start}]",
                    )
                    tested_last += 1
                elif has_rc:
                    # Intermediate turn with thinking: stripped → </think> at start
                    self.assertEqual(
                        full_tokens[start],
                        end_think_id,
                        f"Intermediate thinking turn {turn_idx}: expected </think> at "
                        f"input_ids[{start}] (thinking stripped by truncate_history_thinking)",
                    )
                    tested_intermediate += 1
                self.assertEqual(
                    label_ids[start - 1], IGN, "Fence before turn must be masked"
                )
                self.assertNotEqual(
                    label_ids[start], IGN, "First content token must be trained"
                )
        self.assertGreater(tested_last, 0, "No last-turn thinking turns verified")
        self.assertGreater(
            tested_intermediate, 0, "No intermediate thinking turns verified"
        )

    def test_mixed_thinking_multi_turn_fences(self):
        """Mixed-thinking records: fence holds and correct boundary token for every turn.

        ``truncate_history_thinking=True`` (the default) strips thinking from
        all but the last assistant turn in the full sequence.  So:
        - Last turn **with** ``reasoning_content``: ``input_ids[start] == \\n``
          (thinking preserved).
        - Every other turn (intermediate thinking or any no-thinking turn):
          ``input_ids[start] == </think>`` (empty or absent thinking block).
        """
        end_think_id = self._tokenizer.token_to_id("</think>")
        IGN = self._IGNORE_INDEX

        tested_records = 0
        for r in self._mixed_rc:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            full_tokens, label_ids = result
            asst_idxs = self._asst_indices(messages)
            last_asst_idx = max(asst_idxs)
            for turn_idx in asst_idxs:
                start, end = self._asst_range(messages, turn_idx)
                is_last = turn_idx == last_asst_idx
                has_rc = "reasoning_content" in messages[turn_idx]
                self._assert_fence(label_ids, start, end, is_last=is_last)
                if is_last and has_rc:
                    self.assertNotEqual(
                        full_tokens[start],
                        end_think_id,
                        f"Last thinking turn {turn_idx}: expected \\n at start={start}",
                    )
                else:
                    self.assertEqual(
                        full_tokens[start],
                        end_think_id,
                        f"Turn {turn_idx} (is_last={is_last}, has_rc={has_rc}): "
                        f"expected </think> at start={start}",
                    )
            tested_records += 1
        self.assertGreater(tested_records, 0, "No mixed-rc records were verified")

    def test_fence_invariant_all_turns_multi_turn(self):
        """Fence invariant holds for every assistant turn in every multi-turn record."""
        tested = 0
        for r in self._multi_turn:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            _, label_ids = result
            asst_idxs = self._asst_indices(messages)
            last_asst_idx = max(asst_idxs)
            for turn_idx in asst_idxs:
                start, end = self._asst_range(messages, turn_idx)
                self._assert_fence(
                    label_ids, start, end, is_last=(turn_idx == last_asst_idx)
                )
                tested += 1
        self.assertGreater(tested, 0, "No multi-turn assistant turns were verified")

    def test_inter_turn_gap_fully_masked(self):
        """All labels between consecutive assistant spans are IGNORE_INDEX."""
        IGN = self._IGNORE_INDEX

        tested = 0
        for r in self._multi_turn:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            _, label_ids = result
            asst_idxs = self._asst_indices(messages)
            for k in range(len(asst_idxs) - 1):
                _, end_k = self._asst_range(messages, asst_idxs[k])
                start_next, _ = self._asst_range(messages, asst_idxs[k + 1])
                trained_in_gap = [l for l in label_ids[end_k:start_next] if l != IGN]
                self.assertEqual(
                    trained_in_gap,
                    [],
                    f"Gap between turns {k} and {k + 1} has {len(trained_in_gap)} trained labels",
                )
                tested += 1
        self.assertGreater(tested, 0, "No inter-turn gaps were verified")

    def test_im_end_is_last_trained_in_intermediate_turn(self):
        """For every intermediate assistant turn, <|im_end|> is the last trained label."""
        im_end_id = self._tokenizer.token_to_id("<|im_end|>")
        self.assertIsNotNone(im_end_id, "<|im_end|> must be a known token")

        tested = 0
        for r in self._multi_turn:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            _, label_ids = result
            asst_idxs = self._asst_indices(messages)
            last_asst_idx = max(asst_idxs)
            for turn_idx in asst_idxs:
                if turn_idx == last_asst_idx:
                    continue
                _, end = self._asst_range(messages, turn_idx)
                self.assertEqual(
                    label_ids[end - 1],
                    im_end_id,
                    f"Intermediate turn {turn_idx}: last trained label should be <|im_end|>",
                )
                tested += 1
        self.assertGreater(tested, 0, "No intermediate assistant turns were verified")

    def test_tool_turns_fully_masked(self):
        """All labels outside assistant spans are IGNORE_INDEX in tool-message records."""
        IGN = self._IGNORE_INDEX

        tested = 0
        for r in self._tool_recs:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            _, label_ids = result
            asst_idxs = self._asst_indices(messages)
            asst_spans = [self._asst_range(messages, i) for i in asst_idxs]
            # Check pre-first-span and all inter-span gaps.
            non_asst_regions = [(0, asst_spans[0][0])]
            non_asst_regions += [
                (asst_spans[k][1], asst_spans[k + 1][0])
                for k in range(len(asst_spans) - 1)
            ]
            for lo, hi in non_asst_regions:
                if lo >= hi:
                    continue
                trained = [l for l in label_ids[lo:hi] if l != IGN]
                self.assertEqual(
                    trained,
                    [],
                    f"Non-assistant region [{lo}, {hi}) has {len(trained)} trained labels",
                )
            tested += 1
        self.assertGreater(tested, 0, "No tool-message records were verified")

    def test_eos_trained_in_last_turn(self):
        """Last label in every tokenized sample is the EOS token and is trained."""
        eos_id = self._tokenizer.eos_id

        tested = 0
        for r in self._records:
            messages = r["messages"]
            result = self._tokenize(messages)
            if result is None:
                continue
            _, label_ids = result
            self.assertEqual(label_ids[-1], eos_id, "Last label must be the EOS token")
            self.assertNotEqual(
                label_ids[-1], self._IGNORE_INDEX, "EOS must be trained"
            )
            tested += 1
        self.assertGreater(tested, 0, "No records were verified")


if __name__ == "__main__":
    unittest.main()
