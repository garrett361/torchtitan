# Granite SFT

## Chat Template Behavior

Template: `chat_template.jinja` (ChatML-style with thinking support).

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `enable_thinking` | `True` | Generation prompt ends with `<\|im_start\|>assistant\n<think>\n` (model fills reasoning). When `False`, emits `<think></think>` (skip reasoning). |
| `truncate_history_thinking` | `True` | Strips reasoning from "historical" assistant turns (see below). |
| `low_effort` | `False` | Appends `{reasoning effort: low}` to the final user message. |

### Turn structure

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

Assistant turns with `reasoning_content`:
```
<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n{response}<|im_end|>\n
```

Without `reasoning_content`, the template injects empty tags **with no newlines**:
```
<|im_start|>assistant\n<think></think>{response}<|im_end|>\n
```

Note the three distinct forms:
- **Full thinking**: `<think>\n{reasoning}\n</think>\n{response}` — newlines flank the reasoning AND follow `</think>`
- **Truncated (from full)**: `<think></think>\n{response}` — reasoning removed, but the `\n` after `</think>` survives (it was part of the response prefix in the original)
- **No reasoning / thinking off**: `<think></think>{response}` — no newlines at all between tags and response

### `truncate_history_thinking` semantics

The template computes `last_user_idx` by scanning `loop_messages` for `role == "user"`.
`role == "tool"` messages do not update this index. Any assistant turn at
`loop.index0 < last_user_idx` is "historical" and has its thinking stripped.

Note on tool rendering: `tool` messages are wrapped in `<|im_start|>user … <|im_end|>`
blocks in the output — that is, they appear inside a user-role ChatML turn, not as a
distinct role. Consecutive tool messages share one such block. But because `last_user_idx`
is computed from the original message list (not the rendered output), tool messages have
no effect on it.

**Pure multi-turn** (multiple user messages):
```
user0 → assistant0(think+response) → user1 → assistant1(think+response)
```
`last_user_idx = 2` (index of `user1` in loop_messages). `assistant0` is at index 1
(`< 2`) → stripped to `<think></think>response`. `assistant1` is at index 3 (`> 2`)
→ thinking preserved.

**Tool-use chain** (single user message):
```
user → assistant(think+tool_call) → tool → assistant(think+tool_call) → tool → assistant(think+response)
```
`last_user_idx = 0` (only the initial user message). Every assistant turn has
`loop.index0 > 0`, so **no thinking is ever stripped** — the full agentic loop is
treated as a single "current turn."

Training implication: for tool-call records with a single initial user message, all
intermediate tool-call turns keep their full thinking in the rendered training sequence.
This is intentional: each call represents real model output, so all thinking is trained.
Records where intermediate tool-call turns carry `reasoning_content` produce training
sequences where that thinking is preserved, which diverges from the inference behavior for
pure multi-turn records and confuses tokenization tests that assume uniform stripping.

### BPE boundary guarantee

The pre-tokenizer regex splits on `\s*[\r\n]+` (newlines are always their own word) and
`<think>`/`</think>` are added tokens (always atomic split points). This means the
tokenization of response text after `</think>\n` is identical to after `</think>` alone —
enabling reconstruction of truncated sequences from the full tokenization via token-level
splicing without re-tokenization.

## Tokenization Strategy

Strategy is chosen at pre-tokenization time and recorded in the output manifest
(`manifest["strategy"]`). The dataloader dispatches to the matching dataset class
at runtime.

### TruncateLastStrategy

Labels only the final assistant turn. All earlier turns (user, tool, and intermediate
assistant) are masked (`IGNORE_INDEX`). Uses `truncate_history_thinking=True`, matching
the vLLM/SGLang inference default.

**Trailing non-assistant turns** (tool-last, user-last) are accepted. In both cases,
messages after the last assistant turn are dropped before tokenization:

- **User-last** (e.g. injected "max iterations" scaffolding): a correctness requirement.
  A trailing `user` message shifts `last_user_idx` past the last assistant, causing
  `truncate_history_thinking` to strip that turn's thinking traces. Dropping the trailing
  message restores the correct `last_user_idx`.

- **Tool-last** (agentic trajectories cut off after a tool response): efficiency only.
  Trailing `tool` messages do not affect `last_user_idx` (the template excludes tool role
  from that scan), so thinking is unaffected. The tokens are dropped purely to avoid
  wasting packing budget at training time — they are all-`IGNORE_INDEX` and cannot be
  attended to by any labeled position.

The reasoning in the last assistant turn (the `reasoning_content` / `<think>` block) lives
in the assistant message itself, not in the following tool response. Dropping the trailing
tool response does not affect it.

## Pre-Tokenized Data Pipeline

Two-phase workflow: offline pre-tokenization produces Arrow shards, online dataloader
packs and serves them.

### Offline: `pretokenize_sft.py`

```bash
# Single node
python -m torchtitan.models.granite.scripts.pretokenize_sft \
    --input-dir /path/to/jsonl/ \
    --output-dir /path/to/output/ \
    --tokenizer-path /path/to/tokenizer/ \
    --strategy truncate_last

# Multi-node (each node processes a disjoint subset of shards)
python -m ... --rank 0 --world-size 4
```

Resumable and idempotent. Each JSONL file becomes one Arrow shard under
`output_dir/shards/`. The last rank to finish writes `manifest.json`.

### Online: `GranitePreTokenizedDataLoader`

Reads `manifest.json` and packs examples into fixed-length sequences.

| Config field | Default | Description |
|---|---|---|
| `manifest_path` | (required) | Path to `manifest.json` |
| `packing` | `"buffer"` | `"buffer"` (~99.9% efficiency at 128k) or `"greedy"` (~86%) |
| `buffer_size` | `64` | Lookahead buffer per worker (buffer packing only) |
| `infinite` | `True` | Loop dataset indefinitely |
| `shuffle_in_memory` | `True` | Avoid filesystem contention on shuffle |

Buffer packing maintains a sorted lookahead buffer and fills each sequence with the
largest-fitting example (FIFO anchor + best-fit remainder). Greedy packing appends
examples sequentially until the sequence is full.

Supports multi-worker DataLoader (`num_workers > 0`) — each worker gets a disjoint
slice of the DP-sharded data via `worker_info`.

# Inference Notes

## How Thinking Tokens are Handled in Multi-Turn Serving

Both vLLM and SGLang strip thinking from history **at the chat-template level** before
re-tokenizing the prompt for the next turn. Response tokens are re-encoded in a context
that does NOT include their original thinking trace.

### vLLM (@ `f3fef123`)

Stripping happens inside the model-specific tokenizer's `apply_chat_template`. For
DeepSeek V3.2 (`vllm/tokenizers/deepseek_v32.py:44-46`):

```python
# Historical reasoning content is dropped when a new user message is introduced
drop_thinking = messages[-1]["role"] == "user"
```

Which calls `drop_thinking_messages` (`vllm/tokenizers/deepseek_v32_encoding.py:294-311`)
to pop the `reasoning` field from assistant messages before `last_user_idx`.

vLLM has **no thinking-specific KV eviction**. It relies on prefix caching (token-hash
block reuse) for multi-turn efficiency. Response tokens from prior turns get prefix-cache
hits only if the template renders identically (the Qwen3.6 consistency fix,
QwenLM/Qwen3.6#48).

### SGLang (@ `ece8a1a7`)

Same template-level stripping as vLLM for DeepSeek
(`sglang/.../encoding_dsv32.py:289-307`).

Additionally, SGLang offers a `--strip-thinking-cache` flag
(`sglang/.../server_args.py:454`) which evicts thinking KVs after generation:

```python
# schedule_batch.py:923-928
def _cache_commit_len(self) -> int:
    # Report only the prompt prefix so thinking + answer fall into the
    # overallocated range and are reclaimed by release_kv_cache.
    if get_global_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
        return min(self.kv_committed_len, len(self.origin_input_ids))
    return self.kv_committed_len
```

This evicts **all** output KVs (thinking + response), not selectively. It is a
memory optimization, not a multi-turn semantic one.

### Implication for Training

In production multi-turn inference, response tokens are re-encoded without their
thinking context — their KV representations do NOT carry information from the thinking
trace that produced them. No engine implements "evict thinking KVs but keep response
KVs as-is" (proposed by NVIDIA Dynamo but not shipped).

This creates a subtle train/inference mismatch for any training scheme where response
tokens attend to their thinking trace (as in flex-attention-based thinking masking).
The mismatch is indirect: later turns attend to response token hidden states that
encoded thinking info during training but won't at inference. See the "Tokenization
Strategy" section for how this informs the training design choice.

