# Granite SFT

## Float8 Training

Float8 quantized training is supported via `torchao` (>= 0.18.0, built from source for
GB200). Config registry entries:

- `granite_debugmodel_float8` — tensorwise with FSDP all-gather (single-GPU unit tests)
- `granite_debugmodel_float8_rowwise` — rowwise (multi-GPU integration tests)
- `granite_4_1_8b_sft_pretokenized_float8_filteroutput` — tensorwise + all-gather, output filtered
- `granite_4_1_8b_sft_pretokenized_float8_rowwise` — rowwise recipe

### Known issue: tensorwise + FSDP float8 all-gather + weight tying

Tensorwise with `enable_fsdp_float8_all_gather=True` (without filtering `output`)
crashes during FSDP lazy init:

```
RuntimeError: Attempted to access the data pointer on an invalid python storage.
  File "torch/distributed/fsdp/_fully_shard/_fsdp_param.py", line 950, in reset_sharded_param
```

Root cause: FSDP2's float8 all-gather path calls `reset_sharded_param()` which accesses
the storage data pointer of the parameter. Granite's weight tying
(`tok_embeddings.weight = output.weight`) results in the same underlying storage being
referenced by two FSDP param groups. When one group processes it, the other's reference
becomes invalid.

**Why only `output` needs filtering**: `tok_embeddings` is `nn.Embedding`, not `nn.Linear`.
`Float8LinearConverter` (via torchao's `swap_linear_with_float8_linear`) only converts
`nn.Linear` modules, so the embedding is never a conversion target. Filtering `output`
prevents it from becoming `Float8Linear`, keeping both sides of the weight tie as plain
parameters that FSDP2 handles correctly.

**Workaround**: Use the rowwise recipe (`granite_4_1_8b_sft_pretokenized_float8_rowwise`)
which does not use float8 all-gather and works correctly. Alternatively, filter the output
layer from float8 conversion (`filter_fqns=["output"]`) to avoid the double-registration.

Observed: torch 2.13.0.dev20260422+cu130, torchao 0.18.0+git6367fd63, 4xGB200.

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

### FullThinkingStrategy

Uses `truncate_history_thinking=False` and loss-unmasks all assistant turns that have
`reasoning_content`. The full thinking context from every turn is preserved in the token
sequence — matching an agentic inference setup where the model sees full conversation
history including prior reasoning.

**Loss masking rules:**
- Assistant turns WITH `reasoning_content`: unmasked (reasoning + response + `</think>` +
  `<|im_end|>`)
- Assistant turns WITHOUT `reasoning_content`: masked (present as context only)
- All other roles (system, user, tool): masked

**Why no-reasoning turns are masked:** The template renders them as
`<think></think>{response}` — a token sequence that never occurs at inference time (the
model always receives `<think>\n` from the generation prompt, not adjacent
`<think></think>`). Unmasking would train prediction under a context the model never sees
during generation.

**Trailing non-assistant turns** are handled identically to `TruncateLastStrategy` (dropped
before tokenization).

**Trade-offs vs other strategies:**
- vs `truncate_last`: sequences are longer (thinking not stripped from history), more
  assistant turns contribute training signal, but fewer examples fit per packed sequence
- vs `backbone_suffix`: simpler (no flex attention needed), but historical thinking
  occupies regular sequence positions and competes with training content for seq_len budget

### BackboneSuffixStrategy

Backbone identical to `TruncateLastStrategy` (uses `truncate_history_thinking=True`, only
the last assistant turn is loss-unmasked). Additionally produces per-turn suffix sequences
that recover thinking traces from historical assistant turns. Designed for flex attention
where suffixes are attended to without consuming backbone positions.

Output columns: `input_ids`, `labels`, `positions`, `suffix_starts`, `insertion_limits`,
`n_tokens`. The packing layer expands `suffix_starts`/`insertion_limits` into per-token
tensors at runtime for the attention mask.

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

## TODO

### Loss Weighting for Packed Sequences

**Current behavior (token-uniform):** The loss is `cross_entropy(reduction="sum") /
global_valid_tokens`, where `global_valid_tokens` counts all non-masked labels across
all DP ranks. Every trained token contributes equally to the gradient regardless of
which packed sequence it belongs to. Short sequences get less per-example influence
than long ones — a 50-token sequence has 10x less impact than a 500-token sequence
packed alongside it.

**Per-sequence-uniform weighting:** Each token in sequence `i` (with `N_i` trained
tokens) gets weight `1/N_i`. The denominator becomes the total number of sequences
(sum of weights = number of sequences across all ranks). This makes each sequence
contribute equally regardless of length — equivalent to training on sequences in
isolation and averaging gradients. Implementable dynamically at pack time from labels
and position resets (no pretokenization schema changes needed).

**Open question — backbone+suffix interaction:** When `sequence_uniform` is combined
with the backbone+suffix packing strategy, how should suffix tokens be weighted
relative to backbone tokens?

- **Per-segment normalization:** Treat backbone and each suffix as independent segments,
  each with its own `1/N_segment` weight. A sample with 3 suffixes contributes 4x the
  influence of a backbone-only sample. Matches isolated-training semantics per-segment
  but creates variable per-example influence.
- **Suffix discount factor:** A configurable `suffix_loss_scale` (default 1.0) multiplies
  all suffix token weights. At 0.0, suffixes provide attention context only (no gradient
  signal). Simple knob to tune backbone vs suffix signal balance.
- **Fixed per-example budget:** Each example gets total weight 1.0 regardless of segment
  count. Tokens within the example split that budget proportionally. Strictly
  sequence-uniform across examples, but adding suffixes dilutes backbone signal.

