"""
Visual debugging tool for Granite SFT tokenization and label masking.

Three rows per token: input ID / decoded subword / label ID.
Middle row is red when label is -100 (masked, no loss).
"""

import argparse
import json
import os
import shutil
import tempfile

from datasets import Dataset
from dotenv import load_dotenv

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import IGNORE_INDEX
from torchtitan.models.granite.sft_dataset import GraniteSFTDataLoader, GraniteSFTDataset

parser = argparse.ArgumentParser()
parser.add_argument("--plain", action="store_true", help="Print escaped token text only; no table or colors.")
args = parser.parse_args()

load_dotenv()
hf_assets_path = os.getenv("HF_ASSETS_PATH")
if not hf_assets_path:
    raise EnvironmentError("HF_ASSETS_PATH not set in .env or environment")

SAMPLE_MSGS = [
    [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Tell me a joke."},
        {
            "role": "assistant",
            "content": "Why did the chicken cross the road?",
            "reasoning_content": "The user want me to tell them a funny joke. I should be consise and hilarious.",
        },
    ],
    [
        {"role": "system", "content": ""},
        {"role": "user", "content": "What is 1 + 1?"},
        {
            "role": "assistant",
            "content": "The answer is: 2",
            "reasoning_content": "This is a simple question. I will answer directly and succinctly",
        },
    ],
]

sample_proc = lambda s: s["messages"]  # noqa: E731

tokenizer = HuggingFaceTokenizer(tokenizer_path=hf_assets_path)

# Probe via _tokenize_sample to sum token counts and set a tight seq_len.
probe_ds = GraniteSFTDataset(
    dataset=Dataset.from_list([{"messages": msgs} for msgs in SAMPLE_MSGS]),
    tokenizer=tokenizer,
    sample_processor=sample_proc,
    seq_len=8192,
    infinite=False,
)
n_total = 0
for msgs in SAMPLE_MSGS:
    result = probe_ds._tokenize_sample({"messages": msgs})
    assert result is not None, f"Sample dropped (exceeds seq_len?): {msgs}"
    n_total += len(result[0])

N_PAD = 8
seq_len = n_total + N_PAD

# Write samples to a temp JSONL file and build via GraniteSFTDataLoader,
# matching the exact path used during training.
with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as tmp:
    for msgs in SAMPLE_MSGS:
        tmp.write(json.dumps({"messages": msgs}) + "\n")
    tmp.flush()

    config = GraniteSFTDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": tmp.name, "split": "train"},
        sample_processor=sample_proc,
        infinite=False,
    )
    loader = GraniteSFTDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=1,
    )
    batch, labels = next(iter(loader))

input_ids = batch["input"][0].tolist()
label_ids = labels[0].tolist()

if args.plain:
    # Each packed sample contributes input_ids = full_tokens[:-1], so the
    # closing EOS of every sample lives in label_ids (not input_ids) due to
    # the LM next-token prediction shift. Reconstruct the full token sequence
    # by inserting each boundary EOS: at position i, label_ids[i] is a
    # boundary EOS when it's not IGNORE_INDEX and differs from input_ids[i+1].
    last_content = max(i for i, l in enumerate(label_ids) if l != IGNORE_INDEX)
    full_toks: list[int] = []
    for i in range(last_content + 1):
        inp, lbl = input_ids[i], label_ids[i]
        full_toks.append(inp)
        if lbl != IGNORE_INDEX:
            next_inp = input_ids[i + 1] if i + 1 <= last_content else None
            if next_inp is None or lbl != next_inp:
                full_toks.append(lbl)
    print(
        "".join(
            tokenizer.tokenizer.decode([t], skip_special_tokens=False)
            .encode("unicode_escape")
            .decode("ascii")
            for t in full_toks
        )
    )
    raise SystemExit(0)

RED = "\033[31m"
RESET = "\033[0m"

entries = []
for i, (tok_id, label) in enumerate(zip(input_ids, label_ids)):
    tok_str = tokenizer.tokenizer.decode([tok_id], skip_special_tokens=False).encode("unicode_escape").decode("ascii")
    id_str = str(tok_id)
    lbl_str = str(label)
    col_w = max(len(id_str), len(tok_str), len(lbl_str))
    # Red if this token is not itself a prediction target: the first token
    # is never predicted, and token i is predicted iff label_ids[i-1] != -100.
    masked = i == 0 or label_ids[i - 1] == IGNORE_INDEX
    entries.append((id_str, tok_str, lbl_str, col_w, masked))

print(
    "Each column is one token: input ID / decoded subword / label ID.\n"
    "Middle row is red when this token is NOT being predicted (first token, or preceding label was -100).\n"
)

term_width = shutil.get_terminal_size().columns
GAP = 1
i = 0
while i < len(entries):
    block = []
    used = 0
    while i < len(entries):
        *_, col_w, _ = entries[i]
        needed = col_w if not block else col_w + GAP
        if block and used + needed > term_width:
            break
        block.append(entries[i])
        used += needed
        i += 1

    id_row = tok_row = lbl_row = ""
    for j, (id_str, tok_str, lbl_str, col_w, masked) in enumerate(block):
        sep = " " * GAP if j > 0 else ""
        tok_cell = f"{RED}{tok_str.ljust(col_w)}{RESET}" if masked else tok_str.ljust(col_w)
        id_row  += sep + id_str.ljust(col_w)
        tok_row += sep + tok_cell
        lbl_row += sep + lbl_str.ljust(col_w)

    print(id_row)
    print(tok_row)
    print(lbl_row)
    print()

total = len(input_ids)
masked_count = sum(1 for l in label_ids if l == IGNORE_INDEX)
print(f"total={total}  masked(red)={masked_count}  trained={total - masked_count}")
