"""
Comparing the HF and torchitan impl generations.
"""
import argparse
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel
from torchtitan.models.granite.state_dict_adapter import GraniteStateDictAdapter

def _greedy_generate(model, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
    tokens = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
        next_tok = logits[0, -1, :].argmax().reshape(1, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens


load_dotenv()
CKPT = os.getenv("GRANITE_BASE_CKPT_PATH")
if not CKPT:
    raise EnvironmentError("GRANITE_BASE_CKPT_PATH not set in .env or environment")

_DEFAULT_PROMPT = "The three laws of thermodynamics are:"
_parser = argparse.ArgumentParser()
_parser.add_argument("prompt", nargs="?", default=_DEFAULT_PROMPT)
_parser.add_argument("--max-new-tokens", type=int, default=100)
_args = _parser.parse_args()

PROMPT = _args.prompt
MAX_NEW = _args.max_new_tokens

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

tokenizer = AutoTokenizer.from_pretrained(CKPT)
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
prompt_len = input_ids.shape[1]
print(f"prompt ({prompt_len} tokens): {PROMPT!r}\n")

# ── HF generation ─────────────────────────────────────────────────────────────
print("[1/2] HF model …")
hf_model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).to(device).eval()
hf_sd_cpu = {k: v.cpu() for k, v in hf_model.state_dict().items()}

with torch.no_grad():
    hf_out = hf_model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False)
hf_response = tokenizer.decode(hf_out[0, prompt_len:], skip_special_tokens=True)

del hf_model
if device == "cuda":
    torch.cuda.empty_cache()

# ── TorchTitan generation ─────────────────────────────────────────────────────
print("[2/2] TorchTitan model …")
config = granite_configs["8B"]()
adapter = GraniteStateDictAdapter(config, hf_assets_path=CKPT)
tt_sd = adapter.from_hf(hf_sd_cpu)

with torch.device("cpu"):
    tt_model = GraniteModel(config)
tt_model.to_empty(device=device)
tt_model.init_states()
tt_model.load_state_dict(tt_sd, strict=True)
tt_model.eval()

tt_out = _greedy_generate(tt_model, input_ids, max_new_tokens=MAX_NEW)
tt_response = tokenizer.decode(tt_out[0, prompt_len:], skip_special_tokens=True)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"PROMPT: {PROMPT!r}")
print("=" * 70)
print(f"\n[HF]\n{hf_response}")
print(f"\n[TorchTitan]\n{tt_response}")
print("=" * 70)
print(f"\nExact match: {hf_response == tt_response}")
