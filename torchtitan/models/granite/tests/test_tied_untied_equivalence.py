"""E2E test: tied and untied HF asset checkpoints produce identical logits.

Loads pre-built granite 4.1 assets (tied and _untied variants) via HuggingFace
from_pretrained and asserts bitwise-identical forward pass outputs. The _untied
assets were created by cloning embed_tokens.weight into an independent lm_head.weight.
"""

import os

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

from dtest import DTest


def _get_paths(size: str):
    from dotenv import load_dotenv

    load_dotenv()
    tied = os.getenv(f"HF_ASSETS_PATH_{size}")
    untied = os.getenv(f"HF_ASSETS_PATH_{size}_UNTIED")
    if tied is None or untied is None:
        pytest.skip(
            f"HF_ASSETS_PATH_{size} and/or HF_ASSETS_PATH_{size}_UNTIED not set"
        )
    return tied, untied


def _hf_logits(path: str, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    with torch.no_grad():
        logits = model(tokens.to(device)).logits.cpu()
    del model
    torch.cuda.empty_cache()
    return logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_3b_logits_match():
    tied_path, untied_path = _get_paths("3B")
    tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
    device = torch.device("cuda:0")

    tied_logits = _hf_logits(tied_path, tokens, device)
    untied_logits = _hf_logits(untied_path, tokens, device)

    torch.testing.assert_close(tied_logits, untied_logits, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_8b_logits_match():
    tied_path, untied_path = _get_paths("8B")
    tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
    device = torch.device("cuda:0")

    tied_logits = _hf_logits(tied_path, tokens, device)
    untied_logits = _hf_logits(untied_path, tokens, device)

    torch.testing.assert_close(tied_logits, untied_logits, atol=0.0, rtol=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class Test30BLogitsMatch(DTest):
    default_world_size = 8

    def _fsdp_wrap(self, hf_model, mesh):
        for block in hf_model.model.layers:
            block.to(self.device)
            fully_shard(block, mesh=mesh)
        hf_model.model.embed_tokens.to(self.device)
        hf_model.model.norm.to(self.device)
        hf_model.lm_head.to(self.device)
        fully_shard(
            [hf_model.model.embed_tokens, hf_model.model.norm, hf_model.lm_head],
            mesh=mesh,
        )
        fully_shard(hf_model, mesh=mesh)

    def _logits(self, path: str, tokens: torch.Tensor, mesh) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        self._fsdp_wrap(model, mesh)
        with torch.no_grad():
            logits = model(tokens).logits.cpu()
        del model
        torch.cuda.empty_cache()
        return logits

    def test_30b_logits_match(self):
        tied_path, untied_path = _get_paths("30B")
        mesh = init_device_mesh("cuda", (self.world_size,))
        tokens = torch.arange(1, 9, dtype=torch.long).unsqueeze(0).to(self.device)

        tied_logits = self._logits(tied_path, tokens, mesh)
        untied_logits = self._logits(untied_path, tokens, mesh)

        torch.testing.assert_close(tied_logits, untied_logits, atol=0.0, rtol=0.0)
