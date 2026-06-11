"""E2E DCP checkpoint/resume test: save → load → verify bitwise-identical training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dtest import DTest
from torchtitan.components.loss import IGNORE_INDEX, cross_entropy_loss
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel
from torchtitan.models.granite.pretokenized_dataset import PlannedPackingDataset
from torchtitan.models.granite.scripts.plan_packing import plan_packing
from torchtitan.tools.utils import has_cuda_capability

SEQ_LEN = 512
VOCAB_SIZE = 2048
CONFIG = "debugmodel"
_TOKENIZER_PATH = "tests/assets/tokenizer"

torch._dynamo.config.capture_scalar_outputs = True


def _get_tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)


def _build_model(device, mesh, seed=42, compile=False, float8=False):
    torch.manual_seed(seed)
    config = granite_configs[CONFIG](attn_backend="flex")
    model = GraniteModel(config)
    model.init_states()
    model.to(device)

    if float8:
        world_size = mesh.size()
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=world_size, cp=1, tp=1, pp=1, ep=1, etp=1,
            world_size=world_size,
        )
        float8_cfg = Float8LinearConverter.Config(
            enable_fsdp_float8_all_gather=True,
            precompute_float8_dynamic_scale_for_fsdp=True,
        )
        converter = Float8LinearConverter(
            float8_cfg, parallel_dims=parallel_dims, model_compile_enabled=compile,
        )
        converter.convert(model)

    if compile:
        for _layer_id, block in model.layers.named_children():
            block.compile(backend="inductor", fullgraph=True)

    fsdp_config = {"mesh": mesh}
    fully_shard([model.tok_embeddings, model.norm, model.output], **fsdp_config)
    for block in model.layers.values():
        fully_shard(block, **fsdp_config)
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)

    return model


def _forward_backward(model, input_dict, labels, device, global_valid_tokens):
    input_dict = {
        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
        for k, v in input_dict.items()
    }
    labels_gpu = labels.unsqueeze(0).to(device)

    inputs = input_dict["input"]
    extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
    positions = extra_inputs.pop("positions", None)
    extra_inputs.pop("attn_cost", None)

    extra_kwargs: dict = {}
    if positions is not None:
        extra_kwargs["positions"] = positions
    extra_kwargs["attention_masks"] = model.get_attention_masks(
        input_batch=inputs,
        tokenizer=_get_tokenizer(),
        extra_inputs=extra_inputs,
        positions=positions,
    )

    pred = model(inputs, **extra_inputs, **extra_kwargs)
    loss_sum = cross_entropy_loss(pred, labels_gpu)
    loss = loss_sum / global_valid_tokens
    del pred
    loss.backward()
    return loss.detach()


def _train_step(model, optimizer, data_iter, device, dp_world_size):
    optimizer.zero_grad()

    input_dict, labels, _stats = next(data_iter)
    local_valid = (labels != IGNORE_INDEX).sum().to(device)
    global_valid = local_valid.clone()
    dist.all_reduce(global_valid)
    global_valid = global_valid.float()

    loss = _forward_backward(model, input_dict, labels, device, global_valid)
    optimizer.step()
    return loss


def _create_test_pretok_dir(tmp_dir: Path, n_examples: int = 100, seed: int = 42) -> Path:
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(seed)
    pretok_dir = tmp_dir / "pretok"
    shards_dir = pretok_dir / "shards"
    shards_dir.mkdir(parents=True)

    max_tokens = SEQ_LEN - 50
    n_tokens_arr = rng.integers(80, max_tokens, size=n_examples).astype(np.int32)
    attn_cost_arr = (
        n_tokens_arr.astype(np.int64) * (n_tokens_arr.astype(np.int64) + 1) // 2
    )

    def _make_labels(n):
        labs = rng.integers(0, VOCAB_SIZE, size=int(n))
        mask = rng.random(size=int(n)) < 0.5
        labs[mask] = IGNORE_INDEX
        return labs.tolist()

    data = {
        "input_ids": [
            list(rng.integers(0, VOCAB_SIZE, size=int(n))) for n in n_tokens_arr
        ],
        "labels": [_make_labels(n) for n in n_tokens_arr],
        "positions": [list(range(int(n))) for n in n_tokens_arr],
        "suffix_starts": [[] for _ in range(n_examples)],
        "insertion_limits": [[] for _ in range(n_examples)],
        "n_tokens": n_tokens_arr.tolist(),
        "train_tokens": (n_tokens_arr // 2).tolist(),
        "attn_cost": attn_cost_arr.tolist(),
    }
    full_ds = HFDataset.from_dict(data)
    shard_name = "shard_0000"
    full_ds.save_to_disk(str(shards_dir / shard_name))

    manifest = {
        "version": 1,
        "strategy": "backbone_suffix",
        "tokenizer": {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE},
        "shards": {"completed": [shard_name]},
        "stats": {
            "total_examples": n_examples,
            "total_tokens": int(n_tokens_arr.sum()),
        },
        "length_stats": {"mean": float(n_tokens_arr.mean())},
    }
    with open(pretok_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return pretok_dir


def _get_flat_model_sd(model):
    return get_model_state_dict(model)


def _get_flat_optim_sd(model, optimizer):
    return get_optimizer_state_dict(
        model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
    )


def _set_flat_optim_sd(model, optimizer, state_dict):
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=state_dict,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )


class TestCheckpointResume(DTest):
    default_world_size = 4

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4, reason="Requires 4 GPUs"
    )
    @pytest.mark.parametrize(
        "compile,float8",
        [(False, False), (True, False), (True, True)],
        ids=["compile=False", "compile=True", "compile=True+float8"],
    )
    def test_resume(self, compile: bool, float8: bool):
        self._run_resume_test(compile=compile, float8=float8)

    def _run_resume_test(self, compile: bool, float8: bool = False):
        if float8 and not has_cuda_capability(8, 9):
            pytest.skip("Float8 requires SM89+")
        dp_world_size = self.world_size
        seed = 42
        n_train_steps = 3

        mesh = init_device_mesh("cuda", (dp_world_size,))

        with self.temp_dir() as tmp_dir:
            tmp_path = Path(tmp_dir)

            if self.rank == 0:
                pretok_dir = _create_test_pretok_dir(tmp_path)
                pack_plan_dir = pretok_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"
                plan_packing(pretok_dir, SEQ_LEN, pack_plan_dir)
            dist.barrier()

            pretok_dir = tmp_path / "pretok"
            pack_plan_dir = pretok_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"
            ckpt_dir = tmp_path / "checkpoint" / "step-3"

            # === Phase 1: Train N steps, save checkpoint ===
            model_a = _build_model(self.device, mesh, seed=seed, compile=compile, float8=float8)
            opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)

            ds_a = PlannedPackingDataset(
                pack_plan_path=pack_plan_dir,
                seq_len=SEQ_LEN,
                packing="prepacked_random",
                dp_rank=self.rank,
                dp_world_size=dp_world_size,
                accum_steps=1,
                seed=seed,
            )
            iter_a = iter(ds_a)

            for _step in range(n_train_steps):
                _train_step(model_a, opt_a, iter_a, self.device, dp_world_size)

            # Save model + optimizer state via DCP
            model_sd = _get_flat_model_sd(model_a)
            optim_sd = _get_flat_optim_sd(model_a, opt_a)
            state_dict_to_save = {**model_sd, "optimizer": optim_sd}
            dcp.save(state_dict_to_save, checkpoint_id=str(ckpt_dir))
            dist.barrier()

            # === Phase 1b: Continue 1 more step as reference ===
            ref_loss = _train_step(model_a, opt_a, iter_a, self.device, dp_world_size)

            ref_weights = {}
            for name, param in model_a.named_parameters():
                ref_weights[name] = param.full_tensor().clone()

            del model_a, opt_a

            # Production resume restarts the process, so no dynamo cache carries over.
            if compile:
                torch._dynamo.reset()

            # === Phase 2: Fresh model, load checkpoint, train 1 step ===
            model_b = _build_model(self.device, mesh, seed=seed, compile=compile, float8=float8)
            opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)

            # Load model state
            model_sd_b = _get_flat_model_sd(model_b)
            dcp.load(model_sd_b, checkpoint_id=str(ckpt_dir))
            set_model_state_dict(model_b, model_state_dict=model_sd_b)

            # Load optimizer state
            optim_sd_b = {"optimizer": _get_flat_optim_sd(model_b, opt_b)}
            dcp.load(optim_sd_b, checkpoint_id=str(ckpt_dir))
            _set_flat_optim_sd(model_b, opt_b, optim_sd_b["optimizer"])

            # Recreate dataset at same position (skip first N steps)
            ds_b = PlannedPackingDataset(
                pack_plan_path=pack_plan_dir,
                seq_len=SEQ_LEN,
                packing="prepacked_random",
                dp_rank=self.rank,
                dp_world_size=dp_world_size,
                accum_steps=1,
                seed=seed,
            )
            iter_b = iter(ds_b)
            for _ in range(n_train_steps):
                next(iter_b)

            # Train the (N+1)th step
            resumed_loss = _train_step(model_b, opt_b, iter_b, self.device, dp_world_size)

            resumed_weights = {}
            for name, param in model_b.named_parameters():
                resumed_weights[name] = param.full_tensor().clone()

            # === Assertions ===
            torch.testing.assert_close(
                ref_loss, resumed_loss, atol=1e-5, rtol=1e-5,
                msg="Loss mismatch between continuous and resumed training",
            )
            for name in ref_weights:
                torch.testing.assert_close(
                    ref_weights[name],
                    resumed_weights[name],
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"Weight mismatch for {name}",
                )
