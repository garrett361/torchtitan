"""Distributed correctness tests for packing modes and strategy equivalence.

These tests verify that mathematically-equivalent training configurations
produce identical training dynamics on multiple GPUs.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch
import torch.distributed as dist
from datasets import Dataset as HFDataset
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dtest import DTest
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.loss import IGNORE_INDEX, cross_entropy_loss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.models.granite import granite_configs
from torchtitan.models.granite.model import GraniteModel
from torchtitan.models.granite.pretokenized_dataset import PlannedPackingDataset
from torchtitan.models.granite.scripts.plan_packing import plan_packing

SEQ_LEN = 512
VOCAB_SIZE = 2048
CONFIG = "debugmodel"
_TOKENIZER_PATH = "tests/assets/tokenizer"

torch._dynamo.config.capture_scalar_outputs = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_fsdp_model(device, mesh, seed=42):
    """Build debug GraniteModel with FSDP, flex attention, no gradient division."""
    torch.manual_seed(seed)
    config = granite_configs[CONFIG](attn_backend="flex")
    model = GraniteModel(config)
    model.init_states()
    model.to(device)

    fsdp_config = {"mesh": mesh}
    fully_shard([model.tok_embeddings, model.norm, model.output], **fsdp_config)
    for block in model.layers.values():
        fully_shard(block, **fsdp_config)
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)

    return model


def _get_tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)


def _forward_backward(model, input_dict, labels, device, global_valid_tokens):
    """Run one microbatch forward+backward replicating the trainer exactly.

    Mirrors trainer.post_dataloading_process + forward_backward_step:
    - Move to GPU, extract inputs/extra_inputs/positions
    - Build attention mask via model.get_attention_masks (pops mask fields)
    - Forward, loss/global_valid_tokens, backward
    """
    # Move all tensors to device (trainer line 867-870)
    input_dict = {
        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
        for k, v in input_dict.items()
    }
    labels_gpu = labels.unsqueeze(0).to(device)

    # post_dataloading_process (trainer line 608-661)
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

    # forward_backward_step (trainer line 813-824)
    pred = model(inputs, **extra_inputs, **extra_kwargs)
    loss_sum = cross_entropy_loss(pred, labels_gpu)
    loss = loss_sum / global_valid_tokens
    del pred
    loss.backward()
    return loss.detach()


def _collect_gradients(model) -> dict[str, torch.Tensor]:
    """Collect full-tensor gradients from FSDP model, deduplicating tied params."""
    grads = {}
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen or param.grad is None:
            continue
        seen.add(id(param))
        grads[name] = param.grad.full_tensor().clone()
    return grads


def _compute_grad_norm(model) -> torch.Tensor:
    """Compute the total L2 grad norm across all parameters."""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    seen = set()
    for param in model.parameters():
        if id(param) in seen or param.grad is None:
            continue
        seen.add(id(param))
        grad = param.grad.full_tensor()
        total += grad.float().pow(2).sum()
    return total.sqrt()


def _create_test_pretok_dir(
    tmp_dir: Path,
    n_examples: int = 100,
    seed: int = 42,
) -> Path:
    """Create a minimal pretokenized backbone_suffix dataset that fits within SEQ_LEN."""
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
        """Generate labels: ~50% IGNORE_INDEX, rest valid token IDs."""
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


def _write_pack_plan(
    pack_plan_dir: Path,
    example_indices: list[list[int]],
    total_tokens: list[int],
    attn_costs: list[int],
    seq_len: int,
    source_pretok_dir: Path,
) -> None:
    """Write a pack plan directly (bypassing BFD) to control example-to-pack assignment."""
    pack_plan_dir.mkdir(parents=True, exist_ok=True)

    pack_ids = pa.array(np.arange(len(example_indices), dtype=np.int32))
    example_indices_col = pa.array(
        [pa.array(indices, type=pa.int32()) for indices in example_indices],
        type=pa.list_(pa.int32()),
    )
    total_tokens_col = pa.array(np.array(total_tokens, dtype=np.int64))
    attn_cost_col = pa.array(np.array(attn_costs, dtype=np.int64))

    table = pa.table({
        "pack_id": pack_ids,
        "example_indices": example_indices_col,
        "total_tokens": total_tokens_col,
        "attn_cost": attn_cost_col,
    })

    plan_path = pack_plan_dir / "pack_plan.arrow"
    with pa.OSFile(str(plan_path), "wb") as f:
        writer = ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()

    metadata = {
        "seq_len": seq_len,
        "total_packs": len(example_indices),
        "total_examples_packed": sum(len(ei) for ei in example_indices),
        "overlong_examples_dropped": 0,
        "padding_fraction": 0.0,
        "bucket_width": 128,
        "source_pretok_dir": str(source_pretok_dir),
        "source_manifest_sha256": "test",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    with open(pack_plan_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Test 2: prepacked_random vs prepacked_random_balanced
# ---------------------------------------------------------------------------


class TestPackingModeEquivalence(DTest):
    """Verify prepacked_random and prepacked_random_balanced produce identical
    training dynamics when using the same global batch size."""

    default_world_size = 4

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4, reason="Requires 4 GPUs"
    )
    def test_random_vs_balanced_training(self):
        dp_world_size = self.world_size
        accum_steps = 2
        n_train_steps = 3
        seed = 42

        mesh = init_device_mesh("cuda", (dp_world_size,))

        # --- Create data and pack plan (rank 0 only, broadcast path) ---
        with self.temp_dir() as tmp_dir:
            tmp_path = Path(tmp_dir)
            if self.rank == 0:
                pretok_dir = _create_test_pretok_dir(tmp_path)
                pack_plan_dir = pretok_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"
                plan_packing(pretok_dir, SEQ_LEN, pack_plan_dir)
            dist.barrier()
            pretok_dir = tmp_path / "pretok"
            pack_plan_dir = pretok_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"

            # --- Run both modes, collect metrics ---
            results = {}
            for mode in ("prepacked_random", "prepacked_random_balanced"):
                model = _build_fsdp_model(self.device, mesh, seed=seed)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                dataset = PlannedPackingDataset(
                    pack_plan_path=pack_plan_dir,
                    seq_len=SEQ_LEN,
                    packing=mode,
                    dp_rank=self.rank,
                    dp_world_size=dp_world_size,
                    accum_steps=accum_steps,
                    seed=seed,
                )
                data_iter = iter(dataset)

                step_losses = []
                step_grad_norms = []
                step_grads = []

                for _step in range(n_train_steps):
                    optimizer.zero_grad()

                    local_valid = torch.tensor(
                        0, dtype=torch.int64, device=self.device
                    )
                    microbatches = []
                    for _ in range(accum_steps):
                        input_dict, labels, _stats = next(data_iter)
                        local_valid += (labels != IGNORE_INDEX).sum().to(
                            self.device
                        )
                        microbatches.append((input_dict, labels))

                    global_valid = local_valid.clone()
                    dist.all_reduce(global_valid)
                    global_valid = global_valid.float()

                    step_loss = torch.tensor(0.0, device=self.device)
                    for input_dict, labels in microbatches:
                        loss = _forward_backward(
                            model, input_dict, labels, self.device, global_valid
                        )
                        step_loss += loss

                    # All-reduce loss for global comparison (local losses
                    # differ between modes due to different rank assignments)
                    dist.all_reduce(step_loss)
                    step_losses.append(step_loss)
                    step_grad_norms.append(_compute_grad_norm(model))
                    step_grads.append(_collect_gradients(model))

                    optimizer.step()

                results[mode] = {
                    "losses": step_losses,
                    "grad_norms": step_grad_norms,
                    "grads": step_grads,
                }

            # --- Assertions ---
            atol, rtol = 1e-6, 1e-5
            random_r = results["prepacked_random"]
            balanced_r = results["prepacked_random_balanced"]

            for step in range(n_train_steps):
                torch.testing.assert_close(
                    random_r["losses"][step],
                    balanced_r["losses"][step],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Loss mismatch at step {step}",
                )
                torch.testing.assert_close(
                    random_r["grad_norms"][step],
                    balanced_r["grad_norms"][step],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Grad norm mismatch at step {step}",
                )
                for name in random_r["grads"][step]:
                    torch.testing.assert_close(
                        random_r["grads"][step][name],
                        balanced_r["grads"][step][name],
                        atol=atol,
                        rtol=rtol,
                        msg=f"Gradient mismatch for {name} at step {step}",
                    )


# ---------------------------------------------------------------------------
# Test 1: TruncateEveryTurn vs BackboneSuffix gradient equivalence
# ---------------------------------------------------------------------------


def _create_strategy_pretok_dirs(
    tmp_dir: Path,
    n_conversations: int = 8,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Create aligned pretok dirs for BackboneSuffix and TruncateEveryTurn strategies.

    Each conversation is a 2-turn dialogue. Constructs both representations from
    the same logical conversation so they produce identical training signals:

    BackboneSuffix (per conversation):
        - Backbone: [context | <think></think>response_1 | user_2 | turn_2_full]
        - Suffix: [turn_1_thinking]
        - Labels on: response_1 (in backbone) + turn_1_thinking (in suffix) + turn_2 (in backbone)

    TruncateEveryTurn (per conversation → 2 examples):
        - Example A: [context | <think>turn_1_thinking</think>response_1]
          Labels on: turn_1_thinking + response_1
        - Example B: [context | <think></think>response_1 | user_2 | turn_2_full]
          Labels on: turn_2_full

    Both train exactly the same tokens with the same visible context.
    """
    rng = np.random.default_rng(seed)

    # Per-conversation structure (kept short so TE examples fit in one pack)
    context_len = 40  # system + user_1
    think_1_len = 30  # turn_1 historical thinking
    response_1_len = 20  # turn_1 response (after thinking)
    user_2_len = 20  # user_2 message
    turn_2_len = 50  # final turn (thinking + response)
    # Truncated thinking placeholder: we represent <think></think> as 2 tokens
    trunc_placeholder_len = 2

    # BackboneSuffix: backbone has truncated think + response_1 + user_2 + turn_2
    bs_backbone_len = context_len + trunc_placeholder_len + response_1_len + user_2_len + turn_2_len
    bs_total_len = bs_backbone_len + think_1_len
    # TruncateEveryTurn example A: context + <think> + thinking + </think> + response
    # We represent <think> as 1 token, </think> as 1 token
    te_ex_a_len = context_len + 1 + think_1_len + 1 + response_1_len
    # TruncateEveryTurn example B: same as backbone (no suffix)
    te_ex_b_len = bs_backbone_len
    te_total_packed = te_ex_a_len + te_ex_b_len

    assert bs_total_len <= SEQ_LEN
    assert te_total_packed <= SEQ_LEN

    # Generate token sequences for each conversation
    bs_examples = []
    te_examples = []

    for _conv_idx in range(n_conversations):
        # Shared token pools
        context_tokens = rng.integers(0, VOCAB_SIZE, size=context_len).tolist()
        think_1_tokens = rng.integers(0, VOCAB_SIZE, size=think_1_len).tolist()
        response_1_tokens = rng.integers(0, VOCAB_SIZE, size=response_1_len).tolist()
        user_2_tokens = rng.integers(0, VOCAB_SIZE, size=user_2_len).tolist()
        turn_2_tokens = rng.integers(0, VOCAB_SIZE, size=turn_2_len).tolist()
        # Special tokens for think markers
        think_open = rng.integers(0, VOCAB_SIZE)
        think_close = rng.integers(0, VOCAB_SIZE)

        # --- BackboneSuffix example ---
        # Backbone: context + [think_open, think_close] + response_1 + user_2 + turn_2
        bs_input_ids = (
            context_tokens
            + [int(think_open), int(think_close)]
            + response_1_tokens
            + user_2_tokens
            + turn_2_tokens
            + think_1_tokens  # suffix appended after backbone
        )
        # Labels: train response_1 (in backbone), turn_2 (in backbone), think_1 (in suffix)
        bs_labels = [IGNORE_INDEX] * bs_total_len
        # response_1 starts after context + trunc_placeholder
        resp_1_start = context_len + trunc_placeholder_len
        for i in range(response_1_len):
            bs_labels[resp_1_start + i] = bs_input_ids[resp_1_start + i]
        # turn_2 starts after context + trunc + response_1 + user_2
        turn_2_start = context_len + trunc_placeholder_len + response_1_len + user_2_len
        for i in range(turn_2_len):
            bs_labels[turn_2_start + i] = bs_input_ids[turn_2_start + i]
        # suffix (think_1) labels
        suffix_start = bs_backbone_len
        for i in range(think_1_len):
            bs_labels[suffix_start + i] = bs_input_ids[suffix_start + i]

        # Positions: sequential for backbone, insertion_limit+1 for suffix
        ins_limit = context_len  # position of think_open in backbone
        bs_positions = list(range(bs_backbone_len))
        bs_positions += [ins_limit + 1 + i for i in range(think_1_len)]

        bs_suffix_starts = [bs_backbone_len]
        bs_insertion_limits = [ins_limit]
        bs_n_tokens = bs_total_len
        bs_attn_cost = bs_n_tokens * (bs_n_tokens + 1) // 2

        bs_examples.append({
            "input_ids": bs_input_ids,
            "labels": bs_labels,
            "positions": bs_positions,
            "suffix_starts": bs_suffix_starts,
            "insertion_limits": bs_insertion_limits,
            "n_tokens": bs_n_tokens,
            "train_tokens": response_1_len + turn_2_len + think_1_len,
            "attn_cost": bs_attn_cost,
        })

        # --- TruncateEveryTurn examples ---
        # Example A: context + think_open + thinking + think_close + response
        te_a_input_ids = (
            context_tokens
            + [int(think_open)]
            + think_1_tokens
            + [int(think_close)]
            + response_1_tokens
        )
        te_a_labels = [IGNORE_INDEX] * te_ex_a_len
        # Labels on thinking (after think_open) and response (after think_close)
        # think_1 region:
        train_start_a = context_len + 1  # after think_open
        for i in range(think_1_len):
            te_a_labels[train_start_a + i] = te_a_input_ids[train_start_a + i]
        # response_1 region (after think_close):
        resp_start_a = context_len + 1 + think_1_len + 1  # after think_close
        for i in range(response_1_len):
            te_a_labels[resp_start_a + i] = te_a_input_ids[resp_start_a + i]
        te_a_n_tokens = te_ex_a_len
        te_a_attn_cost = te_a_n_tokens * (te_a_n_tokens + 1) // 2

        # Example B: context + trunc_placeholder + response_1 + user_2 + turn_2
        # (same as backbone — labels on turn_2 only)
        te_b_input_ids = (
            context_tokens
            + [int(think_open), int(think_close)]
            + response_1_tokens
            + user_2_tokens
            + turn_2_tokens
        )
        te_b_labels = [IGNORE_INDEX] * te_ex_b_len
        te_b_turn_2_start = context_len + trunc_placeholder_len + response_1_len + user_2_len
        for i in range(turn_2_len):
            te_b_labels[te_b_turn_2_start + i] = te_b_input_ids[te_b_turn_2_start + i]
        te_b_n_tokens = te_ex_b_len
        te_b_attn_cost = te_b_n_tokens * (te_b_n_tokens + 1) // 2

        te_examples.append({
            "input_ids": te_a_input_ids,
            "labels": te_a_labels,
            "positions": list(range(te_a_n_tokens)),
            "suffix_starts": [],
            "insertion_limits": [],
            "n_tokens": te_a_n_tokens,
            "train_tokens": think_1_len + response_1_len,
            "attn_cost": te_a_attn_cost,
        })
        te_examples.append({
            "input_ids": te_b_input_ids,
            "labels": te_b_labels,
            "positions": list(range(te_b_n_tokens)),
            "suffix_starts": [],
            "insertion_limits": [],
            "n_tokens": te_b_n_tokens,
            "train_tokens": turn_2_len,
            "attn_cost": te_b_attn_cost,
        })

    # --- Write BackboneSuffix pretok dir ---
    bs_dir = tmp_dir / "pretok_bs"
    _write_pretok_dir(bs_dir, bs_examples, strategy="backbone_suffix")

    # --- Write TruncateEveryTurn pretok dir ---
    te_dir = tmp_dir / "pretok_te"
    _write_pretok_dir(te_dir, te_examples, strategy="truncate_every_turn")

    # --- Write aligned pack plans ---
    # BS: 1 pack per conversation (n_conversations packs, 1 example each)
    bs_plan_dir = bs_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"
    _write_pack_plan(
        bs_plan_dir,
        example_indices=[[i] for i in range(n_conversations)],
        total_tokens=[ex["n_tokens"] for ex in bs_examples],
        attn_costs=[ex["attn_cost"] for ex in bs_examples],
        seq_len=SEQ_LEN,
        source_pretok_dir=bs_dir,
    )

    # TE: 1 pack per conversation (groups of 2 consecutive examples)
    te_plan_dir = te_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"
    te_pack_indices = [[2 * i, 2 * i + 1] for i in range(n_conversations)]
    te_pack_tokens = [
        te_examples[2 * i]["n_tokens"] + te_examples[2 * i + 1]["n_tokens"]
        for i in range(n_conversations)
    ]
    te_pack_costs = [
        te_examples[2 * i]["attn_cost"] + te_examples[2 * i + 1]["attn_cost"]
        for i in range(n_conversations)
    ]
    _write_pack_plan(
        te_plan_dir,
        example_indices=te_pack_indices,
        total_tokens=te_pack_tokens,
        attn_costs=te_pack_costs,
        seq_len=SEQ_LEN,
        source_pretok_dir=te_dir,
    )

    return bs_dir, te_dir


def _write_pretok_dir(
    pretok_dir: Path,
    examples: list[dict],
    strategy: str,
) -> None:
    """Write examples to a pretok directory in HF Dataset format."""
    shards_dir = pretok_dir / "shards"
    shards_dir.mkdir(parents=True)

    data = {
        "input_ids": [ex["input_ids"] for ex in examples],
        "labels": [ex["labels"] for ex in examples],
        "positions": [ex["positions"] for ex in examples],
        "suffix_starts": [ex["suffix_starts"] for ex in examples],
        "insertion_limits": [ex["insertion_limits"] for ex in examples],
        "n_tokens": [ex["n_tokens"] for ex in examples],
        "train_tokens": [ex["train_tokens"] for ex in examples],
        "attn_cost": [ex["attn_cost"] for ex in examples],
    }
    ds = HFDataset.from_dict(data)
    shard_name = "shard_0000"
    ds.save_to_disk(str(shards_dir / shard_name))

    total_tokens = sum(ex["n_tokens"] for ex in examples)
    manifest = {
        "version": 1,
        "strategy": strategy,
        "tokenizer": {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE},
        "shards": {"completed": [shard_name]},
        "stats": {
            "total_examples": len(examples),
            "total_tokens": total_tokens,
        },
        "length_stats": {"mean": total_tokens / len(examples)},
    }
    with open(pretok_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


class TestStrategyGradientEquivalence(DTest):
    """Verify BackboneSuffix and TruncateEveryTurn produce identical gradients
    when training on the same logical conversations."""

    default_world_size = 4

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4, reason="Requires 4 GPUs"
    )
    def test_backbone_suffix_vs_truncate_every_turn(self):
        dp_world_size = self.world_size
        accum_steps = 1
        n_train_steps = 3
        n_conversations = dp_world_size * accum_steps * n_train_steps  # 12
        seed = 42

        mesh = init_device_mesh("cuda", (dp_world_size,))

        # --- Create aligned pretok data + pack plans ---
        with self.temp_dir() as tmp_dir:
            tmp_path = Path(tmp_dir)
            if self.rank == 0:
                _create_strategy_pretok_dirs(
                    tmp_path, n_conversations=n_conversations, seed=seed
                )
            dist.barrier()

            bs_dir = tmp_path / "pretok_bs"
            te_dir = tmp_path / "pretok_te"
            bs_plan = bs_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"
            te_plan = te_dir / "pack_plans" / f"seqlen_{SEQ_LEN}"

            # --- Run both configurations ---
            results = {}
            for label, plan_dir in [("bs", bs_plan), ("te", te_plan)]:
                model = _build_fsdp_model(self.device, mesh, seed=seed)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                dataset = PlannedPackingDataset(
                    pack_plan_path=plan_dir,
                    seq_len=SEQ_LEN,
                    packing="prepacked_random",
                    dp_rank=self.rank,
                    dp_world_size=dp_world_size,
                    accum_steps=accum_steps,
                    seed=seed,
                )
                data_iter = iter(dataset)

                step_losses = []
                step_grads = []

                for _step in range(n_train_steps):
                    optimizer.zero_grad()

                    local_valid = torch.tensor(
                        0, dtype=torch.int64, device=self.device
                    )
                    microbatches = []
                    for _ in range(accum_steps):
                        input_dict, labels, _stats = next(data_iter)
                        local_valid += (labels != IGNORE_INDEX).sum().to(
                            self.device
                        )
                        microbatches.append((input_dict, labels))

                    global_valid = local_valid.clone()
                    dist.all_reduce(global_valid)
                    global_valid = global_valid.float()

                    step_loss = torch.tensor(0.0, device=self.device)
                    for input_dict, labels in microbatches:
                        loss = _forward_backward(
                            model, input_dict, labels, self.device, global_valid
                        )
                        step_loss += loss

                    step_losses.append(step_loss)
                    step_grads.append(_collect_gradients(model))
                    optimizer.step()

                results[label] = {
                    "losses": step_losses,
                    "grads": step_grads,
                }

            # --- Assertions ---
            atol, rtol = 1e-4, 1e-4
            for step in range(n_train_steps):
                torch.testing.assert_close(
                    results["bs"]["losses"][step],
                    results["te"]["losses"][step],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Loss mismatch at step {step}",
                )
                for name in results["bs"]["grads"][step]:
                    torch.testing.assert_close(
                        results["bs"]["grads"][step][name],
                        results["te"]["grads"][step][name],
                        atol=atol,
                        rtol=rtol,
                        msg=f"Gradient mismatch for {name} at step {step}",
                    )


# ---------------------------------------------------------------------------
# Test 3: num_workers=0 vs num_workers=2 equivalence for prepacked modes
# ---------------------------------------------------------------------------


class TestMultiWorkerEquivalence(DTest):
    """Verify that num_workers>1 produces identical training dynamics to
    num_workers=0 for PlannedPackingDataset through StatefulDataLoader."""

    default_world_size = 4

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4, reason="Requires 4 GPUs"
    )
    def test_workers_produce_same_training(self):
        dp_world_size = self.world_size
        accum_steps = 2
        n_train_steps = 3
        seed = 42

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

            results = {}
            for num_workers in (0, 2):
                model = _build_fsdp_model(self.device, mesh, seed=seed)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                dataset = PlannedPackingDataset(
                    pack_plan_path=pack_plan_dir,
                    seq_len=SEQ_LEN,
                    packing="prepacked_random",
                    dp_rank=self.rank,
                    dp_world_size=dp_world_size,
                    accum_steps=accum_steps,
                    seed=seed,
                )
                dl = StatefulDataLoader(
                    dataset, batch_size=None, num_workers=num_workers
                )
                data_iter = iter(dl)

                step_losses = []
                step_grad_norms = []
                step_grads = []

                for _step in range(n_train_steps):
                    optimizer.zero_grad()

                    local_valid = torch.tensor(
                        0, dtype=torch.int64, device=self.device
                    )
                    microbatches = []
                    for _ in range(accum_steps):
                        input_dict, labels, _stats = next(data_iter)
                        local_valid += (labels != IGNORE_INDEX).sum().to(
                            self.device
                        )
                        microbatches.append((input_dict, labels))

                    global_valid = local_valid.clone()
                    dist.all_reduce(global_valid)
                    global_valid = global_valid.float()

                    step_loss = torch.tensor(0.0, device=self.device)
                    for input_dict, labels in microbatches:
                        loss = _forward_backward(
                            model, input_dict, labels, self.device, global_valid
                        )
                        step_loss += loss

                    dist.all_reduce(step_loss)
                    step_losses.append(step_loss)
                    step_grad_norms.append(_compute_grad_norm(model))
                    step_grads.append(_collect_gradients(model))

                    optimizer.step()

                results[num_workers] = {
                    "losses": step_losses,
                    "grad_norms": step_grad_norms,
                    "grads": step_grads,
                }

            # Exact match expected — same data, same order
            atol, rtol = 1e-6, 1e-5
            for step in range(n_train_steps):
                torch.testing.assert_close(
                    results[0]["losses"][step],
                    results[2]["losses"][step],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Loss mismatch at step {step}",
                )
                torch.testing.assert_close(
                    results[0]["grad_norms"][step],
                    results[2]["grad_norms"][step],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Grad norm mismatch at step {step}",
                )
                for name in results[0]["grads"][step]:
                    torch.testing.assert_close(
                        results[0]["grads"][step][name],
                        results[2]["grads"][step][name],
                        atol=atol,
                        rtol=rtol,
                        msg=f"Gradient mismatch for {name} at step {step}",
                    )
