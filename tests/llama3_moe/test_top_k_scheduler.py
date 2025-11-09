# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.llama3_moe import (
    Llama3MoE,
    Llama3MoEModelArgs,
    TopKSchedulerArgs,
)
from torchtitan.models.llama3_moe.top_k_scheduler import get_top_k_scheduler
from torchtitan.models.moe import MoE, MoEArgs


class TestSchedulers:
    # Small Defaults
    dim = 128
    moe_inter_dim = 256
    n_layers = 4
    n_heads = 4
    vocab_size = 256
    is_moe_list = n_layers // 2 * [False] + n_layers // 2 * [True]
    bsz = 2
    seqlen = 64
    top_k = 8
    moe_args = MoEArgs(top_k=top_k, num_shared_experts=0)
    model_args = Llama3MoEModelArgs(
        dim=dim,
        moe_inter_dim=moe_inter_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        is_moe_list=is_moe_list,
        moe_args=moe_args,
    )

    def test_no_op(self):
        top_k_args = TopKSchedulerArgs(name="no_op")
        with torch.device("meta"):
            model = Llama3MoE(self.model_args)

        top_k_scheduler = get_top_k_scheduler(
            model_args=self.model_args, top_k_args=top_k_args, model_parts=[model]
        )
        top_k_scheduler.step(0.0)
        top_k_scheduler.state_dict()
        top_k_scheduler.load_state_dict({})

    def test_constant(self):
        warmup_steps = 10
        step_interval = 5
        min_top_k = self.top_k // 2
        top_k_args = TopKSchedulerArgs(
            name="constant",
            warmup_steps=warmup_steps,
            step_interval=step_interval,
            min_top_k=min_top_k,
        )
        with torch.device("meta"):
            model = Llama3MoE(self.model_args)

        top_k_scheduler = get_top_k_scheduler(
            model_args=self.model_args, top_k_args=top_k_args, model_parts=[model]
        )
        assert top_k_scheduler._step == 0

        loss = 0.0
        for _ in range(warmup_steps + step_interval):
            top_k_scheduler.step(loss)

        # Reduce the top_k once
        assert top_k_scheduler._step == warmup_steps + step_interval
        assert (
            top_k_scheduler.layer_idx_to_top_k[str(self.n_layers - 1)] == self.top_k - 1
        )
        for layer_idx, top_k in top_k_scheduler.layer_idx_to_top_k.items():
            moe = model.layers[str(layer_idx)].moe
            assert moe.router.top_k == top_k
            assert moe.reorderer.top_k == top_k

        sd = top_k_scheduler.state_dict()

        # Reduce a second time:
        for _ in range(step_interval):
            top_k_scheduler.step(loss)
        assert top_k_scheduler._step == warmup_steps + 2 * step_interval
        assert (
            top_k_scheduler.layer_idx_to_top_k[str(self.n_layers - 1)] == self.top_k - 2
        )
        for layer_idx, top_k in top_k_scheduler.layer_idx_to_top_k.items():
            moe = model.layers[str(layer_idx)].moe
            assert moe.router.top_k == top_k
            assert moe.reorderer.top_k == top_k

        # Reduce until this should no-op and all moe top_k's are reduced:
        for _ in range(self.n_layers * self.top_k * step_interval):
            top_k_scheduler.step(loss)

        for module in model.modules():
            if isinstance(module, MoE):
                assert module.router.top_k == min_top_k
                assert module.reorderer.top_k == min_top_k

        top_k_scheduler.load_state_dict(sd)

    def test_loss(self):
        warmup_steps = 10
        min_top_k = self.top_k // 2
        top_k_args = TopKSchedulerArgs(
            name="loss",
            warmup_steps=warmup_steps,
            min_steps=1,
            target_loss=10.0,
            beta=0.99,
            min_top_k=min_top_k,
        )
        with torch.device("meta"):
            model = Llama3MoE(self.model_args)

        top_k_scheduler = get_top_k_scheduler(
            model_args=self.model_args, top_k_args=top_k_args, model_parts=[model]
        )
        assert top_k_scheduler._step == 0
        assert top_k_scheduler._mini_step == 0
        assert top_k_scheduler._curr_loss is None

        loss = 0.0
        for _ in range(warmup_steps + 1):
            top_k_scheduler.step(loss)

        # Reduce the top_k once
        assert top_k_scheduler._step == warmup_steps + 1
        assert (
            top_k_scheduler.layer_idx_to_top_k[str(self.n_layers - 1)] == self.top_k - 1
        )
        for layer_idx, top_k in top_k_scheduler.layer_idx_to_top_k.items():
            moe = model.layers[str(layer_idx)].moe
            assert moe.router.top_k == top_k
            assert moe.reorderer.top_k == top_k

        sd = top_k_scheduler.state_dict()

        # Reduce a second time:
        for _ in range(1):
            top_k_scheduler.step(loss)
        assert top_k_scheduler._step == warmup_steps + 2
        assert (
            top_k_scheduler.layer_idx_to_top_k[str(self.n_layers - 1)] == self.top_k - 2
        )
        for layer_idx, top_k in top_k_scheduler.layer_idx_to_top_k.items():
            moe = model.layers[str(layer_idx)].moe
            assert moe.router.top_k == top_k
            assert moe.reorderer.top_k == top_k

        top_k_scheduler.load_state_dict(sd)
