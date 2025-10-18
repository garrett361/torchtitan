# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.llama3_moe import (
    llama3_moe_configs,
    Transformer,
    TransformerModelArgs,
)


class TestModel:
    # Small Defaults
    dim = 128
    moe_inter_dim = 256
    n_layers = 2
    n_heads = 4
    vocab_size = 256
    is_moe_list = None
    bsz = 2
    seqlen = 64
    device = "cuda"

    def test_model_no_moe(self):
        args = TransformerModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=None,
        )
        model = Transformer(args)
        model.init_weights()
        model.to(self.device)
        inputs = torch.randint(
            self.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        model(inputs)

    def test_model_all_moe(self):
        # NOTE: @goon - testing requires cuda, as the histogram op used in the current router impl
        # is not supported on CPU, apparently.
        args = TransformerModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=[True for _ in range(self.n_layers)],
        )
        model = Transformer(args)
        model.init_weights()
        model.to(self.device)
        inputs = torch.randint(
            self.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        model(inputs)

    def test_dev_cfg(self):
        dev_cfg = llama3_moe_configs["3B_dev|n_layers=8|n_moe=4|num_experts=3"]
        assert dev_cfg.n_layers == 8
        assert dev_cfg.moe_args.num_experts == 3
        assert dev_cfg.is_moe_list == 4 * [False] + 4 * [True]
