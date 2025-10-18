# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.models.llama3_moe import (
    JobConfig,
    llama3_moe_configs,
    Llama3MoEStateDictAdapter,
    Transformer,
    TransformerModelArgs,
)
from transformers import AutoModelForCausalLM


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
    hf_assets_path = "/gpfs/goon/models/Llama-3.2-3B-no-tied-weights/"

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

    def test_hf_equivalence(self) -> None:
        model_args = llama3_moe_configs["3B"]
        job_config = JobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path
        with torch.device("meta"):
            model = Transformer(model_args)

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        ckpt_kwargs = {
            "dataloader": None,
            "optimizers": None,  # HACK: @goon - ok to set to None for initial load
            "lr_schedulers": None,  # HACK: @goon - ok to set to None for initial load
            "states": {"train_state": self},
            "checkpoint_config": job_config.checkpoint,
            "sd_adapter": Llama3MoEStateDictAdapter(model_args, self.hf_assets_path),
            "base_folder": "",
            "ft_manager": None,
        }

        checkpointer = CheckpointManager(model_parts=[model], **ckpt_kwargs)
        checkpointer.load()

        model_hf = AutoModelForCausalLM.from_pretrained(
            "/gpfs/goon/models/Llama-3.2-3B/"
        ).to(device=self.device)

        torch.manual_seed(42)
        with torch.no_grad():
            inputs = torch.randint(
                model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
            )
            out = model(inputs)
            out_hf = model_hf(inputs)
            # NOTE: @goon -  current mean error ~ 1%. Might be failing due to the RoPE impl
            # mismatches?
            torch.testing.assert_close(out_hf.logits, out, atol=1e-1, rtol=1e-1)

    def test_dev_cfg(self):
        dev_cfg = llama3_moe_configs["3B_dev|n_layers=8|n_moe=4|num_experts=3"]
        assert dev_cfg.n_layers == 8
        assert dev_cfg.moe_args.num_experts == 3
        assert dev_cfg.is_moe_list == 4 * [False] + 4 * [True]
