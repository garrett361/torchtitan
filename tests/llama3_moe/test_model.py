# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.models.llama3_moe import (
    Llama3MoE,
    Llama3MoEJobConfig,
    Llama3MoEModelArgs,
    Llama3MoEStateDictAdapter,
    apply_custom_init,
    llama3_moe_configs,
)
from torchtitan.models.llama3_moe.metrics import MoEHook
from torchtitan.models.moe import MoE

TEST_TEXT = """
The biggest lesson that can be read from 70 years of AI research is that general methods that leverage
computation are ultimately the most effective, and by a large margin. The ultimate reason for this is
Moore's law, or rather its generalization of continued exponentially falling cost per unit of
computation. Most AI research has been conducted as if the computation available to the agent were
constant (in which case leveraging human knowledge would be one of the only ways to improve
performance) but, over a slightly longer time than a typical research project, massively more
computation inevitably becomes available. Seeking an improvement that makes a difference in the
shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing
that matters in the long run is the leveraging of computation. These two need not run counter to each
other, but in practice they tend to. Time spent on one is time not spent on the other. There are
psychological commitments to investment in one approach or the other. And the human-knowledge
approach tends to complicate methods in ways that make them less suited to taking advantage of
general methods leveraging computation.  There were many examples of AI researchers' belated
learning of this bitter lesson, and it is instructive to review some of the most prominent.
"""

LLAMA_3B_HF_PATH = "/gpfs/goon/models/Llama-3.2-3B/"
LLAMA_3B_HF_NO_TIED_PATH = "/gpfs/goon/models/Llama-3.2-3B-no-tied-weights/"
LLAMA_1B_HF_PATH = "/gpfs/goon/models/Llama-3.2-1B/"
LLAMA_1B_HF_NO_TIED_PATH = "/gpfs/goon/models/Llama-3.2-1B-no-tied-weights/"

HF_ASSETS_PATHS = {
    "1B": LLAMA_1B_HF_NO_TIED_PATH,
    "3B": LLAMA_3B_HF_NO_TIED_PATH,
}
HF_PATHS = {
    "1B": LLAMA_1B_HF_PATH,
    "3B": LLAMA_3B_HF_PATH,
}


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
        args = Llama3MoEModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=None,
        )
        model = Llama3MoE(args)
        model.init_weights()
        model.to(self.device)
        inputs = torch.randint(
            self.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        model(inputs)

    @pytest.mark.parametrize(
        "load_balance_coeff", [None, 1e-3], ids=lambda x: f"load_balance_coeff={x}"
    )
    @pytest.mark.parametrize(
        "n_expert_groups", [1, 2], ids=lambda x: f"n_expert_groups={x}"
    )
    def test_model_all_moe(
        self, load_balance_coeff: float | None, n_expert_groups: int
    ):
        # NOTE: @goon - testing requires cuda, as the histogram op used in the current router impl
        # is not supported on CPU, apparently.
        args = Llama3MoEModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=[True for _ in range(self.n_layers)],
        )
        args.moe_args.load_balance_coeff = load_balance_coeff
        args.moe_args.n_expert_groups = n_expert_groups
        model = Llama3MoE(args)
        model.init_weights()
        model.to(self.device)
        inputs = torch.randint(
            self.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        # Just testing for no errors:
        model(inputs).sum().backward()

    @pytest.mark.parametrize("flavor", ["1B", "3B"], ids=lambda x: f"flavor={x}")
    def test_hf_equivalence(self, flavor: str) -> None:
        torch.manual_seed(42)
        model_args = llama3_moe_configs[flavor]
        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = HF_ASSETS_PATHS[flavor]
        with torch.device("meta"):
            model = Llama3MoE(model_args)

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        ckpt_kwargs = {
            "dataloader": None,
            "optimizers": None,  # HACK: @goon - ok to set to None for initial load
            "lr_schedulers": None,  # HACK: @goon - ok to set to None for initial load
            "states": {"train_state": self},
            "checkpoint_config": job_config.checkpoint,
            "sd_adapter": Llama3MoEStateDictAdapter(
                model_args, HF_ASSETS_PATHS[flavor]
            ),
            "base_folder": "",
            "ft_manager": None,
        }

        checkpointer = CheckpointManager(model_parts=[model], **ckpt_kwargs)
        checkpointer.load()

        model_hf = AutoModelForCausalLM.from_pretrained(HF_PATHS[flavor]).to(
            device=self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(HF_PATHS[flavor])
        inputs = tokenizer(TEST_TEXT, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.cuda()

        with torch.no_grad():
            out = model(inputs["input_ids"])
            out_hf = model_hf(**inputs).logits
            vocab_size = out.shape[-1]
            kl = F.kl_div(
                out_hf.reshape(-1, vocab_size).log_softmax(dim=-1),
                out.reshape(-1, vocab_size).log_softmax(dim=-1),
                reduction="batchmean",
                log_target=True,
            )
            mse = (out_hf - out).pow(2).mean().sqrt()
            assert kl < 1e-8, f"{kl=}"
            assert mse < 1e-5, f"{mse=}"
            torch.testing.assert_close(out_hf, out, atol=1e-2, rtol=1e-5)

    def test_model_dynamic_n_moe_layers(self):
        """
        Test dynamically inserting MoE layers through the job config args
        """
        # NOTE: @goon - testing requires cuda, as the histogram op used in the current router impl
        # is not supported on CPU, apparently.
        args = Llama3MoEModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=4,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=[True for _ in range(self.n_layers)],
        )
        job_config = Llama3MoEJobConfig()
        job_config.model_overrides.n_moe_layers = 2
        args.update_from_config(job_config)
        with torch.device("meta"):
            model = Llama3MoE(args)
        moe_enabled_list = [l.moe_enabled for l in model.layers.values()]
        assert moe_enabled_list == [False, True, True, False]

    def test_model_router_init(self):
        job_config = Llama3MoEJobConfig()
        job_config.moe_overrides.router_init_std = 1.0
        args = Llama3MoEModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=[True for _ in range(self.n_layers)],
        )
        args.moe_args.load_balance_coeff = 1e-4
        args.moe_args.n_expert_groups = 2
        model = Llama3MoE(args)
        model.init_weights()

        before_std = model.layers["0"].moe.router.gate.weight.std()
        apply_custom_init(model, job_config)
        after_std = model.layers["0"].moe.router.gate.weight.std()

        assert not torch.isclose(before_std, after_std)


class TestHooks:
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
    hf_assets_path = LLAMA_3B_HF_NO_TIED_PATH

    def test_router_hook(self) -> None:
        args = Llama3MoEModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            is_moe_list=[True for _ in range(self.n_layers)],
        )
        model = Llama3MoE(args)
        model.init_weights()
        model.to(self.device)

        hooks = []
        for fqn, module in model.named_modules():
            if isinstance(module, MoE):
                hooks.append(MoEHook(module, fqn, parallel_dims=None))

        inputs = torch.randint(
            self.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        model(inputs)
        stats = [h.get_stats_dict() for h in hooks]
        for s in stats:
            assert len(s.values()) > 0
            for v in s.values():
                assert isinstance(v, float)
        for h in hooks:
            h.reset()
        reset_stats = [h.get_stats_dict() for h in hooks]
        assert not any(s for s in reset_stats)


class TestOptim:
    seqlen = 64
    bsz = 1
    atol = 1e-1
    rtol = 1e-1

    def test_per_layer_lrs(self) -> None:
        model_args_moe = llama3_moe_configs["1B_2layer_halfmoe"]
        job_config = Llama3MoEJobConfig()
        lr = 1e-4
        moe_router_lr = 1e-3
        routed_expert_lr = 1e-2

        job_config.optimizer.lr = lr
        job_config.optimizer.moe_router_lr = moe_router_lr
        job_config.optimizer.routed_expert_lr = routed_expert_lr

        train_spec = train_spec_module.get_train_spec(job_config.model.name)
        model_moe = Llama3MoE(model_args_moe)
        optimizers = train_spec.build_optimizers_fn(
            [model_moe],
            job_config.optimizer,
            parallel_dims=None,  # HACK: ok to set to none here.
            ft_manager=None,
        )
        param_groups = optimizers.optimizers[0].param_groups
        # Not a very detailed test, but good enough.
        assert len(param_groups) == 3
        assert set(pg["lr"] for pg in param_groups) == {
            lr,
            moe_router_lr,
            routed_expert_lr,
        }
