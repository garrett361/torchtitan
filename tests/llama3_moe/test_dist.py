# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from dtest import DTest
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.models.llama3_moe import (
    CustomCheckpointManager,
    get_hf_weight_transform_cls,
    llama3_moe_configs,
    Llama3MoE,
    Llama3MoEJobConfig,
    Llama3MoEStateDictAdapter,
    parallelize_llama_moe,
    TransformingHuggingFaceStorageReader,
)
from torchtitan.models.moe import MoE, MoEArgs


class TestHFReader(DTest):
    hf_assets_path = "/gpfs/goon/models/Llama-3.2-3B-no-tied-weights/"
    seqlen = 64
    bsz = 1
    """
    Test loading correctness
    """

    @pytest.mark.parametrize("sharding", ["fsdp", "ep"])
    def test_non_moe_load_equivalence(self, sharding: str) -> None:
        """
        Test e2e equivlance on the full 3B model.
        """
        model_args = llama3_moe_configs["3B"]
        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path
        with torch.device("meta"):
            model = Llama3MoE(model_args)
        model_copy = deepcopy(model)

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size if sharding == "ep" else 1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_copy = parallelize_llama_moe(model_copy, parallel_dims, job_config)

        model.to_empty(device=self.device)
        model_copy.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
            model_copy.init_weights(buffer_device=None)

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

        custom_checkpointer = CustomCheckpointManager(
            model_parts=[model_copy], **ckpt_kwargs
        )
        custom_checkpointer.load()
        torch.manual_seed(42 + dist.get_rank())
        with torch.no_grad():
            inputs = torch.randint(
                model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
            )
            out = model(inputs)
            out_copy = model_copy(inputs)
            torch.testing.assert_close(out, out_copy)

    @pytest.mark.parametrize("sharding", ["fsdp", "ep"])
    def test_small_non_moe_load_equivalence(self, sharding: str) -> None:
        """
        Test equivalence (and that load succeeds) on a truncated version of the model with fewer
        layers.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path
        with torch.device("meta"):
            model = Llama3MoE(model_args)
        model_copy = deepcopy(model)

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size if sharding == "ep" else 1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_copy = parallelize_llama_moe(model_copy, parallel_dims, job_config)

        model.to_empty(device=self.device)
        model_copy.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
            model_copy.init_weights(buffer_device=None)

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

        custom_checkpointer = CustomCheckpointManager(
            model_parts=[model_copy], **ckpt_kwargs
        )
        custom_checkpointer.load()
        torch.manual_seed(42 + dist.get_rank())
        with torch.no_grad():
            inputs = torch.randint(
                model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
            )
            out = model(inputs)
            out_copy = model_copy(inputs)
            torch.testing.assert_close(out, out_copy)

    @pytest.mark.parametrize("sharding", ["fsdp", "ep"])
    def test_small_moe_load_replicate_transform(self, sharding: str) -> None:
        """
        Test  (and that load succeeds) on a truncated version of the model with fewer
        layers.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        model_args_moe = llama3_moe_configs["3B_2layer_halfmoe"]
        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path

        with torch.device("meta"):
            model = Llama3MoE(model_args)
            model_moe = Llama3MoE(model_args_moe)

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size if sharding == "ep" else 1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_moe = parallelize_llama_moe(model_moe, parallel_dims, job_config)

        model.to_empty(device=self.device)
        model_moe.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
            model_moe.init_weights(buffer_device=None)

        # Sanity checks:
        assert not any(isinstance(m, MoE) for m in model.modules())
        assert any(isinstance(m, MoE) for m in model_moe.modules())

        ckpt_kwargs = {
            "dataloader": None,
            "optimizers": None,  # HACK: @goon - ok to set to None for initial load
            "lr_schedulers": None,  # HACK: @goon - ok to set to None for initial load
            "states": {"train_state": self},
            "checkpoint_config": job_config.checkpoint,
            "base_folder": "",
            "ft_manager": None,
        }

        checkpointer = CheckpointManager(
            model_parts=[model],
            sd_adapter=Llama3MoEStateDictAdapter(model_args, self.hf_assets_path),
            **ckpt_kwargs,
        )
        checkpointer.load()

        sd_adapter_moe = Llama3MoEStateDictAdapter(model_args_moe, self.hf_assets_path)
        custom_checkpointer = CustomCheckpointManager(
            hf_storage_reader=TransformingHuggingFaceStorageReader,
            hf_storage_reader_kwargs={
                "transform_fn": get_hf_weight_transform_cls("replicate")(
                    model_args=model_args_moe,
                    hf_to_titan_fqn_map=sd_adapter_moe.from_hf_map,
                ),
                "state_dict": ModelWrapper([model_moe]).state_dict(),
                "sd_adapter": sd_adapter_moe,
            },
            model_parts=[model_moe],
            sd_adapter=sd_adapter_moe,
            **ckpt_kwargs,
        )
        custom_checkpointer.load()
        with torch.no_grad():
            # Verify weight correctness:
            sd = model.state_dict()
            sd_moe = model_moe.state_dict()
            for k_moe in sd_moe:
                if "moe" not in k_moe:
                    w, w_moe = sd[k_moe].full_tensor(), sd_moe[k_moe].full_tensor()
                    torch.testing.assert_close(w, w_moe)

                elif "moe.experts" in k_moe:
                    # Grab the MoE expert weights and the FFN weight they should have originated
                    # from.
                    k = k_moe.replace("moe.experts", "feed_forward") + ".weight"
                    w, w_moe = sd[k].full_tensor(), sd_moe[k_moe].full_tensor()
                    assert (
                        torch.Size((model_args_moe.moe_args.num_experts,)) + w.shape
                        == w_moe.shape
                    ), f"{w.shape=}, {w_moe.shape=}"
                    for w_moe_expert_shard in w_moe:
                        torch.testing.assert_close(w, w_moe_expert_shard)


class TestImpls(DTest):
    hf_assets_path = "/gpfs/goon/models/Llama-3.2-3B-no-tied-weights/"
    seqlen = 64
    bsz = 1
    atol = 1e-1
    rtol = 1e-1
    """
    Test impl correctness
    """

    @pytest.mark.parametrize("sharding", ["fsdp", "ep"])
    def test_basic_replication(self, sharding: str) -> None:
        """
        Test that the dense and MoE models have the same output with FFN weight replication.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        model_args_moe = llama3_moe_configs["3B_2layer_halfmoe"]
        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path

        moe_args = model_args_moe.moe_args
        assert moe_args.score_func == "softmax"
        assert moe_args.route_norm is True
        assert moe_args.score_before_experts is False
        assert model_args.custom_moe_impl is None
        assert model_args_moe.custom_moe_impl is None

        with torch.device("meta"):
            model = Llama3MoE(model_args)
            model_moe = Llama3MoE(model_args_moe)

        # Sanity checks:
        assert not any(isinstance(m, MoE) for m in model.modules())
        assert any(isinstance(m, MoE) for m in model_moe.modules())

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size if sharding == "ep" else 1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_moe = parallelize_llama_moe(model_moe, parallel_dims, job_config)

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        model_moe.to_empty(device=self.device)
        with torch.no_grad():
            model_moe.init_weights(buffer_device=None)

        ckpt_kwargs = {
            "dataloader": None,
            "optimizers": None,  # HACK: @goon - ok to set to None for initial load
            "lr_schedulers": None,  # HACK: @goon - ok to set to None for initial load
            "states": {"train_state": self},
            "checkpoint_config": job_config.checkpoint,
            "base_folder": "",
            "ft_manager": None,
        }

        checkpointer = CheckpointManager(
            model_parts=[model],
            sd_adapter=Llama3MoEStateDictAdapter(model_args, self.hf_assets_path),
            **ckpt_kwargs,
        )
        checkpointer.load()

        sd_adapter_moe = Llama3MoEStateDictAdapter(model_args_moe, self.hf_assets_path)
        custom_checkpointer = CustomCheckpointManager(
            hf_storage_reader=TransformingHuggingFaceStorageReader,
            hf_storage_reader_kwargs={
                "transform_fn": get_hf_weight_transform_cls("replicate")(
                    model_args=model_args_moe,
                    hf_to_titan_fqn_map=sd_adapter_moe.from_hf_map,
                ),
                "state_dict": ModelWrapper([model_moe]).state_dict(),
                "sd_adapter": sd_adapter_moe,
            },
            model_parts=[model_moe],
            sd_adapter=sd_adapter_moe,
            **ckpt_kwargs,
        )
        custom_checkpointer.load()

        with torch.no_grad():
            inputs = torch.randint(
                model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
            )
            out = model(inputs)
            out_moe = model_moe(inputs)
            torch.testing.assert_close(out, out_moe, atol=self.atol, rtol=self.rtol)

    @pytest.mark.parametrize("n_groups", [2, 4], ids=lambda x: f"n_groups={x}")
    @pytest.mark.parametrize("n_replicas", [4, 8], ids=lambda x: f"n_replicas={x}")
    @pytest.mark.parametrize("hf_weight_transform", ["replicate", "replicate_shuffle"])
    @pytest.mark.parametrize("sharding", ["fsdp", "ep"])
    def test_virtual_group(
        self, sharding: str, n_groups: int, n_replicas: int, hf_weight_transform: str
    ) -> None:
        """
        Test that the dense and MoE models have the same output with FFN weight replication when
        using virtual group init and any applicable weight transformation strategy.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        # Dynamically generate a valid moe cfg for a model with one FFN and one MoE layer.
        llama_3b_hidden_dim = 8192
        model_args_moe = deepcopy(model_args)
        moe_args = MoEArgs(
            num_experts=n_replicas * n_groups,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
            top_k=n_groups,
            route_scale=n_groups,
            hf_ffn_hidden_dim=llama_3b_hidden_dim,  # Must specify for virtual_group router init!
        )
        model_args_moe.moe_args = moe_args
        model_args_moe.moe_inter_dim = llama_3b_hidden_dim // n_groups
        model_args_moe.is_moe_list = [True, False]
        model_args_moe.custom_moe_impl = "virtual_group"

        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path

        moe_args = model_args_moe.moe_args
        assert moe_args.score_func == "softmax"
        assert moe_args.route_norm is True
        assert moe_args.score_before_experts is False
        assert moe_args.hf_ffn_hidden_dim is not None
        assert model_args.custom_moe_impl is None
        assert model_args_moe.custom_moe_impl == "virtual_group"

        with torch.device("meta"):
            model = Llama3MoE(model_args)
            model_moe = Llama3MoE(model_args_moe)

        # Sanity checks:
        assert not any(isinstance(m, MoE) for m in model.modules())
        assert any(isinstance(m, MoE) for m in model_moe.modules())

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size if sharding == "ep" else 1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_moe = parallelize_llama_moe(model_moe, parallel_dims, job_config)

        model.to_empty(device=self.device)
        model_moe.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
            model_moe.init_weights(buffer_device=None)

        ckpt_kwargs = {
            "dataloader": None,
            "optimizers": None,  # HACK: @goon - ok to set to None for initial load
            "lr_schedulers": None,  # HACK: @goon - ok to set to None for initial load
            "states": {"train_state": self},
            "checkpoint_config": job_config.checkpoint,
            "base_folder": "",
            "ft_manager": None,
        }

        checkpointer = CheckpointManager(
            model_parts=[model],
            sd_adapter=Llama3MoEStateDictAdapter(model_args, self.hf_assets_path),
            **ckpt_kwargs,
        )
        checkpointer.load()

        sd_adapter_moe = Llama3MoEStateDictAdapter(model_args_moe, self.hf_assets_path)
        custom_checkpointer = CustomCheckpointManager(
            hf_storage_reader=TransformingHuggingFaceStorageReader,
            hf_storage_reader_kwargs={
                "transform_fn": get_hf_weight_transform_cls(hf_weight_transform)(
                    model_args=model_args_moe,
                    hf_to_titan_fqn_map=sd_adapter_moe.from_hf_map,
                ),
                "state_dict": ModelWrapper([model_moe]).state_dict(),
                "sd_adapter": sd_adapter_moe,
            },
            model_parts=[model_moe],
            sd_adapter=sd_adapter_moe,
            **ckpt_kwargs,
        )
        custom_checkpointer.load()

        with torch.no_grad():
            inputs = torch.randint(
                model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
            )
            out = model(inputs)
            out_moe = model_moe(inputs)
            torch.testing.assert_close(out, out_moe, atol=self.atol, rtol=self.rtol)

    # @pytest.mark.parametrize("n_groups", [2, 4], ids=lambda x: f"n_groups={x}")
    # @pytest.mark.parametrize("n_replicas", [4, 8], ids=lambda x: f"n_replicas={x}")
    # @pytest.mark.parametrize("hf_weight_transform", ["replicate", "replicate_shuffle"])
    # @pytest.mark.parametrize("sharding", ["fsdp", "ep"])
    @pytest.mark.parametrize("n_groups", [8], ids=lambda x: f"n_groups={x}")
    @pytest.mark.parametrize("n_replicas", [1], ids=lambda x: f"n_replicas={x}")
    def test_grad_norm(self, n_groups: int, n_replicas: int) -> None:
        """
        Test that the dense and MoE models have the same output with FFN weight replication when
        using virtual group init and any applicable weight transformation strategy.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        # Dynamically generate a valid moe cfg for a model with one FFN and one MoE layer.
        llama_3b_hidden_dim = 8192
        model_args_moe = deepcopy(model_args)
        moe_args = MoEArgs(
            num_experts=n_replicas * n_groups,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
            top_k=n_groups,
            route_scale=n_groups,
            hf_ffn_hidden_dim=llama_3b_hidden_dim,  # Must specify for virtual_group router init!
        )
        model_args_moe.moe_args = moe_args
        model_args_moe.moe_inter_dim = llama_3b_hidden_dim // n_groups
        model_args_moe.is_moe_list = [True, False]
        model_args_moe.custom_moe_impl = "virtual_group"

        job_config = Llama3MoEJobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path

        moe_args = model_args_moe.moe_args
        assert moe_args.score_func == "softmax"
        assert moe_args.route_norm is True
        assert moe_args.score_before_experts is False
        assert moe_args.hf_ffn_hidden_dim is not None
        assert model_args.custom_moe_impl is None
        assert model_args_moe.custom_moe_impl == "virtual_group"

        with torch.device("meta"):
            model_fsdp = Llama3MoE(model_args_moe)
            model_ep = Llama3MoE(model_args_moe)

        # Sanity checks:
        assert any(isinstance(m, MoE) for m in model_fsdp.modules())
        assert any(isinstance(m, MoE) for m in model_ep.modules())

        parallel_dims_fsdp = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )
        parallel_dims_ep = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size,
            etp=1,
            world_size=self.world_size,
        )
        model_fsdp.to_empty(device=self.device)
        model_ep.to_empty(device=self.device)
        with torch.no_grad():
            # Ensure router params are equal, since these aren't loaded in from ckpt.
            torch.manual_seed(42)
            model_fsdp.init_weights(buffer_device=self.device)
            torch.manual_seed(42)
            model_ep.init_weights(buffer_device=self.device)

        model_fsdp = parallelize_llama_moe(model_fsdp, parallel_dims_fsdp, job_config)
        model_ep = parallelize_llama_moe(model_ep, parallel_dims_ep, job_config)

        ckpt_kwargs = {
            "dataloader": None,
            "optimizers": None,  # HACK: @goon - ok to set to None for initial load
            "lr_schedulers": None,  # HACK: @goon - ok to set to None for initial load
            "states": {"train_state": self},
            "checkpoint_config": job_config.checkpoint,
            "base_folder": "",
            "ft_manager": None,
        }

        sd_adapter_moe = Llama3MoEStateDictAdapter(model_args_moe, self.hf_assets_path)
        custom_checkpointer_fsdp = CustomCheckpointManager(
            hf_storage_reader=TransformingHuggingFaceStorageReader,
            hf_storage_reader_kwargs={
                "transform_fn": get_hf_weight_transform_cls("replicate")(
                    model_args=model_args_moe,
                    hf_to_titan_fqn_map=sd_adapter_moe.from_hf_map,
                ),
                "state_dict": ModelWrapper([model_fsdp]).state_dict(),
                "sd_adapter": sd_adapter_moe,
            },
            model_parts=[model_fsdp],
            sd_adapter=sd_adapter_moe,
            **ckpt_kwargs,
        )
        custom_checkpointer_fsdp.load()
        custom_checkpointer_ep = CustomCheckpointManager(
            hf_storage_reader=TransformingHuggingFaceStorageReader,
            hf_storage_reader_kwargs={
                "transform_fn": get_hf_weight_transform_cls("replicate")(
                    model_args=model_args_moe,
                    hf_to_titan_fqn_map=sd_adapter_moe.from_hf_map,
                ),
                "state_dict": ModelWrapper([model_ep]).state_dict(),
                "sd_adapter": sd_adapter_moe,
            },
            model_parts=[model_ep],
            sd_adapter=sd_adapter_moe,
            **ckpt_kwargs,
        )
        custom_checkpointer_ep.load()

        with torch.no_grad():
            for (n, p1), (_, p2) in zip(
                model_fsdp.named_parameters(), model_ep.named_parameters(), strict=True
            ):
                p1 = p1.full_tensor() if isinstance(p1, DTensor) else p1
                p2 = p2.full_tensor() if isinstance(p2, DTensor) else p2
                torch.testing.assert_close(p1, p2)

        inputs = torch.randint(
            model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
        )
        out_fsdp = model_fsdp(inputs)
        out_ep = model_ep(inputs)
        out_fsdp.pow(2).mean().backward()
        out_ep.pow(2).mean().backward()
        torch.testing.assert_close(out_fsdp, out_ep, atol=1e-1, rtol=1e-2)

        grad_norm_fsdp = dist_utils.clip_grad_norm_(
            [p for p in model_fsdp.parameters()],
            1.0,
            foreach=True,
            pp_mesh=None,
            ep_enabled=parallel_dims_fsdp.ep_enabled,
        )
        grad_norm_ep = dist_utils.clip_grad_norm_(
            [p for p in model_ep.parameters()],
            1.0,
            foreach=True,
            pp_mesh=None,
            ep_enabled=parallel_dims_ep.ep_enabled,
        )

        # grad_norm_ep
        for (n, p1), (_, p2) in zip(
            model_fsdp.named_parameters(), model_ep.named_parameters(), strict=True
        ):
            if p1.grad is None:
                assert p2.grad is None
            else:
                g1 = p1.grad.full_tensor() if isinstance(p1.grad, DTensor) else p1.grad
                g2 = p2.grad.full_tensor() if isinstance(p2.grad, DTensor) else p2.grad
                mean_diff = (
                    2 * (g1 - g2).pow(2).sum().sqrt() / (g1 + g2).pow(2).sum().sqrt()
                )
                mean_diff
