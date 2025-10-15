# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.distributed as dist
from dtest import DTest

from torchtitan.components.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.models.llama3_moe import (
    CustomCheckpointManager,
    llama3_moe_configs,
    Llama3MoEStateDictAdapter,
    parallelize_llama_moe,
    ReplicateMoETransform,
    Transformer,
    TransformingHuggingFaceStorageReader,
)
from torchtitan.models.llama3_moe.custom_args import JobConfig


class TestHFReader(DTest):
    hf_assets_path = "/gpfs/goon/models/Llama-3.2-3B-no-tied-weights/"
    seqlen = 64
    bsz = 1
    """
    Test loading correctness
    """

    def test_non_moe_load_equivalence(self) -> None:
        """
        Test e2e equivlance on the full 3B model.
        """
        model_args = llama3_moe_configs["3B"]
        job_config = JobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path
        with torch.device("meta"):
            model = Transformer(model_args)
        model_copy = deepcopy(model)

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_copy = parallelize_llama_moe(model_copy, parallel_dims, job_config)

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        model_copy.to_empty(device=self.device)
        with torch.no_grad():
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
            out_copy = model(inputs)
            torch.testing.assert_close(out, out_copy)

    def test_small_non_moe_load_equivalence(self) -> None:
        """
        Test equivalence (and that load succeeds) on a truncated version of the model with fewer
        layers.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        job_config = JobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path
        with torch.device("meta"):
            model = Transformer(model_args)
        model_copy = deepcopy(model)

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

        model = parallelize_llama_moe(model, parallel_dims, job_config)
        model_copy = parallelize_llama_moe(model_copy, parallel_dims, job_config)

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        model_copy.to_empty(device=self.device)
        with torch.no_grad():
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
        torch.manual_seed(42 + dist.get_rank())
        with torch.no_grad():
            inputs = torch.randint(
                model_args.vocab_size, size=(self.bsz, self.seqlen), device=self.device
            )
            out = model(inputs)
            out_copy = model(inputs)
            torch.testing.assert_close(out, out_copy)

    def test_small_moe_load_replicate_transform(self) -> None:
        """
        Test  (and that load succeeds) on a truncated version of the model with fewer
        layers.
        """
        model_args = llama3_moe_configs["3B_2layer"]
        model_args_moe = llama3_moe_configs["3B_2layer_halfmoe"]
        job_config = JobConfig()
        job_config.checkpoint.enable = True
        job_config.checkpoint.initial_load_in_hf = True
        job_config.model.hf_assets_path = self.hf_assets_path

        with torch.device("meta"):
            model = Transformer(model_args)
            model_moe = Transformer(model_args_moe)

        dist_utils.rank_zero_print(f"{model=}")
        dist_utils.rank_zero_print(f"{model_moe=}")

        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
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
                "transform_fn": ReplicateMoETransform(
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
                        print(f"{w=}\n{w_moe=}\n{w_moe_expert_shard=}")
                        torch.testing.assert_close(w, w_moe_expert_shard)
