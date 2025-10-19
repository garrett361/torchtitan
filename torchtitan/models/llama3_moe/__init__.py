# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.models.llama3_moe.checkpoint import CustomCheckpointManager
from torchtitan.models.llama3_moe.custom_args import (
    Llama3MoEJobConfig,
    TopKSchedulerArgs,
)
from torchtitan.models.llama3_moe.hf_reader import (
    ReplicateMoETransform,
    TransformingHuggingFaceStorageReader,
    get_hf_weight_transform_cls,
)
from torchtitan.models.llama3_moe.infra.parallelize import parallelize_llama_moe
from torchtitan.models.llama3_moe.metrics import (
    CustomMetricsProcessor,
    build_custom_metrics_processor,
)
from torchtitan.models.llama3_moe.model.args import Llama3MoEModelArgs
from torchtitan.models.llama3_moe.model.model import Llama3MoE, VirtualGroupMoE
from torchtitan.models.llama3_moe.model.state_dict_adapter import (
    Llama3MoEStateDictAdapter,
)
from torchtitan.models.llama3_moe.top_k_scheduler import get_top_k_scheduler
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

__all__ = [
    "CustomCheckpointManager",
    "CustomMetricsProcessor",
    "Llama3MoE",
    "Llama3MoEJobConfig",
    "Llama3MoEModelArgs",
    "Llama3MoEStateDictAdapter",
    "ReplicateMoETransform",
    "TopKSchedulerArgs",
    "TopKSchedulerArgs",
    "TransformingHuggingFaceStorageReader",
    "VirtualGroupMoE",
    "build_custom_metrics_processor",
    "get_hf_weight_transform_cls",
    "get_top_k_scheduler",
    "llama3_configs",
    "parallelize_llama_moe",
    "pipeline_llama",
]

DEV_CFG_3B = Llama3MoEModelArgs(
    dim=3072,
    moe_inter_dim=8192,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=1.0,  # Correct?
    multiple_of=256,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
        top_k=2,
        route_scale=1,  # Must have route_scale = top_k; see [Virtual Group Initialization].
        hf_ffn_hidden_dim=8192,
    ),
    is_moe_list=None,
    custom_moe_impl="virtual_group",  # Must specify for virtual_group router init!
)


class devdict(dict):  # noqa: N801
    """
    Hacky dict which lets us generate model configs dynamically.

    Usage: key "3B_moe|num_experts=32|top_k=4|n_moe=8"
    """

    def __missing__(self, key: str) -> Llama3MoEModelArgs:
        # Just supporting (str, int | float) pairs for now
        dev_key = "3B_dev"
        if not key.startswith(dev_key):
            raise ValueError(f"devdict {key=} must start with {dev_key}")
        key = key.replace(dev_key + "|", "")
        cfg = deepcopy(DEV_CFG_3B)
        dev_kwargs = {}
        for pair in key.split("|"):
            k, v = pair.split("=")
            dev_kwargs[k] = eval(v)
        for k in list(dev_kwargs):
            if hasattr(cfg, k):
                setattr(cfg, k, dev_kwargs.pop(k))
            elif hasattr(cfg.moe_args, k):
                setattr(cfg.moe_args, k, dev_kwargs.pop(k))
        # Special args:
        # - n_moe: use this many MoE layers, starting from the last layer backwards
        if "n_moe" in dev_kwargs:
            n_moe = dev_kwargs.pop("n_moe")
            cfg.is_moe_list = [n >= cfg.n_layers - n_moe for n in range(cfg.n_layers)]

        if dev_kwargs:
            raise ValueError(f"Unused {dev_kwargs=}. Probably misconfigured? ")

        return cfg


llama3_moe_configs = devdict()
llama3_moe_configs["debugmodel_1exp"] = Llama3MoEModelArgs(
    dim=256,
    moe_inter_dim=1024,
    n_layers=6,
    n_heads=16,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=1,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    is_moe_list=[True if n == 0 else False for n in range(6)],
)
llama3_moe_configs["debugmodel_2exp"] = Llama3MoEModelArgs(
    dim=256,
    moe_inter_dim=1024,
    n_layers=6,
    n_heads=16,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=2,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    is_moe_list=[True if n == 0 else False for n in range(6)],
)
llama3_moe_configs["debugmodel_4exp"] = Llama3MoEModelArgs(
    dim=256,
    moe_inter_dim=1024,
    n_layers=6,
    n_heads=16,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    is_moe_list=[True if n == 0 else False for n in range(6)],
)
llama3_moe_configs["debugmodel_8exp"] = Llama3MoEModelArgs(
    dim=256,
    moe_inter_dim=1024,
    n_layers=6,
    n_heads=16,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    is_moe_list=[True if n == 0 else False for n in range(6)],
)
llama3_moe_configs["debugmodel_8exp_small"] = Llama3MoEModelArgs(
    dim=64,
    moe_inter_dim=128,
    n_layers=6,
    n_heads=4,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    is_moe_list=[True if n == 0 else False for n in range(6)],
)
# https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
llama3_moe_configs["3B"] = Llama3MoEModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=1.0,  # Correct?
    multiple_of=256,
    rope_theta=500000,
    is_moe_list=None,
)
# NOTE: @goon - the 3B_2layer and 3B_2layer_halfmoe models are used in
# torchtitan/tests/llama3_moe/test_dist.py, do not delete!
#
llama3_moe_configs["3B_2layer"] = Llama3MoEModelArgs(
    dim=3072,
    moe_inter_dim=8192,
    n_layers=2,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=1.0,  # Correct?
    multiple_of=256,
    rope_theta=500000,
    is_moe_list=None,
)
llama3_moe_configs["3B_2layer_halfmoe"] = Llama3MoEModelArgs(
    dim=3072,
    moe_inter_dim=8192,
    n_layers=2,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=1.0,  # Correct?
    multiple_of=256,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
        top_k=2,
    ),
    is_moe_list=[False, True],
)
# See VirtualGroupMoE for necessary cfg requirements for virtual_group init.
llama3_moe_configs["3B_2layer_halfmoe_finegrained"] = Llama3MoEModelArgs(
    dim=3072,
    moe_inter_dim=8192 // 2,
    n_layers=2,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=1.0,  # Correct?
    multiple_of=256,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=8 * 2,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
        top_k=2,
        route_scale=2,  # Must have route_scale = top_k; see [Virtual Group Initialization].
        hf_ffn_hidden_dim=8192,  # Must specify for virtual_group router init!
    ),
    is_moe_list=[False, True],
    custom_moe_impl="virtual_group",  # Must specify for virtual_group router init!
)
llama3_moe_configs["8B"] = Llama3MoEModelArgs(
    dim=4096,
    moe_inter_dim=14336,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=2,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    is_moe_list=None,
)
llama3_moe_configs["8B_2exp"] = Llama3MoEModelArgs(
    dim=4096,
    moe_inter_dim=14336,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=2,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
)
llama3_moe_configs["8B_2exp_4_layer"] = Llama3MoEModelArgs(
    dim=4096,
    moe_inter_dim=14336,
    n_layers=4,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=2,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
)
llama3_moe_configs["8B_4exp"] = Llama3MoEModelArgs(
    dim=4096,
    moe_inter_dim=14336,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
)
llama3_moe_configs["8B_8exp"] = Llama3MoEModelArgs(
    dim=4096,
    moe_inter_dim=14336,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
)


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="llama3_moe",
        model_cls=Llama3MoE,
        model_args=llama3_moe_configs,
        parallelize_fn=parallelize_llama_moe,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3MoEStateDictAdapter,
        build_metrics_processor_fn=build_custom_metrics_processor,
    )
