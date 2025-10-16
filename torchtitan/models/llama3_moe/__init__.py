# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.models.llama3_moe.checkpoint import CustomCheckpointManager
from torchtitan.models.llama3_moe.custom_args import JobConfig
from torchtitan.models.llama3_moe.hf_reader import (
    ReplicateMoETransform,
    TransformingHuggingFaceStorageReader,
    get_hf_weight_transform_cls,
)
from torchtitan.models.llama3_moe.infra.parallelize import parallelize_llama_moe
from torchtitan.models.llama3_moe.model.args import TransformerModelArgs
from torchtitan.models.llama3_moe.model.model import Transformer
from torchtitan.models.llama3_moe.model.state_dict_adapter import (
    Llama3MoEStateDictAdapter,
)
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

__all__ = [
    "CustomCheckpointManager",
    "JobConfig",
    "Llama3MoEStateDictAdapter",
    "ReplicateMoETransform",
    "Transformer",
    "TransformerModelArgs",
    "TransformingHuggingFaceStorageReader",
    "get_hf_weight_transform_cls",
    "llama3_configs",
    "parallelize_llama_moe",
    "pipeline_llama",
]


llama3_moe_configs = {
    "debugmodel_1exp": TransformerModelArgs(
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
    ),
    "debugmodel_2exp": TransformerModelArgs(
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
    ),
    "debugmodel_4exp": TransformerModelArgs(
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
    ),
    "debugmodel_8exp": TransformerModelArgs(
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
    ),
    "debugmodel_8exp_small": TransformerModelArgs(
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
    ),
    # https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
    "3B": TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.0,  # Correct?
        multiple_of=256,
        rope_theta=500000,
        is_moe_list=None,
    ),
    # NOTE: @goon - the 3B_2layer and 3B_2layer_halfmoe models are used in
    # torchtitan/tests/llama3_moe/test_dist.py, do not delete!
    #
    "3B_2layer": TransformerModelArgs(
        dim=3072,
        moe_inter_dim=8192,
        n_layers=2,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.0,  # Correct?
        multiple_of=256,
        rope_theta=500000,
        is_moe_list=None,
    ),
    "3B_2layer_halfmoe": TransformerModelArgs(
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
    ),
    "3B_2layer_halfmoe_finegrained": TransformerModelArgs(
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
            hf_ffn_hidden_dim=8192,  # Must specify for replicated router init!
        ),
        is_moe_list=[False, True],
        custom_moe_impl="replicated",  # Must specify for replicated router init!
    ),
    "8B": TransformerModelArgs(
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
    ),
    "8B_2exp": TransformerModelArgs(
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
    ),
    "8B_2exp_4_layer": TransformerModelArgs(
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
    ),
    "8B_4exp": TransformerModelArgs(
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
    ),
    "8B_8exp": TransformerModelArgs(
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
    ),
    # can add other version from torchtitan/models/llama3/__init__.py
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="llama3_moe",
        model_cls=Transformer,
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
    )
