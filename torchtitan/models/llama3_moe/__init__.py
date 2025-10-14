# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama_moe
from .model.args import TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Llama3MoEStateDictAdapter

__all__ = [
    "parallelize_llama_moe",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


llama3_moe_configs = {
    "debugmodel_1exp": TransformerModelArgs(
        dim=256,
        moe_inter_dim=1024,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        moe_args=MoEArgs(
            num_experts=1,
            num_shared_experts=0,
        ),
    ),
    "debugmodel_2exp": TransformerModelArgs(
        dim=256,
        moe_inter_dim=1024,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        moe_args=MoEArgs(
            num_experts=2,
            num_shared_experts=0,
        ),
    ),
    "debugmodel_4exp": TransformerModelArgs(
        dim=256,
        moe_inter_dim=1024,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=0,
        ),
    ),
    "debugmodel_8exp": TransformerModelArgs(
        dim=256,
        moe_inter_dim=1024,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=0,
        ),
    ),
    "debugmodel_8exp_small": TransformerModelArgs(
        dim=64,
        moe_inter_dim=128,
        n_layers=8,
        n_heads=4,
        vocab_size=2048,
        rope_theta=500000,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=0,
        ),
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
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3MoEStateDictAdapter,
    )
