# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.experiments.llama4.optimizer import build_llama4_optimizers
from torchtitan.models.hybrid_moe.infra.parallelize import parallelize_hybrid_moe
from torchtitan.models.hybrid_moe.infra.pipeline import pipeline_hybrid_moe
from torchtitan.models.hybrid_moe.model.args import HybridMoEModelArgs
from torchtitan.models.hybrid_moe.model.model import HybridMoEModel
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

__all__ = [
    "parallelize_hybrid_moe",
    "HybridMoEModelArgs",
    "HybridMoEModel",
    "hybrid_moe_configs",
]


hybrid_moe_configs = {
    "debugmodel": HybridMoEModelArgs(
        vocab_size=2000,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=3,
        n_dense_layers=1,
        n_heads=16,
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=3,
        route_scale=1.0,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    ),
    "debugmodel_nope": HybridMoEModelArgs(
        vocab_size=2000,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=3,
        n_dense_layers=1,
        n_heads=16,
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=3,
        route_scale=1.0,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        nope=True,
    ),
}


register_train_spec(
    TrainSpec(
        name="hybrid_moe",
        model_cls=HybridMoEModel,
        model_args=hybrid_moe_configs,
        parallelize_fn=parallelize_hybrid_moe,
        pipelining_fn=pipeline_hybrid_moe,
        build_optimizers_fn=build_llama4_optimizers,  # use optimizer hooks to update expert weights
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
