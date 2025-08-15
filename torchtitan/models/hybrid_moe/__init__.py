# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from copy import deepcopy

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_metrics_processor
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.hybrid_moe._metrics import (
    HybridMoEMetricsProcessor,
    build_hybrid_moe_metrics_processor,
)
from torchtitan.models.hybrid_moe.infra.parallelize import parallelize_hybrid_moe
from torchtitan.models.hybrid_moe.infra.pipeline import pipeline_hybrid_moe
from torchtitan.models.hybrid_moe.model.args import HybridMoEModelArgs
from torchtitan.models.hybrid_moe.model.model import HybridMoEModel
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

__all__ = [
    "parallelize_hybrid_moe",
    "HybridMoEModelArgs",
    "HybridMoEModel",
    "hybrid_moe_configs",
]

# HACK: create a dict subclass which we can use to easily generate configs based on a given key.
# Dev keys must be in the form dev.key=value|key=value|


# Default modified from from DSv3 16B
DEV_KWARG_DEFAULTS = dict(
    dim=2048,
    inter_dim=10944,
    moe_inter_dim=1408,
    n_layers=27,
    n_dense_layers=1,
    n_heads=16,
    moe_args=MoEArgs(
        num_experts=64,
        num_shared_experts=2,
        top_k=6,
        score_func="softmax",
        route_norm=True,
        score_before_experts=False,
    ),
    q_lora_rank=0,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    mscale=0.70,
    # TODO: @goon - DELETE - temp no mamba layers
    mha_layer_interval=1,

)


class devdict(dict):
    def __missing__(self, key: str) -> HybridMoEModelArgs:
        # Just supporting (str, int | float) pairs for now
        kwargs = deepcopy(DEV_KWARG_DEFAULTS)
        for pair in key.split("|"):
            split_pair = pair.split("=")
            if len(split_pair) != 2:
                raise ValueError(
                    f"Invalid {key=} does not exist is not a dev key of the form k0=v0|k1=v1|..."
                )
            key, value = pair.split("=")
            value = eval(value)
            kwargs[key] = value
        return HybridMoEModelArgs(**kwargs)


hybrid_moe_configs = devdict(
    debugmodel=HybridMoEModelArgs(
        vocab_size=2000,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=3,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=2,
            top_k=3,
            route_scale=1.0,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        mha_layer_interval=3,
    ),
    dsv3=HybridMoEModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        n_heads=128,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            route_scale=2.5,
            score_func="sigmoid",
        ),
        n_expert_groups=8,
        n_limited_groups=4,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dtype="fp8",
    ),
)

register_train_spec(
    TrainSpec(
        name="hybrid_moe",
        model_cls=HybridMoEModel,
        model_args=hybrid_moe_configs,
        parallelize_fn=parallelize_hybrid_moe,
        pipelining_fn=pipeline_hybrid_moe,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,  # use optimizer hooks to update expert weights
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_metrics_processor_fn=build_hybrid_moe_metrics_processor,
    )
)
