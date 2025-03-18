from typing import Any

import torch
from torch.distributed.device_mesh import init_device_mesh

from dtest import DTest
from torchtitan.models.deepseek_v3.model import (
    ModelArgs,
    MoE,
)


class TestModel(DTest):
    """
    Basic functionality tests.
    """

    batch_size = 2
    seq_len = 32
    # ModelArgs args
    vocab_size: int = 512
    dim: int = 256
    inter_dim: int = 4 * dim
    moe_inter_dim: int = dim // 2
    n_layers: int = 2
    n_dense_layers: int = 1
    n_heads: int = 8
    # moe
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    # mla
    kv_lora_rank: int = dim // 4
    qk_nope_head_dim: int = dim // 8
    qk_rope_head_dim: int = dim // 16
    v_head_dim: int = dim // 8

    dtype = torch.bfloat16

    @property
    def factory_kwargs(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "device": self.device}

    @property
    def n_routed_experts(self) -> int:
        return 4 * self.world_size

    @property
    def model_args(self) -> ModelArgs:
        return ModelArgs(
            vocab_size=self.vocab_size,
            dim=self.dim,
            inter_dim=self.inter_dim,
            moe_inter_dim=self.moe_inter_dim,
            n_layers=self.n_layers,
            n_dense_layers=self.n_dense_layers,
            n_heads=self.n_heads,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_activated_experts,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )

    def test_moe(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        moe = MoE(self.model_args, ep_mesh).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        outputs = moe(inputs)
        assert outputs.shape == inputs.shape
