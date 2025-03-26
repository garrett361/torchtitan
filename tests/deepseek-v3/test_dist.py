from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from dtest import DTest
from torchtitan.models.deepseek_v3.model import (
    ModelArgs,
    MoE,
)


def _copy_params(dst: nn.Module, src: nn.Module) -> None:
    with torch.no_grad():
        for p_dest, p_src in zip(dst.parameters(), src.parameters()):
            p_dest.data.copy_(p_src.data)


class TestEP(DTest):
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
    tol = 1e-2

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
        moe = MoE(self.model_args).to(**self.factory_kwargs)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        moe_ep = MoE(self.model_args, ep_mesh).to(**self.factory_kwargs)

        # Force models equal
        _copy_params(moe_ep.gate, moe.gate)
        if self.n_shared_experts > 0:
            _copy_params(moe_ep.shared_experts, moe.shared_experts)
        if self.n_routed_experts > 0:
            for idx in moe_ep.experts:
                _copy_params(moe_ep.experts[idx], moe.experts[idx])

        torch.manual_seed(42 + self.rank)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        outputs = moe(inputs)
        outputs_ep = moe_ep(inputs)
        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)
