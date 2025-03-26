from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate

from dtest import DTest
from torchtitan.models.deepseek_v3.model import (
    ModelArgs,
    MoE,
)


def _copy_params(dst: nn.Module, src: nn.Module) -> None:
    with torch.no_grad():
        for p_dest, p_src in zip(dst.parameters(), src.parameters()):
            p_dest.data.copy_(p_src.data)


def _test_grads(
    mod: nn.Module, mod_fsdp: nn.Module, tol: float, mesh: DeviceMesh
) -> None:
    for (n, p), (_, p_fsdp) in zip(mod.named_parameters(), mod_fsdp.named_parameters()):
        if p.grad is None:
            assert p_fsdp.grad is None
            return
        grad = p.grad
        grad_fsdp = p_fsdp.grad
        if isinstance(grad_fsdp, DTensor):
            grad_fsdp = grad_fsdp.redistribute(
                mesh, placements=[Replicate() for _ in grad_fsdp.placements]
            ).to_local()
        try:
            torch.testing.assert_close(grad, grad_fsdp, atol=tol, rtol=tol)
        except Exception as e:
            raise RuntimeError(f"Failed on {n=}") from e


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

    def test_moe_ep_fwd(self) -> None:
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

    @pytest.mark.gpu
    def test_moe_ep_bwd(self) -> None:
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

        fully_shard(moe_ep.gate, mesh=ep_mesh)
        if moe_ep.shared_experts is not None:
            fully_shard(moe_ep.shared_experts, mesh=ep_mesh)
        fully_shard(moe_ep, mesh=ep_mesh, ignored_params=moe_ep.experts.parameters())

        inputs = torch.randn(
            self.world_size * self.batch_size,
            self.seq_len,
            self.dim,
            **self.factory_kwargs,
        )
        outputs = moe(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = moe_ep(inputs_ep)

        # Grads should match with an aver-over-batches type loss
        outputs.pow(2).mean().backward()
        outputs_ep.pow(2).mean().backward()

        _test_grads(moe.gate, moe_ep.gate, tol=self.tol, mesh=ep_mesh)
        if moe.shared_experts is not None:
            _test_grads(
                moe.shared_experts, moe_ep.shared_experts, tol=self.tol, mesh=ep_mesh
            )
        for exp_idx in moe_ep.experts:
            _test_grads(
                moe.experts[exp_idx],
                moe_ep.experts[exp_idx],
                tol=self.tol,
                mesh=ep_mesh,
            )
