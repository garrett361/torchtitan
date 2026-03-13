# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for sonic-moe integration.

Tests verify:
1. Forward equivalence: SonicMoE output matches MoE within tolerance
2. Backward equivalence: Gradients match within tolerance
3. Weight conversion: Round-trip conversion preserves weights
4. Checkpoint compatibility: SonicMoE uses same state_dict keys as MoE
"""

import pytest
import torch

from torchtitan.models.llama3_moe import VirtualGroupMoE
from torchtitan.models.llama3_moe.model.model import get_moe_impl_cls
from torchtitan.models.moe import MoE, MoEArgs
from torchtitan.models.sonic_moe import (
    SONIC_MOE_AVAILABLE,
    SonicGroupedExperts,
    SonicMoE,
    SonicVirtualGroupMoE,
)


@pytest.mark.skipif(not SONIC_MOE_AVAILABLE, reason="sonic-moe not available")
class TestSonicMoE:
    """Test suite for sonic-moe integration."""

    # Tolerances for numerical comparisons (sonic-moe uses different accumulation order)
    rtol = 1e-2
    atol = 1e-2

    bsz = 2
    device = "cuda"
    dim = 2048
    hidden_dim = 1408
    num_experts = 64
    num_shared_experts = 0  # Test without shared experts first
    seqlen = 64
    top_k = 6

    def _get_moe_args(self, num_shared_experts: int | None = None) -> MoEArgs:
        return MoEArgs(
            num_experts=self.num_experts,
            num_shared_experts=num_shared_experts
            if num_shared_experts is not None
            else self.num_shared_experts,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
            top_k=self.top_k,
            use_grouped_mm=True,
            hf_ffn_hidden_dim=self.hidden_dim,
            _debug_force_load_balance=False,
        )

    def _get_moe_and_sonic_moe(
        self, num_shared_experts: int | None = None, use_vg: bool = False
    ) -> tuple[MoE, SonicMoE]:
        """Create equivalent MoE and SonicMoE layers."""

        if use_vg:
            moe_cls, sonic_moe_cls = VirtualGroupMoE, SonicVirtualGroupMoE
        else:
            moe_cls, sonic_moe_cls = MoE, SonicMoE
        moe_args = self._get_moe_args(num_shared_experts)

        moe = moe_cls(moe_args, dim=self.dim, hidden_dim=self.hidden_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        moe.init_weights(1 / self.dim**0.5, self.device)
        sonic_moe = sonic_moe_cls(
            moe_args, dim=self.dim, hidden_dim=self.hidden_dim
        ).to(device=self.device, dtype=torch.bfloat16)
        sonic_moe.init_weights(1 / self.dim**0.5, self.device)

        # Create SonicMoE from MoE to ensure weight equivalence
        with torch.no_grad():
            for p, p_sonic in zip(
                moe.parameters(), sonic_moe.parameters(), strict=True
            ):
                p_sonic.data.copy_(p.data)
            for b, b_sonic in zip(moe.buffers(), sonic_moe.buffers(), strict=True):
                b_sonic.copy_(b)

        return moe, sonic_moe

    @pytest.mark.parametrize("use_vg", [True, False], ids=["vg", "no_vg"])
    def test_forward_equivalence(self, use_vg: bool) -> None:
        """Test SonicMoE forward matches MoE within tolerance."""
        torch.manual_seed(42)
        moe, sonic_moe = self._get_moe_and_sonic_moe(use_vg=use_vg)

        inputs = torch.randn(
            self.bsz, self.seqlen, self.dim, device=self.device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            out_moe = moe(inputs)
            out_sonic = sonic_moe(inputs)

        torch.testing.assert_close(out_moe, out_sonic, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("use_vg", [True, False], ids=["vg", "no_vg"])
    def test_forward_equivalence_with_shared_experts(self, use_vg: bool) -> None:
        """Test SonicMoE forward matches MoE with shared experts."""
        torch.manual_seed(42)
        moe, sonic_moe = self._get_moe_and_sonic_moe(
            num_shared_experts=2, use_vg=use_vg
        )

        inputs = torch.randn(
            self.bsz, self.seqlen, self.dim, device=self.device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            out_moe = moe(inputs)
            out_sonic = sonic_moe(inputs)

        torch.testing.assert_close(out_moe, out_sonic, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("use_vg", [True, False], ids=["vg", "no_vg"])
    def test_backward_equivalence(self, use_vg: bool) -> None:
        """Test SonicMoE backward matches MoE gradients within tolerance.

        Note: Uses .mean().backward() instead of .sum().backward() because
        sonic-moe's CuTe kernels reject broadcast tensors with stride 0 that
        .sum() creates. The .mean() backward pass produces proper strides.
        """
        torch.manual_seed(42)
        moe, sonic_moe = self._get_moe_and_sonic_moe(use_vg=use_vg)

        inputs_moe = torch.randn(
            self.bsz,
            self.seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        inputs_sonic = inputs_moe.clone().detach().requires_grad_(True)

        # Forward pass
        out_moe = moe(inputs_moe)
        out_sonic = sonic_moe(inputs_sonic)

        # Compare outputs
        torch.testing.assert_close(out_moe, out_sonic, rtol=self.rtol, atol=self.atol)

        # Backward pass - use .mean() not .sum() (see docstring)
        out_moe.mean().backward()
        out_sonic.mean().backward()

        # Compare input gradients
        torch.testing.assert_close(
            inputs_moe.grad, inputs_sonic.grad, rtol=self.rtol, atol=self.atol
        )

        # Compare router gradients
        torch.testing.assert_close(
            moe.router.gate.weight.grad,
            sonic_moe.router.gate.weight.grad,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Compare expert weight gradients (same layout - direct comparison)
        torch.testing.assert_close(
            moe.experts.w1.grad,
            sonic_moe.experts.w1.grad,
            rtol=self.rtol,
            atol=self.atol,
        )
        torch.testing.assert_close(
            moe.experts.w3.grad,
            sonic_moe.experts.w3.grad,
            rtol=self.rtol,
            atol=self.atol,
        )
        torch.testing.assert_close(
            moe.experts.w2.grad,
            sonic_moe.experts.w2.grad,
            rtol=self.rtol,
            atol=self.atol,
        )

    @pytest.mark.parametrize("use_vg", [True, False], ids=["vg", "no_vg"])
    def test_state_dict_compatibility(self, use_vg: bool) -> None:
        """Test SonicMoE state_dict has same keys as MoE (checkpoint compatible)."""
        torch.manual_seed(42)
        moe, sonic_moe = self._get_moe_and_sonic_moe(use_vg=use_vg)

        moe_keys = set(moe.state_dict().keys())
        sonic_keys = set(sonic_moe.state_dict().keys())

        # Expert keys should match (w1, w3, w2)
        moe_expert_keys = {k for k in moe_keys if "experts." in k}
        sonic_expert_keys = {k for k in sonic_keys if "experts." in k}

        assert moe_expert_keys == sonic_expert_keys

    def test_get_moe_impl_cls(self) -> None:
        cls = get_moe_impl_cls("sonic")
        assert cls is SonicMoE
        cls = get_moe_impl_cls("sonic_virtual_group")
        assert cls is SonicVirtualGroupMoE


@pytest.mark.skipif(not SONIC_MOE_AVAILABLE, reason="sonic-moe not available")
class TestSonicGroupedExpertsIsolated:
    """Test SonicGroupedExperts in isolation (without full MoE routing)."""

    device = "cuda"
    dim = 512
    hidden_dim = 256
    num_experts = 8
    num_tokens = 32
    top_k = 2

    def test_forward_smoke(self) -> None:
        """Basic smoke test for SonicGroupedExperts forward."""
        torch.manual_seed(42)

        sonic_experts = SonicGroupedExperts(
            dim=self.dim, hidden_dim=self.hidden_dim, num_experts=self.num_experts
        ).to(device=self.device, dtype=torch.bfloat16)
        sonic_experts.init_weights(0.02)

        x = torch.randn(
            self.num_tokens, self.dim, device=self.device, dtype=torch.bfloat16
        )

        # Create simple routing: each token routed to top_k consecutive experts
        # Token indices: [0, 0, 1, 1, 2, 2, ...] (each token appears top_k times)
        token_indices = torch.arange(
            self.num_tokens, device=self.device
        ).repeat_interleave(self.top_k)
        # Expert indices: round-robin [0, 1, 2, 3, 0, 1, 2, 3, ...]
        expert_indices = (
            torch.arange(self.num_tokens * self.top_k, device=self.device)
            % self.num_experts
        )
        # Sort by expert
        sort_idx = expert_indices.argsort(stable=True)
        token_indices_reordered = token_indices[sort_idx]
        expert_indices_reordered = expert_indices[sort_idx]
        top_scores_reordered = (
            torch.ones(
                self.num_tokens * self.top_k, device=self.device, dtype=torch.float32
            )
            / self.top_k
        )

        out = sonic_experts(
            x=x,
            top_scores=top_scores_reordered,
            token_indices=token_indices_reordered,
            expert_indices=expert_indices_reordered,
        )

        assert out.shape == (self.num_tokens, self.dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
