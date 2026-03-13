# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sonic-MoE integration for torchtitan.

Installation
------------
Sonic-MoE requires Python 3.12+ and has strict dependency versions that conflict
with torchtitan's nightly PyTorch. Install with --no-deps to avoid downgrading torch:

    # 1. Install torchtitan dependencies first (includes nightly torch)
    uv pip install -e .

    # 2. Install sonic-moe WITHOUT dependencies (avoids torch downgrade)
    uv pip install --ignore-requires-python --no-deps -e /path/to/sonic-moe

    # 3. Install sonic-moe's pinned dependencies manually
    uv pip install nvidia-cutlass-dsl==4.4.0 quack-kernels==0.2.5

    # 4. Verify installation
    python -c "from torchtitan.models.sonic_moe import SONIC_MOE_AVAILABLE; print(SONIC_MOE_AVAILABLE)"

Key constraints:
- nvidia-cutlass-dsl==4.4.0 (4.4.1 has breaking API changes in tile scheduler)
- quack-kernels==0.2.5 (0.3.x has incompatible VarlenMTileSchedulerArguments API)
- Torch nightly works despite sonic-moe's pyproject.toml claiming torch<=2.9.1

Replaces expert computation with sonic-moe kernels while keeping torchtitan's
router and token reordering logic. This targets the main MoE bottleneck:
the scatter/gather operations that account for ~45% of MoE time.

Sonic-MoE addresses this via fused gather-and-sum kernels that handle token
dispatch and accumulation in a single operation.

Weight storage:
    Weights are stored in original torchtitan format (w1, w3, w2) for checkpoint
    compatibility. Conversion to sonic-moe's interleaved format happens at forward
    time with negligible overhead.
"""

from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.llama3_moe import VirtualGroupMoE
from torchtitan.models.llama3_moe.model.model import _CustomMoE
from torchtitan.models.moe import GroupedExperts, MoEArgs

if TYPE_CHECKING:
    pass


def _check_sonic_moe_available() -> bool:
    try:
        import sonicmoe  # noqa: F401

        return True
    except ImportError:
        return False


SONIC_MOE_AVAILABLE = _check_sonic_moe_available()


class SonicGroupedExperts(GroupedExperts):
    """Expert computation using sonic-moe kernels.

    Stores weights in original torchtitan format (w1, w3, w2) for checkpoint
    compatibility. Conversion to sonic-moe's interleaved format happens at
    forward time with negligible overhead.

    Args:
        dim: Input/output dimension (model hidden size).
        hidden_dim: Expert intermediate dimension.
        num_experts: Number of experts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,  # Add a default value to avoid errors; unused.
    ):
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
        )
        if not SONIC_MOE_AVAILABLE:
            raise ImportError(
                "sonic-moe is required for SonicGroupedExperts. "
                "Install from: /proj/data-eng/goon/garrett361/sonic-moe"
            )
        self._stream_id: int | None = None
        self.dim = dim
        self.hidden_dim = hidden_dim

        from sonicmoe.enums import ActivationType
        from sonicmoe.functional import moe_general_routing_inputs

        self._act_type = ActivationType.SWIGLU
        self._sonic_routing = moe_general_routing_inputs

    @property
    def stream_id(self) -> int:
        if self._stream_id is None:
            self._stream_id = torch.cuda.current_stream().cuda_stream
        return self._stream_id

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run expert computation using sonic-moe kernels.

        Converts w1/w3 to interleaved format at forward time, then calls
        sonic-moe's fused gather-compute-scatter kernel.

        Note: Unlike GroupedExperts which expects pre-gathered tokens, this takes
        the original tokens and routing indices. The sonic-moe kernel handles
        gather/scatter internally for better performance.

        Args:
            x: Input tokens, shape (num_tokens, dim). Original order (not reordered).
            top_scores: Routing scores, shape (num_tokens * top_k,).
            token_indices: Token indices for each routing slot, shape (num_tokens * top_k,).
            expert_indices: Expert indices for each routing slot, shape (num_tokens * top_k,).

        Returns:
            Expert output, shape (num_tokens, dim).
        """

        # Convert w1/w3 to interleaved format: (E, 2*hidden, dim)
        # Then permute to sonic-moe layout: (2*hidden, dim, E).

        # Because the actual weight shapes may differ from their supposed init valued (due to EP,
        # say), derive them from the actual weight shapes.
        if isinstance(self.w1, DTensor):
            assert isinstance(self.w2, DTensor)
            assert isinstance(self.w3, DTensor)
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3
        num_experts, hidden_dim, dim = w1.shape
        w1_w3 = (
            torch.stack([w1, w3], dim=2)
            .reshape(num_experts, 2 * hidden_dim, dim)
            .permute(1, 2, 0)
        )  # (2*hidden, dim, E)
        w2_sonic = w2.permute(1, 2, 0)  # (dim, hidden, E)

        out, _ = self._sonic_routing(
            x=x,
            router_scores=top_scores,
            token_indices=token_indices.int(),
            expert_indices=expert_indices.int(),
            w1=w1_w3,
            b1=None,
            w2=w2_sonic,
            b2=None,
            E=self.num_experts,
            stream_id=self.stream_id,
            activation_type=self._act_type,
            is_inference_mode_enabled=not self.training,
        )
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_experts={self.num_experts}, "
            f"dim={self.dim}, "
            f"hidden_dim={self.hidden_dim})"
        )


class SonicMoE(_CustomMoE):
    name = "sonic"

    """MoE using sonic-moe kernels for expert computation.

    Inherits routing, load balancing, and shared experts from torchtitan's MoE.
    Replaces the expert forward computation with sonic-moe's fused kernels.

    Design note: Unlike standard MoE which uses TokenReorderer to pre-gather
    tokens before expert computation, SonicMoE passes original tokens and
    routing indices directly to the sonic-moe kernel, which handles
    gather-compute-scatter in a single fused operation. This is why SonicMoE
    overrides forward() rather than just swapping the experts module.

    Stores expert weights in original w1/w3/w2 format for seamless checkpoint
    compatibility. Conversion to sonic-moe format happens at forward time with
    negligible overhead.

    Args:
        moe_args: MoE configuration.
        dim: Model hidden dimension.
        hidden_dim: Expert intermediate dimension.
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        # Initialize parent MoE (creates router, reorderer, shared experts)
        super().__init__(moe_args, dim, hidden_dim)

        # Replace grouped experts with sonic-moe version
        self.experts = SonicGroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using sonic-moe expert kernels.

        Uses torchtitan's router, but replaces expert computation with sonic-moe's
        fused gather-compute-scatter kernels.

        Args:
            x: Input tensor with shape (bs, slen, dim).

        Returns:
            Output tensor with shape (bs, slen, dim).
        """
        bs, slen, dim = x.shape
        num_tokens = bs * slen
        x_flat = x.view(-1, dim)

        # Router: compute scores and expert assignments
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_flat, self.expert_bias
        )

        # Track expert usage for load balancing
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # Token indices: each token repeated top_k times
        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.router.top_k)
            .reshape(-1)
        )

        # Sonic-MoE expert forward: fused gather-compute-scatter
        out_experts = self.experts(
            x=x_flat,
            top_scores=top_scores.view(-1),
            token_indices=token_indices,
            expert_indices=selected_experts_indices.view(-1),
        )

        # Shared experts (if any)
        if self.shared_experts is not None:
            out = self.shared_experts(x_flat) + out_experts
        else:
            out = out_experts

        return out.view(bs, slen, dim)


class SonicVirtualGroupMoE(SonicMoE, VirtualGroupMoE):
    name = "sonic_virtual_group"
