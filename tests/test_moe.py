import pytest
import torch

from torchtitan.experiments.kernels.moe.indices import generate_permute_indices
from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_backward import (
    cg_grouped_gemm,
)
from torchtitan.experiments.llama4.infra.expert_parallel import (
    ALIGN_SIZE_M,
)
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.deepseek_v3.model.moe import GroupedExperts, MoE


class TestGroupedExperts:
    dim = 2048
    moe_inter_dim = 1408
    batch_size = 2
    seqlen = ALIGN_SIZE_M
    dtype = torch.bfloat16
    n_routed_experts: int = 64
    n_shared_experts: int = 0
    n_activated_experts: int = 8

    def _get_args(self, moe_mm_impl: str = "grouped_mm") -> DeepSeekV3ModelArgs:
        return DeepSeekV3ModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            moe_mm_impl=moe_mm_impl,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_activated_experts,
        )

    def _get_moe(self, moe_mm_impl: str = "grouped_mm") -> MoE:
        moe = MoE(self._get_args(moe_mm_impl)).to(dtype=self.dtype, device="cuda")
        moe.init_weights(init_std=0.02, buffer_device="cuda")
        return moe

    def _get_inputs(self) -> GroupedExperts:
        return torch.randn(
            self.batch_size,
            self.seqlen,
            self.dim,
            dtype=self.dtype,
            device="cuda",
        )

    def _get_moe_tensors(
        self, x: torch.Tensor, moe: MoE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs, slen, dim = x.shape

        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            actual_num_tokens_per_expert,
        ) = moe.router(x.reshape(bs * slen, dim), moe.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # Prevent extra local tokens accumulation on evaluation or activation recomputation.
        if moe.load_balance_coeff is not None and torch.is_grad_enabled():
            with torch.no_grad():
                moe.tokens_per_expert.add_(actual_num_tokens_per_expert)
        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        x = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )

        experts_per_ep_rank = moe.experts.w1.shape[0]
        num_ep_ranks = actual_num_tokens_per_expert.shape[0] // experts_per_ep_rank

        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert,
                _,  # offsets,
            ) = generate_permute_indices(
                actual_num_tokens_per_expert,
                experts_per_ep_rank,
                num_ep_ranks,
                x.shape[0] + experts_per_ep_rank * ALIGN_SIZE_M,
                ALIGN_SIZE_M,
            )

        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        x = x[permuted_indices, :]

        return x, num_tokens_per_expert, permuted_indices, actual_num_tokens_per_expert

    def test_grouped_mm(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="grouped_mm")
        torch.manual_seed(42)
        moe_fl = self._get_moe(moe_mm_impl="for_loop")
        inputs = self._get_inputs()
        with torch.no_grad():
            outputs = moe(inputs)
            outputs_for_loop = moe_fl(inputs)
        rel_err = (outputs - outputs_for_loop).abs().mean() / outputs.abs().mean()
        assert rel_err <= 1e-2

    def test_cg_grouped_gemm(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="cg_grouped_gemm")
        torch.manual_seed(42)
        moe_fl = self._get_moe(moe_mm_impl="for_loop")
        # Check same weights:
        with torch.no_grad():
            for (n, p_for_loop), (_, p) in zip(
                moe_fl.named_parameters(), moe.named_parameters(), strict=True
            ):
                if any(k in n for k in ("w1", "w2", "w3")):
                    (
                        torch.testing.assert_close(
                            p_for_loop.data.swapdims(-1, -2).contiguous(),
                            p.data,
                            msg=f"{n=}",
                        )
                    )
                else:
                    torch.testing.assert_close(
                        p_for_loop.data,
                        p.data,
                        msg=f"{n=}",
                    )
        inputs = self._get_inputs()
        with torch.no_grad():
            outputs = moe(inputs)
            outputs_for_loop = moe_fl(inputs)
        rel_err = (outputs - outputs_for_loop).abs().mean() / outputs.abs().mean()
        assert rel_err <= 1e-2

    @pytest.mark.parametrize("alignment", (128, 64, 32, 16))
    def test_zeros(self, alignment: int) -> None:
        torch.manual_seed(42)
        n_experts = 2
        tok_per_expert = alignment
        d_model = 64
        x = torch.randn(
            n_experts * tok_per_expert, d_model, dtype=torch.bfloat16, device="cuda"
        )
        w = torch.randn(
            n_experts, d_model, d_model, dtype=torch.bfloat16, device="cuda"
        )
        with torch.no_grad():
            # Zero out everything but expert zero's weights.
            torch.nn.init.zeros_(w[1:])
            # And zero out the first expert's inputs
            torch.nn.init.zeros_(x[:tok_per_expert])

            num_tokens_per_expert = torch.full(
                (n_experts,), alignment, dtype=torch.int32, device="cuda"
            )

            # Matmul outputs should be all zeros.

            # For-loop.

            # Setup
            M_total = n_experts * tok_per_expert
            num_groups = M_total // alignment
            expert_indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
            for group_idx in range(num_groups):
                start_idx = group_idx * alignment
                end_idx = start_idx + alignment
                # Assign this entire group to one expert
                expert_idx = group_idx % n_experts
                expert_indices[start_idx:end_idx] = expert_idx

            # For-loop compute
            out_fl = torch.zeros((M_total, d_model), device="cuda")
            for g in range(num_groups):
                group_start = g * alignment
                group_end = (g + 1) * alignment
                expert_idx = expert_indices[group_start].item()

                # Compute output for this group
                out_fl[group_start:group_end] = (
                    x[group_start:group_end] @ w[expert_idx].t()
                )
            torch.testing.assert_close(out_fl, torch.zeros_like(out_fl))

            # _grouped_mm
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            out_gm = torch._grouped_mm(x, w, offs=offsets)
            torch.testing.assert_close(out_gm, torch.zeros_like(out_gm))

            # cg_grouped_gemm

            # Create the expert indices in two ways to check correctness.

            # https://github.com/gpu-mode/triton-tutorials/blob/dc1532fc128377fc108146e78b4b49439d4bef45/lesson_5/triton_contiguous_group_gemm/benchmark_grouped_gemm.py#L59

            # cg_grouped_gemm

            # First an alternative expert_indices computation:
            expert_indices_alt = torch.arange(
                n_experts, dtype=torch.int32, device=x.device
            )
            expert_indices_alt = expert_indices_alt.repeat_interleave(
                num_tokens_per_expert, output_size=num_tokens_per_expert.sum()
            )

            # Check equivalence
            torch.testing.assert_close(expert_indices_alt, expert_indices_alt)

            # Then test the outputs
            out_cg = cg_grouped_gemm(
                x,
                w.swapdims(-1, -2).contiguous(),
                expert_indices=expert_indices,
                group_size_m=alignment,
            )
            torch.testing.assert_close(out_cg, torch.zeros_like(out_cg))  # FAILS
