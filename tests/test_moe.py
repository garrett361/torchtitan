import torch
import torch.nn.functional as F

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
    # batch_size = 2
    batch_size = 1
    # seqlen = 128
    seqlen = 16
    dtype = torch.bfloat16
    n_routed_experts: int = 64
    n_shared_experts: int = 0
    # NOTE: @goon - generate_permute_indices seems to have a bug when n_activated_experts = 1, as it
    # returns permuted_indices as all -1's
    # n_activated_experts: int = 8
    n_activated_experts: int = 1

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

    def test(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="grouped_mm")
        torch.manual_seed(42)
        moe_cg = self._get_moe(moe_mm_impl="cg_grouped_gemm")
        inputs = self._get_inputs()
        with torch.no_grad():
            x, num_tokens_per_expert, permuted_indices = self._get_moe_tensors(
                inputs, moe
            )

            # grouped_mm
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            h = F.silu(
                torch._grouped_mm(x.bfloat16(), moe.experts.w1.bfloat16(), offs=offsets)
            )
            h = h * torch._grouped_mm(
                x.bfloat16(), moe.experts.w3.bfloat16(), offs=offsets
            )
            out = torch._grouped_mm(h, moe.experts.w2.bfloat16(), offs=offsets).type_as(
                x
            )

            # cg_grouped_gemm
            n_experts = moe_cg.experts.w1.shape[0]
            expert_indices = torch.arange(n_experts, dtype=torch.int32, device=x.device)
            expert_indices = expert_indices.repeat_interleave(
                num_tokens_per_expert, output_size=num_tokens_per_expert.sum()
            )
            # NOTE: @goon -  need to pad out to the shape of the actual input tensor. Don't fully
            # understand the padding that is going on in the @expert_parallel wrapper with the
            # torch.vstack call.
            n_padding = x.shape[0] - expert_indices.shape[0]
            padding = torch.zeros(
                n_padding, dtype=expert_indices.dtype, device=expert_indices.device
            )
            expert_indices = torch.cat([expert_indices, padding], dim=0)

            h = F.silu(
                cg_grouped_gemm(
                    x.bfloat16(),
                    moe_cg.experts.w1.bfloat16(),
                    expert_indices=expert_indices,
                    group_size_m=ALIGN_SIZE_M,
                )
            )
            h = h * cg_grouped_gemm(
                x.bfloat16(),
                moe_cg.experts.w3.bfloat16(),
                expert_indices=expert_indices,
                group_size_m=ALIGN_SIZE_M,
            )
            out_cg = cg_grouped_gemm(
                h,
                moe_cg.experts.w2.bfloat16(),
                expert_indices=expert_indices,
                group_size_m=ALIGN_SIZE_M,
            ).type_as(x)

        # Checks. Isolate the trivial and non-trivial indices
        out_zeros = out[permuted_indices == -1]
        torch.testing.assert_close(out_zeros, torch.zeros_like(out_zeros))
        out_cg_zeros = out_cg[permuted_indices == -1]
        torch.testing.assert_close(out_cg_zeros, torch.zeros_like(out_cg_zeros))

        out_non_zeros = out[permuted_indices != -1]
        out_cg_non_zeros = out_cg[permuted_indices != -1]

        rel_err = (out - out_cg).abs().mean() / out.abs().mean()
        assert rel_err <= 1e-2

    def test2(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="grouped_mm")
        torch.manual_seed(42)
        moe_cg = self._get_moe(moe_mm_impl="cg_grouped_gemm")
        inputs = self._get_inputs()
        with torch.no_grad():
            x, num_tokens_per_expert, permuted_indices = self._get_moe_tensors(
                inputs, moe
            )

            # grouped_mm
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            h = torch._grouped_mm(x.bfloat16(), moe.experts.w1.bfloat16(), offs=offsets)
            # cg_grouped_gemm
            n_experts = moe_cg.experts.w1.shape[0]
            expert_indices = torch.arange(n_experts, dtype=torch.int32, device=x.device)
            expert_indices = expert_indices.repeat_interleave(
                num_tokens_per_expert, output_size=num_tokens_per_expert.sum()
            )
            # NOTE: @goon -  need to pad out to the shape of the actual input tensor. Don't fully
            # understand the padding that is going on in the @expert_parallel wrapper with the
            # torch.vstack call.
            n_padding = x.shape[0] - expert_indices.shape[0]
            padding = torch.zeros(
                n_padding, dtype=expert_indices.dtype, device=expert_indices.device
            )
            expert_indices = torch.cat([expert_indices, padding], dim=0)

            h_cg = cg_grouped_gemm(
                x.bfloat16(),
                moe.experts.w1.swapdims(-1, -2).contiguous().bfloat16(),
                expert_indices=expert_indices,
                group_size_m=ALIGN_SIZE_M,
            )
            h_non_zero = h[permuted_indices != -1]
            h_cg_non_zero = h_cg[permuted_indices != -1]
            h_cg_non_zero

    def test3(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="grouped_mm")
        torch.manual_seed(42)
        moe_cg = self._get_moe(moe_mm_impl="cg_grouped_gemm")
        inputs = self._get_inputs()
        with torch.no_grad():
            x, num_tokens_per_expert, permuted_indices, actual_num_tokens_per_expert = (
                self._get_moe_tensors(inputs, moe)
            )
            # TODO: @goon - DELETE
            # x = torch.ones_like(x)
            torch.nn.init.normal_(moe.experts.w1)
            torch.nn.init.zeros_(moe.experts.w1[1:])

            # grouped_mm
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            h = torch._grouped_mm(x.bfloat16(), moe.experts.w1.bfloat16(), offs=offsets)
            # cg_grouped_gemm
            n_experts = moe_cg.experts.w1.shape[0]
            expert_indices = torch.arange(n_experts, dtype=torch.int32, device=x.device)
            expert_indices = expert_indices.repeat_interleave(
                num_tokens_per_expert, output_size=num_tokens_per_expert.sum()
            )
            # NOTE: @goon -  need to pad out to the shape of the actual input tensor. Don't fully
            # understand the padding that is going on in the @expert_parallel wrapper with the
            # torch.vstack call.
            n_padding = x.shape[0] - expert_indices.shape[0]
            padding = torch.full(
                (n_padding,),
                n_experts - 1,
                dtype=expert_indices.dtype,
                device=expert_indices.device,
            )
            expert_indices = torch.cat([expert_indices, padding], dim=0)

            h_cg = cg_grouped_gemm(
                x.bfloat16(),
                moe.experts.w1.swapdims(-1, -2).contiguous().bfloat16(),
                expert_indices=expert_indices,
                group_size_m=ALIGN_SIZE_M,
            )
            h_non_zero = h[permuted_indices != -1]
            h_cg_non_zero = h_cg[permuted_indices != -1]
            diff = h_non_zero - h_cg_non_zero
            diff

    def test_grouped_mm2(self) -> None:
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

    def test4(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="grouped_mm")
        torch.manual_seed(42)
        moe_cg = self._get_moe(moe_mm_impl="cg_grouped_gemm")
        inputs = self._get_inputs()
        with torch.no_grad():
            x, num_tokens_per_expert, permuted_indices, actual_num_tokens_per_expert = (
                self._get_moe_tensors(inputs, moe)
            )
            # TODO: @goon - DELETE
            # x = torch.ones_like(x)
            torch.nn.init.normal_(moe.experts.w1)
            torch.nn.init.zeros_(moe.experts.w1[1:])

            # grouped_mm
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            h = torch._grouped_mm(x.bfloat16(), moe.experts.w1.bfloat16(), offs=offsets)
            # cg_grouped_gemm
            n_experts = moe_cg.experts.w1.shape[0]
            expert_indices = torch.arange(n_experts, dtype=torch.int32, device=x.device)
            expert_indices = expert_indices.repeat_interleave(
                num_tokens_per_expert, output_size=num_tokens_per_expert.sum()
            )
            # NOTE: @goon -  need to pad out to the shape of the actual input tensor. Don't fully
            # understand the padding that is going on in the @expert_parallel wrapper with the
            # torch.vstack call.
            n_padding = x.shape[0] - expert_indices.shape[0]
            padding = torch.full(
                (n_padding,),
                n_experts - 1,
                dtype=expert_indices.dtype,
                device=expert_indices.device,
            )
            expert_indices = torch.cat([expert_indices, padding], dim=0)

            h_cg = cg_grouped_gemm(
                x.bfloat16(),
                moe.experts.w1.swapdims(-1, -2).contiguous().bfloat16(),
                expert_indices=expert_indices,
                group_size_m=ALIGN_SIZE_M,
            )
            h_non_zero = h[permuted_indices != -1]
            h_cg_non_zero = h_cg[permuted_indices != -1]
            diff = h_non_zero - h_cg_non_zero
            diff
