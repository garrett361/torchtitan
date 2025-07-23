import torch

from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.deepseek_v3.model.moe import GroupedExperts, MoE


class TestGroupedExperts:
    dim = 2048
    moe_inter_dim = 1408
    # batch_size = 2
    batch_size = 1
    # seqlen = 128
    seqlen = 4
    dtype = torch.bfloat16
    n_routed_experts: int = 64
    n_shared_experts: int = 0
    # n_activated_experts: int = 8
    n_activated_experts: int = 1

    def _get_args(self, moe_mm_impl: str = "grouped_mm") -> DeepSeekV3ModelArgs:
        return DeepSeekV3ModelArgs(
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            moe_mm_impl=moe_mm_impl,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_routed_experts,
        )

    def _get_moe(self, moe_mm_impl: str = "grouped_mm") -> GroupedExperts:
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

    def test_grouped_mm(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="grouped_mm")
        torch.manual_seed(42)
        moe_for_loop = self._get_moe(moe_mm_impl="for_loop")
        inputs = self._get_inputs()
        with torch.no_grad():
            outputs = moe(inputs)
            outputs_for_loop = moe_for_loop(inputs)
        rel_err = (outputs - outputs_for_loop).abs().mean() / outputs.abs().mean()
        assert rel_err <= 1e-2

    def test_cg_grouped_gemm(self) -> None:
        torch.manual_seed(42)
        moe = self._get_moe(moe_mm_impl="cg_grouped_gemm")
        torch.manual_seed(42)
        moe_for_loop = self._get_moe(moe_mm_impl="for_loop")
        # Check same weights:
        with torch.no_grad():
            for (n, p_for_loop), (_, p) in zip(
                moe_for_loop.named_parameters(), moe.named_parameters(), strict=True
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
            outputs_for_loop = moe_for_loop(inputs)
        rel_err = (outputs - outputs_for_loop).abs().mean() / outputs.abs().mean()
        assert rel_err <= 1e-2
