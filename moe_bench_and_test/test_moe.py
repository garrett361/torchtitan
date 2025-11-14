# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Union

import torch
from einops import rearrange
from triton.testing import do_bench

from torchtitan.models.moe.moe import FeedForward, MoE, MoEArgs, MoEClone, MoEEmptyLike


class CUDAMemContext:
    def __init__(
        self,
        use_GiB: bool = True,
        filter_patterns: tuple[str, ...] = ("allocated_bytes.all",),
        mem_only: bool = True,
    ) -> None:
        self.use_GiB = use_GiB
        self.filter_patterns = filter_patterns
        self.mem_only = mem_only
        self.before: dict[str, Union[int, float]] = {}
        self.after: dict[str, Union[int, float]] = {}
        self.delta: dict[str, Union[int, float]] = {}

    def _get_mem_dict(self) -> dict[str, Union[int, float]]:
        mem_dict = torch.cuda.memory_stats()
        if self.filter_patterns:
            mem_dict = {
                k: v
                for k, v in mem_dict.items()
                if any(p in k for p in self.filter_patterns)
            }
        if self.mem_only:
            mem_dict = {k: v for k, v in mem_dict.items() if "bytes" in k}
        if self.use_GiB:
            mem_dict = {
                k.replace("bytes", "GiB"): v / 2**30 if "bytes" in k else v
                for k, v in mem_dict.items()
            }
        return mem_dict

    def __enter__(self) -> "CUDAMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        torch.cuda.synchronize()
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}


# NOTE: @goon -  torch.testing.assert_close is very strict and hard to pass. Use the more-lenient
# assert_close from FLA, slightly modified.
# https://github.com/fla-org/flash-linear-attention/blob/3ddba2a043100837a1f6499b5eb6692de71a477b/fla/utils.py?plain=1#L82
def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    assert error_rate < ratio, msg


class TestModel:
    # Default following DSv3 16B
    assert_close_ratio = 1e-2
    bsz = 2
    device = "cuda"
    dim = 2048
    is_moe_list = None
    moe_inter_dim = 1408
    num_experts = 64
    num_shared_experts = 2
    perf_reps = 1000
    perf_warmups = 100
    route_norm = False
    score_before_experts = False
    seqlen = 64
    top_k = 6
    use_grouped_mm = True

    def _get_moe_empty_like_moe_clone_and_moe_layers(
        self, score_before_experts: bool | None = None
    ) -> tuple[MoEEmptyLike, MoEClone, MoE]:
        """
        Create MoEClone and MOE layers with equivalent parameters.
        """
        score_before_experts = score_before_experts or self.score_before_experts
        moe_args = MoEArgs(
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            score_func="softmax",
            route_norm=self.route_norm,
            score_before_experts=score_before_experts,
            top_k=self.top_k,
            use_grouped_mm=self.use_grouped_mm,
        )
        moe_empty_like = MoEEmptyLike(
            moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim
        ).to(device=self.device, dtype=torch.bfloat16)
        moe_clone = MoEClone(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        moe = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )

        moe_empty_like.init_weights(1 / self.dim**0.5, self.device)

        with torch.no_grad():
            # Set the MoE model params equal to the MoEClone ones.
            for p, p2, p3 in zip(
                moe_empty_like.parameters(),
                moe_clone.parameters(),
                moe.parameters(),
                strict=True,
            ):
                p2.data.copy_(p.data)
                p3.data.copy_(p.data)

        return moe_empty_like, moe_clone, moe

    def _get_equiv_layers(self) -> tuple[MoEClone, MoE, FeedForward]:
        """
        Create MoEClone, MOE, and FeedForward layers which are all configured so that they should have
        the same outputs. Accomplished by breaking the FeedForward weights into experts, choosing
        top_k = num_shared_experts, and ensuring that the router gives every expert weight 1.
        """
        top_k = 4
        moe_args = MoEArgs(
            num_experts=self.num_experts,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
            top_k=self.num_experts,
            route_scale=self.num_experts,  # Required for equivalence
            use_grouped_mm=self.use_grouped_mm,
        )
        moe_clone = MoEClone(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        moe = MoE(moe_args, dim=self.dim, hidden_dim=self.moe_inter_dim).to(
            device=self.device, dtype=torch.bfloat16
        )
        ffn = FeedForward(
            dim=self.dim, hidden_dim=self.moe_inter_dim * self.num_experts
        ).to(device=self.device, dtype=torch.bfloat16)

        moe_clone.init_weights(1 / self.dim**0.5, self.device)
        ffn.init_weights(1 / self.dim**0.5)

        with torch.no_grad():
            ffn_w1_experts = rearrange(
                ffn.w1.weight, "(e h) d -> e h d", e=self.num_experts
            )
            ffn_w2_experts = rearrange(
                ffn.w2.weight, "d (e h) -> e d h", e=self.num_experts
            )
            ffn_w3_experts = rearrange(
                ffn.w3.weight, "(e h) d -> e h d", e=self.num_experts
            )
            moe_clone.experts.w1.data.copy_(ffn_w1_experts)
            moe_clone.experts.w2.data.copy_(ffn_w2_experts)
            moe_clone.experts.w3.data.copy_(ffn_w3_experts)

            # Zero out the router weights, so every expert has equal weighting.
            moe_clone.router.gate.weight.zero_()
            # Set the MoE model params equal to the MoEClone ones.
            for p, p2 in zip(moe_clone.parameters(), moe.parameters(), strict=True):
                p2.data.copy_(p.data)

        return moe_clone, moe, ffn

    def test_moe_equivalence(
        self, score_before_experts: bool = True
    ) -> tuple[float, float]:
        torch.manual_seed(42)
        moe_empty_like, moe_clone, moe = (
            self._get_moe_empty_like_moe_clone_and_moe_layers(score_before_experts)
        )
        inputs = torch.randn(
            self.bsz,
            self.seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        out_moe_empty_like = moe_empty_like(inputs)
        out_moe_clone = moe_clone(inputs)
        out_moe = moe(inputs)

        assert_close(
            "moe_clone vs moe", out_moe_clone, out_moe, self.assert_close_ratio
        )
        assert_close(
            "moe_empty_like vs moe",
            out_moe_empty_like,
            out_moe,
            self.assert_close_ratio,
        )

        out_moe_empty_like.pow(2).mean().backward()
        out_moe_clone.pow(2).mean().backward()
        out_moe.pow(2).mean().backward()

        for p1, p2, p3 in zip(
            moe_empty_like.parameters(),
            moe_clone.parameters(),
            moe.parameters(),
            strict=True,
        ):
            assert_close(
                "grad",
                p1.grad,
                p2.grad,
                self.assert_close_ratio,
            )
            assert_close(
                "grad",
                p1.grad,
                p3.grad,
                self.assert_close_ratio,
            )

    def test_moe_ffn_equivalence(self, iteration: int = 0) -> tuple[float, float]:
        torch.manual_seed(42 + iteration)
        moe_clone, moe, ffn = self._get_equiv_layers()
        with torch.no_grad():
            inputs = torch.randn(
                self.bsz,
                self.seqlen,
                self.dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            out_moe_clone = moe_clone(inputs)
            out_moe = moe(inputs)
            out_ffn = ffn(inputs)

            assert_close(
                "moe_clone vs ffn", out_ffn, out_moe_clone, self.assert_close_ratio
            )
            assert_close("moe vs ffn", out_ffn, out_moe, self.assert_close_ratio)

            moe_clone_rel_err = get_err_ratio(out_ffn, out_moe_clone)
            moe_rel_err = get_err_ratio(out_ffn, out_moe)
            return moe_clone_rel_err, moe_rel_err

    def test_perf(
        self, bsz: int | None = None, seqlen: int | None = None
    ) -> tuple[float, float]:
        seqlen = seqlen or self.seqlen
        bsz = bsz or self.bsz
        torch.manual_seed(42)
        moe_empty_like, moe_clone, moe = (
            self._get_moe_empty_like_moe_clone_and_moe_layers()
        )
        inputs = torch.randn(
            bsz,
            seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        moe_empty_like_time_ms = do_bench(
            lambda: moe_empty_like(inputs).sum().backward(),
            warmup=self.perf_warmups,
            rep=self.perf_reps,
        )
        moe_clone_time_ms = do_bench(
            lambda: moe_clone(inputs).sum().backward(),
            warmup=self.perf_warmups,
            rep=self.perf_reps,
        )
        moe_time_ms = do_bench(
            lambda: moe(inputs).sum().backward(),
            warmup=self.perf_warmups,
            rep=self.perf_reps,
        )
        print(f"{moe_empty_like_time_ms=}")
        print(f"{moe_clone_time_ms=}")
        print(f"{moe_time_ms=}")

        print(f"Speedup vs empty_like: {moe_empty_like_time_ms/moe_time_ms=}")
        print(f"Speedup vs clone: {moe_clone_time_ms/moe_time_ms=}")

    def test_determinism(self):
        torch.manual_seed(42)
        moe_empty_like, moe_clone, moe = (
            self._get_moe_empty_like_moe_clone_and_moe_layers(
                score_before_experts=False
            )
        )
        inputs = torch.randn(
            self.bsz,
            self.seqlen,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        out_moe_empty_like = []
        out_moe_clone = []
        out_moe = []
        with torch.no_grad():
            for _ in range(100):
                out_moe_empty_like.append(moe_empty_like(inputs))
                out_moe_clone.append(moe_clone(inputs))
                out_moe.append(moe(inputs))

        out_moe_empty_like = torch.stack(out_moe_empty_like, dim=0)
        out_moe_clone = torch.stack(out_moe_clone, dim=0)
        out_moe = torch.stack(out_moe, dim=0)

        out_empty_like_std = out_moe_empty_like.std(dim=0).mean()
        out_clone_std = out_moe_clone.std(dim=0).mean()
        out_std = out_moe.std(dim=0).mean()

        print(f"{out_empty_like_std=}")
        print(f"{out_clone_std=}")
        print(f"{out_std=}")

        torch.testing.assert_close(out_std, torch.zeros_like(out_std))
        torch.testing.assert_close(out_clone_std, torch.zeros_like(out_clone_std))
        torch.testing.assert_close(
            out_empty_like_std, torch.zeros_like(out_empty_like_std)
        )


if __name__ == "__main__":
    t = TestModel()

    # Collect some accuracy stats
    moe_clone_rel_errs = []
    moe_rel_errs = []
    accuracy_iters = 10
    for idx in range(accuracy_iters):
        moe_clone_rel_err, moe_rel_err = t.test_moe_ffn_equivalence(idx)
        moe_clone_rel_errs.append(moe_clone_rel_err)
        moe_rel_errs.append(moe_rel_err)
    mean_moe_clone_rel_err = torch.tensor(moe_clone_rel_errs)
    mean_moe_rel_err = torch.tensor(moe_rel_errs)

    print(f"\nACCURACY VS FFN: {accuracy_iters} iterations\n")
    print(f"{mean_moe_clone_rel_err.mean()=}, {mean_moe_clone_rel_err.std()=}")
    print(f"{mean_moe_rel_err.mean()=}, {mean_moe_rel_err.std()=}")
    print(f"{mean_moe_clone_rel_err.mean()/mean_moe_rel_err.mean()=}")

    # Perf bsz and seqlen as in torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml
    perf_seqlen = 4096
    perf_bsz = 4
    print(
        f"\nTRITON BENCH: {perf_seqlen=} {perf_bsz=} warmups={t.perf_warmups} repeats={t.perf_reps}\n"
    )
    t.test_perf(bsz=perf_bsz, seqlen=perf_seqlen)

    t.test_moe_equivalence(True)
    print("\nMoEClone AND MoE CLOSE: score_before_experts=True")
    t.test_moe_equivalence(False)
    print("\nMoEClone AND MoE CLOSE: score_before_experts=False")

    print("\nDeterminism")
    t.test_determinism()
