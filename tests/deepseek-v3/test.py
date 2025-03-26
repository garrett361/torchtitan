import torch

from torchtitan.models.deepseek_v3.model import (
    MLA,
    MLP,
    Block,
    DeepSeekV3,
    Expert,
    Gate,
    ModelArgs,
    MoE,
    precompute_freqs_cis,
)


class TestLayers:
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
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    # mla
    kv_lora_rank: int = dim // 4
    qk_nope_head_dim: int = dim // 8
    qk_rope_head_dim: int = dim // 16
    v_head_dim: int = dim // 8

    model_args = ModelArgs(
        vocab_size=vocab_size,
        dim=dim,
        inter_dim=inter_dim,
        moe_inter_dim=moe_inter_dim,
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        n_heads=n_heads,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        n_activated_experts=n_activated_experts,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
    )
    dtype = torch.bfloat16
    device = "cuda"
    factory_kwargs = {"dtype": dtype, "device": device}

    def test_mlp(self) -> None:
        mlp = MLP(self.dim, self.inter_dim).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        outputs = mlp(inputs)
        assert outputs.shape == inputs.shape

    def test_expert(self) -> None:
        expert = Expert(self.dim, self.inter_dim).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        outputs = expert(inputs)
        assert outputs.shape == inputs.shape

    def test_gate(self) -> None:
        gate = Gate(self.model_args).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size * self.seq_len, self.dim, **self.factory_kwargs
        )
        weights, indices = gate(inputs)
        assert weights.shape == torch.Size(
            (self.batch_size * self.seq_len, self.n_activated_experts)
        )
        assert indices.shape == torch.Size(
            (self.batch_size * self.seq_len, self.n_activated_experts)
        )

    def test_moe(self) -> None:
        moe = MoE(self.model_args).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        outputs = moe(inputs)
        assert outputs.shape == inputs.shape

    def test_mla(self) -> None:
        mla = MLA(self.model_args).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        freqs_cis = precompute_freqs_cis(self.model_args).to(self.device)
        freqs_cis_slice = freqs_cis[: self.seq_len]

        outputs = mla(inputs, freqs_cis_slice)
        assert outputs.shape == inputs.shape

    def test_block_dense(self) -> None:
        block_dense = Block(
            layer_id=self.model_args.n_dense_layers - 1, args=self.model_args
        ).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        freqs_cis = precompute_freqs_cis(self.model_args).to(self.device)
        freqs_cis_slice = freqs_cis[: self.seq_len]

        outputs = block_dense(inputs, freqs_cis_slice)
        assert outputs.shape == inputs.shape

    def test_block_moe(self) -> None:
        block_moe = Block(
            layer_id=self.model_args.n_dense_layers + 1, args=self.model_args
        ).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )

        freqs_cis = precompute_freqs_cis(self.model_args).to(self.device)
        freqs_cis_slice = freqs_cis[: self.seq_len]

        outputs = block_moe(inputs, freqs_cis_slice)
        assert outputs.shape == inputs.shape

    def test_model(self) -> None:
        model = DeepSeekV3(args=self.model_args).to(**self.factory_kwargs)
        inputs = torch.randint(
            self.vocab_size,
            size=(
                self.batch_size,
                self.seq_len,
            ),
            device=self.device,
        )

        outputs = model(inputs)
        assert outputs.shape == torch.Size(
            (self.batch_size, self.seq_len, self.vocab_size)
        )

    def test_model_bwd(self) -> None:
        model = DeepSeekV3(args=self.model_args).to(**self.factory_kwargs)
        inputs = torch.randint(
            self.vocab_size,
            size=(
                self.batch_size,
                self.seq_len,
            ),
            device=self.device,
        )

        model(inputs).sum().backward()

    def test_mock_ep(self) -> None:
        torch.manual_seed(42)
        expert = Gate(self.model_args).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size * self.seq_len, self.dim, **self.factory_kwargs
        )
        # Original
        weights, indices = expert(inputs)
        outputs = torch.zeros_like(inputs)
        for i in range(self.n_routed_experts):
            idx, top = torch.where(indices == i)
            outputs[idx] += (2 * i - 1) * inputs[idx] * weights[idx, top, None]
            # outputs[idx] += inputs[idx]

        # EP
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        inputs_flat_sorted = inputs[flat_sorted_indices // self.n_activated_experts]
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        local_expert_idxs = torch.arange(
            counts.numel(),
            device=counts.device,
        )

        local_expert_idxs = local_expert_idxs.repeat_interleave(counts)
        temp = torch.empty_like(inputs_flat_sorted)
        temp2 = torch.empty_like(inputs_flat_sorted)
        for i in range(self.n_routed_experts):
            temp[local_expert_idxs == i] = (2 * i - 1) * inputs_flat_sorted[
                local_expert_idxs == i
            ]
            # temp[local_expert_idxs == i] = inputs_flat_sorted[local_expert_idxs == i]
        # unsort
        temp2[flat_sorted_indices] = temp
        temp2 = temp2.reshape(*(weights.shape + temp2.shape[-1:]))
        outputs_ep = torch.bmm(weights[:, None], temp2).squeeze(1)

        tol = 1e-2
        torch.testing.assert_close(outputs, outputs_ep, atol=1e-2, rtol=1e-2)
