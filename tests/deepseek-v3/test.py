from torchtitan.models.deepseek_v3.model import (
    MLA,
    Block,
    Gate,
    MoE,
    ModelArgs,
    DeepSeekV3,
    MLP,
    Expert,
    precompute_freqs_cis,
)
import torch


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
        expert = Gate(self.model_args).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        weights, indices = expert(inputs)
        assert weights.shape == torch.Size(
            (self.batch_size, self.seq_len, self.n_activated_experts)
        )
        assert indices.shape == torch.Size(
            (self.batch_size, self.seq_len, self.n_activated_experts)
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
        start_pos = 0
        freqs_cis = precompute_freqs_cis(self.model_args).to(self.device)
        freqs_cis_slice = freqs_cis[start_pos : start_pos + self.seq_len]
        mask = None

        outputs = mla(inputs, start_pos, freqs_cis_slice, mask)
        assert outputs.shape == inputs.shape

    def test_block_dense(self) -> None:
        block_dense = Block(
            layer_id=self.model_args.n_dense_layers - 1, args=self.model_args
        ).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        start_pos = 0
        freqs_cis = precompute_freqs_cis(self.model_args).to(self.device)
        freqs_cis_slice = freqs_cis[start_pos : start_pos + self.seq_len]
        mask = None

        outputs = block_dense(inputs, start_pos, freqs_cis_slice, mask)
        assert outputs.shape == inputs.shape

    def test_block_moe(self) -> None:
        block_moe = Block(
            layer_id=self.model_args.n_dense_layers + 1, args=self.model_args
        ).to(**self.factory_kwargs)
        inputs = torch.randn(
            self.batch_size, self.seq_len, self.dim, **self.factory_kwargs
        )
        start_pos = 0
        freqs_cis = precompute_freqs_cis(self.model_args).to(self.device)
        freqs_cis_slice = freqs_cis[start_pos : start_pos + self.seq_len]
        mask = None

        outputs = block_moe(inputs, start_pos, freqs_cis_slice, mask)
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
