import torch

from torchtitan.models.hybrid_moe.model.args import HybridMoEModelArgs
from torchtitan.models.hybrid_moe.model.model import (
    Attention,
    HybridMoEModel,
    LinearAttention,
    precompute_freqs_cis,
)


class TestLayers:
    vocab_size = 49160
    dim = 256
    inter_dim = 1024
    moe_inter_dim = 256
    n_layers = 8  # 3
    n_dense_layers = 1
    n_heads = 16
    n_routed_experts = 8
    n_shared_experts = 2
    n_activated_experts = 3
    route_scale = 1.0
    q_lora_rank = 0
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    mscale = 0.70
    mha_layer_interval = 2  # 3

    seq_len = 128
    batch_size = 2
    device = "cuda"
    dtype = torch.bfloat16

    weight_init_std = 0.02 / 2**0.5

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    @property
    def factory_kwargs(self) -> dict:
        return {"device": self.device, "dtype": self.dtype}

    def _get_inputs(
        self, batch_size: int | None = None, seq_len: int | None = None
    ) -> torch.Tensor:
        return torch.randint(
            self.vocab_size,
            size=(batch_size or self.batch_size, seq_len or self.seq_len),
            device=self.device,
        )

    def _get_activations(
        self, batch_size: int | None = None, seq_len: int | None = None
    ) -> torch.Tensor:
        return torch.randn(
            batch_size or self.batch_size,
            seq_len or self.seq_len,
            self.dim,
            **self.factory_kwargs,
        )

    def _get_freq_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(self._get_model_args()).to(self.device)

    def _get_model_args(self) -> HybridMoEModelArgs:
        return HybridMoEModelArgs(
            max_seq_len=self.seq_len,  # NOTE: @goon - need to update explicitly. Shape errs otherwise.
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
            route_scale=self.route_scale,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            mscale=self.mscale,
            mha_layer_interval=self.mha_layer_interval,
        )

    def test_model(self) -> None:
        model = HybridMoEModel(self._get_model_args()).to(**self.factory_kwargs)
        model.init_weights(buffer_device=self.device)
        inputs = self._get_inputs()
        outputs = model(inputs)
        outputs.sum().backward()
        for n, p in model.named_parameters():
            if torch.isnan(p.grad).any():
                print(f"{n=}, {p.grad=}")

    def test_attn(self) -> None:
        model = Attention(self._get_model_args()).to(**self.factory_kwargs)
        model.init_weights(self.weight_init_std)
        inputs = self._get_activations()
        freqs_cis = self._get_freq_cis()
        outputs = model(inputs, freqs_cis=freqs_cis)
        outputs

    def test_lin_attn(self) -> None:
        model = LinearAttention(self._get_model_args()).to(**self.factory_kwargs)
        model.init_weights(self.weight_init_std)
        inputs = self._get_activations()
        outputs = model(inputs)
        outputs
