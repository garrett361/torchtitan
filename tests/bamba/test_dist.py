from copy import deepcopy

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from dtest import DTest
from torchtitan.models.llama.model import (
    Attention,
    ModelArgs,
    Transformer,
    precompute_freqs_cis,
)
from torchtitan.parallelisms.parallelize_llama import apply_tp as apply_tp_llama

# Attention TP (no seq parallel)
attn_layer_plan_tp = {
    "wq": ColwiseParallel(),
    "wk": ColwiseParallel(),
    "wv": ColwiseParallel(),
    "wo": RowwiseParallel(),
}


# Transformer tests for reference
class TestTPTransformers(DTest):
    requires_cuda_env = True
    model_args = ModelArgs(n_layers=1, dim=256, n_heads=8, vocab_size=128)
    batch_size = 1
    seq_len = 32
    dtype = torch.bfloat16

    def test_attn_tp(self) -> None:
        torch.manual_seed(42)
        mesh = init_device_mesh(
            "cuda", mesh_shape=(self.world_size,), mesh_dim_names=("tp",)
        )
        freqs_cis = precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        ).cuda()

        attn = Attention(self.model_args).to(dtype=self.dtype, device=self.device)
        model_copy = deepcopy(attn)
        parallelize_module(
            module=attn,
            device_mesh=mesh,
            parallelize_plan=attn_layer_plan_tp,
        )

        inputs = torch.randn(
            self.batch_size,
            self.seq_len,
            self.model_args.dim,
            device=self.device,
            dtype=self.dtype,
        )
        outputs_tp = attn(inputs, freqs_cis)
        outputs = model_copy(inputs, freqs_cis)

        tol = 1e-2
        torch.testing.assert_close(outputs, outputs_tp, atol=tol, rtol=tol)

    def test_model_tp(self) -> None:
        torch.manual_seed(42)
        mesh = init_device_mesh(
            "cuda", mesh_shape=(self.world_size,), mesh_dim_names=("tp",)
        )
        model = Transformer(self.model_args).to(dtype=self.dtype, device=self.device)
        model_copy = deepcopy(model)
        apply_tp_llama(
            model,
            mesh,
            loss_parallel=False,
            enable_float8=False,
            enable_async_tp=False,
        )
        inputs = torch.randint(
            self.model_args.vocab_size,
            size=(self.batch_size, self.seq_len),
            device=self.device,
        )
        outputs_tp = model(inputs)
        outputs = model_copy(inputs)

        tol = 1e-2  # Fails on 1e-3
        torch.testing.assert_close(outputs, outputs_tp, atol=tol, rtol=tol)


class TestTPBamba(DTest):
    requires_cuda_env = True
    model_args = ModelArgs(n_layers=1, dim=256, n_heads=8, vocab_size=128)
    batch_size = 1
    seq_len = 32
    dtype = torch.bfloat16

    def test_basic_tp(self) -> None:
        torch.manual_seed(42)
        mesh = init_device_mesh(
            "cuda", mesh_shape=(self.world_size,), mesh_dim_names=("tp",)
        )
        model = Transformer(self.model_args).to(dtype=self.dtype, device=self.device)
        model_copy = deepcopy(model)
        apply_tp_llama(
            model,
            mesh,
            loss_parallel=False,
            enable_float8=False,
            enable_async_tp=False,
        )
        inputs = torch.randint(
            self.model_args.vocab_size,
            size=(self.batch_size, self.seq_len),
            device=self.device,
        )
        outputs_tp = model(inputs)
        outputs = model_copy(inputs)

        tol = 1e-2  # Fails on 1e-3
        torch.testing.assert_close(outputs, outputs_tp, atol=tol, rtol=tol)
