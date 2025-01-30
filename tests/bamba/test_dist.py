import pytest
from dtest import DTest
import torch
import torch.distributed as dist
from copy import deepcopy
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.llama.model import ModelArgs, Transformer
from torchtitan.parallelisms.parallelize_llama import apply_tp as apply_tp_llama


class TestDTest(DTest):
    requires_cuda_env = False

    def test_basic(self) -> None:
        print(f"{self.get_rank()=}")

    def test_all_reduce(self) -> None:
        t = torch.arange(self.get_world_size(), device=self.get_device())
        dist.all_reduce(t)
        self.print_rank(f"{t=}")

    def test_skip(self) -> None:
        pytest.skip("I should be skipped")

    def test_fail(self) -> None:
        self.print_rank0_only("I should fail")
        assert False

    @pytest.mark.world_size([2, 3, 4])
    def test_world_sizes(self) -> None:
        self.print_rank0_only(f"{self.get_world_size()=}")

    @pytest.mark.parametrize("n", (2, 3, 4))
    def test_parametrize(self, n) -> None:
        self.print_rank0_only(f"{n=}")


class TestTPTransformers(DTest):
    requires_cuda_env = True
    model_args = ModelArgs(n_layers=1, dim=256, n_heads=8, vocab_size=128)
    batch_size = 1
    seq_len = 32

    def test_basic_tp(self) -> None:
        torch.manual_seed(42)
        mesh = init_device_mesh(
            "cuda", mesh_shape=(self.get_world_size(),), mesh_dim_names=("tp",)
        )
        model = Transformer(self.model_args).to(
            dtype=torch.bfloat16, device=self.get_device()
        )
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
            device=self.get_device(),
        )
        outputs_tp = model(inputs)
        outputs = model_copy(inputs)

        tol = 1e-2  # Fails on 1e-3
        torch.testing.assert_close(outputs, outputs_tp, atol=tol, rtol=tol)
