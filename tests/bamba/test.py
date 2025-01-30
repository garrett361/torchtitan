import torch
import torch.nn.functional as F

from torchtitan.models.bamba.model import (
    BambaModelArgs,
    Mamba2,
    BambaBlock,
    Bamba,
    torch_chunk_scan_combined,
    torch_chunk_scan_combined_linear,
    torch_scan,
)


def test_model_args() -> None:
    BambaModelArgs(max_seq_len=256, chunk_size=16)


class TestMamba2:
    model_args = BambaModelArgs(max_seq_len=256, chunk_size=16)
    batch_size = 2

    def test_model(self):
        mamba2 = Mamba2(self.model_args).cuda()
        inputs = torch.randn(
            self.batch_size,
            self.model_args.max_seq_len,
            self.model_args.dim,
            device="cuda",
        )
        outputs = mamba2(inputs)
        assert outputs.shape == inputs.shape


class TestBambaBlock:
    model_args = BambaModelArgs(max_seq_len=256, chunk_size=16)
    batch_size = 2

    def test_model(self):
        block = BambaBlock(0, self.model_args).cuda()
        inputs = torch.randn(
            self.batch_size,
            self.model_args.max_seq_len,
            self.model_args.dim,
            device="cuda",
        )
        outputs = block(inputs)
        assert outputs.shape == inputs.shape


class TestBamba:
    model_args = BambaModelArgs(
        max_seq_len=256,
        chunk_size=16,
        vocab_size=64,
        dim=256,
        n_layers=8,
        attn_layer_indices=[7],
        use_mamba_kernels=True,
    )
    batch_size = 2

    def test_model(self):
        model = Bamba(self.model_args).cuda()
        inputs = torch.randint(
            self.model_args.vocab_size,
            size=(self.batch_size, self.model_args.max_seq_len),
            device="cuda",
        )
        outputs = model(inputs)
        assert outputs.shape == torch.Size(
            (self.batch_size, self.model_args.max_seq_len, self.model_args.vocab_size)
        )


class TestScan:
    batch_size = 1
    dim = 128
    n_heads = 4
    d_head = dim // n_heads
    seq_len = 256
    chunk_size = 32
    d_state = 16
    # Getting failures with n_groups 2. TODO: @goon - Investigate. Also happens when testing against
    # mamba-ssm's own torch reference impl.
    n_groups = 1

    def _get_args(
        self,
        dtype=torch.float32,
        device="cuda",
        requires_grad: bool = False,
        use_D: bool = True,
    ):
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.n_heads,
            self.d_head,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        dt = F.softplus(
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.n_heads,
                dtype=dtype,
                device=device,
            )
            - 4
        )
        A = -torch.exp(
            torch.rand(
                self.n_heads,
                dtype=dtype,
                device=device,
            )
        )
        if requires_grad:
            # Set dt and A as requires_grad, and not the tensors they're built from, so that they
            # are leaf tensors which accumulate gradients.
            dt.requires_grad_()
            A.requires_grad_()
        B = torch.randn(
            self.batch_size,
            self.seq_len,
            self.n_groups,
            self.d_state,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        C = torch.randn(
            self.batch_size,
            self.seq_len,
            self.n_groups,
            self.d_state,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        if use_D:
            D = torch.rand(
                self.n_heads,
                dtype=dtype,
                device=device,
            )
        else:
            D = None
        return x, dt, A, B, C, self.chunk_size, D

    def test_torch_scan(self):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        torch.manual_seed(42)

        args = self._get_args()
        out = torch_scan(*args)
        out_chunked = torch_chunk_scan_combined(*args)
        out_chunked_linear = torch_chunk_scan_combined_linear(*args)

        torch.testing.assert_close(out, out_chunked)
        torch.testing.assert_close(out, out_chunked_linear)

        tol = 1e-2
        mamba_out_chunked = mamba_chunk_scan_combined(*args)
        torch.testing.assert_close(out_chunked, mamba_out_chunked, atol=tol, rtol=tol)

        assert out is not None
