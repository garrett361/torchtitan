import torch

from torchtitan.models.llama3gdn.model.args import Llama3GDNModelArgs
from torchtitan.models.llama3gdn.model.model import LinearAttention


class TestLinearAttention:
    bsz = 1
    # NOTE: @goon - seqlen must be larger than 128
    seqlen = 128
    dim = 256
    num_heads = 4
    head_dim = 64

    def test(self) -> None:
        args = Llama3GDNModelArgs(dim=self.dim)
        args.gdn_layer_args.hidden_size = self.dim
        args.gdn_layer_args.num_heads = self.num_heads
        args.gdn_layer_args.head_dim = self.head_dim

        model = LinearAttention(args).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(
            self.bsz, self.seqlen, self.dim, device="cuda", dtype=torch.bfloat16
        )
        out = model(x, None, None)
