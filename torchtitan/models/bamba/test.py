import torch

from torchtitan.models.bamba.model import BambaModelArgs, Mamba2, BambaBlock, Bamba


def test_config() -> None:
    BambaModelArgs(max_seq_len=256, chunk_size=16)


class TestMamba2:
    config = BambaModelArgs(max_seq_len=256, chunk_size=16)
    batch_size = 2

    def test_model(self):
        mamba2 = Mamba2(self.config)
        inputs = torch.randn(self.batch_size, self.config.max_seq_len, self.config.dim)
        outputs = mamba2(inputs)
        assert outputs.shape == inputs.shape


class TestBambaBlock:
    config = BambaModelArgs(max_seq_len=256, chunk_size=16)
    batch_size = 2

    def test_model(self):
        block = BambaBlock(0, self.config)
        inputs = torch.randn(self.batch_size, self.config.max_seq_len, self.config.dim)
        outputs = block(inputs)
        assert outputs.shape == inputs.shape


class TestBamba:
    config = BambaModelArgs(
        max_seq_len=256,
        chunk_size=16,
        vocab_size=64,
        dim=256,
        n_layers=2,
        attn_layer_indices=[1],
    )
    batch_size = 2

    def test_model(self):
        model = Bamba(self.config)
        inputs = torch.randint(
            self.config.vocab_size, size=(self.batch_size, self.config.max_seq_len)
        )
        outputs = model(inputs)
        assert outputs.shape == torch.Size(
            (self.batch_size, self.config.max_seq_len, self.config.vocab_size)
        )
