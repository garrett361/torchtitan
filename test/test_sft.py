from torchtitan.datasets import build_sft_data_loader


class TestData:
    dataset_path = "/proj/data-eng/goon/garrett361/torchtitan-yarn-sft/data/1b8d0c88fe"
    dataset_weights = "1.0"
    batch_size = 1
    seq_len = 512
    naive_padding_free = True
    max_out_tokens = True

    def test_data(self) -> None:
        data_loader = build_sft_data_loader(
            dataset_path=self.dataset_path,
            dataset_weights=self.dataset_weights,
            dp_rank=0,
            dp_degree=1,
            cp_rank=0,
            cp_degree=1,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            naive_padding_free=True,
            max_out_tokens=True,
        )
        diter =iter(data_loader)
        out = next(diter)
        print(f"{len(out)=}")
