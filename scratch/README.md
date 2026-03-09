# MoE BMM Alternatives

For testing alternatives to bmm. Results for `python scratch/bmm_bench.py`:

```
Device: NVIDIA H100 80GB HBM3
Tokens: 65536
```

`dim` and `top_k`  taken from deepseek cfgs.

Timing of different methods.
* bmm: `torch.bmm(top_scores.unsqueeze(1), routed_output).squeeze(1)`
* bmm_compiled: the above, with `torch.compile`
* bcast_sum: `(top_scores.unsqueeze(-1) * routed_output).sum(dim=1)`
* bcast_sum_compiled: the above, with `torch.compile`
* einsum: `torch.einsum("tk,tkd->td", top_scores, routed_output)`

Reporting the absolute bmm time in milliseconds and time ratios relative to bmm for all other cases. 🚀 = faster.

```
❯ python scratch/bmm_bench.py
PyTorch: 2.12.0.dev20260309+cu128
Device: NVIDIA H100 80GB HBM3
Tokens: 65536
```


### Forward
| Model | bmm fwd ms | bmm_compiled/bmm | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- | --- |
| dsv3_16B | 4.58 | 0.82x 🚀 | 1.39x | 0.15x 🚀 | 1.00x |
| dsv3_236B | 11.35 | 0.82x 🚀 | 1.39x | 0.17x 🚀 | 1.00x |
| dsv3_671B | 18.38 | 0.80x 🚀 | 1.59x | 0.69x 🚀 | 1.00x |
| llama4_17bx16e | 9.98 | 0.12x 🚀 | 0.34x 🚀 | 0.07x 🚀 | 0.21x 🚀 |
| llama4_17bx128e | 9.98 | 0.12x 🚀 | 0.34x 🚀 | 0.07x 🚀 | 0.21x 🚀 |
| qwen3_30B_A3B | 5.27 | 0.79x 🚀 | 1.60x | 0.69x 🚀 | 1.00x |
| qwen3_235B_A22B | 10.49 | 0.79x 🚀 | 1.59x | 0.69x 🚀 | 1.00x |
| gpt_oss_20b | 4.34 | 0.82x 🚀 | 1.40x | 1.17x | 1.00x |
| gpt_oss_120b | 4.34 | 0.82x 🚀 | 1.40x | 1.17x | 1.00x |

### Forward + Backward
| Model | bmm fwd+bwd ms | bmm_compiled/bmm | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- | --- |
| dsv3_16B | 15.06 | 0.41x 🚀 | 0.90x 🚀 | 0.16x 🚀 | 1.00x |
| dsv3_236B | 37.50 | 0.52x 🚀 | 0.90x 🚀 | 0.18x 🚀 | 1.00x |
| dsv3_671B | 61.21 | 0.53x 🚀 | 1.02x | 0.37x 🚀 | 1.00x |
| llama4_17bx16e | 22.82 | 0.19x 🚀 | 0.30x 🚀 | 0.12x 🚀 | 0.24x 🚀 |
| llama4_17bx128e | 22.82 | 0.19x 🚀 | 0.30x 🚀 | 0.12x 🚀 | 0.24x 🚀 |
| qwen3_30B_A3B | 16.67 | 0.56x 🚀 | 1.08x | 0.42x 🚀 | 1.00x |
| qwen3_235B_A22B | 33.25 | 0.56x 🚀 | 1.07x | 0.39x 🚀 | 1.00x |
| gpt_oss_20b | 17.94 | 0.43x 🚀 | 0.72x 🚀 | 0.43x 🚀 | 1.00x |
| gpt_oss_120b | 17.94 | 0.43x 🚀 | 0.72x 🚀 | 0.43x 🚀 | 1.00x |

