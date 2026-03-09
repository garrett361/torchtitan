# MoE BMM Alternatives

For testing alternatives to bmm. Results for `python scratch/bmm_bench.py`:

```
Device: NVIDIA H100 80GB HBM3
Tokens: 65536
```

`dim` and `top_k`  taken from deepseek cfgs.

Timing of different methods.
* bmm: `torch.bmm(top_scores.unsqueeze(1), routed_output).squeeze(1)`
* bcast_sum: `(top_scores.unsqueeze(-1) * routed_output).sum(dim=1)`
* bcast_sum_compiled: the above, with `torch.compile`
* einsum: `torch.einsum("tk,tkd->td", top_scores, routed_output)`

Reporting the absolute bmm time in milliseconds and time ratios relative to bmm for all other cases. 🚀 = faster.

### Forward
| Model | bmm fwd ms | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- |
| dsv3_16B | 2.16 | 1.83x | 0.56x 🚀 | 1.00x |
| dsv3_236B | 5.29 | 1.82x | 0.57x 🚀 | 1.00x |
| dsv3_671B | 7.05 | 2.52x | 1.25x | 1.00x 🚀 |
| llama4_17bx16e | 9.11 | 0.26x 🚀 | 0.10x 🚀 | 0.12x 🚀 |
| llama4_17bx128e | 9.11 | 0.26x 🚀 | 0.10x 🚀 | 0.12x 🚀 |
| qwen3_30B_A3B | 2.02 | 2.57x | 1.25x | 1.00x |
| qwen3_235B_A22B | 4.00 | 2.56x | 1.26x | 1.00x |
| gpt_oss_20b | 2.05 | 1.83x | 1.74x | 1.00x 🚀 |
| gpt_oss_120b | 2.05 | 1.83x | 1.73x | 1.00x 🚀 |

### Forward + Backward
| Model | bmm fwd+bwd ms | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- |
| dsv3_16B | 11.17 | 0.86x 🚀 | 0.36x 🚀 | 1.00x |
| dsv3_236B | 27.76 | 0.85x 🚀 | 0.36x 🚀 | 1.00x 🚀 |
| dsv3_671B | 43.26 | 1.01x | 0.50x 🚀 | 1.00x |
| llama4_17bx16e | 21.40 | 0.24x 🚀 | 0.16x 🚀 | 0.18x 🚀 |
| llama4_17bx128e | 21.40 | 0.24x 🚀 | 0.16x 🚀 | 0.18x 🚀 |
| qwen3_30B_A3B | 11.37 | 1.11x | 0.55x 🚀 | 1.00x |
| qwen3_235B_A22B | 22.65 | 1.10x | 0.55x 🚀 | 1.00x |
| gpt_oss_20b | 14.28 | 0.64x 🚀 | 0.46x 🚀 | 1.00x |
| gpt_oss_120b | 14.28 | 0.64x 🚀 | 0.46x 🚀 | 1.00x 🚀 |

