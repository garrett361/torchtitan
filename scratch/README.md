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
| dsv3_16B | 4.59 | 1.39x | 0.15x 🚀 | 1.00x |
| dsv3_236B | 11.39 | 1.38x | 0.16x 🚀 | 1.00x |
| dsv3_671B | 18.42 | 1.58x | 0.48x 🚀 | 1.00x 🚀 |
| llama4_17bx16e | 10.12 | 0.33x 🚀 | 0.07x 🚀 | 0.21x 🚀 |
| llama4_17bx128e | 10.12 | 0.33x 🚀 | 0.07x 🚀 | 0.21x 🚀 |
| qwen3_30B_A3B | 5.26 | 1.60x | 0.48x 🚀 | 1.00x |
| qwen3_235B_A22B | 10.47 | 1.59x | 0.48x 🚀 | 1.00x |
| gpt_oss_20b | 4.33 | 1.39x | 0.82x 🚀 | 1.00x |
| gpt_oss_120b | 4.33 | 1.40x | 0.82x 🚀 | 1.00x |

### Forward + Backward
| Model | bmm fwd+bwd ms | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- |
| dsv3_16B | 15.21 | 0.89x 🚀 | 0.16x 🚀 | 1.00x |
| dsv3_236B | 37.86 | 0.89x 🚀 | 0.18x 🚀 | 1.00x |
| dsv3_671B | 61.85 | 1.01x | 0.28x 🚀 | 1.00x |
| llama4_17bx16e | 23.06 | 0.29x 🚀 | 0.12x 🚀 | 0.24x 🚀 |
| llama4_17bx128e | 23.07 | 0.29x 🚀 | 0.12x 🚀 | 0.24x 🚀 |
| qwen3_30B_A3B | 16.68 | 1.08x | 0.30x 🚀 | 1.00x |
| qwen3_235B_A22B | 33.28 | 1.07x | 0.30x 🚀 | 1.00x |
| gpt_oss_20b | 18.01 | 0.71x 🚀 | 0.32x 🚀 | 1.00x |
| gpt_oss_120b | 18.01 | 0.71x 🚀 | 0.32x 🚀 | 1.00x |

