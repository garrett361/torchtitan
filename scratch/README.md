# MoE BMM Alternatives

For testing alternatives to bmm. Results for `python scratch/bmm_bench.py`:

```
Device: NVIDIA H100 80GB HBM3
Tokens: 65536
```

`dim` and `top_k`  taken from deepseek cfgs.

### Forward
| Model | bmm fwd ms | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- |
| dsv3_16B | 2.15 | 1.83x | 0.56x | 1.00x |
| dsv3_236B | 5.29 | 1.83x | 0.57x | 1.00x |
| dsv3_671B | 7.06 | 2.52x | 1.79x | 1.00x |
| llama4_17bx16e | 8.97 | 0.26x | 0.10x | 0.12x |
| llama4_17bx128e | 8.97 | 0.26x | 0.10x | 0.12x |
| qwen3_30B_A3B | 2.02 | 2.57x | 1.78x | 1.00x |
| qwen3_235B_A22B | 4.00 | 2.56x | 1.80x | 1.00x |
| gpt_oss_20b | 2.05 | 1.84x | 2.46x | 1.00x |
| gpt_oss_120b | 2.05 | 1.84x | 2.46x | 1.00x |

### Forward + Backward
| Model | bmm fwd+bwd ms | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- |
| dsv3_16B | 11.13 | 0.86x | 0.36x | 1.00x |
| dsv3_236B | 27.65 | 0.86x | 0.37x | 1.00x |
| dsv3_671B | 42.81 | 1.02x | 0.61x | 1.00x |
| llama4_17bx16e | 21.16 | 0.24x | 0.16x | 0.18x |
| llama4_17bx128e | 21.17 | 0.24x | 0.16x | 0.18x |
| qwen3_30B_A3B | 11.36 | 1.11x | 0.67x | 1.00x |
| qwen3_235B_A22B | 22.62 | 1.11x | 0.66x | 1.00x |
| gpt_oss_20b | 14.20 | 0.64x | 0.58x | 1.00x |
| gpt_oss_120b | 14.20 | 0.64x | 0.58x | 1.00x |

