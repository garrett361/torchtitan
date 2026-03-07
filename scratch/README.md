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
| 16B | 2.16 | 1.83x | 0.56x | 1.00x |
| 236B | 5.30 | 1.82x | 0.57x | 1.00x |
| 671B | 7.05 | 2.52x | 1.79x | 1.00x |

### Forward + Backward
| Model | bmm fwd+bwd ms | bcast_sum/bmm | bcast_sum_compiled/bmm | einsum/bmm |
| --- | --- | --- | --- | --- |
| 16B | 11.13 | 0.86x | 0.36x | 1.00x |
| 236B | 27.66 | 0.86x | 0.37x | 1.00x |
| 671B | 42.81 | 1.02x | 0.61x | 1.00x |

