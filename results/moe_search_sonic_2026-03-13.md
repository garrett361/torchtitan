# MoE Throughput Search: sonic

**Date:** 2026-03-13T19:26:03.679335
**Git commit:** a7588194
**Base config:** flavor=1B, batch_size=8, seq_len=2048, n_groups=64, n_moe_layers=14

## Baseline (Dense)

**TPS/GPU:** 56,119 | **Memory:** 30.0 GiB

## Best Configuration

```bash
torchrun --nproc_per_node=8 scripts/benchmark_moe.py \
    --flavor 1B --batch-size 8 --seq-len 2048 \
    --n-groups 64 --n-moe-layers 14 --force-balance \
    --ep 1 --ac-mode none --no-moe-reshard-after-forward \
    --custom-moe-impl sonic_virtual_group 
```

**TPS/GPU:** 39,567 | **Memory:** 25.6 GiB active / 30.2 GiB reserved

## All Results (by TPS/GPU)

| Rank | EP | AC Mode | AC Opt | Reshard | TPS/GPU | vs Baseline | Active GiB | Reserved GiB | Status |
|------|----|---------|--------|---------|---------|-------------|------------|--------------|--------|
| 1 | 1 | none | - | no | 39,567 | -29% | 25.6 | 30.2 | ok |
| 2 | 1 | selective | 2 | no | 35,586 | -37% | 16.5 | 26.0 | ok |
| 3 | 1 | selective | op | no | 32,830 | -41% | 10.7 | 13.3 | ok |
| 4 | 1 | full | - | no | 31,916 | -43% | 8.5 | 12.8 | ok |
| 5 | 1 | selective | 1 | no | 31,848 | -43% | 8.5 | 12.8 | ok |
| 6 | 1 | selective | op | no | 7,434 | -87% | 31.0 | 36.5 | ok |
| 7 | 1 | selective | 1 | no | 7,388 | -87% | 28.9 | 36.3 | ok |
| 8 | 1 | full | - | no | 7,381 | -87% | 28.9 | 36.3 | ok |
| 9 | 1 | none | - | no | - | - | - | - | oom |
| 10 | 1 | selective | 2 | no | - | - | - | - | oom |

## Failed Configurations

| Config | Error |
|--------|-------|
| ep=1, ac=none, no-reshard | CUDA out of memory |
| ep=1, ac=selective/2, no-reshard | CUDA out of memory |