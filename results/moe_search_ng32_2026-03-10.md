# MoE Throughput Search: ng32

**Date:** 2026-03-10T18:17:43.332581
**Git commit:** 231219f0
**Base config:** flavor=1B, batch_size=8, seq_len=2048, n_groups=32, n_moe_layers=14

## Baseline (Dense)

**TPS/GPU:** 50,038 | **Memory:** 18.5 GiB

## Best Configuration

```bash
torchrun --nproc_per_node=8 scripts/benchmark_moe.py \
    --flavor 1B --batch-size 8 --seq-len 2048 \
    --n-groups 32 --n-moe-layers 14 --force-balance \
    --ep 1 --ac-mode selective --ac-option 2 --moe-reshard-after-forward
```

**TPS/GPU:** 15,277 | **Memory:** 63.1 GiB active / 67.1 GiB reserved

## All Results (by TPS/GPU)

| Rank | EP | AC Mode | AC Opt | Reshard | TPS/GPU | vs Baseline | Active GiB | Reserved GiB | Status |
|------|----|---------|--------|---------|---------|-------------|------------|--------------|--------|
| 1 | 1 | selective | 2 | yes | 15,277 | -69% | 63.1 | 67.1 | ok |
| 2 | 1 | selective | 2 | no | 15,274 | -69% | 63.1 | 67.1 | ok |
| 3 | 1 | selective | op | yes | 13,174 | -74% | 19.0 | 26.5 | ok |
| 4 | 1 | selective | op | no | 13,162 | -74% | 19.0 | 26.5 | ok |
| 5 | 1 | selective | 1 | yes | 13,021 | -74% | 16.8 | 20.3 | ok |
| 6 | 1 | full | - | no | 13,007 | -74% | 16.8 | 20.3 | ok |
| 7 | 1 | selective | 1 | no | 13,004 | -74% | 16.8 | 20.3 | ok |
| 8 | 1 | full | - | yes | 12,997 | -74% | 16.8 | 20.3 | ok |
| 9 | 8 | selective | 2 | no | 11,143 | -78% | 62.6 | 72.9 | ok |
| 10 | 8 | selective | 2 | yes | 11,136 | -78% | 62.3 | 72.9 | ok |
| 11 | 8 | selective | 1 | no | 9,398 | -81% | 16.6 | 28.3 | ok |
| 12 | 8 | full | - | no | 9,396 | -81% | 16.6 | 28.3 | ok |
| 13 | 8 | selective | 1 | yes | 9,392 | -81% | 16.3 | 22.3 | ok |
| 14 | 8 | full | - | yes | 9,390 | -81% | 16.3 | 22.3 | ok |
| 15 | 1 | none | - | yes | - | - | - | - | oom |
| 16 | 1 | none | - | no | - | - | - | - | oom |
| 17 | 8 | none | - | yes | - | - | - | - | oom |
| 18 | 8 | none | - | no | - | - | - | - | oom |
| 19 | 8 | selective | op | yes | - | - | - | - | oom |
| 20 | 8 | selective | op | no | - | - | - | - | oom |

## Failed Configurations

| Config | Error |
|--------|-------|
| ep=1, ac=none, reshard | CUDA out of memory |
| ep=1, ac=none, no-reshard | CUDA out of memory |
| ep=8, ac=none, reshard | CUDA out of memory |
| ep=8, ac=none, no-reshard | CUDA out of memory |
| ep=8, ac=selective/op, reshard | CUDA out of memory |
| ep=8, ac=selective/op, no-reshard | CUDA out of memory |