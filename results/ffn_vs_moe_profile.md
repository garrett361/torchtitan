# FFN vs MoE Layer Kernel Profiling

This document analyzes kernel-level performance differences between dense FFN and MoE layers to understand why MoE shows ~13x lower throughput despite equivalent FLOPs.

## Commands

```bash
# Profile FFN baseline
python scripts/profile_ffn_moe.py --layer ffn

# Profile MoE layer
python scripts/profile_ffn_moe.py --layer moe --n-groups 64
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Tokens | 16,384 (batch=8 × seq=2048) |
| Model dim | 2048 |
| FFN hidden_dim | 8192 |
| MoE experts | 128 (2 replicas × 64 groups) |
| MoE top_k | 64 (full replica) |
| MoE inter_dim | 128 (8192 / 64) |
| Device | NVIDIA H100 80GB HBM3 |
| Dtype | bfloat16 |
| Profile iters | 3 (fwd + bwd each) |

**FLOPs are identical:**

```
FFN FLOPs:  6 × tokens × dim × hidden_dim
          = 6 × 16384 × 2048 × 8192
          = 1.649 TFLOPs

MoE FLOPs:  6 × tokens × top_k × dim × moe_inter_dim
          = 6 × 16384 × 64 × 2048 × 128
          = 6 × 16384 × 2048 × (64 × 128)
          = 6 × 16384 × 2048 × 8192
          = 1.649 TFLOPs
```

When `top_k = n_groups`, each token activates a full replica worth of experts, matching the dense FFN compute exactly. The 12.9x slowdown is **pure overhead from memory operations**, not extra compute.

## Summary

| Metric | FFN | MoE | Ratio |
|--------|-----|-----|-------|
| **Total CUDA time** | 22.7 ms | 293.2 ms | **12.9x slower** |
| **Compute time** | 18.2 ms | 47.2 ms | 2.6x slower |
| **Compute % of total** | 80.3% | 16.1% | — |
| **Memory ops % of total** | 1.1% | ~66% | — |

Percentages indicate share of total CUDA time. MoE spends more *absolute* time in compute (47.2 ms vs 18.2 ms), but memory operations dominate its runtime:

```
FFN:  [=======compute (80%)=======][other]        22.7 ms total
MoE:  [compute (16%)][=====memory ops (~66%)=====]  293.2 ms total
```

## FFN Kernel Breakdown

```
┌─────────────────────────────────────────────────────┐
│                     FFN (22.7 ms)                   │
├─────────────────────────────────────────────────────┤
│ aten::mm (GEMMs)                    80.3%  18.2 ms  │ ████████████████████████
│ aten::mul (SwiGLU gate)             10.3%   2.3 ms  │ ███
│ aten::silu_backward                  3.5%   0.8 ms  │ █
│ aten::silu                           2.3%   0.5 ms  │ █
│ aten::add_ (grad accum)              2.1%   0.5 ms  │ █
│ aten::copy_                          1.1%   0.3 ms  │ ░
└─────────────────────────────────────────────────────┘
```

**Top kernels by CUDA time:**

| Kernel | Time | % | Calls |
|--------|------|---|-------|
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | 5.82 ms | 25.7% | 9 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 4.26 ms | 18.8% | 6 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 4.00 ms | 17.7% | 6 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | 2.06 ms | 9.1% | 3 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 2.02 ms | 8.9% | 3 |
| `vectorized_elementwise_kernel (mul)` | 2.34 ms | 10.3% | 9 |

**Characteristics:**
- 9 large GEMMs total (3 fwd + 6 bwd) × 3 iterations = 27 calls
- Each GEMM: ~670-710 μs (well-optimized nvjet kernels)
- Shape: `(16384, 2048) @ (2048, 8192)` — large contiguous tiles
- High arithmetic intensity, compute-bound

## MoE Kernel Breakdown

```
┌─────────────────────────────────────────────────────┐
│                     MoE (293.2 ms)                  │
├─────────────────────────────────────────────────────┤
│ aten::_index_put_impl_ (scatter)    33.6%  98.6 ms  │ ██████████
│ index_elementwise_kernel            20.7%  60.6 ms  │ ██████
│ aten::_grouped_mm (expert compute)  16.1%  47.2 ms  │ █████
│ aten::copy_ (dtype/layout)          14.1%  41.3 ms  │ ████
│ indexing_backward_kernel            12.8%  37.6 ms  │ ████
│ aten::index (gather)                12.1%  35.5 ms  │ ████
│ aten::bmm (router score combine)    10.6%  31.1 ms  │ ███
│ Memcpy DtoD                          5.8%  16.9 ms  │ ██
│ aten::add_ (grad accum)              4.5%  13.3 ms  │ █
└─────────────────────────────────────────────────────┘
```

**Top kernels by CUDA time:**

| Kernel | Time | % | Calls |
|--------|------|---|-------|
| `index_elementwise_kernel` | 60.6 ms | 20.7% | 18 |
| `_grouped_mm (cutlass)` | 47.2 ms | 16.1% | 27 |
| `indexing_backward_kernel` | 37.6 ms | 12.8% | 6 |
| `vectorized_gather_kernel` | 35.5 ms | 12.1% | 36 |
| `bmm (sm80_xmma_gemm)` | 31.1 ms | 10.6% | 9 |
| `unrolled_elementwise_kernel (copy)` | 19.9 ms | 6.8% | 15 |
| `Memcpy DtoD` | 16.9 ms | 5.8% | 12 |

## Root Cause Analysis

### 1. Token Reordering Dominates (~46% of time)

The MoE forward/backward requires permuting tokens to/from expert order:

```
Forward:
  tokens[permute_indices] → sorted_tokens      (aten::index / gather)
  expert_outputs → tokens[scatter_indices]     (aten::_index_put_impl_ / scatter)

Backward:
  grad[scatter_indices] → grad_sorted          (indexing_backward_kernel)
  grad_sorted → grad_tokens                    (index_put backward)
```

| Operation | Time | % |
|-----------|------|---|
| `aten::index` (gather) | 35.5 ms | 12.1% |
| `aten::_index_put_impl_` (scatter) | 98.6 ms | 33.6% |
| **Total reordering** | **134.1 ms** | **45.7%** |

These operations have **random memory access patterns** with poor cache locality. Each token may route to a different expert, creating scattered reads/writes across GPU memory.

### 2. Compute is a Small Fraction of MoE Runtime

| Layer | Compute Op | CUDA Time | % of Total Time |
|-------|------------|-----------|-----------------|
| FFN | `aten::mm` | 18.2 ms | 80.3% |
| MoE | `aten::_grouped_mm` | 47.2 ms | 16.1% |

MoE compute takes 2.6x longer in absolute terms (47.2 ms vs 18.2 ms), but represents only 16% of total runtime vs FFN's 80%. The remaining MoE time is consumed by memory operations.

**Why grouped_mm is slower:**
- 64 small GEMMs per forward instead of 3 large ones
- Per-expert shape: `(~256 tokens, 2048) @ (2048, 128)` — smaller tiles
- Lower arithmetic intensity per expert
- Higher kernel launch overhead

### 3. Dtype Conversions (14% overhead)

```
aten::copy_ (14.1%): bf16 → fp32 for routing, fp32 → bf16 for experts
```

The router operates in fp32 for numerical stability, requiring dtype conversions at boundaries.

### 4. Router Score Application (bmm, 10.6%)

```python
combined_output = bmm(top_scores.unsqueeze(1), routed_output)
# Shape: (16384, 1, 64) @ (16384, 64, 128) → (16384, 1, 128)
```

This batched matmul to weight expert outputs by router scores adds significant overhead.

## Data Flow Comparison

```
FFN Data Flow:
  x ──────────────────► [GEMM w1] ──► [GEMM w2] ──► [GEMM w3] ──► y
  (contiguous)            │              │              │
                     one large       one large      one large
                       kernel          kernel         kernel

MoE Data Flow:
  x ──► [router] ──► [permute] ──► [64×expert] ──► [scatter] ──► [combine] ──► y
           │            │              │              │              │
        small mm    random I/O     64 small        random I/O     bmm
                                   GEMMs
```

**The fundamental tradeoff:** MoE trades compute efficiency for conditional computation (sparse activation).

## Optimization Opportunities

1. **Fused permute-expert-scatter kernels**: Avoid materializing intermediate permuted tensors
2. **Block-sparse patterns**: If tokens cluster to experts, exploit locality
3. **Reduced precision routing**: fp16 router to eliminate dtype conversions
4. **Expert parallelism (EP)**: Distribute experts across GPUs to reduce per-GPU scatter/gather volume
5. **Capacity factor tuning**: Limit max tokens per expert to bound worst-case memory patterns
6. **Persistent kernels**: Keep expert weights in registers/shared memory across token batches

## Conclusions

The 12.9x slowdown from FFN to MoE is explained by:

1. **Memory operations dominate**: ~66% of MoE CUDA time spent in index/copy ops vs 1% for FFN
2. **Token reordering overhead**: Scatter/gather for expert routing costs ~134 ms (46% of MoE time)
3. **Slower compute**: grouped_mm takes 47.2 ms vs mm's 18.2 ms due to smaller per-expert tiles
4. **Dtype conversion overhead**: fp32 routing adds 14% overhead

MoE efficiency gains require either:
- Fused kernels that eliminate intermediate tensor materialization
- Expert parallelism to distribute memory operations across devices
- Hardware/compiler improvements for scatter/gather patterns

## Memory Profiling

Peak memory usage for forward + backward pass (same configuration as above):

| Layer | Peak (GiB) | Reserved (GiB) |
|-------|------------|----------------|
| FFN   | 2.13       | 2.19           |
| MoE   | 25.93      | 38.20          |
| Sonic | 5.99       | 38.20          |

**Key observations:**

1. **MoE uses 12x more peak memory than FFN** (25.93 GiB vs 2.13 GiB) — the memory overhead from token reordering tensors, router scores, and intermediate buffers dwarfs the compute tensors.

2. **SonicMoE reduces peak memory by 4.3x** (5.99 GiB vs 25.93 GiB) — fused kernels eliminate intermediate tensor materialization for permute/scatter operations.

3. **Reserved memory is similar for MoE and SonicMoE** (38.20 GiB) — the CUDA allocator reserves similar pools, but SonicMoE's lower peak allocation leaves more headroom for larger batches or longer sequences.

Run memory profiling with:
```bash
python scripts/bench_ffn_moe.py --sonic
```
