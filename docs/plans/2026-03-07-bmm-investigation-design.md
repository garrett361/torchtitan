# BMM Performance Investigation Design

**Date:** 2026-03-07
**Issue:** https://github.com/pytorch/torchtitan/issues/2225#issuecomment-4015292343

## Problem

The backward pass for `torch.bmm` in MoE token combining dispatches an unexpectedly slow kernel (`sm80_xmma_gemm_f32f32`) that runs on CUDA cores instead of Tensor cores, even on H100/GH200 hardware.

## Goal

Create diagnostic scripts to:
1. Verify the issue by capturing kernel names in forward/backward
2. Establish timing baseline for forward and fwd+bwd

## Shapes (16B DeepSeek config, bs=2, seqlen=4096)

From `torchtitan/models/deepseek_v3/__init__.py`:
- `dim = 2048`
- `top_k = 6`
- `score_before_experts = False` (BMM path taken)

Derived shapes:
```
N = bs * seqlen = 8192
K = top_k = 6
D = dim = 2048

A: (N, 1, K)  = (8192, 1, 6)    # top_scores
B: (N, K, D)  = (8192, 6, 2048) # routed_output
C: (N, 1, D)  = (8192, 1, 2048) # output
```

## Scripts

### Script 1: `scripts/bmm_profile.py`

Captures kernel names via torch profiler.

**Flow:**
1. Create tensors A, B with `requires_grad=True`
2. Warmup iterations (no profiling)
3. Profile forward: `C = torch.bmm(A, B)`
4. Profile backward: `C.sum().backward()`
5. Export Chrome trace + print kernel summary

**Output:**
- `bmm_profile_trace.json` for `chrome://tracing`
- Console table of kernel names and durations

### Script 2: `scripts/bmm_bench.py`

Times forward and fwd+bwd using `triton.testing.do_bench`.

**Measurements:**
1. **forward**: `torch.bmm(A, B)` only
2. **fwd+bwd**: forward then `.backward()`

**Output:**
```
Operation   Median (ms)   Min (ms)   Max (ms)   TFLOP/s
forward     X.XX          X.XX       X.XX       X.XX
fwd+bwd     X.XX          X.XX       X.XX       X.XX
```

## Future Work

Once baseline is established, compare alternative implementations:
- `torch.einsum`
- Manual matmul + squeeze
- Custom Triton kernel

**Note:** The current input shapes `(N, 1, K)` and `(N, K, D)` are structured specifically for BMM. If an alternative approach performs better, we may need to reorganize the upstream code that prepares these tensors (reshaping `top_scores` and `routed_output_unsorted`) to match the new operation's preferred layout.
