# BMM Investigation Scripts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create two diagnostic scripts to verify BMM kernel dispatch issues and establish timing baselines.

**Architecture:** Two standalone scripts in `scripts/` that isolate the BMM operation from the full MoE forward pass. Both use the same shape constants derived from 16B DeepSeek config.

**Tech Stack:** PyTorch profiler, triton.testing.do_bench, CUDA

**Design Doc:** `docs/plans/2026-03-07-bmm-investigation-design.md`

---

## Shared Constants

Both scripts use these shapes (16B DeepSeek, bs=2, seqlen=4096):

```python
N = 8192      # bs * seqlen
K = 6         # top_k
D = 2048      # dim
```

---

### Task 1: Create bmm_profile.py

**Files:**
- Create: `scripts/bmm_profile.py`

**Step 1: Create the profiler script**

```python
"""Profile torch.bmm kernel dispatch for MoE token combining."""

import torch
from torch.profiler import profile, ProfilerActivity

# 16B DeepSeek shapes: bs=2, seqlen=4096
N = 8192  # bs * seqlen
K = 6     # top_k
D = 2048  # dim

WARMUP = 5
PROFILE_ITERS = 3


def main():
    device = torch.device("cuda")

    # A: (N, 1, K) top_scores, B: (N, K, D) routed_output
    A = torch.randn(N, 1, K, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(N, K, D, device=device, dtype=torch.float32, requires_grad=True)

    # Warmup
    for _ in range(WARMUP):
        C = torch.bmm(A, B)
        C.sum().backward()
        A.grad = None
        B.grad = None

    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(PROFILE_ITERS):
            C = torch.bmm(A, B)
            C.sum().backward()
            A.grad = None
            B.grad = None
        torch.cuda.synchronize()

    # Print CUDA kernels from FunctionEvent.kernels
    print(f"BMM Shapes: A=({N}, 1, {K}), B=({N}, {K}, {D})")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print("CUDA Kernels by PyTorch Op:")
    print("=" * 100)
    for event in prof.events():
        if event.kernels:
            print(f"\n{event.name}:")
            for kernel in event.kernels:
                print(f"  {kernel.name:<70} {kernel.duration:>8.1f} us")
    print()

    # Also print summary table
    print("PyTorch Op Summary (sorted by CUDA time):")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs**

Run: `python scripts/bmm_profile.py`

Expected output:
- CUDA kernel names grouped by PyTorch op (aten::bmm, etc.)
- Look for `sm80_xmma_gemm` patterns in backward kernels (the problematic dispatch)
- PyTorch op summary table

**Step 3: Commit**

```bash
git add scripts/bmm_profile.py
git commit -m "feat(scripts): add bmm_profile.py for kernel dispatch analysis

Profiles torch.bmm forward/backward and prints CUDA kernel names.
Related to pytorch/torchtitan#2225.
"
```

---

### Task 2: Create bmm_bench.py

**Files:**
- Create: `scripts/bmm_bench.py`

**Step 1: Create the benchmark script**

```python
"""Benchmark torch.bmm forward and fwd+bwd for MoE token combining."""

import torch
from triton.testing import do_bench

# 16B DeepSeek shapes: bs=2, seqlen=4096
N = 8192  # bs * seqlen
K = 6     # top_k
D = 2048  # dim


def main():
    device = torch.device("cuda")

    # A: (N, 1, K) top_scores, B: (N, K, D) routed_output
    A = torch.randn(N, 1, K, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(N, K, D, device=device, dtype=torch.float32, requires_grad=True)

    def fwd():
        return torch.bmm(A, B)

    def fwd_bwd():
        C = torch.bmm(A, B)
        C.sum().backward()
        A.grad = None
        B.grad = None

    # do_bench returns (median, min, max) in ms when quantiles provided
    fwd_ms = do_bench(fwd, quantiles=[0.5, 0.0, 1.0])
    fwd_bwd_ms = do_bench(fwd_bwd, quantiles=[0.5, 0.0, 1.0])

    # FLOPs calculation
    # Forward: 2 * N * 1 * K * D (standard matmul)
    # Backward: ~2x forward (dA and dB gradients)
    fwd_flops = 2 * N * 1 * K * D
    bwd_flops = 2 * fwd_flops  # approximate
    total_flops = fwd_flops + bwd_flops

    def tflops(flops, ms):
        return (flops / 1e12) / (ms / 1e3) if ms > 0 else 0

    print(f"BMM Shapes: A=({N}, 1, {K}), B=({N}, {K}, {D})")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print(f"{'Operation':<12} {'Median (ms)':>12} {'Min (ms)':>10} {'Max (ms)':>10} {'TFLOP/s':>10}")
    print("-" * 56)
    print(f"{'forward':<12} {fwd_ms[0]:>12.3f} {fwd_ms[1]:>10.3f} {fwd_ms[2]:>10.3f} {tflops(fwd_flops, fwd_ms[0]):>10.2f}")
    print(f"{'fwd+bwd':<12} {fwd_bwd_ms[0]:>12.3f} {fwd_bwd_ms[1]:>10.3f} {fwd_bwd_ms[2]:>10.3f} {tflops(total_flops, fwd_bwd_ms[0]):>10.2f}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs**

Run: `python scripts/bmm_bench.py`

Expected output:
```
BMM Shapes: A=(8192, 1, 6), B=(8192, 6, 2048)
Device: <GPU name>

Operation    Median (ms)   Min (ms)   Max (ms)   TFLOP/s
--------------------------------------------------------
forward            X.XXX      X.XXX      X.XXX       X.XX
fwd+bwd            X.XXX      X.XXX      X.XXX       X.XX
```

**Step 3: Commit**

```bash
git add scripts/bmm_bench.py
git commit -m "feat(scripts): add bmm_bench.py for timing baseline

Benchmarks torch.bmm forward and fwd+bwd using triton.testing.do_bench.
Related to pytorch/torchtitan#2225.
"
```

---

## Verification

After both scripts are committed, run both to capture baseline data:

```bash
python scripts/bmm_profile.py  # Check for sm80_xmma_gemm in backward
python scripts/bmm_bench.py    # Record timing baseline
```

Document findings in the GitHub issue thread.
