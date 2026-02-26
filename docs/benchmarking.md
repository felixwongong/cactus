# Kernel Benchmarking

Cactus benchmarks its quantized matmul kernels against other mobile/edge CPU inference frameworks to demonstrate our pure performance. This document covers the methodology, current results, and analysis.

## Methodology

### What we measure

We benchmark the core quantized matmul operation — the single most expensive operation in transformer inference, accounting for ~90% of decode time. Two shapes are tested:

- **1x1024x1024** — a single-row matrix-vector multiply (GEMV), representing autoregressive token decoding where one token is generated at a time
- **1024x1024x1024** — a batched matrix multiply (GEMM), representing prompt prefill or batched inference

Both INT4 and INT8 precision are tested. Most backends use group-wise quantization with a group size of 32, where each group of 32 weight elements along the K dimension shares a single scale factor.

**Note:** LiteRT and Ruy do not support group-wise (block-wise) INT8 quantization — they use **per-channel** quantization (one scale per output row) instead. LiteRT's INT4 backend (`litert_4bit_neon`) also uses per-channel weight quantization and additionally quantizes activations to 4 bits internally, which significantly increases quantization error (NRMSE ~1.4 vs ~0.07 for other INT4 backends).

### Cache pressure simulation

A naive benchmark that runs the same matmul 1024 times would keep the weight matrix hot in L2 cache the entire time — unrealistic, since real inference loads different weights for each transformer layer. To simulate this, we pre-generate 64 distinct random weight matrices (~1 MB each in INT8) and cycle through them round-robin during the timed loop. At 64 MB total, this exceeds both the L2 cache (16 MB on M4 Pro) and the system-level cache (SLC, ~36 MB), forcing realistic DRAM fetch patterns on every call.

### Timing

Each backend's `run_kernel` function is called 1024 times in the timed loop, preceded by 100 warmup calls. The reported latency is the average across all 1024 iterations.

### Accuracy

We run accuracy measurements PURELY as a sanity check, not as a proper gauge of performance. During warmup, before timing, each backend runs once with output capture enabled. The captured fp32 output is compared against a naive fp64-accumulated reference matmul using Normalized Root Mean Square Error (NRMSE). INT8 backends must achieve NRMSE < 0.05; INT4 backends must achieve NRMSE < 0.20.

### Frameworks compared

| Framework | Backends | Description |
|-----------|----------|-------------|
| **Cactus** | `cactus_int8`, `cactus_int4` | Cactus ARM NEON SIMD kernels with interleaved NK4 weight layout |
| **GGML** | `ggml_q4_0`, `ggml_q8_0`, `ggml_q4_0_graph` , `ggml_q8_0_graph` | llama.cpp's quantization engine; manual vec_dot dispatch and graph-based variants |
| **MLX** | `mlx_q4_cpu`, `mlx_q8_cpu`, `mlx_q4_gpu`, `mlx_q8_gpu` | Apple's ML framework with CPU and GPU quantized matmul |
| **LiteRT** | `litert_neon`, `ruy`, `litert_4bit_neon` | TFLite's NEON GEMV kernel, Ruy GEMM engine, and optimized 4-bit FC |
| **ONNX Runtime** | `onnxrt_int8`, `onnxrt_int4` | Microsoft's MatMulNBits operator |
| **ExecuTorch** | `executorch_int8`, `executorch_int4` | Meta's XNNPACK fully-connected operators |
| **MLC-LLM** | `mlc_int4`, `mlc_int8` | TVM-compiled quantized matmul kernels |

## Results

**Hardware:** Apple M4 Pro (14 cores, 16 MB L2, ~36 MB SLC)
**Settings:** 100 warmup, 1024 iterations, 64 matrices

### INT8 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT NEON ^ | 23.6 us | 88.76 | 0.0052 |
| **Cactus INT8** | **32.5 us** | **64.51** | **0.0051** |
| Ruy ^ | 35.3 us | 59.36 | 0.0052 |
| GGML Q8_0 | 57.0 us | 36.78 | 0.0053 |
| MLX Q8 GPU | 115.7 us | 18.13 | 0.0036 |
| MLX Q8 CPU | 259.1 us | 8.10 | 0.0036 |
| ONNX Runtime INT8 | 587.0 us | 3.57 | 0.0037 |

### INT8 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q8 GPU | 545.4 us | 3937.39 | 0.0037 |
| **Cactus INT8** | **828.2 us** | **2592.91** | **0.0055** |
| Ruy ^ | 1348.1 us | 1593.01 | 0.0056 |
| ONNX Runtime INT8 | 1616.1 us | 1328.85 | 0.0038 |
| GGML Q8_0 | 3268.6 us | 657.01 | 0.0053 |
| LiteRT NEON ^ | 24816.6 us | 86.53 | 0.0056 |
| MLX Q8 CPU | 142528.1 us | 15.07 | 0.0037 |

### INT4 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT 4bit NEON | 10.8 us | 194.06 | 1.4193 |
| **Cactus INT4** | **33.6 us** | **62.44** | **0.0691** |
| GGML Q4_0 | 55.1 us | 38.05 | 0.0665 |
| MLX Q4 GPU | 121.5 us | 17.26 | 0.0665 |
| MLX Q4 CPU | 248.5 us | 8.44 | 0.0665 |
| ONNX Runtime INT4 | 784.2 us | 2.67 | 0.0689 |

### INT4 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q4 GPU | 541.8 us | 3963.46 | 0.0650 |
| **Cactus INT4** | **1192.2 us** | **1801.34** | **0.0683** |
| ONNX Runtime INT4 | 1743.5 us | 1231.68 | 0.0682 |
| GGML Q4_0 | 3484.9 us | 616.22 | 0.0653 |
| LiteRT 4bit NEON | 6851.5 us | 313.43 | 1.4144 |
| MLX Q4 CPU | 133928.7 us | 16.03 | 0.0650 |

^ LiteRT NEON and Ruy use per-channel quantization (one scale per output row) rather than per-group (one scale per 32 elements). LiteRT does not support group-wise INT8 quantization. LiteRT 4bit NEON has high NRMSE (~1.4) because its kernel quantizes activations to 4 bits internally, introducing significantly more error than backends that keep activations at INT8 or FP16.

## Performance Analysis

Cactus is among the fastest general-purpose CPU matmul kernel across both precisions and both shapes. For GEMV, LiteRT NEON is marginally faster in INT8 (23.6 vs 32.5 us) and significantly faster in INT4 (10.8 vs 33.6 us), but at least part of its INT4 advantage comes from using per-channel quantization (less compute per element) rather than the per-group quantization every other backend uses, resulting in reduced output quality. LiteRT also collapses for GEMM, dropping to last place (24 ms INT8, 6.8 ms INT4) since its kernel is a single-threaded GEMV-only path with no batched matmul support.

MLX GPU dominates GEMM by dispatching to Apple's Metal/AMX/GPU hardware, but falls behind for GEMV due to GPU dispatch overhead, and its CPU backend is among the slowest overall (259 us for INT8 GEMV, 142 ms for INT8 GEMM).

All backends are compared against the same naive fp64 reference matmul for NRMSE.
