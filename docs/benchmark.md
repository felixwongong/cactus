# Kernel Benchmarking

Cactus benchmarks its quantized matmul and attention kernels against other mobile/edge CPU inference frameworks. This document covers the methodology, current results, and analysis.

## Methodology

### What we measure

We benchmark the two operations that dominate transformer inference time:

**Matmul** — the quantized matrix multiply, accounting for ~90% of decode time. Two shapes are tested:
- **1x1024x1024** (GEMV) — single-row matrix-vector multiply, representing autoregressive token decoding
- **1024x1024x1024** (GEMM) — batched matrix multiply, representing prompt prefill

Both INT4 and INT8 precision are tested. Most backends use group-wise quantization with a group size of 32.

**Attention** — the full scaled dot-product attention operation. Two modes are tested:
- **Prefill** — processing 1024 input tokens with causal masking (seq_len=1024)
- **Decode** — generating 1 token with 511 cached KV tokens (seq_len=1, cache_len=511)

Attention uses GQA dimensions matching Qwen3/Llama-class models: head_dim=128, 32 query heads, 8 KV heads (GQA ratio 4).

**Note:** LiteRT and Ruy do not support group-wise (block-wise) INT8 quantization — they use **per-channel** quantization (one scale per output row) instead. LiteRT's INT4 matmul backend (`litert_4bit_neon`) also uses per-channel weight quantization and additionally quantizes activations to 4 bits internally, which significantly increases quantization error (NRMSE ~1.4 vs ~0.07 for other INT4 backends). LiteRT's INT4 attention backend (`litert_q4`) uses a similar per-channel scheme, resulting in elevated NRMSE (~0.10-0.12).

### Cache pressure simulation

A naive matmul benchmark that runs the same operation 1024 times would keep the weight matrix hot in L2 cache — unrealistic, since real inference loads different weights for each transformer layer. To simulate this, we pre-generate 64 distinct random weight matrices (~1 MB each in INT8) and cycle through them round-robin during the timed loop. At 64 MB total, this exceeds both the L2 cache (16 MB on M4 Pro) and the system-level cache (SLC, ~36 MB), forcing realistic DRAM fetch patterns.

Attention benchmarks use a single set of Q/K/V tensors since the working set is already large enough to exceed cache at the tested dimensions.

### Timing and accuracy

Each kernel is called in a timed loop (1024 iterations for matmul, 512 for attention) preceded by warmup calls. The reported latency is the per-call average.

Before timing, each backend runs once with output capture to verify correctness against a naive fp64-accumulated reference. INT8 backends must achieve NRMSE < 0.05 (matmul) or < 0.10 (attention); INT4 must achieve < 2.0. This is a sanity check, not a precision benchmark.

### Threading

Results are reported at two thread configurations:
- **Default** — each backend uses its own tuned thread configuration
- **12 threads** — all backends forced to 12 threads via `--threads 12`, enabling fair cross-framework comparison at matched thread counts

| Backend | Default threading | Override mechanism |
|---------|------------------|-------------------|
| Cactus | Heuristic per shape | `CactusThreading::set_gemm_threads()` |
| GGML matmul | `CactusThreading::parallel_for` | Already uses full pool |
| GGML attention | `hardware_concurrency()` | Built at prepare-time |
| LiteRT/Ruy | GEMV=1, GEMM=4 | `ruy::Context::set_max_num_threads()` |
| ONNX Runtime | GEMV=2, GEMM=3 | `SessionOptions::SetIntraOpNumThreads()` |
| ExecuTorch | `hardware_concurrency()` | `pthreadpool_create(n)` |
| MLC | TVM runtime managed | `TVM_NUM_THREADS` env var |
| MLX GPU | Metal dispatch | N/A |
| MLX CPU | Accelerate/BNNS managed | N/A |

### Frameworks compared

| Framework | Matmul backends | Attention backends | Description |
|-----------|----------------|-------------------|-------------|
| **Cactus** | `cactus_int8`, `cactus_int4` | `cactus_prefill` (INT8/FP16), `cactus_decode` (INT8/FP16) | ARM NEON SIMD kernels with interleaved NK4 weight layout; INT8 quantized KV cache with FP16 computation |
| **GGML** | `ggml_q4_0`, `ggml_q8_0` | `ggml_fa_{f16,q8,q4}_{prefill,decode}`, `ggml_mm_{q8,q4}_{prefill,decode}` | llama.cpp's quantization engine; flash attention (`fa`) via `ggml_flash_attn_ext`, matmul-composed (`mm`) via `ggml_mul_mat` + `ggml_soft_max_ext` |
| **MLX** | `mlx_q{4,8}_{gpu,cpu}` | `mlx_{gpu,cpu}_{prefill,decode}`, `mlx_q{4,8}_{gpu,cpu}_{prefill,decode}` | Apple's ML framework; GPU uses Metal kernels, CPU uses Accelerate/BNNS; quantized attention mirrors mlx-lm's `quantized_scaled_dot_product_attention` |
| **LiteRT** | `litert_neon`, `ruy`, `litert_4bit_neon` | `litert_ruy_int8_{prefill,decode}`, `litert_neon_int8_decode`, `litert_q4_{prefill,decode}` | TFLite's NEON GEMV kernel, Ruy GEMM engine, and optimized 4-bit FC; attention uses Ruy/NEON INT8 matmul or 4-bit matmul for Q@K and scores@V |
| **ONNX Runtime** | `onnxrt_int8`, `onnxrt_int4` | — | Microsoft's MatMulNBits operator |
| **ExecuTorch** | `executorch_int8`, `executorch_int4` | — | Meta's XNNPACK fully-connected operators with KleidiAI ARM kernels |
| **MLC-LLM** | `mlc_int4`, `mlc_int8` | `mlc_q{4,8}_{prefill,decode}` | TVM-compiled quantized matmul kernels; attention uses two quantized matmul calls with C++ softmax |

## Matmul Results

**Hardware:** Apple M4 Pro (14 cores, 16 MB L2, ~36 MB SLC)
**Settings:** 100 warmup, 1024 iterations, 64 matrices

### Default Threads

#### INT8 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT NEON ^ | 26.6 us | 78.94 | 0.0052 |
| **Cactus INT8** | **28.8 us** | **72.78** | **0.0051** |
| ExecuTorch INT8 | 32.4 us | 64.65 | 0.0052 |
| Ruy ^ | 38.7 us | 54.20 | 0.0052 |
| GGML Q8_0 | 54.8 us | 38.25 | 0.0053 |
| MLX Q8 GPU | 120.7 us | 17.38 | 0.0036 |
| MLX Q8 CPU | 266.3 us | 7.88 | 0.0036 |
| ONNX Runtime INT8 | 606.3 us | 3.46 | 0.0037 |

#### INT8 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q8 GPU | 621.9 us | 3452.85 | 0.0037 |
| ExecuTorch INT8 | 664.9 us | 3229.80 | 0.0056 |
| **Cactus INT8** | **833.6 us** | **2576.04** | **0.0055** |
| Ruy ^ | 1331.4 us | 1612.94 | 0.0056 |
| ONNX Runtime INT8 | 1706.2 us | 1258.62 | 0.0038 |
| GGML Q8_0 | 3414.3 us | 628.96 | 0.0053 |
| LiteRT NEON ^ | 25078.5 us | 85.63 | 0.0056 |
| MLX Q8 CPU | 140302.9 us | 15.31 | 0.0037 |

#### INT4 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT 4bit NEON ^ | 12.0 us | 174.83 | 1.4193 |
| **Cactus INT4** | **28.2 us** | **74.45** | **0.0691** |
| ExecuTorch INT4 | 30.1 us | 69.66 | 0.0689 |
| GGML Q4_0 | 56.0 us | 37.48 | 0.0665 |
| MLX Q4 GPU | 124.4 us | 16.86 | 0.0665 |
| MLX Q4 CPU | 262.6 us | 7.99 | 0.0665 |
| ONNX Runtime INT4 | 848.5 us | 2.47 | 0.0689 |

#### INT4 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q4 GPU | 563.0 us | 3814.13 | 0.0650 |
| ExecuTorch INT4 | 1065.7 us | 2015.08 | 0.0680 |
| **Cactus INT4** | **1171.7 us** | **1832.74** | **0.0683** |
| ONNX Runtime INT4 | 1672.9 us | 1283.72 | 0.0682 |
| GGML Q4_0 | 3195.9 us | 671.96 | 0.0651 |
| LiteRT 4bit NEON ^ | 6723.1 us | 319.42 | 1.4136 |
| MLX Q4 CPU | 132419.7 us | 16.22 | 0.0650 |

### 12 Threads

#### INT8 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT NEON ^ | 26.5 us | 79.24 | 0.0052 |
| **Cactus INT8** | **30.3 us** | **69.24** | **0.0051** |
| ExecuTorch INT8 | 31.2 us | 67.24 | 0.0052 |
| GGML Q8_0 | 53.4 us | 39.27 | 0.0053 |
| MLX Q8 GPU | 147.3 us | 14.23 | 0.0036 |
| MLX Q8 CPU | 265.6 us | 7.90 | 0.0036 |
| ONNX Runtime INT8 | 1031.2 us | 2.03 | 0.0037 |
| Ruy ^ | 1760.3 us | 1.19 | 0.0052 |

#### INT8 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q8 GPU | 678.8 us | 3163.58 | 0.0037 |
| ExecuTorch INT8 | 655.2 us | 3277.78 | 0.0056 |
| **Cactus INT8** | **862.1 us** | **2490.94** | **0.0055** |
| Ruy ^ | 2596.0 us | 827.23 | 0.0056 |
| ONNX Runtime INT8 | 2714.5 us | 791.12 | 0.0038 |
| GGML Q8_0 | 3028.1 us | 709.19 | 0.0053 |
| LiteRT NEON ^ | 25329.3 us | 84.78 | 0.0056 |
| MLX Q8 CPU | 141565.0 us | 15.17 | 0.0037 |

#### INT4 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT 4bit NEON ^ | 11.6 us | 181.31 | 1.4193 |
| ExecuTorch INT4 | 28.3 us | 74.14 | 0.0689 |
| **Cactus INT4** | **30.7 us** | **68.40** | **0.0691** |
| GGML Q4_0 | 48.9 us | 42.92 | 0.0665 |
| MLX Q4 GPU | 134.6 us | 15.58 | 0.0665 |
| MLX Q4 CPU | 254.0 us | 8.26 | 0.0665 |
| ONNX Runtime INT4 | 1511.7 us | 1.39 | 0.0689 |

#### INT4 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q4 GPU | 616.6 us | 3482.97 | 0.0650 |
| ExecuTorch INT4 | 924.8 us | 2322.11 | 0.0680 |
| **Cactus INT4** | **1225.8 us** | **1751.92** | **0.0683** |
| GGML Q4_0 | 3471.0 us | 618.70 | 0.0651 |
| ONNX Runtime INT4 | 3179.9 us | 675.33 | 0.0682 |
| LiteRT 4bit NEON ^ | 7141.7 us | 300.69 | 1.4136 |
| MLX Q4 CPU | 132372.4 us | 16.22 | 0.0650 |

^ LiteRT NEON and Ruy use per-channel quantization (one scale per output row) rather than per-group (one scale per 32 elements). LiteRT does not support group-wise INT8 quantization. LiteRT 4bit NEON has high NRMSE (~1.4) because its kernel quantizes activations to 4 bits internally, introducing significantly more error than backends that keep activations at INT8 or FP16. Ruy degrades severely at 12 threads for GEMV (1760 us vs 38.7 us default) because its thread pool overhead dominates at single-row sizes when over-provisioned.

## Attention Results

**Hardware:** Apple M4 Pro (14 cores, 16 MB L2, ~36 MB SLC)
**Settings:** 100 warmup, 512 iterations, head_dim=128, q_heads=32, kv_heads=8 (GQA ratio 4)

### Default Threads

#### Prefill — seq_len=1024, causal

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| MLX FP16 CPU | 1598.0 us | 10855.57 | 0.0003 |
| MLX FP16 GPU | 1687.4 us | 10280.97 | 0.0003 |
| MLX Q4 GPU | 4212.4 us | 4118.20 | 0.0672 |
| MLX Q8 GPU | 4265.3 us | 4067.15 | 0.0039 |
| MLX Q4 CPU | 30001.6 us | 578.22 | 0.0672 |
| MLX Q8 CPU | 30051.9 us | 577.26 | 0.0039 |
| GGML Flash Attn FP16 | 33104.1 us | 524.03 | 0.0002 |
| GGML Matmul Q8_0 | 41164.3 us | 421.42 | 0.0044 |
| GGML Matmul Q4_0 | 43556.3 us | 398.28 | 0.0701 |
| **Cactus INT8** | **44018.3 us** | **394.10** | **0.0076** |
| GGML Flash Attn Q8_0 | 63009.9 us | 275.32 | 0.0038 |
| GGML Flash Attn Q4_0 | 66065.2 us | 262.58 | 0.0647 |
| LiteRT Ruy INT8 ^ | 98679.2 us | 175.80 | 0.0046 |
| LiteRT Q4 ^ | 160400.5 us | 108.15 | 0.1047 |

#### Decode — seq_len=1, cache_len=511

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| LiteRT NEON INT8 ^ | 127.6 us | 66.39 | 0.0049 |
| GGML Matmul Q4_0 | 150.5 us | 56.28 | 0.0729 |
| MLX Q4 GPU | 151.2 us | 56.03 | 0.0699 |
| MLX FP16 GPU | 153.3 us | 55.25 | 0.0003 |
| LiteRT Q4 ^ | 153.8 us | 55.07 | 0.1170 |
| MLX FP16 CPU | 177.8 us | 47.64 | 0.0003 |
| MLX Q8 GPU | 216.4 us | 39.13 | 0.0041 |
| LiteRT Ruy INT8 ^ | 217.8 us | 38.90 | 0.0049 |
| **Cactus INT8** | **223.2 us** | **37.95** | **0.0039** |
| GGML Matmul Q8_0 | 229.1 us | 36.97 | 0.0048 |
| MLX Q4 CPU | 279.6 us | 30.30 | 0.0698 |
| MLX Q8 CPU | 283.2 us | 29.91 | 0.0041 |
| GGML Flash Attn Q4_0 † | 325.4 us | 26.03 | 0.0680 |
| GGML Flash Attn Q8_0 † | 1999.8 us | 4.24 | 0.0039 |
| GGML Flash Attn FP16 † | 3392.9 us | 2.50 | 0.0010 |

### 12 Threads

#### Prefill — seq_len=1024, causal

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| MLX FP16 CPU | 1602.0 us | 10828.52 | 0.0003 |
| MLX FP16 GPU | 1631.4 us | 10633.53 | 0.0003 |
| MLX Q4 GPU | 4279.9 us | 4053.32 | 0.0672 |
| MLX Q8 GPU | 4287.1 us | 4046.48 | 0.0039 |
| MLX Q4 CPU | 29522.7 us | 587.60 | 0.0672 |
| MLX Q8 CPU | 29563.0 us | 586.80 | 0.0039 |
| GGML Flash Attn FP16 | 33095.7 us | 524.17 | 0.0002 |
| GGML Matmul Q8_0 | 40850.0 us | 424.67 | 0.0044 |
| GGML Matmul Q4_0 | 42671.4 us | 406.54 | 0.0701 |
| **Cactus INT8** | **43758.2 us** | **396.44** | **0.0076** |
| GGML Flash Attn Q8_0 | 61782.1 us | 280.79 | 0.0038 |
| GGML Flash Attn Q4_0 | 65478.5 us | 264.94 | 0.0647 |
| LiteRT Ruy INT8 ^ | 99286.0 us | 174.72 | 0.0046 |
| LiteRT Q4 ^ | 158093.3 us | 109.73 | 0.1047 |

#### Decode — seq_len=1, cache_len=511

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| GGML Matmul Q4_0 | 111.8 us | 75.79 | 0.0729 |
| MLX FP16 GPU | 121.2 us | 69.91 | 0.0003 |
| LiteRT NEON INT8 ^ | 130.0 us | 65.18 | 0.0049 |
| LiteRT Q4 ^ | 130.2 us | 65.04 | 0.1170 |
| GGML Flash Attn Q4_0 | 147.1 us | 57.57 | 0.0680 |
| GGML Flash Attn Q8_0 | 154.3 us | 54.90 | 0.0039 |
| MLX Q4 GPU | 156.1 us | 54.26 | 0.0699 |
| LiteRT Ruy INT8 ^ | 158.8 us | 53.36 | 0.0049 |
| MLX FP16 CPU | 163.5 us | 51.82 | 0.0003 |
| GGML Matmul Q8_0 | 198.3 us | 42.73 | 0.0048 |
| MLX Q8 GPU | 206.2 us | 41.07 | 0.0041 |
| **Cactus INT8** | **216.0 us** | **39.21** | **0.0039** |
| GGML Flash Attn FP16 | 243.2 us | 34.83 | 0.0010 |
| MLX Q8 CPU | 258.2 us | 32.80 | 0.0041 |
| MLX Q4 CPU | 273.7 us | 30.95 | 0.0698 |

^ LiteRT attention backends use per-channel quantization rather than per-group, same as their matmul backends. LiteRT Q4 has elevated NRMSE (~0.10-0.12) due to per-channel weight quantization with internal 4-bit activation quantization.

† GGML flash attention decode results at default threads are misleadingly slow — the FA kernel requires a minimum thread count to activate its fast path, and falls back to a scalar path otherwise. The 12-thread results reflect its actual performance.

## Analysis

### Matmul

**INT8:** Cactus is the second-fastest CPU matmul kernel for GEMV regardless of thread count, barely trailing LiteRT NEON (28.8 vs 26.6 us at default threads). LiteRT uses per-channel quantization rather than the per-group quantization every other backend uses — a simpler computation with fewer scale factors to apply. For GEMM, Cactus is modestly behind ExecuTorch's KleidiAI-backed kernel (834 vs 665 us), whose ARM I8MM-optimized tiling achieves higher utilization at large batch sizes.

**INT4:** Similar story. For GEMV, LiteRT's 4-bit kernel is significantly faster (12.0 vs 28.2 us) but produces unusable output (NRMSE 1.4 vs 0.07) due to internal 4-bit activation quantization — not a fair comparison. Among accurate backends, Cactus leads GEMV. For GEMM, Cactus is ~9% slower than ExecuTorch (1172 vs 1066 us).

LiteRT collapses for GEMM in both precisions (25ms INT8, 6.7ms INT4) — its kernel is a GEMV-only path with no batched matmul support. MLX GPU dominates GEMM via Metal/AMX hardware but loses GEMV to GPU launch overhead. At 12 threads, GEMV results are mostly stable while Ruy degrades catastrophically (38.7 us → 1760 us) due to thread pool overhead on single-row operations.

### Attention

**Prefill:** Among CPU-only backends, Cactus INT8 (44ms) is only behind GGML — specifically GGML's flash attention FP16 (33ms) and matmul-composed paths (41-44ms) — and not by much. MLX's AMX-backed SDPA is in a different category entirely (1.6ms) thanks to Apple's AMX coprocessor.

**Decode:** This is Cactus's weakest result. At 223 us (default) / 216 us (12 threads), we're significantly slower than LiteRT's optimized NEON INT8 kernel (128 us) and GGML's fastest kernels. At 12 threads, GGML's matmul Q4_0 (112 us) is nearly 2x as fast. Our INT8/FP16 approach — INT8 quantized KV cache with FP16 computation — trades latency for a ~2x reduction in KV cache memory, a worthwhile tradeoff on memory-constrained mobile devices, but decode performance is a clear area for improvement.

## Running benchmarks

```bash
# Matmul (default threads)
./tests/run_benchmark.sh --external-frameworks

# Matmul (12 threads)
./tests/run_benchmark.sh --external-frameworks --threads 12

# Attention (default threads)
./tests/run_benchmark.sh --attention --external-frameworks

# Attention (12 threads)
./tests/run_benchmark.sh --attention --external-frameworks --threads 12

# Direct executable (from tests/build/)
./matmul_bench [--iterations N] [--warmup N] [--matrices N] [--backends fw1,fw2] [--threads N|max]
./attn_bench [--iterations N] [--warmup N] [--backends fw1,fw2] [--threads N|max]
             [--prefill_len N] [--cache_len N] [--head_dim N] [--q_heads N] [--kv_heads N]
```
