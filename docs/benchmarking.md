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

**Note:** LiteRT and Ruy do not support group-wise (block-wise) INT8 quantization — they use **per-channel** quantization (one scale per output row) instead. LiteRT's INT4 backend (`litert_4bit_neon`) also uses per-channel weight quantization and additionally quantizes activations to 4 bits internally, which significantly increases quantization error (NRMSE ~1.4 vs ~0.07 for other INT4 backends).

### Cache pressure simulation

A naive matmul benchmark that runs the same operation 1024 times would keep the weight matrix hot in L2 cache — unrealistic, since real inference loads different weights for each transformer layer. To simulate this, we pre-generate 64 distinct random weight matrices (~1 MB each in INT8) and cycle through them round-robin during the timed loop. At 64 MB total, this exceeds both the L2 cache (16 MB on M4 Pro) and the system-level cache (SLC, ~36 MB), forcing realistic DRAM fetch patterns.

Attention benchmarks use a single set of Q/K/V tensors since the working set is already large enough to exceed cache at the tested dimensions.

### Timing and accuracy

Each kernel is called in a timed loop (1024 iterations for matmul, 512 for attention) preceded by warmup calls. The reported latency is the per-call average.

Before timing, each backend runs once with output capture to verify correctness against a naive fp64-accumulated reference. INT8 backends must achieve NRMSE < 0.05 (matmul) or < 0.10 (attention); INT4 must achieve < 0.20. This is a sanity check, not a precision benchmark.

### Threading

By default, each backend uses its own tuned thread configuration. The `--threads` flag overrides all backends to use a fixed thread count (`--threads max` uses all hardware threads). This allows fair cross-framework comparisons at matched thread counts.

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
| **Cactus** | `cactus_int8`, `cactus_int4` | `cactus_prefill` (FP16), `cactus_decode` (hybrid INT8/FP16) | ARM NEON SIMD kernels with interleaved NK4 weight layout; hybrid decode uses INT8 quantized KV cache with FP16 new tokens |
| **GGML** | `ggml_q4_0`, `ggml_q8_0` | `ggml_fa_{f16,q8,q4}_{prefill,decode}`, `ggml_mm_{q8,q4}_{prefill,decode}` | llama.cpp's quantization engine; flash attention (`fa`) via `ggml_flash_attn_ext`, matmul-composed (`mm`) via `ggml_mul_mat` + `ggml_soft_max_ext` |
| **MLX** | `mlx_q{4,8}_{gpu,cpu}` | `mlx_{gpu,cpu}_{prefill,decode}`, `mlx_q{4,8}_{gpu,cpu}_{prefill,decode}` | Apple's ML framework; GPU uses Metal kernels, CPU uses Accelerate/BNNS; quantized attention mirrors mlx-lm's `quantized_scaled_dot_product_attention` |
| **LiteRT** | `litert_neon`, `ruy`, `litert_4bit_neon` | — | TFLite's NEON GEMV kernel, Ruy GEMM engine, and optimized 4-bit FC |
| **ONNX Runtime** | `onnxrt_int8`, `onnxrt_int4` | — | Microsoft's MatMulNBits operator |
| **ExecuTorch** | `executorch_int8`, `executorch_int4` | — | Meta's XNNPACK fully-connected operators |
| **MLC-LLM** | `mlc_int4`, `mlc_int8` | `mlc_q{4,8}_{prefill,decode}` | TVM-compiled quantized matmul kernels; attention uses two quantized matmul calls with C++ softmax |

## Matmul Results

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

## Attention Results

**Hardware:** Apple M4 Pro (14 cores, 16 MB L2, ~36 MB SLC)
**Settings:** 30 warmup, 100 iterations, head_dim=128, q_heads=32, kv_heads=8 (GQA ratio 4)

### Cactus vs GGML

Cactus uses FP16 attention for prefill and a hybrid INT8/FP16 kernel for decode (INT8 quantized KV cache with FP16 new tokens).

GGML provides two attention paths: **flash attention** (`fa`) using `ggml_flash_attn_ext` (a fused kernel), and **matmul-composed** (`mm`) using `ggml_mul_mat` + `ggml_soft_max_ext` + `ggml_mul_mat` (the llama.cpp fallback when flash attention is disabled). Both paths are tested with FP16, Q8_0, and Q4_0 KV cache quantization.

#### Prefill — seq_len=1024, causal

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| GGML Flash Attn FP16 | 33792.4 us | 513.36 | 0.0002 |
| **Cactus FP16** | **39171.5 us** | **442.86** | **0.0076** |
| GGML Matmul Q8_0 | 45294.8 us | 382.99 | 0.0044 |
| GGML Matmul Q4_0 | 45978.2 us | 377.30 | 0.0701 |
| GGML Flash Attn Q8_0 | 58309.4 us | 297.51 | 0.0038 |
| GGML Flash Attn Q4_0 | 64228.7 us | 270.09 | 0.0647 |

#### Decode — seq_len=1, cache_len=511

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| GGML Flash Attn FP16 | 96.1 us | 88.17 | 0.0010 |
| GGML Matmul Q8_0 | 175.6 us | 48.25 | 0.0048 |
| **Cactus Hybrid INT8/FP16** | **207.9 us** | **40.74** | **0.0039** |
| GGML Flash Attn Q8_0 | 214.9 us | 39.42 | 0.0039 |
| GGML Matmul Q4_0 | 358.0 us | 23.66 | 0.0729 |
| GGML Flash Attn Q4_0 | 416.6 us | 20.33 | 0.0680 |

#### GGML attention path analysis

GGML's flash attention (`ggml_flash_attn_ext`) has two internal fast paths on CPU:
- **Tiled path** (prefill): activated when KV type is F16 or F32, K/V types match, and `kv_seq_len` is aligned to the tile size
- **Split-KV path** (decode): activated when `seq_len=1`, KV type is F16 or F32, and `kv_seq_len >= 512`

Quantized KV types (Q8_0, Q4_0) fall through to a generic scalar path, which is why `fa_q8` and `fa_q4` are significantly slower than `fa_f16`.

The matmul-composed path benefits from ggml's heavily optimized ARM NEON quantized matmul kernels (with I8MM and DOTPROD support), making `mm_q8` and `mm_q4` ~1.3x faster than their `fa_q8`/`fa_q4` flash attention equivalents for prefill. For decode, `mm_q8` also outperforms `fa_q8`.

### MLX

**Settings:** 50 warmup, 256 iterations, same attention dimensions as above

MLX provides three attention strategies: **FP16 SDPA** using `mx::fast::scaled_dot_product_attention` (fused flash attention on GPU, Accelerate-decomposed on CPU), **quantized INT8** and **quantized INT4** using `mx::quantized_matmul` for Q@K.T and scores@V with quantized KV cache (mirroring mlx-lm's `quantized_scaled_dot_product_attention`).

On CPU, the quantized path uses pre-dequantized FP16 KV with Accelerate/BNNS matmul rather than MLX's naive CPU `quantized_matmul` kernels, since the latter lack SIMD optimization for the `transpose=false` path and run ~700x slower than Accelerate. On GPU, the optimized Metal `quantized_matmul` kernels are used directly.

#### Prefill — seq_len=1024, causal

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| MLX FP16 CPU | 1586.2 us | 10936.81 | 0.0003 |
| MLX FP16 GPU | 1605.4 us | 10805.56 | 0.0003 |
| MLX Q8 GPU | 4248.9 us | 4082.82 | 0.0039 |
| MLX Q4 GPU | 4247.0 us | 4084.69 | 0.0672 |
| MLX Q8 CPU | 29410.4 us | 589.85 | 0.0039 |
| MLX Q4 CPU | 29561.0 us | 586.84 | 0.0672 |

#### Decode — seq_len=1, cache_len=511

| Backend | Latency | GFLOPS | NRMSE |
|---------|---------|--------|-------|
| MLX FP16 CPU | 126.3 us | 67.07 | 0.0003 |
| MLX Q8 GPU | 137.7 us | 61.52 | 0.0041 |
| MLX FP16 GPU | 162.3 us | 52.20 | 0.0003 |
| MLX Q4 GPU | 170.5 us | 49.67 | 0.0699 |
| MLX Q8 CPU | 270.0 us | 31.37 | 0.0041 |
| MLX Q4 CPU | 272.7 us | 31.07 | 0.0698 |

#### MLX attention analysis

FP16 SDPA is fastest for prefill on both GPU and CPU — the fused Metal kernel and Accelerate/AMX decomposition outperform the unfused quantized matmul decomposition. CPU FP16 slightly edges out GPU for prefill due to AMX throughput matching Metal at this scale.

For decode, GPU quantized INT8 is competitive with FP16 (138 vs 162 us), since the smaller working set makes the quantized Metal kernels efficient. CPU quantized decode (270 us) is ~2x slower than CPU FP16 (126 us) due to the overhead of Accelerate matmul dispatch on small matrices.

The GPU quantized prefill gap (4.2ms vs 1.6ms for FP16) reflects the cost of the unfused two-`quantized_matmul` decomposition vs the fused flash attention kernel. A fused quantized SDPA Metal kernel has been proposed upstream ([PR #3026](https://github.com/ml-explore/mlx/pull/3026)) but is not yet merged.

## Analysis

### Matmul

Cactus is among the fastest general-purpose CPU matmul kernel across both precisions and both shapes. For GEMV, LiteRT NEON is marginally faster in INT8 (23.6 vs 32.5 us) and significantly faster in INT4 (10.8 vs 33.6 us), but at least part of its INT4 advantage comes from using per-channel quantization (less compute per element) rather than the per-group quantization every other backend uses, resulting in reduced output quality. LiteRT also collapses for GEMM, dropping to last place (24 ms INT8, 6.8 ms INT4) since its kernel is a single-threaded GEMV-only path with no batched matmul support.

MLX GPU dominates GEMM by dispatching to Apple's Metal/AMX/GPU hardware, but falls behind for GEMV due to GPU dispatch overhead, and its CPU backend is among the slowest overall (259 us for INT8 GEMV, 142 ms for INT8 GEMM).

### Attention

For CPU-only attention, Cactus FP16 prefill (39ms) is competitive with GGML's best quantized paths (45ms for mm_q8) while maintaining higher precision. GGML's flash attention FP16 (34ms) is the fastest prefill path due to its tiled kernel hitting the optimized fast path.

For decode, GGML's flash attention FP16 (96 us) leads, followed by GGML mm_q8 (176 us) and Cactus hybrid INT8/FP16 (208 us). Cactus's hybrid decode approach — INT8 quantized KV cache with FP16 new tokens — trades ~2x latency vs GGML's FP16 flash attention for a ~2x reduction in KV cache memory, a worthwhile tradeoff for memory-constrained mobile devices.

MLX's Accelerate/AMX-backed FP16 SDPA is dramatically faster for prefill (1.6ms vs 34-39ms for CPU-only backends), demonstrating the advantage of Apple's AMX coprocessor. However, this path is only available on Apple Silicon and is not portable.

All backends are compared against the same naive fp64 reference for NRMSE.

## Running benchmarks

```bash
# Matmul (default)
./tests/run_benchmark.sh --external-frameworks

# Attention
./tests/run_benchmark.sh --attention --external-frameworks

# With thread override
./tests/run_benchmark.sh --threads max --backends cactus,ggml
./tests/run_benchmark.sh --attention --threads 4 --backends cactus

# Direct executable (from tests/build/)
./matmul_bench [--iterations N] [--warmup N] [--matrices N] [--backends fw1,fw2] [--threads N|max]
./attn_bench [--iterations N] [--warmup N] [--backends fw1,fw2] [--threads N|max]
             [--prefill_len N] [--cache_len N] [--head_dim N] [--q_heads N] [--kv_heads N]
```
