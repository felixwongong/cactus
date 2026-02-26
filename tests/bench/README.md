# Benchmark Suite

Compares Cactus quantized matmul and attention kernels against other inference frameworks on identical workloads.

**Matmul** — 1x1024x1024 (GEMV) and 1024x1024x1024 (GEMM), INT8 and INT4 precision.
**Attention** — prefill (seq_len=1024) and decode (seq_len=1, cache_len=511), GQA with 32 query heads and 8 KV heads.

## Quick Start (Cactus-only, no third-party deps)

```bash
cactus build
cd tests && mkdir -p build && cd build
cmake ..
make -j matmul_bench attn_bench

./matmul_bench                  # cactus_int8, cactus_int4
./attn_bench                    # cactus_prefill (FP16), cactus_decode (hybrid INT8/FP16)
```

Or use the wrapper script:

```bash
./tests/run_benchmark.sh                    # matmul benchmark
./tests/run_benchmark.sh --attention        # attention benchmark
```

## Adding Third-Party Frameworks

Each framework is an opt-in CMake option. Clone/download into `../third_party/` (one level above the repo root), then enable the flag at configure time.

### GGML

```bash
git clone https://github.com/ggml-org/ggml.git ../third_party/ggml
cmake .. -DWITH_GGML=ON
```

Builds GGML from source. Enables matmul backends (`ggml_q4_0`, `ggml_q8_0`) and attention backends (flash attention and matmul-composed paths with FP16/Q8_0/Q4_0 KV types).

### LiteRT (TFLite)

```bash
git clone https://github.com/google-ai-edge/LiteRT.git ../third_party/litert
cmake .. -DWITH_LITERT=ON
```

Fetches FlatBuffers + TFLite deps on first build (requires network). Enables `litert_neon`, `ruy`, and `litert_4bit_neon` matmul backends. No attention backends.

### MLX (Apple-only)

```bash
git clone https://github.com/ml-explore/mlx.git ../third_party/mlx
cmake .. -DWITH_MLX=ON
```

Requires macOS with Metal. Enables matmul backends (`mlx_q{4,8}_{cpu,gpu}`) and attention backends (FP16 SDPA and quantized INT4/INT8 on both CPU and GPU).

### MLC-LLM (TVM runtime)

```bash
git clone --recursive https://github.com/mlc-ai/mlc-llm ../third_party/mlc
cd ../third_party/mlc/3rdparty/tvm && mkdir -p build && cd build
cmake .. -DUSE_LLVM=OFF && make tvm_runtime -j
cd ../../../../..
```

Then compile the benchmark matmul kernels using TVM's compiler and set `MLC_MATMUL_LIB` to the resulting `.so`:

```bash
python tests/bench/compile_mlc_kernels.py  # produces bench_kernels.so
export MLC_MATMUL_LIB=path/to/bench_kernels.so
cmake .. -DWITH_MLC=ON
```

The compiled module must export `quantized_matmul_int4` and/or `quantized_matmul_int8` functions. Enables matmul backends (`mlc_int4`, `mlc_int8`) and attention backends (`mlc_q{4,8}_{prefill,decode}`).

### ONNX Runtime (prebuilt)

Download the prebuilt release for your platform from https://github.com/microsoft/onnxruntime/releases and extract into `../third_party/onnxruntime/` so you have:

```
../third_party/onnxruntime/
  include/
  lib/
    libonnxruntime.dylib   (macOS)
    libonnxruntime.so      (Linux)
```

```bash
cmake .. -DWITH_ONNXRT=ON
```

Enables `onnxrt_int8` and `onnxrt_int4` matmul backends. No attention backends.

### Executorch (XNNPACK)

```bash
cmake .. -DWITH_EXECUTORCH=ON
```

Fetches XNNPACK from GitHub at configure time (no manual clone needed). Enables `executorch_int8` and `executorch_int4` matmul backends. No attention backends.

## Enabling Multiple Frameworks

Combine flags, or use `--external-frameworks` to auto-detect:

```bash
# Manual
cmake .. -DWITH_GGML=ON -DWITH_LITERT=ON -DWITH_MLX=ON
make -j matmul_bench attn_bench

# Auto-detect
./tests/run_benchmark.sh --external-frameworks
./tests/run_benchmark.sh --attention --external-frameworks
```

## Quantization Granularity

Most backends use **group-wise** quantization (group size 32), where each group of 32 elements along K shares a single scale factor. The exceptions are noted below.

| Backend | Precision | Granularity | Notes |
|---------|-----------|-------------|-------|
| **Cactus** | INT4, INT8 | Group-wise (32) | |
| **GGML** | Q4_0, Q8_0 | Group-wise (32) | |
| **LiteRT 4bit_neon** | INT4 | Per-channel | Per-channel weights + 4-bit activations |
| **LiteRT neon / Ruy** | INT8 | Per-channel | LiteRT does not support group-wise INT8 |
| **MLX** | INT4, INT8 | Group-wise (32) | |
| **MLC** | INT4, INT8 | Group-wise (32) | |
| **ONNX Runtime** | INT4, INT8 | Group-wise (32) | MatMulNBits (com.microsoft) |
| **ExecuTorch** | INT4 | Group-wise (32) | XNNPACK `qb4w` kernel |
| **ExecuTorch** | INT8 | Per-channel | XNNPACK `qc8w` kernel; no block-wise INT8 API exists in XNNPACK |

## CLI Options

```
./matmul_bench [options]

  --warmup N          Warmup iterations (default: 100)
  --iterations N      Timed iterations (default: 1024)
  --matrices N        Distinct weight matrices to cycle through (default: 64)
  --backends FILTER   Comma-separated framework names to run
  --threads N|max     Override thread count for all backends (default: each backend's own)
```

```
./attn_bench [options]

  --warmup N          Warmup iterations (default: 100)
  --iterations N      Timed iterations (default: 512)
  --backends FILTER   Comma-separated framework names to run
  --threads N|max     Override thread count for all backends (default: each backend's own)
  --prefill_len N     Prefill sequence length (default: 1024)
  --cache_len N       Decode KV cache length (default: 511)
  --head_dim N        Head dimension (default: 128)
  --q_heads N         Number of query heads (default: 32)
  --kv_heads N        Number of KV heads (default: 8)
```

```
./tests/run_benchmark.sh [options]

  --attention             Run attention benchmark instead of matmul
  --external-frameworks   Auto-detect and enable third-party frameworks
  --threads N|max         Override thread count (also sets TVM_NUM_THREADS for MLC)
  (all other flags are passed through to the benchmark executable)
```

## Results

See [docs/benchmarking.md](../../docs/benchmarking.md) for full results, methodology, and analysis.
