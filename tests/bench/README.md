# Matmul Benchmark Suite

Compares Cactus INT8/INT4 matmul kernels against other inference frameworks on identical workloads (1x1024x1024 and 1024x1024x1024).
This only compares CPU performance; NPU & GPU are not benchmarked because they are too power hungry for our use cases.

## Quick Start (Cactus-only, no third-party deps)

```bash
cactus build
cd tests && mkdir -p build && cd build
cmake ..
make -j matmul_bench
./matmul_bench
```

This runs the `cactus_int8` and `cactus_int4` backends with no external dependencies.

## Adding Third-Party Frameworks

Each framework is an opt-in CMake option. Clone/download into `../third_party/` (one level above the repo root), then enable the flag at configure time.

### GGML

```bash
git clone https://github.com/ggml-org/ggml.git ../third_party/ggml
cmake .. -DWITH_GGML=ON
```

Builds GGML from source. Enables `ggml_q4_0`, `ggml_q8_0`, `ggml_q4_0_graph`, and `ggml_q8_0_graph` backends.

### LiteRT (TFLite)

```bash
git clone https://github.com/google-ai-edge/LiteRT.git ../third_party/litert
cmake .. -DWITH_LITERT=ON
```

Fetches FlatBuffers + TFLite deps on first build (requires network). Enables `litert_neon`, `ruy_mc`, `ruy_1c`, and `litert_4bit_neon` backends.

### MLX (Apple-only)

```bash
git clone https://github.com/ml-explore/mlx.git ../third_party/mlx
cmake .. -DWITH_MLX=ON
```

Requires macOS with Metal. Enables `mlx_q4_cpu`, `mlx_q8_cpu`, `mlx_q4_gpu`, and `mlx_q8_gpu` backends.

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

The compiled module must export `quantized_matmul_int4` and/or `quantized_matmul_int8` functions. Enables `mlc_int4` and `mlc_int8` backends.

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

Enables `onnxrt_int8` and `onnxrt_int4` backends.

### Executorch (XNNPACK)

```bash
cmake .. -DWITH_EXECUTORCH=ON
```

Fetches XNNPACK from GitHub at configure time (no manual clone needed). Enables `executorch_int8` and `executorch_int4` backends.

## Enabling Multiple Frameworks

Combine flags:

```bash
cmake .. -DWITH_GGML=ON -DWITH_LITERT=ON -DWITH_MLX=ON
make -j matmul_bench
./matmul_bench
```

## Quantization Granularity

Most backends use **group-wise** quantization (group size 32), where each group of 32 elements along K shares a single scale factor. The exceptions are noted below.

| Backend | Precision | Granularity | Notes |
|---------|-----------|-------------|-------|
| **Cactus** | INT4, INT8 | Group-wise (32) | |
| **GGML** | Q4_0, Q8_0 | Group-wise (32) | |
| **LiteRT 4bit_neon** | INT4 | Per-channel | One scale per output row |
| **LiteRT neon / Ruy** | INT8 | Group-wise (32) | |
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
```

The benchmark runs 1x1024x1024 and 1024x1024x1024 matmuls. Each timed run cycles through `--matrices` distinct weight matrices to simulate realistic cache pressure (64 matrices x ~1MB each exceeds L2 cache).

## Results (Apple M4 Pro, 14 cores)

Default settings: 100 warmup, 1024 iterations, 64 matrices.

### INT8 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT NEON | 23.6 us | 88.76 | 0.0000 |
| **Cactus INT8** | **32.5 us** | **64.51** | **0.0051** |
| Ruy | 35.3 us | 59.36 | — |
| GGML Q8_0 | 57.0 us | 36.78 | 0.0000 |
| MLX Q8 | 140.5 us | 14.92 | 0.0036 |
| ONNX Runtime INT8 | 587.0 us | 3.57 | 0.0037 |

### INT8 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q8 | 606.0 us | 3543.57 | 0.0037 |
| **Cactus INT8** | **828.2 us** | **2592.91** | **0.0055** |
| Ruy | 1348.1 us | 1593.01 | — |
| ONNX Runtime INT8 | 1616.1 us | 1328.85 | 0.0038 |
| GGML Q8_0 | 3268.6 us | 657.01 | 0.0000 |
| LiteRT NEON | 24816.6 us | 86.53 | 0.0000 |

### INT4 — 1x1024x1024 (GEMV)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| LiteRT 4bit NEON | 10.8 us | 194.06 | 0.0000 |
| **Cactus INT4** | **33.6 us** | **62.44** | **0.0691** |
| GGML Q4_0 | 55.1 us | 38.05 | 0.0000 |
| MLX Q4 | 135.0 us | 15.54 | 0.0665 |
| ONNX Runtime INT4 | 784.2 us | 2.67 | 0.0689 |

### INT4 — 1024x1024x1024 (GEMM)

| Backend | Latency | GOPS | NRMSE |
|---------|---------|------|-------|
| MLX Q4 | 594.7 us | 3611.25 | 0.0650 |
| **Cactus INT4** | **1192.2 us** | **1801.34** | **0.0683** |
| ONNX Runtime INT4 | 1743.5 us | 1231.68 | 0.0682 |
| GGML Q4_0 | 3484.9 us | 616.22 | 0.0000 |
| LiteRT 4bit NEON | 6851.5 us | 313.43 | 0.0000 |
