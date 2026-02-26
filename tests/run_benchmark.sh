#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/../third_party"
BUILD_DIR="$SCRIPT_DIR/build"

EXTERNAL=false
BENCH_MODE="matmul"
THREAD_VALUE=""
PASSTHROUGH_ARGS=()
NEXT_IS_THREADS=false

for arg in "$@"; do
    if [ "$NEXT_IS_THREADS" = true ]; then
        THREAD_VALUE="$arg"
        NEXT_IS_THREADS=false
        PASSTHROUGH_ARGS+=("$arg")
        continue
    fi
    case "$arg" in
        --external-frameworks) EXTERNAL=true ;;
        --attention) BENCH_MODE="attention" ;;
        --threads) NEXT_IS_THREADS=true; PASSTHROUGH_ARGS+=("$arg") ;;
        *) PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

if [ "$BENCH_MODE" = "attention" ]; then
    echo "=== Cactus Attention Benchmark ==="
else
    echo "=== Cactus Matmul Benchmark ==="
fi

# Build libcactus if needed
if [ ! -f "$REPO_ROOT/cactus/build/libcactus.a" ]; then
    echo "Building libcactus..."
    (cd "$REPO_ROOT" && source ./setup)
fi

mkdir -p "$BUILD_DIR"

CMAKE_FLAGS=(
    "-DWITH_GGML=OFF"
    "-DWITH_LITERT=OFF"
    "-DWITH_MLX=OFF"
    "-DWITH_MLC=OFF"
    "-DWITH_ONNXRT=OFF"
    "-DWITH_EXECUTORCH=OFF"
)

if [ "$EXTERNAL" = true ]; then
    echo "Detecting third-party frameworks in $THIRD_PARTY ..."

    if [ -d "$THIRD_PARTY/ggml/src" ]; then
        echo "  Found GGML"
        CMAKE_FLAGS+=("-DWITH_GGML=ON")
    fi

    if [ -f "$THIRD_PARTY/litert/tflite/CMakeLists.txt" ]; then
        echo "  Found LiteRT"
        CMAKE_FLAGS+=("-DWITH_LITERT=ON")
    fi

    if [ -d "$THIRD_PARTY/mlx/mlx" ]; then
        if [[ "$(uname)" == "Darwin" ]]; then
            echo "  Found MLX"
            CMAKE_FLAGS+=("-DWITH_MLX=ON")
        else
            echo "  Found MLX (skipping — requires macOS)"
        fi
    fi

    if [ -d "$THIRD_PARTY/mlc/3rdparty/tvm/build" ]; then
        echo "  Found MLC"
        CMAKE_FLAGS+=("-DWITH_MLC=ON")
    fi

    if [ -d "$THIRD_PARTY/onnxruntime/lib" ] && [ -d "$THIRD_PARTY/onnxruntime/include" ]; then
        echo "  Found ONNX Runtime"
        CMAKE_FLAGS+=("-DWITH_ONNXRT=ON")
    fi

    echo "  Enabling Executorch (XNNPACK)"
    CMAKE_FLAGS+=("-DWITH_EXECUTORCH=ON")
fi

echo "Configuring CMake..."
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" "${CMAKE_FLAGS[@]+"${CMAKE_FLAGS[@]}"}"

if [ "$BENCH_MODE" = "attention" ]; then
    BENCH_TARGET="attn_bench"
else
    BENCH_TARGET="matmul_bench"
fi

echo "Building $BENCH_TARGET..."
cmake --build "$BUILD_DIR" --target "$BENCH_TARGET" -j

echo ""
echo "=== Running Benchmark ==="

if [ -n "$THREAD_VALUE" ]; then
    if [ "$THREAD_VALUE" = "max" ]; then
        THREAD_COUNT=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 1)
    else
        THREAD_COUNT="$THREAD_VALUE"
    fi
    export TVM_NUM_THREADS="$THREAD_COUNT"
    echo "Set TVM_NUM_THREADS=$THREAD_COUNT"
fi

"$BUILD_DIR/$BENCH_TARGET" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"
