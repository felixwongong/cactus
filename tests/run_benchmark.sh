#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/../third_party"
BUILD_DIR="$SCRIPT_DIR/build"

EXTERNAL=false
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --external-frameworks) EXTERNAL=true ;;
        *) PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

echo "=== Cactus Matmul Benchmark ==="

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

    # Executorch/XNNPACK is fetched automatically, always enable when external
    echo "  Enabling Executorch (XNNPACK)"
    CMAKE_FLAGS+=("-DWITH_EXECUTORCH=ON")
fi

echo "Configuring CMake..."
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" "${CMAKE_FLAGS[@]+"${CMAKE_FLAGS[@]}"}"

echo "Building test_matmul_bench..."
cmake --build "$BUILD_DIR" --target test_matmul_bench -j

echo ""
echo "=== Running Benchmark ==="
"$BUILD_DIR/test_matmul_bench" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"
