#include "bench_driver.h"

#ifndef WITH_MLX

namespace {
[[maybe_unused]] static int reg = [] {
    return 0;
}();
} // namespace

#else

#include <mlx/mlx.h>
#include <cstring>

namespace mx = mlx::core;

namespace {

struct MLXWeights {
    size_t K, N;
    int bits;
    mx::Device device;
    mx::array w_q;
    mx::array scales;
    mx::array biases;
};

struct MLXActivations {
    mx::array x;
};

void* prepare_impl(const float* fp32, size_t N, size_t K, int bits, mx::Device device) {
    auto w = mx::array(fp32, {static_cast<int>(N), static_cast<int>(K)}, mx::float32);
    auto parts = mx::quantize(w, static_cast<int>(bench::kGroupSize), bits, "affine", device);
    mx::eval(parts);
    return new MLXWeights{K, N, bits, device,
        std::move(parts[0]), std::move(parts[1]), std::move(parts[2])};
}

void* prepare_q4(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, 4, mx::Device::gpu); }
void* prepare_q8(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, 8, mx::Device::gpu); }

void* prepare_act(const float* fp32, size_t M, size_t K, void* weights) {
    auto* w = static_cast<MLXWeights*>(weights);
    auto x = mx::astype(
        mx::array(fp32, {static_cast<int>(M), static_cast<int>(K)}, mx::float32),
        mx::float16, w->device);
    mx::eval(x);
    return new MLXActivations{std::move(x)};
}

void run_kernel(size_t M, void* weights, void* activations,
                const int8_t*, const float*,
                float* output, float*) {
    auto* w = static_cast<MLXWeights*>(weights);
    auto* a = static_cast<MLXActivations*>(activations);
    auto y = mx::quantized_matmul(a->x, w->w_q, w->scales, w->biases,
                                   true, static_cast<int>(bench::kGroupSize), w->bits,
                                   "affine", w->device);
    if (output) {
        y = mx::astype(y, mx::float32, w->device);
        mx::eval(y);
        std::memcpy(output, y.data<float>(), M * w->N * sizeof(float));
    } else {
        mx::eval(y);
    }
}

void cleanup(void* weights, void* activations) {
    delete static_cast<MLXWeights*>(weights);
    if (activations) delete static_cast<MLXActivations*>(activations);
}

static int reg = [] {
    bench::register_backend({
        "mlx_q4", "mlx", bench::QuantCategory::INT4, 0,
        prepare_q4, prepare_act, run_kernel, cleanup
    });
    bench::register_backend({
        "mlx_q8", "mlx", bench::QuantCategory::INT8, 0,
        prepare_q8, prepare_act, run_kernel, cleanup
    });
    return 0;
}();

} // namespace

#endif
