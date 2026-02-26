#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../cactus/cactus.h"

namespace bench {

constexpr size_t kGroupSize = 32;
constexpr size_t kBlockSize = 4;
constexpr size_t kK = 1024;
constexpr size_t kN = 1024;

struct BenchResult {
    double avg_us = 0.0;
    double gops = 0.0;
};

struct BenchOptions {
    int warmup = 100;
    int iterations = 1024;
    int num_threads = 0;
    int num_matrices = 64;
    std::vector<size_t> batch_sizes = {1, 1024};
    std::string backends_filter;
    float* capture_output = nullptr;
    float* capture_reference = nullptr;
};

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double compute_gops(size_t M, size_t K, size_t N, int iters, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    return (2.0 * M * K * N * iters) / (total_ms * 1e6);
}

void quantize_int8_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales);

void quantize_int4_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales);

void quantize_int4_per_channel(const std::vector<float>& src, size_t N, size_t K,
                                std::vector<int8_t>& dst, std::vector<float>& scales);

std::vector<int8_t> interleave_weights_nk4(const std::vector<int8_t>& rowmajor, size_t N, size_t K);

std::vector<__fp16> interleave_scales_n4(const std::vector<float>& scales, size_t N, size_t num_groups);

std::vector<uint8_t> pack_int4_pairs(const std::vector<int8_t>& interleaved);

struct CactusActivations {
    std::vector<int8_t> int8;
    std::vector<float> scales;
    std::vector<__fp16> fp16;
    std::vector<float> fp32;
};

CactusActivations prepare_cactus_activations(size_t M, size_t K, std::mt19937& gen);

void reference_matmul_fp32(const float* A, const float* B_rowmajor_NK,
                            float* C, size_t M, size_t K, size_t N);

struct AccuracyResult {
    float max_abs_error = 0.0f;
    float nrmse = 0.0f;
    bool passed = false;
};

AccuracyResult check_accuracy(const float* reference, const float* actual,
                               size_t count, float nrmse_tolerance);

bool parse_bench_args(int argc, char** argv, BenchOptions& opt, std::string& err);

} // namespace bench

#endif
