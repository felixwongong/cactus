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

struct MatmulBenchOptions {
    int warmup = 100;
    int iterations = 1024;
    int num_matrices = 64;
    int num_threads = 0;
    std::vector<size_t> batch_sizes = {1, 1024};
    std::string backends_filter;
};

enum class AttnMode { PREFILL, DECODE };

struct AttnDims {
    size_t head_dim = 128;
    size_t num_q_heads = 32;
    size_t num_kv_heads = 8;
};

struct AttnBenchOptions {
    int warmup = 100;
    int iterations = 512;
    int num_threads = 0;
    size_t prefill_seq_len = 1024;
    size_t decode_cache_len = 511;
    AttnDims dims;
    std::string backends_filter;
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

bool parse_matmul_bench_args(int argc, char** argv, MatmulBenchOptions& opt, std::string& err);

bool framework_matches_filter(const char* framework, const std::string& filter);

void reference_attention_fp32(const float* Q, const float* K, const float* V,
                               float* output,
                               size_t num_q_heads, size_t num_kv_heads,
                               size_t seq_len, size_t kv_seq_len,
                               size_t head_dim, float scale);

void fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void fp16_to_fp32(const __fp16* src, float* dst, size_t count);

void quantize_rows_int8(const float* src, int8_t* dst, float* scales,
                         size_t rows, size_t cols);

void transpose_2d(const float* src, float* dst, size_t rows, size_t cols);

void pack_int4_unsigned(const int8_t* signed_vals, uint8_t* packed, size_t count);

void set_thread_override(int n);
int get_thread_override();
int get_effective_threads(int backend_default);

} // namespace bench

#endif
