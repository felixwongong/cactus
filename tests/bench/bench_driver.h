#ifndef BENCH_DRIVER_H
#define BENCH_DRIVER_H

#include "bench_common.h"
#include "../test_utils.h"

#include <string>
#include <vector>

namespace bench {

enum class QuantCategory { INT8, INT4 };

struct MatmulBackendVariant {
    const char* name;
    const char* framework;
    QuantCategory category;
    size_t max_M = 0;

    void* (*prepare_weights)(const float* fp32, size_t N, size_t K);
    void* (*prepare_activations)(const float* fp32, size_t M, size_t K, void* weights);
    void (*run_kernel)(size_t M, void* weights, void* activations,
                       const int8_t* act_int8, const float* act_scales,
                       float* output, float* reference);
    void (*cleanup)(void* weights, void* activations);
};

void register_matmul_backend(MatmulBackendVariant v);
const std::vector<MatmulBackendVariant>& get_matmul_backends();

bool run_matmul_benchmark(TestUtils::TestRunner& runner, const MatmulBenchOptions& opt);

struct AttnBackendVariant {
    const char* name;
    const char* framework;
    AttnMode mode;

    void* (*prepare)(const AttnDims& dims, size_t seq_len, size_t cache_len,
                     const float* fp32_q, const float* fp32_k, const float* fp32_v);
    void (*run)(void* state, float* output);
    void (*cleanup)(void* state);
};

void register_attn_backend(AttnBackendVariant v);
const std::vector<AttnBackendVariant>& get_attn_backends();

bool run_attn_benchmark(TestUtils::TestRunner& runner, const AttnBenchOptions& opt);
bool parse_attn_bench_args(int argc, char** argv, AttnBenchOptions& opt, std::string& err);

} // namespace bench

#endif
