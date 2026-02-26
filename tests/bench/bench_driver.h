#ifndef BENCH_DRIVER_H
#define BENCH_DRIVER_H

#include "bench_common.h"
#include "../test_utils.h"

#include <string>
#include <vector>

namespace bench {

enum class QuantCategory { INT8, INT4 };

struct BackendVariant {
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

void register_backend(BackendVariant v);
const std::vector<BackendVariant>& get_backends();

bool run_benchmark(TestUtils::TestRunner& runner, const BenchOptions& opt);

} // namespace bench

#endif
