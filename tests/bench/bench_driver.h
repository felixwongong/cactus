#ifndef BENCH_DRIVER_H
#define BENCH_DRIVER_H

#include "bench_common.h"
#include "../test_utils.h"

#include <functional>
#include <string>
#include <vector>

namespace bench {

enum class QuantCategory { INT8, INT4 };

struct BackendVariant {
    const char* name;
    const char* framework;
    QuantCategory category;

    void* (*prepare_weights)(const float* fp32, size_t N, size_t K);
    void* (*prepare_activations)(const float* fp32, size_t M, size_t K, void* weights);
    BenchResult (*bench_fn)(size_t M, void* weights, void* activations,
                            const int8_t* act_int8, const float* act_scales,
                            const BenchOptions& opt);
    void (*cleanup)(void* weights, void* activations);

    size_t max_M = 0;

    void (*run_once)(size_t M, void* weights, void* activations,
                     const int8_t* act_int8, const float* act_scales);
};

void register_backend(BackendVariant v);
const std::vector<BackendVariant>& get_backends();

bool run_benchmark(TestUtils::TestRunner& runner, const BenchOptions& opt);

} // namespace bench

#endif
