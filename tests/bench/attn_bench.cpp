#include "../test_utils.h"
#include "bench_common.h"
#include "bench_driver.h"

#include <iostream>

int main(int argc, char** argv) {
    TestUtils::TestRunner runner("Attention Benchmark Suite");

    bench::AttnBenchOptions opt;
    std::string err;
    if (!bench::parse_attn_bench_args(argc, argv, opt, err)) {
        std::cerr << "Error: " << err << "\n"
                  << "Usage: " << argv[0]
                  << " [--iterations N] [--warmup N]"
                  << " [--backends fw1,fw2] [--threads N|max]"
                  << " [--prefill_len N] [--cache_len N]"
                  << " [--head_dim N] [--q_heads N] [--kv_heads N]\n";
        return 1;
    }

    const auto& backends = bench::get_attn_backends();
    runner.log_performance("Backends registered", std::to_string(backends.size()));
    for (const auto& b : backends)
        runner.log_performance("  Backend", std::string(b.name) + " (" + b.framework + ")");

    runner.run_test("Attention Benchmark", bench::run_attn_benchmark(runner, opt));

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
