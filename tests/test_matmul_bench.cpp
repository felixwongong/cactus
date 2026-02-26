#include "test_utils.h"
#include "bench/bench_common.h"
#include "bench/bench_driver.h"

#include <iostream>

int main(int argc, char** argv) {
    TestUtils::TestRunner runner("Matmul Benchmark Suite");

    bench::BenchOptions opt;
    std::string err;
    if (!bench::parse_bench_args(argc, argv, opt, err)) {
        std::cerr << "Error: " << err << "\n"
                  << "Usage: " << argv[0]
                  << " [--iterations N] [--warmup N] [--matrices N]"
                  << " [--threads N] [--backends fw1,fw2]\n";
        return 1;
    }

    const auto& backends = bench::get_backends();
    runner.log_performance("Backends registered", std::to_string(backends.size()));
    for (const auto& b : backends)
        runner.log_performance("  Backend", std::string(b.name) + " (" + b.framework + ")");

    runner.run_test("Matmul Benchmark", bench::run_benchmark(runner, opt));

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
