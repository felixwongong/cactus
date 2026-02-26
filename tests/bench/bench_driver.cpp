#include "bench_driver.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace bench {

static std::vector<BackendVariant>& backend_registry() {
    static std::vector<BackendVariant> backends;
    return backends;
}

void register_backend(BackendVariant v) {
    backend_registry().push_back(v);
}

const std::vector<BackendVariant>& get_backends() {
    return backend_registry();
}

static bool framework_matches_filter(const char* framework, const std::string& filter) {
    if (filter.empty()) return true;
    std::string f = filter;
    std::string fw(framework);
    size_t pos = 0;
    while (pos < f.size()) {
        size_t comma = f.find(',', pos);
        if (comma == std::string::npos) comma = f.size();
        std::string token = f.substr(pos, comma - pos);
        if (token == fw) return true;
        pos = comma + 1;
    }
    return false;
}

bool run_benchmark(TestUtils::TestRunner& runner, const BenchOptions& opt_in) {
    BenchOptions opt = opt_in;
    const size_t NM = static_cast<size_t>(opt.num_matrices);
    const auto& all_backends = get_backends();

    std::vector<const BackendVariant*> active;
    for (const auto& b : all_backends) {
        if (!b.run_kernel) continue;
        if (framework_matches_filter(b.framework, opt.backends_filter))
            active.push_back(&b);
    }

    if (active.empty()) {
        runner.log_performance("Error", "No backends matched filter");
        return false;
    }

    {
        std::ostringstream cfg;
        cfg << "warmup=" << opt.warmup
            << ", iterations=" << opt.iterations
            << ", matrices=" << NM
            << ", backends=";
        for (size_t i = 0; i < active.size(); ++i) {
            if (i > 0) cfg << ",";
            cfg << active[i]->name;
        }
        runner.log_performance("Config", cfg.str());
    }

    std::mt19937 gen(270270u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t M : opt.batch_sizes) {
        runner.log_performance("",
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────");

        std::string shape = std::to_string(M) + "x" + std::to_string(kK) + "x" + std::to_string(kN);

        std::vector<std::vector<float>> fp32_weights(NM);
        for (size_t i = 0; i < NM; ++i) {
            fp32_weights[i].resize(kN * kK);
            for (auto& v : fp32_weights[i]) v = dist(gen);
        }

        std::mt19937 agen(static_cast<uint32_t>(42 + M));
        auto act = prepare_cactus_activations(M, kK, agen);

        std::ostringstream perf_line;
        std::ostringstream acc_line;
        perf_line << std::fixed;
        acc_line << std::fixed;
        bool first_perf = true;
        bool first_acc = true;

        for (size_t bi = 0; bi < active.size(); ++bi) {
            const auto* backend = active[bi];

            if (backend->max_M > 0 && M > backend->max_M) {
                if (!first_perf) perf_line << "  ";
                first_perf = false;
                perf_line << backend->name << "=skipped(M>" << backend->max_M << ")";
                continue;
            }

            struct WeightEntry {
                void* weights = nullptr;
                void* activations = nullptr;
            };
            std::vector<WeightEntry> entries(NM);

            for (size_t i = 0; i < NM; ++i) {
                entries[i].weights = backend->prepare_weights(
                    fp32_weights[i].data(), kN, kK);
                if (backend->prepare_activations)
                    entries[i].activations = backend->prepare_activations(
                        act.fp32.data(), M, kK, entries[i].weights);
            }

            {
                std::vector<float> reference(M * kN);
                reference_matmul_fp32(act.fp32.data(), fp32_weights[0].data(),
                                      reference.data(), M, kK, kN);

                const size_t out_count = M * kN;
                std::vector<float> captured(out_count, 0.0f);
                std::vector<float> backend_ref(out_count, 0.0f);

                backend->run_kernel(M, entries[0].weights, entries[0].activations,
                                    act.int8.data(), act.scales.data(),
                                    captured.data(), backend_ref.data());

                bool has_output = false;
                for (size_t i = 0; i < out_count && !has_output; i++)
                    if (captured[i] != 0.0f) has_output = true;

                if (has_output) {
                    bool has_backend_ref = false;
                    for (size_t i = 0; i < out_count && !has_backend_ref; i++)
                        if (backend_ref[i] != 0.0f) has_backend_ref = true;

                    const float* ref = has_backend_ref ? backend_ref.data() : reference.data();
                    float tol = has_backend_ref ? 0.01f
                              : (backend->category == QuantCategory::INT8) ? 0.05f : 0.20f;
                    auto acc = check_accuracy(ref, captured.data(), out_count, tol);

                    if (!first_acc) acc_line << "  ";
                    first_acc = false;
                    acc_line << backend->name << "="
                             << (acc.passed ? "PASS" : "FAIL")
                             << " nrmse=" << std::setprecision(4) << acc.nrmse
                             << " max=" << std::setprecision(2) << acc.max_abs_error;
                }
            }

            for (int w = 0; w < opt.warmup; ++w) {
                size_t idx = static_cast<size_t>(w) % NM;
                backend->run_kernel(M, entries[idx].weights, entries[idx].activations,
                                    act.int8.data(), act.scales.data(),
                                    nullptr, nullptr);
            }

            // Timed: cycle through distinct weight matrices so each call forces an L2
            // cache miss on weights, matching real inference where every layer has unique weights.
            double total_ms = 0.0;
            for (int iter = 0; iter < opt.iterations; ++iter) {
                size_t idx = static_cast<size_t>(iter) % NM;
                double t0 = now_ms();
                backend->run_kernel(M, entries[idx].weights, entries[idx].activations,
                                    act.int8.data(), act.scales.data(),
                                    nullptr, nullptr);
                total_ms += now_ms() - t0;
            }

            double avg_us = (total_ms * 1000.0) / opt.iterations;
            double gops = compute_gops(M, kK, kN, opt.iterations, total_ms);

            if (!first_perf) perf_line << "  ";
            first_perf = false;
            perf_line << backend->name << "="
                      << std::setprecision(1) << avg_us << "us"
                      << " (" << std::setprecision(2) << gops << " GOPS)";

            for (auto& e : entries) {
                if (backend->cleanup)
                    backend->cleanup(e.weights, e.activations);
            }
        }

        runner.log_performance(shape, perf_line.str());
        runner.log_performance("Accuracy M=" + std::to_string(M), acc_line.str());
    }

    return true;
}

} // namespace bench
