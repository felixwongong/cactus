#include "bench_driver.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace bench {

static std::vector<MatmulBackendVariant>& backend_registry() {
    static std::vector<MatmulBackendVariant> backends;
    return backends;
}

void register_matmul_backend(MatmulBackendVariant v) {
    backend_registry().push_back(v);
}

const std::vector<MatmulBackendVariant>& get_matmul_backends() {
    return backend_registry();
}

bool run_matmul_benchmark(TestUtils::TestRunner& runner, const MatmulBenchOptions& opt_in) {
    MatmulBenchOptions opt = opt_in;
    const size_t NM = static_cast<size_t>(opt.num_matrices);
    const auto& all_backends = get_matmul_backends();

    std::vector<const MatmulBackendVariant*> active;
    for (const auto& b : all_backends) {
        if (!b.run_kernel) continue;
        if (framework_matches_filter(b.framework, opt.backends_filter))
            active.push_back(&b);
    }

    if (active.empty()) {
        runner.log_performance("Error", "No backends matched filter");
        return false;
    }

    set_thread_override(opt.num_threads);

    {
        std::ostringstream cfg;
        cfg << "warmup=" << opt.warmup
            << ", iterations=" << opt.iterations
            << ", matrices=" << NM
            << ", threads=";
        if (opt.num_threads == 0)
            cfg << "default";
        else if (opt.num_threads == static_cast<int>(std::thread::hardware_concurrency()))
            cfg << "max(" << opt.num_threads << ")";
        else
            cfg << opt.num_threads;
        cfg << ", backends=";
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
                              : (backend->category == QuantCategory::INT8) ? 0.05f : 2.0f;
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

    set_thread_override(0);
    return true;
}

static std::vector<AttnBackendVariant>& attn_backend_registry() {
    static std::vector<AttnBackendVariant> backends;
    return backends;
}

void register_attn_backend(AttnBackendVariant v) {
    attn_backend_registry().push_back(v);
}

const std::vector<AttnBackendVariant>& get_attn_backends() {
    return attn_backend_registry();
}

static double compute_attention_gflops(size_t num_q_heads, size_t seq_len,
                                        size_t kv_seq_len, size_t head_dim,
                                        int iterations, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    double flops_per_call = static_cast<double>(num_q_heads) *
        (4.0 * seq_len * kv_seq_len * head_dim + 5.0 * seq_len * kv_seq_len);
    return (flops_per_call * iterations) / (total_ms * 1e6);
}

static float attn_tolerance(const char* name) {
    std::string n(name);
    if (n.find("q4") != std::string::npos) return 0.20f;
    if (n.find("q8") != std::string::npos) return 0.10f;
    if (n.find("int8") != std::string::npos || n.find("hybrid") != std::string::npos) return 0.10f;
    return 0.05f;
}

bool run_attn_benchmark(TestUtils::TestRunner& runner, const AttnBenchOptions& opt) {
    const auto& all_backends = get_attn_backends();

    std::vector<const AttnBackendVariant*> prefill_backends;
    std::vector<const AttnBackendVariant*> decode_backends;

    for (const auto& b : all_backends) {
        if (!b.run) continue;
        if (!framework_matches_filter(b.framework, opt.backends_filter)) continue;
        if (b.mode == AttnMode::PREFILL)
            prefill_backends.push_back(&b);
        else
            decode_backends.push_back(&b);
    }

    if (prefill_backends.empty() && decode_backends.empty()) {
        runner.log_performance("Error", "No attention backends matched filter");
        return false;
    }

    set_thread_override(opt.num_threads);

    {
        std::ostringstream cfg;
        cfg << "warmup=" << opt.warmup
            << ", iterations=" << opt.iterations
            << ", threads=";
        if (opt.num_threads == 0)
            cfg << "default";
        else if (opt.num_threads == static_cast<int>(std::thread::hardware_concurrency()))
            cfg << "max(" << opt.num_threads << ")";
        else
            cfg << opt.num_threads;
        cfg << ", head_dim=" << opt.dims.head_dim
            << ", q_heads=" << opt.dims.num_q_heads
            << ", kv_heads=" << opt.dims.num_kv_heads
            << ", prefill_len=" << opt.prefill_seq_len
            << ", cache_len=" << opt.decode_cache_len;
        runner.log_performance("Config", cfg.str());

        std::ostringstream backends_str;
        bool first = true;
        for (auto* b : prefill_backends) {
            if (!first) backends_str << ",";
            first = false;
            backends_str << b->name;
        }
        for (auto* b : decode_backends) {
            if (!first) backends_str << ",";
            first = false;
            backends_str << b->name;
        }
        runner.log_performance("Backends", backends_str.str());
    }

    std::mt19937 gen(270270u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const auto& dims = opt.dims;
    float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

    if (!prefill_backends.empty()) {
        runner.log_performance("",
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────");
        runner.log_performance("PREFILL", "seq_len=" + std::to_string(opt.prefill_seq_len));

        size_t seq_len = opt.prefill_seq_len;
        size_t q_count = dims.num_q_heads * seq_len * dims.head_dim;
        size_t kv_count = dims.num_kv_heads * seq_len * dims.head_dim;

        std::vector<float> fp32_q(q_count), fp32_k(kv_count), fp32_v(kv_count);
        for (auto& v : fp32_q) v = dist(gen);
        for (auto& v : fp32_k) v = dist(gen);
        for (auto& v : fp32_v) v = dist(gen);

        std::vector<float> reference(q_count);
        reference_attention_fp32(fp32_q.data(), fp32_k.data(), fp32_v.data(),
                                  reference.data(), dims.num_q_heads, dims.num_kv_heads,
                                  seq_len, seq_len, dims.head_dim, scale);

        for (const auto* backend : prefill_backends) {
            void* state = backend->prepare(dims, seq_len, 0,
                                            fp32_q.data(), fp32_k.data(), fp32_v.data());
            if (!state) {
                runner.log_performance(backend->name, "FAILED (prepare returned null)");
                continue;
            }

            {
                std::vector<float> captured(q_count, 0.0f);
                backend->run(state, captured.data());

                auto acc = check_accuracy(reference.data(), captured.data(), q_count,
                                           attn_tolerance(backend->name));

                std::ostringstream acc_line;
                acc_line << std::fixed
                         << (acc.passed ? "PASS" : "FAIL")
                         << " nrmse=" << std::setprecision(4) << acc.nrmse
                         << " max=" << std::setprecision(2) << acc.max_abs_error;
                runner.log_performance(std::string(backend->name) + " accuracy", acc_line.str());
            }

            for (int w = 0; w < opt.warmup; ++w)
                backend->run(state, nullptr);

            double total_ms = 0.0;
            for (int iter = 0; iter < opt.iterations; ++iter) {
                double t0 = now_ms();
                backend->run(state, nullptr);
                total_ms += now_ms() - t0;
            }

            double avg_us = (total_ms * 1000.0) / opt.iterations;
            double gflops = compute_attention_gflops(dims.num_q_heads, seq_len, seq_len,
                                                      dims.head_dim, opt.iterations, total_ms);

            std::ostringstream perf_line;
            perf_line << std::fixed << std::setprecision(1) << avg_us << "us"
                      << " (" << std::setprecision(2) << gflops << " GFLOPS)";
            runner.log_performance(backend->name, perf_line.str());

            backend->cleanup(state);
        }
    }

    if (!decode_backends.empty()) {
        std::vector<size_t> cache_lens;
        if (opt.sweep)
            cache_lens = {8, 16, 32, 64, 128, 256, 512};
        else
            cache_lens = {opt.decode_cache_len};

        constexpr size_t TARGET_BYTES = 64 * 1024 * 1024;

        for (size_t cache_len : cache_lens) {
            size_t bytes_per_state = 2 * cache_len * dims.num_kv_heads * dims.head_dim
                + 2 * kv_scales_count(cache_len, dims.num_kv_heads, dims.head_dim) * sizeof(float)
                + dims.num_q_heads * dims.head_dim * sizeof(__fp16);
            size_t NS = std::max(static_cast<size_t>(4),
                        std::min(static_cast<size_t>(512),
                        TARGET_BYTES / std::max(bytes_per_state, static_cast<size_t>(1))));

            runner.log_performance("",
                "─────────────────────────────────────────────────────────────────────────────────────────────────────────");
            runner.log_performance("DECODE",
                "seq_len=1, cache_len=" + std::to_string(cache_len) + ", states=" + std::to_string(NS));

            size_t kv_seq_len = cache_len + 1;
            size_t q_count = dims.num_q_heads * 1 * dims.head_dim;
            size_t kv_count = dims.num_kv_heads * kv_seq_len * dims.head_dim;

            std::vector<std::vector<float>> all_q(NS), all_k(NS), all_v(NS);
            for (size_t si = 0; si < NS; ++si) {
                all_q[si].resize(q_count);
                all_k[si].resize(kv_count);
                all_v[si].resize(kv_count);
                for (auto& v : all_q[si]) v = dist(gen);
                for (auto& v : all_k[si]) v = dist(gen);
                for (auto& v : all_v[si]) v = dist(gen);
            }

            std::vector<float> reference(q_count);
            reference_attention_fp32(all_q[0].data(), all_k[0].data(), all_v[0].data(),
                                      reference.data(), dims.num_q_heads, dims.num_kv_heads,
                                      1, kv_seq_len, dims.head_dim, scale);

            for (const auto* backend : decode_backends) {
                std::vector<void*> states(NS);
                for (size_t si = 0; si < NS; ++si) {
                    states[si] = backend->prepare(dims, 1, cache_len,
                                                   all_q[si].data(), all_k[si].data(), all_v[si].data());
                }

                if (!states[0]) {
                    runner.log_performance(backend->name, "FAILED (prepare returned null)");
                    continue;
                }

                {
                    std::vector<float> captured(q_count, 0.0f);
                    backend->run(states[0], captured.data());

                    auto acc = check_accuracy(reference.data(), captured.data(), q_count,
                                               attn_tolerance(backend->name));

                    std::ostringstream acc_line;
                    acc_line << std::fixed
                             << (acc.passed ? "PASS" : "FAIL")
                             << " nrmse=" << std::setprecision(4) << acc.nrmse
                             << " max=" << std::setprecision(2) << acc.max_abs_error;
                    runner.log_performance(std::string(backend->name) + " accuracy", acc_line.str());
                }

                for (int w = 0; w < opt.warmup; ++w) {
                    size_t si = static_cast<size_t>(w) % NS;
                    backend->run(states[si], nullptr);
                }

                double total_ms = 0.0;
                for (int iter = 0; iter < opt.iterations; ++iter) {
                    size_t si = static_cast<size_t>(iter) % NS;
                    double t0 = now_ms();
                    backend->run(states[si], nullptr);
                    total_ms += now_ms() - t0;
                }

                double avg_us = (total_ms * 1000.0) / opt.iterations;
                double gflops = compute_attention_gflops(dims.num_q_heads, 1, kv_seq_len,
                                                          dims.head_dim, opt.iterations, total_ms);

                std::ostringstream perf_line;
                perf_line << std::fixed << std::setprecision(1) << avg_us << "us"
                          << " (" << std::setprecision(2) << gflops << " GFLOPS)";
                runner.log_performance(backend->name, perf_line.str());

                for (auto* s : states) backend->cleanup(s);
            }
        }
    }

    set_thread_override(0);
    return true;
}

bool parse_attn_bench_args(int argc, char** argv, AttnBenchOptions& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--iterations") {
            if (++i >= argc) { err = "Missing --iterations value"; return false; }
            opt.iterations = std::max(1, std::stoi(argv[i]));
        } else if (a == "--warmup") {
            if (++i >= argc) { err = "Missing --warmup value"; return false; }
            opt.warmup = std::max(0, std::stoi(argv[i]));
        } else if (a == "--backends") {
            if (++i >= argc) { err = "Missing --backends value"; return false; }
            opt.backends_filter = argv[i];
        } else if (a == "--prefill_len") {
            if (++i >= argc) { err = "Missing --prefill_len value"; return false; }
            opt.prefill_seq_len = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--cache_len") {
            if (++i >= argc) { err = "Missing --cache_len value"; return false; }
            opt.decode_cache_len = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--head_dim") {
            if (++i >= argc) { err = "Missing --head_dim value"; return false; }
            opt.dims.head_dim = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--q_heads") {
            if (++i >= argc) { err = "Missing --q_heads value"; return false; }
            opt.dims.num_q_heads = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--kv_heads") {
            if (++i >= argc) { err = "Missing --kv_heads value"; return false; }
            opt.dims.num_kv_heads = static_cast<size_t>(std::max(1, std::stoi(argv[i])));
        } else if (a == "--sweep") {
            opt.sweep = true;
        } else if (a == "--threads") {
            if (++i >= argc) { err = "Missing --threads value"; return false; }
            std::string val(argv[i]);
            if (val == "max")
                opt.num_threads = static_cast<int>(std::thread::hardware_concurrency());
            else
                opt.num_threads = std::max(1, std::stoi(val));
        } else {
            err = "Unknown argument: " + a;
            return false;
        }
    }
    return true;
}

} // namespace bench
