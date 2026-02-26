#include "bench_driver.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstring>

namespace {

struct GgmlWeights {
    size_t K, N;
    ggml_type type;
    std::vector<uint8_t> quantized;
    size_t row_stride;

    size_t cached_M = 0;
    std::vector<uint8_t> q8_input;
    size_t q8_row_stride = 0;
    std::vector<float> output;
};

static void* prepare_impl(const float* fp32, size_t N, size_t K, ggml_type type) {
    auto* w = new GgmlWeights();
    w->K = K;
    w->N = N;
    w->type = type;
    w->row_stride = ggml_row_size(type, K);
    w->quantized.resize(w->row_stride * N);

    const auto* cpu_traits = ggml_get_type_traits_cpu(type);
    auto from_float = cpu_traits->from_float;
    if (!from_float) {
        const auto* traits = ggml_get_type_traits(type);
        from_float = traits->from_float_ref;
    }

    for (size_t n = 0; n < N; n++)
        from_float(fp32 + n * K, w->quantized.data() + n * w->row_stride, static_cast<int64_t>(K));

    return w;
}

void* prepare_q4(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, GGML_TYPE_Q4_0); }
void* prepare_q8(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, GGML_TYPE_Q8_0); }

static void gemv_slice(ggml_vec_dot_t vec_dot, int64_t nrows,
                        const uint8_t* wdata, size_t w_stride,
                        const uint8_t* q8_input, size_t q8_row_stride,
                        float* output, size_t M, size_t K, size_t N,
                        size_t n_begin, size_t n_end) {
    const int Kint = static_cast<int>(K);
    for (size_t m = 0; m < M; m++) {
        const uint8_t* act_row = q8_input + m * q8_row_stride;
        float* out_row = output + m * N;
        bool can_pair_m = (nrows >= 2 && m + 1 < M);

        size_t n = n_begin;
        if (can_pair_m) {
            float* out_row_next = output + (m + 1) * N;
            for (; n + 1 < n_end; n += 2) {
                float tmp[4];
                vec_dot(Kint, tmp, 2,
                        wdata + n * w_stride, w_stride,
                        act_row, q8_row_stride, 2);
                out_row[n]          = tmp[0];
                out_row[n + 1]      = tmp[1];
                out_row_next[n]     = tmp[2];
                out_row_next[n + 1] = tmp[3];
            }
            for (; n < n_end; n++) {
                vec_dot(Kint, out_row + n, 0,
                        wdata + n * w_stride, 0, act_row, 0, 1);
                vec_dot(Kint, out_row_next + n, 0,
                        wdata + n * w_stride, 0, act_row + q8_row_stride, 0, 1);
            }
            m++;
        } else {
            for (; n < n_end; n++)
                vec_dot(Kint, out_row + n, 0,
                        wdata + n * w_stride, 0, act_row, 0, 1);
        }
    }
}

static void run_gemv(GgmlWeights* w, size_t M,
                      const uint8_t* q8_input, size_t q8_row_stride,
                      float* output) {
    const auto* cpu_traits = ggml_get_type_traits_cpu(w->type);
    size_t chunk = std::max(size_t(16), w->N / std::max(size_t(1), static_cast<size_t>(std::thread::hardware_concurrency())));
    CactusThreading::ParallelConfig par_cfg{chunk, 16};
    CactusThreading::parallel_for(w->N, par_cfg,
        [&](size_t n_begin, size_t n_end) {
            gemv_slice(cpu_traits->vec_dot, cpu_traits->nrows,
                       w->quantized.data(), w->row_stride,
                       q8_input, q8_row_stride,
                       output, M, w->K, w->N, n_begin, n_end);
        });
}

static void quantize_activations(const float* fp32, size_t M, size_t K,
                                  std::vector<uint8_t>& q8_input, size_t& q8_row_stride) {
    q8_row_stride = ggml_row_size(GGML_TYPE_Q8_0, K);
    q8_input.resize(q8_row_stride * M);
    auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (size_t m = 0; m < M; m++)
        quantize(fp32 + m * K, q8_input.data() + m * q8_row_stride, static_cast<int64_t>(K));
}

static void ensure_activations(GgmlWeights* w, size_t M) {
    if (w->cached_M == M) return;
    w->cached_M = M;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_input(M * w->K);
    for (auto& v : fp32_input) v = dist(gen);
    quantize_activations(fp32_input.data(), M, w->K, w->q8_input, w->q8_row_stride);
    w->output.resize(M * w->N);
}

void run_kernel(size_t M, void* weights, void*,
                const int8_t*, const float*,
                float* output, float* reference) {
    auto* w = static_cast<GgmlWeights*>(weights);
    ensure_activations(w, M);
    run_gemv(w, M, w->q8_input.data(), w->q8_row_stride, w->output.data());

    if (output)
        std::memcpy(output, w->output.data(), M * w->N * sizeof(float));

    if (reference) {
        auto to_float_w = ggml_get_type_traits(w->type)->to_float;
        auto to_float_a = ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float;

        std::vector<float> deq_w(w->N * w->K);
        for (size_t n = 0; n < w->N; n++)
            to_float_w(w->quantized.data() + n * w->row_stride,
                        deq_w.data() + n * w->K, static_cast<int64_t>(w->K));

        std::vector<float> deq_a(M * w->K);
        for (size_t m = 0; m < M; m++)
            to_float_a(w->q8_input.data() + m * w->q8_row_stride,
                        deq_a.data() + m * w->K, static_cast<int64_t>(w->K));

        bench::reference_matmul_fp32(deq_a.data(), deq_w.data(), reference, M, w->K, w->N);
    }
}

void cleanup(void* weights, void*) {
    delete static_cast<GgmlWeights*>(weights);
}

// ---------------------------------------------------------------------------
// Graph-based variant: uses ggml_mul_mat + ggml_graph_plan + ggml_graph_compute
// Our hand-written vec_dot loop above is ~6x faster at M=1 and ~1.3x faster
// at M=1024 vs GGML's native graph path, so we keep the loop as the default.
// ---------------------------------------------------------------------------

// struct GgmlGraphWeights {
//     size_t K, N;
//     ggml_type type;
//     std::vector<uint8_t> quantized;
//     size_t row_stride;
//     int n_threads;
//
//     size_t cached_M = 0;
//     ggml_context* ctx = nullptr;
//     ggml_cgraph* graph = nullptr;
//     ggml_tensor* result = nullptr;
//     ggml_cplan plan = {};
//     std::vector<uint8_t> work_buf;
//     std::vector<float> fp32_act;
//
//     void teardown() {
//         if (ctx) { ggml_free(ctx); ctx = nullptr; }
//         graph = nullptr;
//         result = nullptr;
//         work_buf.clear();
//         plan = {};
//     }
// };
//
// static void* prepare_graph_impl(const float* fp32, size_t N, size_t K, ggml_type type) {
//     auto* w = new GgmlGraphWeights();
//     w->K = K;
//     w->N = N;
//     w->type = type;
//     w->row_stride = ggml_row_size(type, K);
//     w->quantized.resize(w->row_stride * N);
//     w->n_threads = static_cast<int>(std::thread::hardware_concurrency());
//     if (w->n_threads < 1) w->n_threads = 1;
//
//     const auto* cpu_traits = ggml_get_type_traits_cpu(type);
//     auto from_float = cpu_traits->from_float;
//     if (!from_float)
//         from_float = ggml_get_type_traits(type)->from_float_ref;
//
//     for (size_t n = 0; n < N; n++)
//         from_float(fp32 + n * K, w->quantized.data() + n * w->row_stride, static_cast<int64_t>(K));
//
//     return w;
// }
//
// void* prepare_graph_q4(const float* fp32, size_t N, size_t K) { return prepare_graph_impl(fp32, N, K, GGML_TYPE_Q4_0); }
// void* prepare_graph_q8(const float* fp32, size_t N, size_t K) { return prepare_graph_impl(fp32, N, K, GGML_TYPE_Q8_0); }
//
// static void ensure_graph(GgmlGraphWeights* w, size_t M) {
//     if (w->cached_M == M) return;
//     w->teardown();
//     w->cached_M = M;
//
//     std::mt19937 gen(42);
//     std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//     w->fp32_act.resize(M * w->K);
//     for (auto& v : w->fp32_act) v = dist(gen);
//
//     size_t mem = ggml_tensor_overhead() * 3 + ggml_graph_overhead()
//                + w->row_stride * w->N
//                + w->K * M * sizeof(float)
//                + w->N * M * sizeof(float)
//                + 1024;
//
//     ggml_init_params params = { mem, nullptr, false };
//     w->ctx = ggml_init(params);
//
//     auto* wt = ggml_new_tensor_2d(w->ctx, w->type, w->K, w->N);
//     std::memcpy(wt->data, w->quantized.data(), w->row_stride * w->N);
//
//     auto* at = ggml_new_tensor_2d(w->ctx, GGML_TYPE_F32, w->K, M);
//     std::memcpy(at->data, w->fp32_act.data(), w->K * M * sizeof(float));
//
//     w->result = ggml_mul_mat(w->ctx, wt, at);
//     w->graph = ggml_new_graph(w->ctx);
//     ggml_build_forward_expand(w->graph, w->result);
//
//     w->plan = ggml_graph_plan(w->graph, w->n_threads, nullptr);
//     if (w->plan.work_size > 0) {
//         w->work_buf.resize(w->plan.work_size);
//         w->plan.work_data = w->work_buf.data();
//     }
// }
//
// void run_kernel_graph(size_t M, void* weights, void*,
//                       const int8_t*, const float*,
//                       float* output, float* reference) {
//     auto* w = static_cast<GgmlGraphWeights*>(weights);
//     ensure_graph(w, M);
//     ggml_graph_compute(w->graph, &w->plan);
//
//     if (output)
//         std::memcpy(output, w->result->data, M * w->N * sizeof(float));
//
//     if (reference) {
//         auto to_float_w = ggml_get_type_traits(w->type)->to_float;
//
//         std::vector<float> deq_w(w->N * w->K);
//         for (size_t n = 0; n < w->N; n++)
//             to_float_w(w->quantized.data() + n * w->row_stride,
//                         deq_w.data() + n * w->K, static_cast<int64_t>(w->K));
//
//         size_t q8_stride = ggml_row_size(GGML_TYPE_Q8_0, w->K);
//         std::vector<uint8_t> q8_act(q8_stride * M);
//         auto quantize_fn = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
//         for (size_t m = 0; m < M; m++)
//             quantize_fn(w->fp32_act.data() + m * w->K,
//                         q8_act.data() + m * q8_stride, static_cast<int64_t>(w->K));
//
//         auto to_float_a = ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float;
//         std::vector<float> deq_a(M * w->K);
//         for (size_t m = 0; m < M; m++)
//             to_float_a(q8_act.data() + m * q8_stride,
//                         deq_a.data() + m * w->K, static_cast<int64_t>(w->K));
//
//         bench::reference_matmul_fp32(deq_a.data(), deq_w.data(), reference, M, w->K, w->N);
//     }
// }
//
// void cleanup_graph(void* weights, void*) {
//     auto* w = static_cast<GgmlGraphWeights*>(weights);
//     w->teardown();
//     delete w;
// }

static int reg = [] {
    bench::register_backend({
        "ggml_q4_0", "ggml", bench::QuantCategory::INT4, 0,
        prepare_q4, nullptr, run_kernel, cleanup
    });
    bench::register_backend({
        "ggml_q8_0", "ggml", bench::QuantCategory::INT8, 0,
        prepare_q8, nullptr, run_kernel, cleanup
    });
    // bench::register_backend({
    //     "ggml_q4_0_graph", "ggml", bench::QuantCategory::INT4, 0,
    //     prepare_graph_q4, nullptr, run_kernel_graph, cleanup_graph
    // });
    // bench::register_backend({
    //     "ggml_q8_0_graph", "ggml", bench::QuantCategory::INT8, 0,
    //     prepare_graph_q8, nullptr, run_kernel_graph, cleanup_graph
    // });
    return 0;
}();

} // namespace
