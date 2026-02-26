#include "bench_driver.h"

#include <cstdlib>
#include <memory>

#include "tflite/kernels/internal/optimized/neon_tensor_utils.h"
#include "tflite/kernels/internal/optimized/fully_connected_4bit.h"
#include "tflite/kernels/internal/optimized/4bit/fully_connected_reference_impl.h"
#include "ruy/ruy.h"

namespace {

static constexpr int kRuyGemvThreads = 1;
static constexpr int kRuyGemmThreads = 4;

struct Int8Weights {
    size_t K, N;
    std::vector<int8_t> int8_rowmajor;
    std::vector<float> neon_output;
    std::vector<int32_t> ruy_output;
    std::unique_ptr<ruy::Context> ctx_gemv;
    std::unique_ptr<ruy::Context> ctx_gemm;
};

void* int8_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);
    auto* w = new Int8Weights();
    w->K = K;
    w->N = N;
    w->int8_rowmajor.resize(N * K);
    std::vector<float> scales;
    bench::quantize_int8_per_group(src, N, K, w->int8_rowmajor, scales);
    return w;
}

void int8_cleanup(void* weights, void*) {
    delete static_cast<Int8Weights*>(weights);
}

void neon_run_kernel(size_t M, void* weights, void*,
                     const int8_t* act_int8, const float* act_scales,
                     float* output, float* reference) {
    auto* w = static_cast<Int8Weights*>(weights);
    w->neon_output.resize(M * w->N);
    std::memset(w->neon_output.data(), 0, M * w->N * sizeof(float));
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        w->int8_rowmajor.data(), static_cast<int>(w->N), static_cast<int>(w->K),
        act_int8, act_scales, static_cast<int>(M), w->neon_output.data());

    if (output)
        std::memcpy(output, w->neon_output.data(), M * w->N * sizeof(float));

    if (reference) {
        const size_t K = w->K, N = w->N;
        for (size_t m = 0; m < M; m++)
            for (size_t n = 0; n < N; n++) {
                int32_t dot = 0;
                for (size_t k = 0; k < K; k++)
                    dot += static_cast<int32_t>(w->int8_rowmajor[n * K + k])
                         * static_cast<int32_t>(act_int8[m * K + k]);
                reference[m * N + n] = static_cast<float>(dot) * act_scales[m];
            }
    }
}

static void ruy_run_kernel_impl(size_t M, Int8Weights* w,
                                const int8_t* act_int8, ruy::Context* ctx) {
    w->ruy_output.resize(M * w->N);

    ruy::Matrix<int8_t> lhs;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->K),
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(act_int8);
    lhs.set_zero_point(0);

    ruy::Matrix<int8_t> rhs;
    ruy::MakeSimpleLayout(static_cast<int>(w->K), static_cast<int>(w->N),
                          ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(w->int8_rowmajor.data());
    rhs.set_zero_point(0);
    rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ruy::Matrix<int32_t> dst;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->N),
                          ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(w->ruy_output.data());

    ruy::MulParams<int32_t, int32_t> mul_params;
    ruy::Mul(lhs, rhs, mul_params, ctx, &dst);
}

void ruy_run_kernel(size_t M, void* weights, void*,
                    const int8_t* act_int8, const float*,
                    float*, float*) {
    auto* w = static_cast<Int8Weights*>(weights);
    if (M <= 1) {
        if (!w->ctx_gemv) {
            w->ctx_gemv = std::make_unique<ruy::Context>();
            w->ctx_gemv->set_max_num_threads(kRuyGemvThreads);
        }
        ruy_run_kernel_impl(M, w, act_int8, w->ctx_gemv.get());
    } else {
        if (!w->ctx_gemm) {
            w->ctx_gemm = std::make_unique<ruy::Context>();
            w->ctx_gemm->set_max_num_threads(kRuyGemmThreads);
        }
        ruy_run_kernel_impl(M, w, act_int8, w->ctx_gemm.get());
    }
}

struct Int4FcWeights {
    size_t K, N;
    uint8_t* prepacked = nullptr;
    uint8_t* ref_prepacked = nullptr;
    std::vector<float> filter_scales;
    int lhs_layout_rows;
    int lhs_layout_cols;
    std::vector<int32_t> dst_buf;
    std::vector<float> output_buf;

    ~Int4FcWeights() { free(prepacked); free(ref_prepacked); }
};

struct Int4FcActivations {
    std::vector<int8_t> rhs;
    std::vector<float> scales;
    std::vector<int32_t> input_offsets;
    std::vector<float> fp32;
    int rhs_width;
    int rhs_layout_rows;
    int rhs_layout_cols;
};

static std::vector<int8_t> pack_litert_source(const std::vector<int8_t>& int4_rowmajor,
                                              size_t N, size_t K) {
    std::vector<int8_t> packed(N * K / 2);
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; k += 2) {
            int8_t upper = int4_rowmajor[n * K + k];
            int8_t lower = int4_rowmajor[n * K + k + 1];
            uint8_t byte = (static_cast<uint8_t>(upper) << 4) | (static_cast<uint8_t>(lower) & 0x0F);
            packed[n * (K / 2) + k / 2] = static_cast<int8_t>(byte);
        }
    return packed;
}

void* int4_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    std::vector<int8_t> int4_rowmajor;
    std::vector<float> filter_scales;
    bench::quantize_int4_per_channel(src, N, K, int4_rowmajor, filter_scales);

    auto litert_source = pack_litert_source(int4_rowmajor, N, K);

    auto* w = new Int4FcWeights();
    w->K = K;
    w->N = N;
    w->filter_scales = filter_scales;

    w->lhs_layout_rows = (static_cast<int>(N) + tflite::optimized_4bit::FilterWidth - 1)
                         & ~(tflite::optimized_4bit::FilterWidth - 1);
    w->lhs_layout_cols = (static_cast<int>(K) + tflite::optimized_4bit::FilterDepth - 1)
                         & ~(tflite::optimized_4bit::FilterDepth - 1);

    size_t prepacked_size = static_cast<size_t>(w->lhs_layout_rows) * w->lhs_layout_cols / 2
                          + tflite::optimized_4bit::kDefaultAlignmentPadding;
    void* raw = nullptr;
    posix_memalign(&raw, 64, prepacked_size);
    w->prepacked = static_cast<uint8_t*>(raw);

    tflite::optimized_4bit::api::Prepack(
        w->prepacked, litert_source.data(),
        w->lhs_layout_rows, w->lhs_layout_cols,
        static_cast<int>(N), static_cast<int>(K),
        tflite::optimized_4bit::FilterWidth,
        tflite::optimized_4bit::FilterDepth);

    posix_memalign(reinterpret_cast<void**>(&raw), 64, prepacked_size);
    w->ref_prepacked = static_cast<uint8_t*>(raw);
    tflite::optimized_4bit::ReferencePrepack(
        w->ref_prepacked, litert_source.data(),
        w->lhs_layout_rows, w->lhs_layout_cols,
        static_cast<int>(N), static_cast<int>(K),
        tflite::optimized_4bit::FilterWidth,
        tflite::optimized_4bit::FilterDepth);

    return w;
}

void* int4_prepare_activations(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new Int4FcActivations();

    a->rhs_width = std::min(static_cast<int>(M),
                            tflite::optimized_4bit::GetMaxSupportedRows());
    a->rhs_layout_rows = (static_cast<int>(M) + a->rhs_width - 1) & ~(a->rhs_width - 1);
    a->rhs_layout_cols = (static_cast<int>(K) + tflite::optimized_4bit::FilterDepth - 1)
                        & ~(tflite::optimized_4bit::FilterDepth - 1);

    a->rhs.resize(static_cast<size_t>(a->rhs_layout_rows) * a->rhs_layout_cols, 0);
    a->scales.resize(a->rhs_layout_rows, 1.0f);
    a->input_offsets.resize(a->rhs_layout_rows, 0);
    a->fp32.assign(fp32, fp32 + M * K);

    tflite::optimized_4bit::api::BatchQuantizeFloats4Bit(
        fp32, static_cast<int>(M), static_cast<int>(K),
        a->rhs.data(), a->scales.data(),
        a->rhs_width, tflite::optimized_4bit::FilterDepth,
        a->input_offsets.data());

    return a;
}

void int4_run_kernel(size_t M, void* weights, void* activations,
                     const int8_t*, const float*,
                     float* output, float* reference) {
    auto* w = static_cast<Int4FcWeights*>(weights);
    auto* a = static_cast<Int4FcActivations*>(activations);

    const int output_depth = static_cast<int>(w->N);
    const int batch_size = static_cast<int>(M);
    const int dst_layout_rows = a->rhs_layout_rows;
    const int dst_layout_cols = w->lhs_layout_rows;

    const size_t dst_count = static_cast<size_t>(dst_layout_rows) * dst_layout_cols;
    w->dst_buf.resize(dst_count);
    std::memset(w->dst_buf.data(), 0, dst_count * sizeof(int32_t));
    w->output_buf.resize(M * w->N);
    std::memset(w->output_buf.data(), 0, M * w->N * sizeof(float));

    tflite::optimized_4bit::api::AssignBiasAndComputeOffsets(
        a->input_offsets.data(), a->scales.data(),
        w->filter_scales.data(), nullptr, w->output_buf.data(), output_depth, batch_size);
    tflite::optimized_4bit::api::RunAndUnpack(
        a->rhs_width, w->prepacked, a->rhs.data(),
        w->dst_buf.data(), output_depth, batch_size,
        w->lhs_layout_rows, w->lhs_layout_cols,
        a->rhs_layout_rows, a->rhs_layout_cols,
        dst_layout_rows, dst_layout_cols,
        w->output_buf.data(), a->scales.data(), w->filter_scales.data());

    if (output)
        std::memcpy(output, w->output_buf.data(), M * w->N * sizeof(float));

    if (reference) {
        int ref_rw = 1;
        int ref_rhs_lr = (batch_size + ref_rw - 1) & ~(ref_rw - 1);
        int ref_dst_lr = ref_rhs_lr;

        std::vector<int8_t> ref_act(static_cast<size_t>(ref_rhs_lr) * a->rhs_layout_cols, 0);
        std::vector<float> ref_scales(ref_rhs_lr, 1.0f);
        std::vector<int32_t> ref_offsets(ref_rhs_lr, 0);
        tflite::optimized_4bit::ReferenceBatchQuantizeFloats4Bit(
            a->fp32.data(), batch_size, static_cast<int>(w->K),
            ref_act.data(), ref_scales.data(), ref_rw,
            tflite::optimized_4bit::FilterDepth, ref_offsets.data());

        std::vector<int32_t> ref_dst_buf(static_cast<size_t>(ref_dst_lr) * dst_layout_cols, 0);
        std::vector<float> fs_ref = w->filter_scales;
        std::memset(reference, 0, M * w->N * sizeof(float));
        tflite::optimized_4bit::ReferenceAssignBiasAndComputeOffsets(
            ref_offsets.data(), ref_scales.data(), fs_ref.data(),
            nullptr, reference, output_depth, batch_size);
        tflite::optimized_4bit::ReferenceRunKernel<4, 1, 32>(
            w->ref_prepacked, ref_act.data(), ref_dst_buf.data(),
            w->lhs_layout_rows, w->lhs_layout_cols,
            ref_rhs_lr, a->rhs_layout_cols, ref_dst_lr, dst_layout_cols);
        tflite::optimized_4bit::ReferenceUnpack<4, 1>(
            reference, ref_dst_buf.data(), batch_size, output_depth,
            ref_scales.data(), fs_ref.data(), ref_dst_lr, dst_layout_cols);
    }
}

void int4_cleanup(void* weights, void* activations) {
    delete static_cast<Int4FcWeights*>(weights);
    delete static_cast<Int4FcActivations*>(activations);
}

static int reg = [] {
    bench::register_backend({
        "litert_neon", "litert", bench::QuantCategory::INT8, 0,
        int8_prepare, nullptr, neon_run_kernel, int8_cleanup
    });
    bench::register_backend({
        "ruy", "litert", bench::QuantCategory::INT8, 0,
        int8_prepare, nullptr, ruy_run_kernel, int8_cleanup
    });
    bench::register_backend({
        "litert_4bit_neon", "litert", bench::QuantCategory::INT4, 0,
        int4_prepare, int4_prepare_activations, int4_run_kernel, int4_cleanup
    });
    return 0;
}();

} // namespace
