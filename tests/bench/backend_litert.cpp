#include "bench_driver.h"

#include <arm_neon.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "tflite/kernels/internal/optimized/neon_tensor_utils.h"
#include "tflite/kernels/internal/optimized/fully_connected_4bit.h"
#include "ruy/ruy.h"

namespace {

static constexpr int kRuyGemvThreads = 1;
static constexpr int kRuyGemmThreads = 4;

struct Int8Weights {
    size_t K, N;
    std::vector<int8_t> int8_rowmajor;
    std::vector<float> weight_scales;
    std::vector<float> neon_output;
    std::vector<int32_t> ruy_output;
    std::unique_ptr<ruy::Context> ctx_gemv;
    std::unique_ptr<ruy::Context> ctx_gemm;
};

void* int8_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new Int8Weights();
    w->K = K;
    w->N = N;
    w->int8_rowmajor.resize(N * K);
    w->weight_scales.resize(N);
    for (size_t n = 0; n < N; ++n) {
        float max_abs = 0.0f;
        for (size_t k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::abs(fp32[n * K + k]));
        float scale = std::max(max_abs / 127.0f, 1e-10f);
        w->weight_scales[n] = scale;
        for (size_t k = 0; k < K; ++k) {
            int q = static_cast<int>(std::round(fp32[n * K + k] / scale));
            w->int8_rowmajor[n * K + k] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
        }
    }
    return w;
}

void int8_cleanup(void* weights, void*) {
    delete static_cast<Int8Weights*>(weights);
}

void neon_run_kernel(size_t M, void* weights, void*,
                     const int8_t* act_int8, const float* act_scales,
                     float* output, float*) {
    auto* w = static_cast<Int8Weights*>(weights);
    w->neon_output.resize(M * w->N);
    std::memset(w->neon_output.data(), 0, M * w->N * sizeof(float));
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        w->int8_rowmajor.data(), static_cast<int>(w->N), static_cast<int>(w->K),
        act_int8, act_scales, static_cast<int>(M), w->neon_output.data());

    const float* ws = w->weight_scales.data();
    for (size_t m = 0; m < M; m++) {
        float* row = w->neon_output.data() + m * w->N;
        size_t n = 0;
        for (; n + 4 <= w->N; n += 4) {
            float32x4_t v = vld1q_f32(row + n);
            float32x4_t s = vld1q_f32(ws + n);
            vst1q_f32(row + n, vmulq_f32(v, s));
        }
        for (; n < w->N; n++)
            row[n] *= ws[n];
    }

    if (output)
        std::memcpy(output, w->neon_output.data(), M * w->N * sizeof(float));
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
                    const int8_t* act_int8, const float* act_scales,
                    float* output, float*) {
    auto* w = static_cast<Int8Weights*>(weights);
    if (M <= 1) {
        int threads = bench::get_effective_threads(kRuyGemvThreads);
        if (!w->ctx_gemv) {
            w->ctx_gemv = std::make_unique<ruy::Context>();
            w->ctx_gemv->set_max_num_threads(threads);
        }
        ruy_run_kernel_impl(M, w, act_int8, w->ctx_gemv.get());
    } else {
        int threads = bench::get_effective_threads(kRuyGemmThreads);
        if (!w->ctx_gemm) {
            w->ctx_gemm = std::make_unique<ruy::Context>();
            w->ctx_gemm->set_max_num_threads(threads);
        }
        ruy_run_kernel_impl(M, w, act_int8, w->ctx_gemm.get());
    }

    w->neon_output.resize(M * w->N);
    const float* ws = w->weight_scales.data();
    for (size_t m = 0; m < M; m++) {
        float32x4_t as = vdupq_n_f32(act_scales[m]);
        const int32_t* src = w->ruy_output.data() + m * w->N;
        float* dst = w->neon_output.data() + m * w->N;
        size_t n = 0;
        for (; n + 4 <= w->N; n += 4) {
            float32x4_t v = vcvtq_f32_s32(vld1q_s32(src + n));
            float32x4_t s = vld1q_f32(ws + n);
            vst1q_f32(dst + n, vmulq_f32(vmulq_f32(v, as), s));
        }
        for (; n < w->N; n++)
            dst[n] = static_cast<float>(src[n]) * act_scales[m] * ws[n];
    }

    if (output)
        std::memcpy(output, w->neon_output.data(), M * w->N * sizeof(float));
}

struct Int4FcWeights {
    size_t K, N;
    uint8_t* prepacked = nullptr;
    std::vector<float> filter_scales;
    int lhs_layout_rows;
    int lhs_layout_cols;
    std::vector<int32_t> dst_buf;
    std::vector<float> output_buf;

    ~Int4FcWeights() { free(prepacked); }
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
}

void int4_cleanup(void* weights, void* activations) {
    delete static_cast<Int4FcWeights*>(weights);
    delete static_cast<Int4FcActivations*>(activations);
}

// ── Attention ───────────────────────────────────────────────────────────────────
//
// Composes attention from LiteRT's best matmul primitives:
//   INT8: Ruy GEMM/GEMV or Neon MatrixBatchVectorMultiplyAccumulate
//   INT4: optimized_4bit FC kernel (NEON SDOT)
// No fused attention op exists in LiteRT, so this mirrors the matmul-composed
// strategy used by GGML's mm_q8/mm_q4 variants.

namespace attn {

// ── INT8 helpers ────────────────────────────────────────────────────────────────

static void quantize_rows_int8(const float* fp32, int8_t* dst, float* scales,
                                size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row = fp32 + r * cols;
        int8_t* out = dst + r * cols;

        float32x4_t vmax = vdupq_n_f32(0.0f);
        size_t c = 0;
        for (; c + 4 <= cols; c += 4)
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(row + c)));
        float max_abs = vmaxvq_f32(vmax);
        for (; c < cols; ++c)
            max_abs = std::max(max_abs, std::abs(row[c]));

        float scale = std::max(max_abs / 127.0f, 1e-10f);
        scales[r] = scale;
        float inv_scale = 1.0f / scale;
        float32x4_t vinv = vdupq_n_f32(inv_scale);

        c = 0;
        for (; c + 8 <= cols; c += 8) {
            float32x4_t f0 = vmulq_f32(vld1q_f32(row + c), vinv);
            float32x4_t f1 = vmulq_f32(vld1q_f32(row + c + 4), vinv);
            int32x4_t i0 = vcvtnq_s32_f32(f0);
            int32x4_t i1 = vcvtnq_s32_f32(f1);
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int8x8_t b = vqmovn_s16(vcombine_s16(s0, s1));
            vst1_s8(out + c, b);
        }
        for (; c < cols; ++c) {
            int q = static_cast<int>(std::round(row[c] * inv_scale));
            out[c] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
        }
    }
}

static void ruy_matmul_int32(const int8_t* A, size_t M, size_t K,
                              const int8_t* B_rowmajor, size_t N,
                              int32_t* dst, ruy::Context* ctx,
                              bool cache_rhs = false) {
    ruy::Matrix<int8_t> lhs;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(K),
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(A);
    lhs.set_zero_point(0);

    ruy::Matrix<int8_t> rhs;
    ruy::MakeSimpleLayout(static_cast<int>(K), static_cast<int>(N),
                          ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(B_rowmajor);
    rhs.set_zero_point(0);
    if (cache_rhs)
        rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ruy::Matrix<int32_t> out;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(N),
                          ruy::Order::kRowMajor, out.mutable_layout());
    out.set_data(dst);

    ruy::MulParams<int32_t, int32_t> mul_params;
    ruy::Mul(lhs, rhs, mul_params, ctx, &out);
}

static void dequant_and_scale(const int32_t* int32_out,
                               const float* row_scales, const float* col_scales,
                               float extra_scale, float* dst,
                               size_t M, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        float rs = row_scales[m] * extra_scale;
        float32x4_t rs_v = vdupq_n_f32(rs);
        const int32_t* src = int32_out + m * N;
        float* out = dst + m * N;
        size_t n = 0;
        for (; n + 4 <= N; n += 4) {
            float32x4_t v = vcvtq_f32_s32(vld1q_s32(src + n));
            float32x4_t cs = vld1q_f32(col_scales + n);
            vst1q_f32(out + n, vmulq_f32(vmulq_f32(v, rs_v), cs));
        }
        for (; n < N; n++)
            out[n] = static_cast<float>(src[n]) * rs * col_scales[n];
    }
}

static void neon_mbvma_dequant(const int8_t* weight, size_t N, size_t K,
                                const float* weight_scales,
                                const int8_t* act_int8, const float* act_scales,
                                float* dst) {
    std::memset(dst, 0, N * sizeof(float));
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weight, static_cast<int>(N), static_cast<int>(K),
        act_int8, act_scales, 1, dst);
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
        float32x4_t v = vld1q_f32(dst + n);
        float32x4_t s = vld1q_f32(weight_scales + n);
        vst1q_f32(dst + n, vmulq_f32(v, s));
    }
    for (; n < N; n++)
        dst[n] *= weight_scales[n];
}

static void softmax_causal(float* scores, size_t seq_len, size_t kv_seq_len) {
    size_t offset = kv_seq_len - seq_len;
    for (size_t sq = 0; sq < seq_len; ++sq) {
        float* row = scores + sq * kv_seq_len;

        if (seq_len > 1)
            for (size_t sk = sq + offset + 1; sk < kv_seq_len; ++sk)
                row[sk] = -1e30f;

        float max_val = row[0];
        for (size_t sk = 1; sk < kv_seq_len; ++sk)
            if (row[sk] > max_val) max_val = row[sk];

        float sum = 0.0f;
        for (size_t sk = 0; sk < kv_seq_len; ++sk) {
            row[sk] = std::exp(row[sk] - max_val);
            sum += row[sk];
        }
        float inv_sum = 1.0f / sum;
        for (size_t sk = 0; sk < kv_seq_len; ++sk)
            row[sk] *= inv_sum;
    }
}

// ── INT8 attention state ────────────────────────────────────────────────────────

struct Int8KVHead {
    std::vector<int8_t> k_int8;       // [kv_seq_len, head_dim] row-major
    std::vector<float> k_scales;      // [kv_seq_len] per-row
    std::vector<int8_t> vt_int8;      // [head_dim, kv_seq_len] row-major (V transposed)
    std::vector<float> vt_scales;     // [head_dim] per-row
};

struct Int8State {
    bench::AttnDims dims;
    size_t seq_len, kv_seq_len;
    float scale;
    std::vector<Int8KVHead> kv_heads;
    std::vector<float> fp32_q;
    ruy::Context ruy_ctx;
    std::vector<int8_t> q_int8_buf;
    std::vector<float> q_scales_buf;
    std::vector<int32_t> int32_buf;
    std::vector<float> scores_buf;
    std::vector<int8_t> scores_int8_buf;
    std::vector<float> scores_scales_buf;
    std::vector<float> sv_buf;
};

static void* int8_attn_prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                                const float* fp32_q, const float* fp32_k, const float* fp32_v,
                                bench::AttnMode mode) {
    auto* s = new Int8State();
    s->dims = dims;
    s->seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : 1;
    s->kv_seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

    size_t sl = s->seq_len, kvl = s->kv_seq_len, hd = dims.head_dim;

    s->fp32_q.assign(fp32_q, fp32_q + dims.num_q_heads * sl * hd);

    s->kv_heads.resize(dims.num_kv_heads);
    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        auto& head = s->kv_heads[h];
        const float* k_head = fp32_k + h * kvl * hd;
        const float* v_head = fp32_v + h * kvl * hd;

        head.k_int8.resize(kvl * hd);
        head.k_scales.resize(kvl);
        quantize_rows_int8(k_head, head.k_int8.data(), head.k_scales.data(), kvl, hd);

        std::vector<float> vt(hd * kvl);
        for (size_t sk = 0; sk < kvl; ++sk)
            for (size_t d = 0; d < hd; ++d)
                vt[d * kvl + sk] = v_head[sk * hd + d];
        head.vt_int8.resize(hd * kvl);
        head.vt_scales.resize(hd);
        quantize_rows_int8(vt.data(), head.vt_int8.data(), head.vt_scales.data(), hd, kvl);
    }

    s->q_int8_buf.resize(sl * hd);
    s->q_scales_buf.resize(sl);
    s->int32_buf.resize(std::max(sl * kvl, sl * hd));
    s->scores_buf.resize(sl * kvl);
    s->scores_int8_buf.resize(sl * kvl);
    s->scores_scales_buf.resize(sl);
    s->sv_buf.resize(sl * hd);

    int threads = bench::get_effective_threads(sl > 1 ? kRuyGemmThreads : kRuyGemvThreads);
    s->ruy_ctx.set_max_num_threads(threads);

    return s;
}

static void int8_attn_run_ruy(void* state, float* output) {
    auto* s = static_cast<Int8State*>(state);
    size_t sl = s->seq_len, kvl = s->kv_seq_len, hd = s->dims.head_dim;
    size_t gqa_ratio = s->dims.num_q_heads / s->dims.num_kv_heads;
    bool causal = (sl > 1);

    for (size_t qh = 0; qh < s->dims.num_q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;
        const float* q_head = s->fp32_q.data() + qh * sl * hd;
        auto& kv = s->kv_heads[kvh];

        quantize_rows_int8(q_head, s->q_int8_buf.data(), s->q_scales_buf.data(), sl, hd);

        ruy_matmul_int32(s->q_int8_buf.data(), sl, hd,
                         kv.k_int8.data(), kvl,
                         s->int32_buf.data(), &s->ruy_ctx, true);
        dequant_and_scale(s->int32_buf.data(), s->q_scales_buf.data(), kv.k_scales.data(),
                          s->scale, s->scores_buf.data(), sl, kvl);

        softmax_causal(s->scores_buf.data(), sl, kvl);

        quantize_rows_int8(s->scores_buf.data(), s->scores_int8_buf.data(),
                           s->scores_scales_buf.data(), sl, kvl);

        ruy_matmul_int32(s->scores_int8_buf.data(), sl, kvl,
                         kv.vt_int8.data(), hd,
                         s->int32_buf.data(), &s->ruy_ctx, true);
        dequant_and_scale(s->int32_buf.data(), s->scores_scales_buf.data(), kv.vt_scales.data(),
                          1.0f, s->sv_buf.data(), sl, hd);

        if (output)
            std::memcpy(output + qh * sl * hd, s->sv_buf.data(), sl * hd * sizeof(float));
    }
}

static void int8_attn_run_neon(void* state, float* output) {
    auto* s = static_cast<Int8State*>(state);
    size_t kvl = s->kv_seq_len, hd = s->dims.head_dim;
    size_t gqa_ratio = s->dims.num_q_heads / s->dims.num_kv_heads;

    for (size_t qh = 0; qh < s->dims.num_q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;
        const float* q_head = s->fp32_q.data() + qh * hd;
        auto& kv = s->kv_heads[kvh];

        quantize_rows_int8(q_head, s->q_int8_buf.data(), s->q_scales_buf.data(), 1, hd);
        s->q_scales_buf[0] *= s->scale;

        neon_mbvma_dequant(kv.k_int8.data(), kvl, hd, kv.k_scales.data(),
                           s->q_int8_buf.data(), s->q_scales_buf.data(),
                           s->scores_buf.data());

        softmax_causal(s->scores_buf.data(), 1, kvl);

        quantize_rows_int8(s->scores_buf.data(), s->scores_int8_buf.data(),
                           s->scores_scales_buf.data(), 1, kvl);

        neon_mbvma_dequant(kv.vt_int8.data(), hd, kvl, kv.vt_scales.data(),
                           s->scores_int8_buf.data(), s->scores_scales_buf.data(),
                           s->sv_buf.data());

        if (output)
            std::memcpy(output + qh * hd, s->sv_buf.data(), hd * sizeof(float));
    }
}

static void int8_attn_cleanup(void* state) { delete static_cast<Int8State*>(state); }

// ── INT4 attention state ────────────────────────────────────────────────────────

struct Int4KVHead {
    uint8_t* k_prepacked = nullptr;
    std::vector<float> k_filter_scales;
    int k_lhs_layout_rows, k_lhs_layout_cols;

    // V stored as INT8 (better accuracy for score×V than INT4)
    std::vector<int8_t> vt_int8;      // [head_dim, kv_seq_len] row-major
    std::vector<float> vt_scales;     // [head_dim] per-row

    ~Int4KVHead() { free(k_prepacked); }
};

struct Int4MatmulBufs {
    std::vector<int8_t> rhs;
    std::vector<float> scales;
    std::vector<int32_t> input_offsets;
    std::vector<int32_t> dst_buf;
};

struct Int4State {
    bench::AttnDims dims;
    size_t seq_len, kv_seq_len;
    float scale;
    std::vector<std::unique_ptr<Int4KVHead>> kv_heads;
    std::vector<float> fp32_q;
    ruy::Context ruy_ctx;
    std::vector<float> scores_buf;
    std::vector<int8_t> scores_int8_buf;
    std::vector<float> scores_scales_buf;
    std::vector<int32_t> int32_buf;
    std::vector<float> sv_buf;
    Int4MatmulBufs qk_bufs;
};

static void int4_prepack_matrix(const float* fp32, size_t N, size_t K,
                                 uint8_t*& prepacked, std::vector<float>& filter_scales,
                                 int& lhs_layout_rows, int& lhs_layout_cols) {
    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> int4_rowmajor;
    bench::quantize_int4_per_channel(src, N, K, int4_rowmajor, filter_scales);
    auto litert_source = pack_litert_source(int4_rowmajor, N, K);

    lhs_layout_rows = (static_cast<int>(N) + tflite::optimized_4bit::FilterWidth - 1)
                      & ~(tflite::optimized_4bit::FilterWidth - 1);
    lhs_layout_cols = (static_cast<int>(K) + tflite::optimized_4bit::FilterDepth - 1)
                      & ~(tflite::optimized_4bit::FilterDepth - 1);

    size_t prepacked_size = static_cast<size_t>(lhs_layout_rows) * lhs_layout_cols / 2
                           + tflite::optimized_4bit::kDefaultAlignmentPadding;
    void* raw = nullptr;
    posix_memalign(&raw, 64, prepacked_size);
    prepacked = static_cast<uint8_t*>(raw);

    tflite::optimized_4bit::api::Prepack(
        prepacked, litert_source.data(),
        lhs_layout_rows, lhs_layout_cols,
        static_cast<int>(N), static_cast<int>(K),
        tflite::optimized_4bit::FilterWidth,
        tflite::optimized_4bit::FilterDepth);
}

static void int4_matmul(uint8_t* prepacked, int lhs_layout_rows, int lhs_layout_cols,
                         float* filter_scales, size_t N,
                         const float* act_fp32, size_t M, size_t K,
                         float* output, Int4MatmulBufs* bufs = nullptr) {
    int rhs_width = std::min(static_cast<int>(M),
                             tflite::optimized_4bit::GetMaxSupportedRows());
    int rhs_layout_rows = (static_cast<int>(M) + rhs_width - 1) & ~(rhs_width - 1);
    int rhs_layout_cols = (static_cast<int>(K) + tflite::optimized_4bit::FilterDepth - 1)
                         & ~(tflite::optimized_4bit::FilterDepth - 1);
    int dst_layout_rows = rhs_layout_rows;
    int dst_layout_cols = lhs_layout_rows;

    Int4MatmulBufs local;
    if (!bufs) bufs = &local;

    size_t rhs_size = static_cast<size_t>(rhs_layout_rows) * rhs_layout_cols;
    bufs->rhs.resize(rhs_size);
    std::memset(bufs->rhs.data(), 0, rhs_size);
    bufs->scales.assign(rhs_layout_rows, 1.0f);
    bufs->input_offsets.assign(rhs_layout_rows, 0);

    tflite::optimized_4bit::api::BatchQuantizeFloats4Bit(
        act_fp32, static_cast<int>(M), static_cast<int>(K),
        bufs->rhs.data(), bufs->scales.data(), rhs_width,
        tflite::optimized_4bit::FilterDepth, bufs->input_offsets.data());

    size_t dst_count = static_cast<size_t>(dst_layout_rows) * dst_layout_cols;
    bufs->dst_buf.resize(dst_count);
    std::memset(bufs->dst_buf.data(), 0, dst_count * sizeof(int32_t));
    std::memset(output, 0, M * N * sizeof(float));

    tflite::optimized_4bit::api::AssignBiasAndComputeOffsets(
        bufs->input_offsets.data(), bufs->scales.data(),
        filter_scales, nullptr, output,
        static_cast<int>(N), static_cast<int>(M));
    tflite::optimized_4bit::api::RunAndUnpack(
        rhs_width, prepacked, bufs->rhs.data(),
        bufs->dst_buf.data(), static_cast<int>(N), static_cast<int>(M),
        lhs_layout_rows, lhs_layout_cols,
        rhs_layout_rows, rhs_layout_cols,
        dst_layout_rows, dst_layout_cols,
        output, bufs->scales.data(), filter_scales);
}

static void* int4_attn_prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                                const float* fp32_q, const float* fp32_k, const float* fp32_v,
                                bench::AttnMode mode) {
    auto* s = new Int4State();
    s->dims = dims;
    s->seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : 1;
    s->kv_seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    s->scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

    size_t sl = s->seq_len, kvl = s->kv_seq_len, hd = dims.head_dim;

    s->fp32_q.assign(fp32_q, fp32_q + dims.num_q_heads * sl * hd);

    s->kv_heads.resize(dims.num_kv_heads);
    for (size_t h = 0; h < dims.num_kv_heads; ++h) {
        auto head = std::make_unique<Int4KVHead>();
        const float* k_head = fp32_k + h * kvl * hd;
        const float* v_head = fp32_v + h * kvl * hd;

        int4_prepack_matrix(k_head, kvl, hd,
                            head->k_prepacked, head->k_filter_scales,
                            head->k_lhs_layout_rows, head->k_lhs_layout_cols);

        std::vector<float> vt(hd * kvl);
        for (size_t sk = 0; sk < kvl; ++sk)
            for (size_t d = 0; d < hd; ++d)
                vt[d * kvl + sk] = v_head[sk * hd + d];
        head->vt_int8.resize(hd * kvl);
        head->vt_scales.resize(hd);
        quantize_rows_int8(vt.data(), head->vt_int8.data(), head->vt_scales.data(), hd, kvl);

        s->kv_heads[h] = std::move(head);
    }

    s->scores_buf.resize(sl * kvl);
    s->scores_int8_buf.resize(sl * kvl);
    s->scores_scales_buf.resize(sl);
    s->int32_buf.resize(sl * hd);
    s->sv_buf.resize(sl * hd);

    int threads = bench::get_effective_threads(sl > 1 ? kRuyGemmThreads : kRuyGemvThreads);
    s->ruy_ctx.set_max_num_threads(threads);

    return s;
}

static void int4_attn_run(void* state, float* output) {
    auto* s = static_cast<Int4State*>(state);
    size_t sl = s->seq_len, kvl = s->kv_seq_len, hd = s->dims.head_dim;
    size_t gqa_ratio = s->dims.num_q_heads / s->dims.num_kv_heads;
    bool causal = (sl > 1);

    for (size_t qh = 0; qh < s->dims.num_q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;
        const float* q_head = s->fp32_q.data() + qh * sl * hd;
        auto& kv = *s->kv_heads[kvh];

        int4_matmul(kv.k_prepacked, kv.k_lhs_layout_rows, kv.k_lhs_layout_cols,
                    kv.k_filter_scales.data(), kvl,
                    q_head, sl, hd,
                    s->scores_buf.data(), &s->qk_bufs);

        // INT4 kernel output already in fp32; apply attn_scale inline
        {
            size_t count = sl * kvl;
            float32x4_t vs = vdupq_n_f32(s->scale);
            size_t i = 0;
            for (; i + 4 <= count; i += 4)
                vst1q_f32(s->scores_buf.data() + i,
                          vmulq_f32(vld1q_f32(s->scores_buf.data() + i), vs));
            for (; i < count; ++i)
                s->scores_buf[i] *= s->scale;
        }

        softmax_causal(s->scores_buf.data(), sl, kvl);

        quantize_rows_int8(s->scores_buf.data(), s->scores_int8_buf.data(),
                           s->scores_scales_buf.data(), sl, kvl);
        ruy_matmul_int32(s->scores_int8_buf.data(), sl, kvl,
                         kv.vt_int8.data(), hd,
                         s->int32_buf.data(), &s->ruy_ctx, true);
        dequant_and_scale(s->int32_buf.data(), s->scores_scales_buf.data(), kv.vt_scales.data(),
                          1.0f, s->sv_buf.data(), sl, hd);

        if (output)
            std::memcpy(output + qh * sl * hd, s->sv_buf.data(), sl * hd * sizeof(float));
    }
}

static void int4_attn_cleanup(void* state) { delete static_cast<Int4State*>(state); }

// ── Prepare wrappers ────────────────────────────────────────────────────────────

void* ruy_int8_prefill(const bench::AttnDims& d, size_t sl, size_t cl,
                       const float* q, const float* k, const float* v) {
    return int8_attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::PREFILL);
}
void* ruy_int8_decode(const bench::AttnDims& d, size_t sl, size_t cl,
                      const float* q, const float* k, const float* v) {
    return int8_attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
}
void* neon_int8_decode(const bench::AttnDims& d, size_t sl, size_t cl,
                       const float* q, const float* k, const float* v) {
    return int8_attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
}
void* q4_prefill(const bench::AttnDims& d, size_t sl, size_t cl,
                 const float* q, const float* k, const float* v) {
    return int4_attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::PREFILL);
}
void* q4_decode(const bench::AttnDims& d, size_t sl, size_t cl,
                const float* q, const float* k, const float* v) {
    return int4_attn_prepare(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
}

} // namespace attn

static int reg = [] {
    bench::register_matmul_backend({
        "litert_neon", "litert", bench::QuantCategory::INT8, 0,
        int8_prepare, nullptr, neon_run_kernel, int8_cleanup
    });
    bench::register_matmul_backend({
        "ruy", "litert", bench::QuantCategory::INT8, 0,
        int8_prepare, nullptr, ruy_run_kernel, int8_cleanup
    });
    bench::register_matmul_backend({
        "litert_4bit_neon", "litert", bench::QuantCategory::INT4, 0,
        int4_prepare, int4_prepare_activations, int4_run_kernel, int4_cleanup
    });

    using P = bench::AttnMode;
    bench::register_attn_backend({
        "litert_ruy_int8_prefill", "litert", P::PREFILL,
        attn::ruy_int8_prefill, attn::int8_attn_run_ruy, attn::int8_attn_cleanup
    });
    bench::register_attn_backend({
        "litert_ruy_int8_decode", "litert", P::DECODE,
        attn::ruy_int8_decode, attn::int8_attn_run_ruy, attn::int8_attn_cleanup
    });
    bench::register_attn_backend({
        "litert_neon_int8_decode", "litert", P::DECODE,
        attn::neon_int8_decode, attn::int8_attn_run_neon, attn::int8_attn_cleanup
    });
    bench::register_attn_backend({
        "litert_q4_prefill", "litert", P::PREFILL,
        attn::q4_prefill, attn::int4_attn_run, attn::int4_attn_cleanup
    });
    bench::register_attn_backend({
        "litert_q4_decode", "litert", P::DECODE,
        attn::q4_decode, attn::int4_attn_run, attn::int4_attn_cleanup
    });
    return 0;
}();

} // namespace
