#include "bench_driver.h"

#ifdef WITH_MLC

#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ffi = tvm::ffi;

namespace {

static ffi::Function s_q4_fn{nullptr};
static ffi::Function s_q8_fn{nullptr};

static bool load_module() {
    const char* path = std::getenv("MLC_MATMUL_LIB");
    if (!path) {
        fprintf(stderr, "[mlc] MLC_MATMUL_LIB not set; skipping MLC backend.\n"
                        "      Set it to a TVM-compiled .so with quantized matmul kernels.\n");
        return false;
    }

    ffi::Module mod{nullptr};
    try {
        mod = ffi::Module::LoadFromFile(path);
    } catch (const std::exception& e) {
        fprintf(stderr, "[mlc] Failed to load %s: %s\n", path, e.what());
        return false;
    }

    auto q4 = mod->GetFunction("quantized_matmul_int4", true);
    if (q4.has_value()) s_q4_fn = *q4;
    auto q8 = mod->GetFunction("quantized_matmul_int8", true);
    if (q8.has_value()) s_q8_fn = *q8;

    if (s_q4_fn == nullptr && s_q8_fn == nullptr) {
        fprintf(stderr, "[mlc] Module lacks quantized_matmul_int4/int8 functions\n");
        return false;
    }

    return true;
}

// ── Shared helpers ──────────────────────────────────────────────────────────────

static DLTensor make_dl(void* data, int64_t* shape, int ndim,
                         uint8_t type_code, uint8_t bits) {
    DLTensor t{};
    t.data = data;
    t.device = {kDLCPU, 0};
    t.ndim = ndim;
    t.dtype = {type_code, bits, 1};
    t.shape = shape;
    return t;
}

struct QuantizedTensor {
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
    int64_t b_shape[2];
    int64_t s_shape[2];
};

static QuantizedTensor quantize_and_pack(const float* fp32, size_t N, size_t K, int bits) {
    QuantizedTensor qt;
    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> quantized;
    std::vector<float> raw_scales;

    if (bits == 4) {
        bench::quantize_int4_per_group(src, N, K, quantized, raw_scales);
        qt.packed.resize(N * K / 2);
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k += 2) {
                uint8_t lo = static_cast<uint8_t>(quantized[n * K + k] + 8);
                uint8_t hi = static_cast<uint8_t>(quantized[n * K + k + 1] + 8);
                qt.packed[n * (K / 2) + k / 2] = lo | (hi << 4);
            }
        qt.b_shape[0] = static_cast<int64_t>(N);
        qt.b_shape[1] = static_cast<int64_t>(K / 2);
    } else {
        bench::quantize_int8_per_group(src, N, K, quantized, raw_scales);
        qt.packed.resize(N * K);
        std::memcpy(qt.packed.data(), quantized.data(), N * K);
        qt.b_shape[0] = static_cast<int64_t>(N);
        qt.b_shape[1] = static_cast<int64_t>(K);
    }

    const size_t num_groups = K / bench::kGroupSize;
    qt.scales.resize(N * num_groups);
    for (size_t i = 0; i < N * num_groups; i++)
        qt.scales[i] = static_cast<__fp16>(raw_scales[i]);
    qt.s_shape[0] = static_cast<int64_t>(N);
    qt.s_shape[1] = static_cast<int64_t>(num_groups);

    return qt;
}

static void softmax_causal(float* scores, size_t sl, size_t kvl, bool causal) {
    size_t offset = kvl - sl;
    for (size_t row = 0; row < sl; row++) {
        float* rp = scores + row * kvl;
        if (causal)
            for (size_t j = row + offset + 1; j < kvl; j++)
                rp[j] = -1e4f;
        float mx = rp[0];
        for (size_t j = 1; j < kvl; j++)
            mx = std::max(mx, rp[j]);
        float sum = 0.0f;
        for (size_t j = 0; j < kvl; j++) {
            rp[j] = std::exp(rp[j] - mx);
            sum += rp[j];
        }
        float inv = 1.0f / sum;
        for (size_t j = 0; j < kvl; j++)
            rp[j] *= inv;
    }
}

static void tvm_matmul(const ffi::Function& fn,
                        __fp16* a, int64_t M, int64_t K,
                        QuantizedTensor& qt, int bits,
                        __fp16* out, int64_t N) {
    int64_t a_shape[] = {M, K};
    int64_t c_shape[] = {M, N};
    uint8_t b_type = (bits == 4) ? kDLUInt : kDLInt;
    DLTensor a_dl = make_dl(a, a_shape, 2, kDLFloat, 16);
    DLTensor b_dl = make_dl(qt.packed.data(), qt.b_shape, 2, b_type, 8);
    DLTensor s_dl = make_dl(qt.scales.data(), qt.s_shape, 2, kDLFloat, 16);
    DLTensor c_dl = make_dl(out, c_shape, 2, kDLFloat, 16);
    fn(&a_dl, &b_dl, &s_dl, &c_dl);
}

// ── Matmul ──────────────────────────────────────────────────────────────────────

struct MlcWeights {
    size_t K, N;
    int bits;
    ffi::Function fn;
    QuantizedTensor qt;
    std::vector<__fp16> output_buf;
};

struct MlcActivations {
    std::vector<__fp16> fp16;
};

static void* prepare_impl(const float* fp32, size_t N, size_t K,
                            int bits, const ffi::Function& fn) {
    auto* w = new MlcWeights();
    w->K = K;
    w->N = N;
    w->bits = bits;
    w->fn = fn;

    w->qt = quantize_and_pack(fp32, N, K, bits);

    return w;
}

void* prepare_q4(const float* fp32, size_t N, size_t K) {
    return prepare_impl(fp32, N, K, 4, s_q4_fn);
}

void* prepare_q8(const float* fp32, size_t N, size_t K) {
    return prepare_impl(fp32, N, K, 8, s_q8_fn);
}

void* prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new MlcActivations();
    a->fp16.resize(M * K);
    bench::fp32_to_fp16(fp32, a->fp16.data(), M * K);
    return a;
}

void run_kernel(size_t M, void* weights, void* activations,
                const int8_t*, const float*,
                float* output, float*) {
    auto* w = static_cast<MlcWeights*>(weights);
    auto* a = static_cast<MlcActivations*>(activations);
    w->output_buf.resize(M * w->N);

    tvm_matmul(w->fn, a->fp16.data(),
               static_cast<int64_t>(M), static_cast<int64_t>(w->K),
               w->qt, w->bits,
               w->output_buf.data(), static_cast<int64_t>(w->N));

    if (output)
        bench::fp16_to_fp32(w->output_buf.data(), output, M * w->N);
}

void cleanup(void* weights, void* activations) {
    delete static_cast<MlcWeights*>(weights);
    if (activations) delete static_cast<MlcActivations*>(activations);
}

// ── Quantized Attention ─────────────────────────────────────────────────────────
// Two quantized_matmul calls (Q@K^T and softmax(scores)@V) with C++ softmax/masking.
// V is transposed before quantization so the existing TVM kernel handles both steps.

namespace qattn {

struct PerHead {
    QuantizedTensor k;
    QuantizedTensor vt;
};

struct State {
    size_t sl, kvl, hd, qh, kvh;
    int bits;
    bool causal;
    ffi::Function fn;
    std::vector<__fp16> q_fp16;
    std::vector<PerHead> heads;
    std::vector<__fp16> scores_fp16;
    std::vector<__fp16> out_fp16;
};

void* prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
              const float* fp32_q, const float* fp32_k, const float* fp32_v,
              bench::AttnMode mode, int bits) {
    size_t sl = (mode == bench::AttnMode::PREFILL) ? seq_len : 1;
    size_t kvl = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    size_t hd = dims.head_dim;
    size_t qh = dims.num_q_heads;
    size_t kvh = dims.num_kv_heads;

    if (kvl % bench::kGroupSize != 0 || hd % bench::kGroupSize != 0)
        return nullptr;

    const ffi::Function& fn = (bits == 4) ? s_q4_fn : s_q8_fn;
    if (fn == nullptr) return nullptr;

    auto* state = new State();
    state->sl = sl;
    state->kvl = kvl;
    state->hd = hd;
    state->qh = qh;
    state->kvh = kvh;
    state->bits = bits;
    state->causal = (mode == bench::AttnMode::PREFILL && sl > 1);
    state->fn = fn;

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    state->q_fp16.resize(qh * sl * hd);
    for (size_t i = 0; i < qh * sl * hd; i++)
        state->q_fp16[i] = static_cast<__fp16>(fp32_q[i] * scale);

    state->heads.resize(kvh);
    for (size_t h = 0; h < kvh; h++) {
        const float* k_head = fp32_k + h * kvl * hd;
        state->heads[h].k = quantize_and_pack(k_head, kvl, hd, bits);

        const float* v_head = fp32_v + h * kvl * hd;
        std::vector<float> vt(hd * kvl);
        for (size_t r = 0; r < kvl; r++)
            for (size_t c = 0; c < hd; c++)
                vt[c * kvl + r] = v_head[r * hd + c];
        state->heads[h].vt = quantize_and_pack(vt.data(), hd, kvl, bits);
    }

    state->scores_fp16.resize(sl * kvl);
    state->out_fp16.resize(qh * sl * hd);

    return state;
}

void run(void* state_ptr, float* output) {
    auto* s = static_cast<State*>(state_ptr);
    size_t sl = s->sl, kvl = s->kvl, hd = s->hd, qh = s->qh;
    size_t gqa_ratio = qh / s->kvh;
    int64_t sl_i = static_cast<int64_t>(sl);
    int64_t kvl_i = static_cast<int64_t>(kvl);
    int64_t hd_i = static_cast<int64_t>(hd);
    std::vector<float> scores_fp32(sl * kvl);

    for (size_t qhi = 0; qhi < qh; qhi++) {
        size_t kvi = qhi / gqa_ratio;
        auto& head = s->heads[kvi];
        __fp16* q_head = s->q_fp16.data() + qhi * sl * hd;

        tvm_matmul(s->fn, q_head, sl_i, hd_i,
                   head.k, s->bits,
                   s->scores_fp16.data(), kvl_i);

        bench::fp16_to_fp32(s->scores_fp16.data(), scores_fp32.data(), sl * kvl);
        softmax_causal(scores_fp32.data(), sl, kvl, s->causal);
        bench::fp32_to_fp16(scores_fp32.data(), s->scores_fp16.data(), sl * kvl);

        __fp16* out_head = s->out_fp16.data() + qhi * sl * hd;
        tvm_matmul(s->fn, s->scores_fp16.data(), sl_i, kvl_i,
                   head.vt, s->bits,
                   out_head, hd_i);
    }

    if (output)
        bench::fp16_to_fp32(s->out_fp16.data(), output, qh * sl * hd);
}

void attn_cleanup(void* state) {
    delete static_cast<State*>(state);
}

template<bench::AttnMode Mode, int Bits>
void* prepare_variant(const bench::AttnDims& d, size_t sl, size_t cl,
                      const float* q, const float* k, const float* v) {
    return prepare(d, sl, cl, q, k, v, Mode, Bits);
}

} // namespace qattn

// ── Registration ────────────────────────────────────────────────────────────────

static int reg = [] {
    if (!load_module()) return 0;

    if (s_q4_fn != nullptr)
        bench::register_matmul_backend({
            "mlc_int4", "mlc", bench::QuantCategory::INT4, 0,
            prepare_q4, prepare_act, run_kernel, cleanup
        });
    if (s_q8_fn != nullptr)
        bench::register_matmul_backend({
            "mlc_int8", "mlc", bench::QuantCategory::INT8, 0,
            prepare_q8, prepare_act, run_kernel, cleanup
        });

    using P = bench::AttnMode;
    if (s_q4_fn != nullptr) {
        bench::register_attn_backend({"mlc_q4_prefill", "mlc", P::PREFILL,
            qattn::prepare_variant<P::PREFILL, 4>, qattn::run, qattn::attn_cleanup});
        bench::register_attn_backend({"mlc_q4_decode", "mlc", P::DECODE,
            qattn::prepare_variant<P::DECODE, 4>, qattn::run, qattn::attn_cleanup});
    }
    if (s_q8_fn != nullptr) {
        bench::register_attn_backend({"mlc_q8_prefill", "mlc", P::PREFILL,
            qattn::prepare_variant<P::PREFILL, 8>, qattn::run, qattn::attn_cleanup});
        bench::register_attn_backend({"mlc_q8_decode", "mlc", P::DECODE,
            qattn::prepare_variant<P::DECODE, 8>, qattn::run, qattn::attn_cleanup});
    }

    return 0;
}();

} // namespace

#else

namespace {
[[maybe_unused]] static int reg = [] { return 0; }();
} // namespace

#endif
