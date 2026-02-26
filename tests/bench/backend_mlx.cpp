#include "bench_driver.h"

#ifndef WITH_MLX

namespace {
[[maybe_unused]] static int reg = [] {
    return 0;
}();
} // namespace

#else

#include <mlx/mlx.h>
#include <cstring>
#include <cmath>

namespace mx = mlx::core;

namespace {

// ── Matmul ──────────────────────────────────────────────────────────────────────

namespace matmul {

struct Weights {
    size_t K, N;
    int bits;
    mx::Device device;
    mx::array w_q;
    mx::array scales;
    mx::array biases;
};

struct Activations {
    mx::array x;
};

void* prepare_impl(const float* fp32, size_t N, size_t K, int bits, mx::Device device) {
    auto w = mx::array(fp32, {static_cast<int>(N), static_cast<int>(K)}, mx::float32);
    auto parts = mx::quantize(w, static_cast<int>(bench::kGroupSize), bits, "affine", device);
    mx::eval(parts);
    return new Weights{K, N, bits, device,
        std::move(parts[0]), std::move(parts[1]), std::move(parts[2])};
}

template<int Bits, bool GPU>
void* prepare_weights(const float* fp32, size_t N, size_t K) {
    return prepare_impl(fp32, N, K, Bits, GPU ? mx::Device::gpu : mx::Device::cpu);
}

void* prepare_act(const float* fp32, size_t M, size_t K, void* weights) {
    auto* w = static_cast<Weights*>(weights);
    auto x = mx::astype(
        mx::array(fp32, {static_cast<int>(M), static_cast<int>(K)}, mx::float32),
        mx::float16, w->device);
    mx::eval(x);
    return new Activations{std::move(x)};
}

void run_kernel(size_t M, void* weights, void* activations,
                const int8_t*, const float*,
                float* output, float*) {
    auto* w = static_cast<Weights*>(weights);
    auto* a = static_cast<Activations*>(activations);
    auto y = mx::quantized_matmul(a->x, w->w_q, w->scales, w->biases,
                                   true, static_cast<int>(bench::kGroupSize), w->bits,
                                   "affine", w->device);
    if (output) {
        y = mx::astype(y, mx::float32, w->device);
        mx::eval(y);
        std::memcpy(output, y.data<float>(), M * w->N * sizeof(float));
    } else {
        mx::eval(y);
    }
}

void cleanup(void* weights, void* activations) {
    delete static_cast<Weights*>(weights);
    if (activations) delete static_cast<Activations*>(activations);
}

} // namespace matmul

// ── Attention Helpers ───────────────────────────────────────────────────────────

mx::array make_fp16(const float* fp32, int d0, int d1, int d2, mx::Device device) {
    return mx::astype(
        mx::reshape(mx::array(fp32, {d0, d1, d2}, mx::float32), {1, d0, d1, d2}),
        mx::float16, device);
}

void finalize(mx::array y, const bench::AttnDims& dims, size_t seq_len,
              mx::Device device, float* output) {
    if (output) {
        y = mx::astype(mx::reshape(y, {static_cast<int>(dims.num_q_heads),
                                        static_cast<int>(seq_len),
                                        static_cast<int>(dims.head_dim)}),
                       mx::float32, device);
        mx::eval(y);
        std::memcpy(output, y.data<float>(),
                    dims.num_q_heads * seq_len * dims.head_dim * sizeof(float));
    } else {
        mx::eval(y);
    }
}

struct AttnShape { int hd, sl, kvl, qh, kvh; };

AttnShape compute_shape(const bench::AttnDims& dims, size_t seq_len,
                         size_t cache_len, bench::AttnMode mode) {
    size_t kv_seq_len = (mode == bench::AttnMode::PREFILL) ? seq_len : cache_len + 1;
    return {static_cast<int>(dims.head_dim), static_cast<int>(seq_len),
            static_cast<int>(kv_seq_len), static_cast<int>(dims.num_q_heads),
            static_cast<int>(dims.num_kv_heads)};
}

template<typename T>
void attn_cleanup(void* state) { delete static_cast<T*>(state); }

// ── FP16 Attention ──────────────────────────────────────────────────────────────

namespace attn {

struct State {
    bench::AttnDims dims;
    mx::Device device;
    size_t seq_len;
    mx::array q, k, v;
    float scale;
};

void* prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
              const float* fp32_q, const float* fp32_k, const float* fp32_v,
              bench::AttnMode mode, mx::Device device) {
    auto [hd, sl, kvl, qh, kvh] = compute_shape(dims, seq_len, cache_len, mode);

    auto q = make_fp16(fp32_q, qh, sl, hd, device);
    auto k = make_fp16(fp32_k, kvh, kvl, hd, device);
    auto v = make_fp16(fp32_v, kvh, kvl, hd, device);
    mx::eval({q, k, v});

    return new State{dims, device, seq_len, std::move(q), std::move(k), std::move(v),
                     1.0f / std::sqrt(static_cast<float>(dims.head_dim))};
}

template<bench::AttnMode Mode, bool GPU>
void* prepare_variant(const bench::AttnDims& d, size_t sl, size_t cl,
                      const float* q, const float* k, const float* v) {
    return prepare(d, sl, cl, q, k, v, Mode, GPU ? mx::Device::gpu : mx::Device::cpu);
}

void run(void* state, float* output) {
    auto* s = static_cast<State*>(state);
    auto y = mx::fast::scaled_dot_product_attention(s->q, s->k, s->v, s->scale, "causal");
    finalize(std::move(y), s->dims, s->seq_len, s->device, output);
}

} // namespace attn

// ── Quantized Attention ─────────────────────────────────────────────────────────
// Mirrors mlx-lm's quantized_scaled_dot_product_attention:
// two quantized_matmul calls (Q@K.T, softmax, scores@V) with quantized KV cache.

namespace qattn {

constexpr int kGroupSize = 64;

struct State {
    bench::AttnDims dims;
    mx::Device device;
    size_t seq_len;
    int bits;
    bool has_mask;
    mx::array q;
    mx::array k_packed, k_scales, k_biases;
    mx::array v_packed, v_scales, v_biases;
    mx::array mask;
    mx::array k_fp16, v_fp16;
};

void* prepare(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
              const float* fp32_q, const float* fp32_k, const float* fp32_v,
              bench::AttnMode mode, mx::Device device, int bits) {
    auto [hd, sl, kvl, qh, kvh] = compute_shape(dims, seq_len, cache_len, mode);
    float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

    auto q = mx::astype(
        mx::multiply(make_fp16(fp32_q, qh, sl, hd, device), mx::array(scale), device),
        mx::float16, device);

    auto k_fp = make_fp16(fp32_k, kvh, kvl, hd, device);
    auto v_fp = make_fp16(fp32_v, kvh, kvl, hd, device);
    if (qh != kvh) {
        k_fp = mx::repeat(k_fp, qh / kvh, 1, device);
        v_fp = mx::repeat(v_fp, qh / kvh, 1, device);
    }
    auto k_parts = mx::quantize(k_fp, kGroupSize, bits, "affine", device);
    auto v_parts = mx::quantize(v_fp, kGroupSize, bits, "affine", device);

    bool has_mask = (mode == bench::AttnMode::PREFILL && sl > 1);
    auto mask = mx::zeros({1});
    if (has_mask) {
        auto rows = mx::reshape(mx::arange(sl, mx::int32, device), {sl, 1});
        auto cols = mx::reshape(mx::arange(kvl, mx::int32, device), {1, kvl});
        mask = mx::reshape(
            mx::where(mx::greater(cols, rows, device),
                      mx::full({1}, -1e4f, mx::float16, device),
                      mx::zeros({1}, mx::float16, device), device),
            {1, 1, sl, kvl});
    }

    auto k_deq = mx::dequantize(k_parts[0], k_parts[1], k_parts[2],
                                 kGroupSize, bits, "affine", mx::float16, device);
    auto v_deq = mx::dequantize(v_parts[0], v_parts[1], v_parts[2],
                                 kGroupSize, bits, "affine", mx::float16, device);

    std::vector<mx::array> to_eval = {q, k_parts[0], k_parts[1], k_parts[2],
                                       v_parts[0], v_parts[1], v_parts[2],
                                       k_deq, v_deq};
    if (has_mask) to_eval.push_back(mask);
    mx::eval(to_eval);

    return new State{dims, device, seq_len, bits, has_mask, std::move(q),
                     std::move(k_parts[0]), std::move(k_parts[1]), std::move(k_parts[2]),
                     std::move(v_parts[0]), std::move(v_parts[1]), std::move(v_parts[2]),
                     std::move(mask), std::move(k_deq), std::move(v_deq)};
}

template<bench::AttnMode Mode, bool GPU, int Bits>
void* prepare_variant(const bench::AttnDims& d, size_t sl, size_t cl,
                      const float* q, const float* k, const float* v) {
    return prepare(d, sl, cl, q, k, v, Mode, GPU ? mx::Device::gpu : mx::Device::cpu, Bits);
}

void run_gpu(void* state, float* output) {
    auto* s = static_cast<State*>(state);

    auto scores = mx::quantized_matmul(s->q, s->k_packed, s->k_scales, s->k_biases,
                                        true, kGroupSize, s->bits, "affine", s->device);
    if (s->has_mask)
        scores = mx::add(scores, s->mask, s->device);
    scores = mx::softmax(scores, std::vector<int>{-1}, false, s->device);

    auto y = mx::quantized_matmul(scores, s->v_packed, s->v_scales, s->v_biases,
                                   false, kGroupSize, s->bits, "affine", s->device);
    finalize(std::move(y), s->dims, s->seq_len, s->device, output);
}

// CPU path: use pre-dequantized fp16 KV with regular matmul through Accelerate/AMX.
// MLX's CPU quantized_matmul is a naive scalar loop; BNNS fp16 matmul is orders of magnitude faster.
void run_cpu(void* state, float* output) {
    auto* s = static_cast<State*>(state);

    auto scores = mx::matmul(s->q, mx::transpose(s->k_fp16, {0, 1, 3, 2}, s->device), s->device);
    if (s->has_mask)
        scores = mx::add(scores, s->mask, s->device);
    scores = mx::softmax(scores, std::vector<int>{-1}, false, s->device);

    auto y = mx::matmul(scores, s->v_fp16, s->device);
    finalize(std::move(y), s->dims, s->seq_len, s->device, output);
}

} // namespace qattn

// ── Registration ────────────────────────────────────────────────────────────────

static int reg = [] {
    bench::register_matmul_backend({"mlx_q4_gpu", "mlx", bench::QuantCategory::INT4, 0,
        matmul::prepare_weights<4,true>, matmul::prepare_act, matmul::run_kernel, matmul::cleanup});
    bench::register_matmul_backend({"mlx_q8_gpu", "mlx", bench::QuantCategory::INT8, 0,
        matmul::prepare_weights<8,true>, matmul::prepare_act, matmul::run_kernel, matmul::cleanup});
    bench::register_matmul_backend({"mlx_q4_cpu", "mlx", bench::QuantCategory::INT4, 0,
        matmul::prepare_weights<4,false>, matmul::prepare_act, matmul::run_kernel, matmul::cleanup});
    bench::register_matmul_backend({"mlx_q8_cpu", "mlx", bench::QuantCategory::INT8, 0,
        matmul::prepare_weights<8,false>, matmul::prepare_act, matmul::run_kernel, matmul::cleanup});

    using P = bench::AttnMode;
    bench::register_attn_backend({"mlx_gpu_prefill", "mlx", P::PREFILL,
        attn::prepare_variant<P::PREFILL,true>, attn::run, attn_cleanup<attn::State>});
    bench::register_attn_backend({"mlx_cpu_prefill", "mlx", P::PREFILL,
        attn::prepare_variant<P::PREFILL,false>, attn::run, attn_cleanup<attn::State>});
    bench::register_attn_backend({"mlx_gpu_decode", "mlx", P::DECODE,
        attn::prepare_variant<P::DECODE,true>, attn::run, attn_cleanup<attn::State>});
    bench::register_attn_backend({"mlx_cpu_decode", "mlx", P::DECODE,
        attn::prepare_variant<P::DECODE,false>, attn::run, attn_cleanup<attn::State>});

    bench::register_attn_backend({"mlx_q8_gpu_prefill", "mlx", P::PREFILL,
        qattn::prepare_variant<P::PREFILL,true,8>, qattn::run_gpu, attn_cleanup<qattn::State>});
    bench::register_attn_backend({"mlx_q8_cpu_prefill", "mlx", P::PREFILL,
        qattn::prepare_variant<P::PREFILL,false,8>, qattn::run_cpu, attn_cleanup<qattn::State>});
    bench::register_attn_backend({"mlx_q8_gpu_decode", "mlx", P::DECODE,
        qattn::prepare_variant<P::DECODE,true,8>, qattn::run_gpu, attn_cleanup<qattn::State>});
    bench::register_attn_backend({"mlx_q8_cpu_decode", "mlx", P::DECODE,
        qattn::prepare_variant<P::DECODE,false,8>, qattn::run_cpu, attn_cleanup<qattn::State>});

    bench::register_attn_backend({"mlx_q4_gpu_prefill", "mlx", P::PREFILL,
        qattn::prepare_variant<P::PREFILL,true,4>, qattn::run_gpu, attn_cleanup<qattn::State>});
    bench::register_attn_backend({"mlx_q4_cpu_prefill", "mlx", P::PREFILL,
        qattn::prepare_variant<P::PREFILL,false,4>, qattn::run_cpu, attn_cleanup<qattn::State>});
    bench::register_attn_backend({"mlx_q4_gpu_decode", "mlx", P::DECODE,
        qattn::prepare_variant<P::DECODE,true,4>, qattn::run_gpu, attn_cleanup<qattn::State>});
    bench::register_attn_backend({"mlx_q4_cpu_decode", "mlx", P::DECODE,
        qattn::prepare_variant<P::DECODE,false,4>, qattn::run_cpu, attn_cleanup<qattn::State>});
    return 0;
}();

} // namespace

#endif
