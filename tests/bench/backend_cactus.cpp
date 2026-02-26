#include "bench_driver.h"

#include <arm_neon.h>

namespace {

enum class CactusQuant { INT8, INT4 };

struct CactusWeights {
    size_t K, N;
    CactusQuant quant;
    std::vector<int8_t> int8_weights;
    std::vector<uint8_t> int4_packed;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;

    void run(size_t M, const int8_t* act_int8, const float* act_scales, __fp16* out) const {
        if (quant == CactusQuant::INT8)
            cactus_matmul_int8(act_int8, act_scales, int8_weights.data(), scales.data(),
                               out, M, K, N, bench::kGroupSize);
        else
            cactus_matmul_int4(act_int8, act_scales,
                               reinterpret_cast<const int8_t*>(int4_packed.data()),
                               scales.data(), out, M, K, N, bench::kGroupSize);
    }
};

static void* prepare_impl(const float* fp32, size_t N, size_t K, CactusQuant quant) {
    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;

    auto* w = new CactusWeights();
    w->K = K;
    w->N = N;
    w->quant = quant;

    if (quant == CactusQuant::INT8) {
        bench::quantize_int8_per_group(src, N, K, rowmajor, raw_scales);
        w->int8_weights = bench::interleave_weights_nk4(rowmajor, N, K);
    } else {
        bench::quantize_int4_per_group(src, N, K, rowmajor, raw_scales);
        w->int4_packed = bench::pack_int4_pairs(bench::interleave_weights_nk4(rowmajor, N, K));
    }
    w->scales = bench::interleave_scales_n4(raw_scales, N, K / bench::kGroupSize);
    return w;
}

void* prepare_int8(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, CactusQuant::INT8); }
void* prepare_int4(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, CactusQuant::INT4); }

void run_kernel(size_t M, void* weights, void*,
                const int8_t* act_int8, const float* act_scales,
                float* output, float*) {
    auto* w = static_cast<CactusWeights*>(weights);
    w->output_buf.resize(M * w->N);
    w->run(M, act_int8, act_scales, w->output_buf.data());
    if (output) {
        const __fp16* src = w->output_buf.data();
        size_t count = M * w->N;
        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            float16x8_t v = vld1q_f16(reinterpret_cast<const float16_t*>(src + i));
            vst1q_f32(output + i,     vcvt_f32_f16(vget_low_f16(v)));
            vst1q_f32(output + i + 4, vcvt_f32_f16(vget_high_f16(v)));
        }
        for (; i < count; i++)
            output[i] = static_cast<float>(src[i]);
    }
}

void cleanup(void* weights, void*) {
    delete static_cast<CactusWeights*>(weights);
}

static int reg = [] {
    bench::register_backend({
        "cactus_int8", "cactus", bench::QuantCategory::INT8, 0,
        prepare_int8, nullptr, run_kernel, cleanup
    });
    bench::register_backend({
        "cactus_int4", "cactus", bench::QuantCategory::INT4, 0,
        prepare_int4, nullptr, run_kernel, cleanup
    });
    return 0;
}();

} // namespace
