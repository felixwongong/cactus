#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>

void cactus_scalar_op_f32(const float* input, float* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    switch (op_type) {
        case ScalarOpType::ADD: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vaddq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] + scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::SUBTRACT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vsubq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] - scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::MULTIPLY: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vmulq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] * scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::DIVIDE: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    float32x4_t scalar_vec = vdupq_n_f32(scalar_value);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_vec = vld1q_f32(&input[i]);
                        float32x4_t result_vec = vdivq_f32(input_vec, scalar_vec);
                        vst1q_f32(&output[i], result_vec);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] / scalar_value;
                    }
                });
            break;
        }
        
        case ScalarOpType::EXP: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_f32 = vld1q_f32(&input[i]);
                        
                        const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
                        const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
                        const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);
                        const float32x4_t c3 = vdupq_n_f32(0.05550410866482158f);

                        float32x4_t x = vmulq_f32(input_f32, log2e);
                        int32x4_t xi = vcvtq_s32_f32(x);
                        float32x4_t xf = vsubq_f32(x, vcvtq_f32_s32(xi));
                        
                        float32x4_t p = vmlaq_f32(c2, c3, xf);
                        p = vmlaq_f32(c1, p, xf);
                        p = vmlaq_f32(vdupq_n_f32(1.0f), p, xf);

                        int32x4_t exponent = vaddq_s32(xi, vdupq_n_s32(127));
                        exponent = vshlq_n_s32(exponent, 23);
                        float32x4_t scale = vreinterpretq_f32_s32(exponent);
                        
                        float32x4_t result_f32 = vmulq_f32(p, scale);
                        
                        vst1q_f32(&output[i], result_f32);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = std::exp(input[i]);
                    }
                });
            break;
        }
        
        case ScalarOpType::SQRT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t input_f32 = vld1q_f32(&input[i]);
                        
                        input_f32 = vmaxq_f32(input_f32, vdupq_n_f32(0.0f));
                        
                        uint32x4_t zero_mask = vceqq_f32(input_f32, vdupq_n_f32(0.0f));
                        
                        float32x4_t rsqrt = vrsqrteq_f32(input_f32);
                        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(input_f32, rsqrt), rsqrt));
                        float32x4_t result_f32 = vmulq_f32(input_f32, rsqrt);
                        
                        result_f32 = vbslq_f32(zero_mask, vdupq_n_f32(0.0f), result_f32);
                        
                        vst1q_f32(&output[i], result_f32);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = std::sqrt(std::max(0.0f, input[i]));
                    }
                });
            break;
        }
        
        case ScalarOpType::COS: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    const float32x4_t two_pi = vdupq_n_f32(2.0f * 3.14159265358979323846f);
                    const float32x4_t inv_two_pi = vdupq_n_f32(1.0f / (2.0f * 3.14159265358979323846f));
                    const float32x4_t c0 = vdupq_n_f32(1.0f);
                    const float32x4_t c2 = vdupq_n_f32(-0.5f);
                    const float32x4_t c4 = vdupq_n_f32(0.04166666666f);
                    const float32x4_t c6 = vdupq_n_f32(-0.00138888888f);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t x = vld1q_f32(&input[i]);
                        
                        x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, inv_two_pi)), two_pi));
                        
                        float32x4_t x2 = vmulq_f32(x, x);
                        float32x4_t x4 = vmulq_f32(x2, x2);
                        float32x4_t x6 = vmulq_f32(x4, x2);
                        
                        float32x4_t result = c0;
                        result = vmlaq_f32(result, x2, c2);
                        result = vmlaq_f32(result, x4, c4);
                        result = vmlaq_f32(result, x6, c6);
                        
                        vst1q_f32(&output[i], result);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = cosf(input[i]);
                    }
                });
            break;
        }
        
        case ScalarOpType::SIN: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 4;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    
                    const float32x4_t two_pi = vdupq_n_f32(2.0f * 3.14159265358979323846f);
                    const float32x4_t inv_two_pi = vdupq_n_f32(1.0f / (2.0f * 3.14159265358979323846f));
                    const float32x4_t c1 = vdupq_n_f32(1.0f);
                    const float32x4_t c3 = vdupq_n_f32(-0.16666666666f);
                    const float32x4_t c5 = vdupq_n_f32(0.00833333333f);
                    const float32x4_t c7 = vdupq_n_f32(-0.00019841269f);
                    
                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float32x4_t x = vld1q_f32(&input[i]);
                        
                        x = vsubq_f32(x, vmulq_f32(vrndnq_f32(vmulq_f32(x, inv_two_pi)), two_pi));
                        
                        float32x4_t x2 = vmulq_f32(x, x);
                        float32x4_t x3 = vmulq_f32(x2, x);
                        float32x4_t x5 = vmulq_f32(x3, x2);
                        float32x4_t x7 = vmulq_f32(x5, x2);
                        
                        float32x4_t result = vmulq_f32(x, c1);
                        result = vmlaq_f32(result, x3, c3);
                        result = vmlaq_f32(result, x5, c5);
                        result = vmlaq_f32(result, x7, c7);
                        
                        vst1q_f32(&output[i], result);
                    }
                    
                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = sinf(input[i]);
                    }
                });
            break;
        }
    }
}

void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements, float scalar_value, ScalarOpType op_type) {
    const __fp16 scalar_f16 = static_cast<__fp16>(scalar_value);

    switch (op_type) {
        case ScalarOpType::ADD: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vaddq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] + scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::SUBTRACT: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vsubq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] - scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::MULTIPLY: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vmulq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] * scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::DIVIDE: {
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_BASIC,
                [&](size_t start_idx, size_t end_idx) {
                    constexpr size_t SIMD_WIDTH = 8;
                    const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
                    const float16x8_t scalar_vec = vdupq_n_f16(scalar_f16);

                    for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                        float16x8_t in_vec = vld1q_f16(&input[i]);
                        float16x8_t result = vdivq_f16(in_vec, scalar_vec);
                        vst1q_f16(&output[i], result);
                    }

                    for (size_t i = vectorized_end; i < end_idx; ++i) {
                        output[i] = input[i] / scalar_f16;
                    }
                });
            break;
        }

        case ScalarOpType::EXP:
        case ScalarOpType::SQRT:
        case ScalarOpType::COS:
        case ScalarOpType::SIN: {
            // For complex operations, convert to float32, compute, then convert back
            CactusThreading::parallel_for(num_elements, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
                [&](size_t start_idx, size_t end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        float val = static_cast<float>(input[i]);
                        float result;
                        switch (op_type) {
                            case ScalarOpType::EXP: result = std::exp(val); break;
                            case ScalarOpType::SQRT: result = std::sqrt(val); break;
                            case ScalarOpType::COS: result = std::cos(val); break;
                            case ScalarOpType::SIN: result = std::sin(val); break;
                            default: result = val; break;
                        }
                        output[i] = static_cast<__fp16>(result);
                    }
                });
            break;
        }
    }
}