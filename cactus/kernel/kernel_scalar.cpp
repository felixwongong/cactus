#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>

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