#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

double cactus_sum_all_f32(const float* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                sum_vec = vaddq_f32(sum_vec, input_vec);
            }
            
            double thread_sum = static_cast<double>(vaddvq_f32(sum_vec));
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_sum += static_cast<double>(data[i]);
            }
            
            return thread_sum;
        },
        0.0,
        [](double a, double b) { return a + b; }
    );
}

void cactus_sum_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                sum_vec = vaddq_f32(sum_vec, input_vec);
            }
            
            float total_sum = vaddvq_f32(sum_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum += input[idx];
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = total_sum;
        });
}

double cactus_sum_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 8;  
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float16x8_t sum_vec = vdupq_n_f16(0.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                sum_vec = vaddq_f16(sum_vec, input_vec);
            }
            
            double thread_sum = 0.0;
            __fp16 sum_array[8];
            vst1q_f16(sum_array, sum_vec);
            for (int j = 0; j < 8; j++) {
                thread_sum += static_cast<double>(sum_array[j]);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_sum += static_cast<double>(data[i]);
            }
            
            return thread_sum;
        },
        0.0,
        [](double a, double b) { return a + b; }
    );
}

double cactus_mean_all_f16(const __fp16* data, size_t num_elements) {
    double sum = cactus_sum_all_f16(data, num_elements);
    return sum / static_cast<double>(num_elements);
}

void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t sum_vec = vdupq_n_f16(0.0f);
            
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                sum_vec = vaddq_f16(sum_vec, input_vec);
            }
            
            __fp16 total_sum = 0.0f;
            __fp16 sum_array[8];
            vst1q_f16(sum_array, sum_vec);
            for (int j = 0; j < 8; j++) {
                total_sum += sum_array[j];
            }
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                total_sum += input[idx];
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = total_sum / static_cast<__fp16>(axis_size);
        });
}

double cactus_mean_all_f32(const float* data, size_t num_elements) {
    double sum = cactus_sum_all_f32(data, num_elements);
    return sum / static_cast<double>(num_elements);
}

void cactus_mean_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    cactus_sum_axis_f32(input, output, outer_size, axis_size, inner_size);
    const float divisor = static_cast<float>(axis_size);
    
    CactusThreading::parallel_for(outer_size * inner_size, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start_idx, size_t end_idx) {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t divisor_vec = vdupq_n_f32(divisor);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t output_vec = vld1q_f32(&output[i]);
                output_vec = vdivq_f32(output_vec, divisor_vec);
                vst1q_f32(&output[i], output_vec);
            }
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                output[i] /= divisor;
            }
        });
}

double cactus_variance_all_f32(const float* data, size_t num_elements) {
    double mean = cactus_mean_all_f32(data, num_elements);
    
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t mean_vec = vdupq_n_f32(static_cast<float>(mean));
            float32x4_t var_vec = vdupq_n_f32(0.0f);
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                float32x4_t diff = vsubq_f32(input_vec, mean_vec);
                var_vec = vmlaq_f32(var_vec, diff, diff);
            }
            
            double thread_var = static_cast<double>(vaddvq_f32(var_vec));
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                double diff = static_cast<double>(data[i]) - mean;
                thread_var += diff * diff;
            }
            
            return thread_var;
        },
        0.0,
        [](double a, double b) { return a + b; }
    ) / static_cast<double>(num_elements);
}

void cactus_variance_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    std::vector<float> means(outer_size * inner_size);
    cactus_mean_axis_f32(input, means.data(), outer_size, axis_size, inner_size);
    
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            size_t output_idx = outer * inner_size + inner;
            float mean_val = means[output_idx];
            
            float32x4_t mean_vec = vdupq_n_f32(mean_val);
            float32x4_t var_vec = vdupq_n_f32(0.0f);
            
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                float32x4_t diff = vsubq_f32(input_vec, mean_vec);
                var_vec = vmlaq_f32(var_vec, diff, diff);
            }
            
            float total_var = vaddvq_f32(var_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                float diff = input[idx] - mean_val;
                total_var += diff * diff;
            }
            
            output[output_idx] = total_var / static_cast<float>(axis_size);
        });
}

float cactus_min_all_f32(const float* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> float {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                min_vec = vminq_f32(min_vec, input_vec);
            }
            
            float thread_min = vminvq_f32(min_vec);
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_min = std::min(thread_min, data[i]);
            }
            
            return thread_min;
        },
        std::numeric_limits<float>::max(),
        [](float a, float b) { return std::min(a, b); }
    );
}

void cactus_min_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float32x4_t min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
            
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                min_vec = vminq_f32(min_vec, input_vec);
            }
            
            float min_val = vminvq_f32(min_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                min_val = std::min(min_val, input[idx]);
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = min_val;
        });
}

float cactus_max_all_f32(const float* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> float {
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;
            
            float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
            
            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float32x4_t input_vec = vld1q_f32(&data[i]);
                max_vec = vmaxq_f32(max_vec, input_vec);
            }
            
            float thread_max = vmaxvq_f32(max_vec);
            
            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_max = std::max(thread_max, data[i]);
            }
            
            return thread_max;
        },
        std::numeric_limits<float>::lowest(),
        [](float a, float b) { return std::max(a, b); }
    );
}

void cactus_max_axis_f32(const float* input, float* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
            
            constexpr size_t SIMD_WIDTH = 4;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;
            
            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                float values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float32x4_t input_vec = vld1q_f32(values);
                max_vec = vmaxq_f32(max_vec, input_vec);
            }
            
            float max_val = vmaxvq_f32(max_vec);
            
            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                max_val = std::max(max_val, input[idx]);
            }
            
            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = max_val;
        });
} 