#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

void cactus_sum_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
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
            output[output_idx] = total_sum;
        });
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

double cactus_variance_all_f16(const __fp16* data, size_t num_elements) {
    double mean = cactus_mean_all_f16(data, num_elements);

    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> double {
            double thread_var = 0.0;

            for (size_t i = start_idx; i < end_idx; ++i) {
                double diff = static_cast<double>(data[i]) - mean;
                thread_var += diff * diff;
            }

            return thread_var;
        },
        0.0,
        [](double a, double b) { return a + b; }
    ) / static_cast<double>(num_elements);
}

void cactus_variance_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    std::vector<__fp16> means(outer_size * inner_size);
    cactus_mean_axis_f16(input, means.data(), outer_size, axis_size, inner_size);

    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            size_t output_idx = outer * inner_size + inner;
            float mean_val = static_cast<float>(means[output_idx]);

            float total_var = 0.0f;

            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                float diff = static_cast<float>(input[idx]) - mean_val;
                total_var += diff * diff;
            }

            output[output_idx] = static_cast<__fp16>(total_var / static_cast<float>(axis_size));
        });
}

__fp16 cactus_min_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> __fp16 {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            float16x8_t min_vec = vdupq_n_f16(static_cast<__fp16>(65504.0f));

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                min_vec = vminq_f16(min_vec, input_vec);
            }

            __fp16 thread_min = static_cast<__fp16>(65504.0f);
            __fp16 min_array[8];
            vst1q_f16(min_array, min_vec);
            for (int j = 0; j < 8; j++) {
                thread_min = std::min(thread_min, min_array[j]);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_min = std::min(thread_min, data[i]);
            }

            return thread_min;
        },
        static_cast<__fp16>(65504.0f),
        [](__fp16 a, __fp16 b) { return std::min(a, b); }
    );
}

void cactus_min_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t min_vec = vdupq_n_f16(static_cast<__fp16>(65504.0f));

            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                min_vec = vminq_f16(min_vec, input_vec);
            }

            __fp16 min_val = static_cast<__fp16>(65504.0f);
            __fp16 min_array[8];
            vst1q_f16(min_array, min_vec);
            for (int j = 0; j < 8; j++) {
                min_val = std::min(min_val, min_array[j]);
            }

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                min_val = std::min(min_val, input[idx]);
            }

            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = min_val;
        });
}

__fp16 cactus_max_all_f16(const __fp16* data, size_t num_elements) {
    return CactusThreading::parallel_reduce(
        num_elements, CactusThreading::Thresholds::ALL_REDUCE,
        [&](size_t start_idx, size_t end_idx) -> __fp16 {
            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_end = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            float16x8_t max_vec = vdupq_n_f16(static_cast<__fp16>(-65504.0f));

            for (size_t i = start_idx; i < vectorized_end; i += SIMD_WIDTH) {
                float16x8_t input_vec = vld1q_f16(&data[i]);
                max_vec = vmaxq_f16(max_vec, input_vec);
            }

            __fp16 thread_max = static_cast<__fp16>(-65504.0f);
            __fp16 max_array[8];
            vst1q_f16(max_array, max_vec);
            for (int j = 0; j < 8; j++) {
                thread_max = std::max(thread_max, max_array[j]);
            }

            for (size_t i = vectorized_end; i < end_idx; ++i) {
                thread_max = std::max(thread_max, data[i]);
            }

            return thread_max;
        },
        static_cast<__fp16>(-65504.0f),
        [](__fp16 a, __fp16 b) { return std::max(a, b); }
    );
}

void cactus_max_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size, size_t inner_size) {
    CactusThreading::parallel_for_2d(outer_size, inner_size, CactusThreading::Thresholds::AXIS_REDUCE,
        [&](size_t outer, size_t inner) {
            float16x8_t max_vec = vdupq_n_f16(static_cast<__fp16>(-65504.0f));

            constexpr size_t SIMD_WIDTH = 8;
            const size_t vectorized_axis = (axis_size / SIMD_WIDTH) * SIMD_WIDTH;

            for (size_t a = 0; a < vectorized_axis; a += SIMD_WIDTH) {
                __fp16 values[SIMD_WIDTH];
                for (size_t j = 0; j < SIMD_WIDTH; j++) {
                    size_t idx = outer * axis_size * inner_size + (a + j) * inner_size + inner;
                    values[j] = input[idx];
                }
                float16x8_t input_vec = vld1q_f16(values);
                max_vec = vmaxq_f16(max_vec, input_vec);
            }

            __fp16 max_val = static_cast<__fp16>(-65504.0f);
            __fp16 max_array[8];
            vst1q_f16(max_array, max_vec);
            for (int j = 0; j < 8; j++) {
                max_val = std::max(max_val, max_array[j]);
            }

            for (size_t a = vectorized_axis; a < axis_size; a++) {
                size_t idx = outer * axis_size * inner_size + a * inner_size + inner;
                max_val = std::max(max_val, input[idx]);
            }

            size_t output_idx = outer * inner_size + inner;
            output[output_idx] = max_val;
        });
}
