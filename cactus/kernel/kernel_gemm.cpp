#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

static void cactus_matmul_int8_worker(
    const int8_t* a,
    const int8_t* b_transposed,
    int8_t* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row,
    float a_scale,
    float b_scale,
    float c_scale
) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int VECTOR_WIDTH = 16;
    const size_t K_aligned = (K / (VECTOR_WIDTH * 2)) * (VECTOR_WIDTH * 2);

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            int32x4_t accumulators[TILE_M][TILE_N];
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n)
                    accumulators[m][n] = vdupq_n_s32(0);

            for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * 2) {
                int8x16_t a_vec_low[TILE_M], a_vec_high[TILE_M];
                int8x16_t b_vec_low[TILE_N], b_vec_high[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        a_vec_low[m] = vld1q_s8(&a[row * K + k_block]);
                        a_vec_high[m] = vld1q_s8(&a[row * K + k_block + VECTOR_WIDTH]);
                    } else {
                        a_vec_low[m] = vdupq_n_s8(0);
                        a_vec_high[m] = vdupq_n_s8(0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        b_vec_low[n] = vld1q_s8(&b_transposed[col * K + k_block]);
                        b_vec_high[n] = vld1q_s8(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                    } else {
                        b_vec_low[n] = vdupq_n_s8(0);
                        b_vec_high[n] = vdupq_n_s8(0);
                    }
                }

                accumulators[0][0] = accum_i8mm(accumulators[0][0], a_vec_low[0], b_vec_low[0]);
                accumulators[0][1] = accum_i8mm(accumulators[0][1], a_vec_low[0], b_vec_low[1]);
                accumulators[0][2] = accum_i8mm(accumulators[0][2], a_vec_low[0], b_vec_low[2]);
                accumulators[0][3] = accum_i8mm(accumulators[0][3], a_vec_low[0], b_vec_low[3]);
                accumulators[1][0] = accum_i8mm(accumulators[1][0], a_vec_low[1], b_vec_low[0]);
                accumulators[1][1] = accum_i8mm(accumulators[1][1], a_vec_low[1], b_vec_low[1]);
                accumulators[1][2] = accum_i8mm(accumulators[1][2], a_vec_low[1], b_vec_low[2]);
                accumulators[1][3] = accum_i8mm(accumulators[1][3], a_vec_low[1], b_vec_low[3]);
                accumulators[2][0] = accum_i8mm(accumulators[2][0], a_vec_low[2], b_vec_low[0]);
                accumulators[2][1] = accum_i8mm(accumulators[2][1], a_vec_low[2], b_vec_low[1]);
                accumulators[2][2] = accum_i8mm(accumulators[2][2], a_vec_low[2], b_vec_low[2]);
                accumulators[2][3] = accum_i8mm(accumulators[2][3], a_vec_low[2], b_vec_low[3]);
                accumulators[3][0] = accum_i8mm(accumulators[3][0], a_vec_low[3], b_vec_low[0]);
                accumulators[3][1] = accum_i8mm(accumulators[3][1], a_vec_low[3], b_vec_low[1]);
                accumulators[3][2] = accum_i8mm(accumulators[3][2], a_vec_low[3], b_vec_low[2]);
                accumulators[3][3] = accum_i8mm(accumulators[3][3], a_vec_low[3], b_vec_low[3]);
                
                accumulators[0][0] = accum_i8mm(accumulators[0][0], a_vec_high[0], b_vec_high[0]);
                accumulators[0][1] = accum_i8mm(accumulators[0][1], a_vec_high[0], b_vec_high[1]);
                accumulators[0][2] = accum_i8mm(accumulators[0][2], a_vec_high[0], b_vec_high[2]);
                accumulators[0][3] = accum_i8mm(accumulators[0][3], a_vec_high[0], b_vec_high[3]);
                accumulators[1][0] = accum_i8mm(accumulators[1][0], a_vec_high[1], b_vec_high[0]);
                accumulators[1][1] = accum_i8mm(accumulators[1][1], a_vec_high[1], b_vec_high[1]);
                accumulators[1][2] = accum_i8mm(accumulators[1][2], a_vec_high[1], b_vec_high[2]);
                accumulators[1][3] = accum_i8mm(accumulators[1][3], a_vec_high[1], b_vec_high[3]);
                accumulators[2][0] = accum_i8mm(accumulators[2][0], a_vec_high[2], b_vec_high[0]);
                accumulators[2][1] = accum_i8mm(accumulators[2][1], a_vec_high[2], b_vec_high[1]);
                accumulators[2][2] = accum_i8mm(accumulators[2][2], a_vec_high[2], b_vec_high[2]);
                accumulators[2][3] = accum_i8mm(accumulators[2][3], a_vec_high[2], b_vec_high[3]);
                accumulators[3][0] = accum_i8mm(accumulators[3][0], a_vec_high[3], b_vec_high[0]);
                accumulators[3][1] = accum_i8mm(accumulators[3][1], a_vec_high[3], b_vec_high[1]);
                accumulators[3][2] = accum_i8mm(accumulators[3][2], a_vec_high[3], b_vec_high[2]);
                accumulators[3][3] = accum_i8mm(accumulators[3][3], a_vec_high[3], b_vec_high[3]);
            }

            for (size_t k_block = K_aligned; k_block < K; k_block += VECTOR_WIDTH) {
                size_t remaining = K - k_block;
                int8x16_t a_vec[TILE_M], b_vec[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        if (remaining >= VECTOR_WIDTH) {
                            a_vec[m] = vld1q_s8(&a[row * K + k_block]);
                        } else {
                            int8_t tmp[VECTOR_WIDTH] = {};
                            memcpy(tmp, &a[row * K + k_block], remaining);
                            a_vec[m] = vld1q_s8(tmp);
                        }
                    } else {
                        a_vec[m] = vdupq_n_s8(0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        if (remaining >= VECTOR_WIDTH) {
                            b_vec[n] = vld1q_s8(&b_transposed[col * K + k_block]);
                        } else {
                            int8_t tmp[VECTOR_WIDTH] = {};
                            memcpy(tmp, &b_transposed[col * K + k_block], remaining);
                            b_vec[n] = vld1q_s8(tmp);
                        }
                    } else {
                        b_vec[n] = vdupq_n_s8(0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n)
                        accumulators[m][n] = accum_i8mm(accumulators[m][n], a_vec[m], b_vec[n]);
            }
            const float scale_factor = (a_scale * b_scale) / c_scale;
            
            for (int m = 0; m < TILE_M; ++m) {
                size_t row = row_block + m;
                if (row >= M) continue;
                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col >= N) continue;
                    int32_t sum = vaddvq_s32(accumulators[m][n]);
                    
                    float scaled_result = static_cast<float>(sum) * scale_factor;
                    int32_t quantized_result = static_cast<int32_t>(scaled_result + (scaled_result >= 0 ? 0.5f : -0.5f));
                    quantized_result = std::min(127, std::max(-128, quantized_result));
                    
                    c[row * N + col] = static_cast<int8_t>(quantized_result);
                }
            }
        }
    }
}

void cactus_matmul_int8(
    const int8_t* a,
    const int8_t* b_transposed,
    int8_t* c,
    size_t M,
    size_t K,
    size_t N,
    float a_scale,
    float b_scale,
    float c_scale
) {
    if (M == 0) return;

    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;
    
    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);
                
                cactus_matmul_int8_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row,
                    a_scale, b_scale, c_scale
                );
            }
        });
}

static void cactus_matmul_f16_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int VECTOR_WIDTH = 8;
    const size_t K_aligned = (K / (VECTOR_WIDTH * 2)) * (VECTOR_WIDTH * 2);

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            float16x8_t accumulators[TILE_M][TILE_N];
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n)
                    accumulators[m][n] = vdupq_n_f16(0.0);

            for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * 2) {
                float16x8_t a_vec_low[TILE_M], a_vec_high[TILE_M];
                float16x8_t b_vec_low[TILE_N], b_vec_high[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        a_vec_low[m] = vld1q_f16(&a[row * K + k_block]);
                        a_vec_high[m] = vld1q_f16(&a[row * K + k_block + VECTOR_WIDTH]);
                    } else {
                        a_vec_low[m] = vdupq_n_f16(0.0);
                        a_vec_high[m] = vdupq_n_f16(0.0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        b_vec_low[n] = vld1q_f16(&b_transposed[col * K + k_block]);
                        b_vec_high[n] = vld1q_f16(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                    } else {
                        b_vec_low[n] = vdupq_n_f16(0.0);
                        b_vec_high[n] = vdupq_n_f16(0.0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n) {
                        accumulators[m][n] = accum_f16_dot(accumulators[m][n], 
                                                          a_vec_low[m], a_vec_high[m],
                                                          b_vec_low[n], b_vec_high[n]);
                    }
            }

            for (size_t k_block = K_aligned; k_block < K; k_block += VECTOR_WIDTH) {
                size_t remaining = K - k_block;
                float16x8_t a_vec[TILE_M], b_vec[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        if (remaining >= VECTOR_WIDTH) {
                            a_vec[m] = vld1q_f16(&a[row * K + k_block]);
                        } else {
                            __fp16 tmp[VECTOR_WIDTH] = {0.0};
                            memcpy(tmp, &a[row * K + k_block], remaining * sizeof(__fp16));
                            a_vec[m] = vld1q_f16(tmp);
                        }
                    } else {
                        a_vec[m] = vdupq_n_f16(0.0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        if (remaining >= VECTOR_WIDTH) {
                            b_vec[n] = vld1q_f16(&b_transposed[col * K + k_block]);
                        } else {
                            __fp16 tmp[VECTOR_WIDTH] = {0.0};
                            memcpy(tmp, &b_transposed[col * K + k_block], remaining * sizeof(__fp16));
                            b_vec[n] = vld1q_f16(tmp);
                        }
                    } else {
                        b_vec[n] = vdupq_n_f16(0.0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n)
                        accumulators[m][n] = vfmaq_f16(accumulators[m][n], a_vec[m], b_vec[n]);
            }

            for (int m = 0; m < TILE_M; ++m) {
                size_t row = row_block + m;
                if (row >= M) continue;
                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col >= N) continue;
                    float16x4_t low = vget_low_f16(accumulators[m][n]);
                    float16x4_t high = vget_high_f16(accumulators[m][n]);
                    float16x4_t sum_vec = vadd_f16(low, high);
                    __fp16 sum = vget_lane_f16(sum_vec, 0) + vget_lane_f16(sum_vec, 1) + 
                                vget_lane_f16(sum_vec, 2) + vget_lane_f16(sum_vec, 3);
                    c[row * N + col] = sum;
                }
            }
        }
    }
}

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {
    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;
    
    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);
                
                cactus_matmul_f16_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row
                );
            }
        });
} 


static void cactus_matmul_f32_worker(
    const float* a,
    const float* b_transposed,
    float* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int VECTOR_WIDTH = 4;
    const size_t K_aligned = (K / (VECTOR_WIDTH * 2)) * (VECTOR_WIDTH * 2);

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            float32x4_t accumulators[TILE_M][TILE_N];
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n)
                    accumulators[m][n] = vdupq_n_f32(0.0f);

            for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * 2) {
                float32x4_t a_vec_low[TILE_M], a_vec_high[TILE_M];
                float32x4_t b_vec_low[TILE_N], b_vec_high[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        a_vec_low[m] = vld1q_f32(&a[row * K + k_block]);
                        a_vec_high[m] = vld1q_f32(&a[row * K + k_block + VECTOR_WIDTH]);
                    } else {
                        a_vec_low[m] = vdupq_n_f32(0.0f);
                        a_vec_high[m] = vdupq_n_f32(0.0f);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        b_vec_low[n] = vld1q_f32(&b_transposed[col * K + k_block]);
                        b_vec_high[n] = vld1q_f32(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                    } else {
                        b_vec_low[n] = vdupq_n_f32(0.0f);
                        b_vec_high[n] = vdupq_n_f32(0.0f);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n) {
                        accumulators[m][n] = accum_f32_dot(accumulators[m][n], 
                                                          a_vec_low[m], a_vec_high[m],
                                                          b_vec_low[n], b_vec_high[n]);
                    }
            }

            for (size_t k_block = K_aligned; k_block < K; k_block += VECTOR_WIDTH) {
                size_t remaining = K - k_block;
                float32x4_t a_vec[TILE_M], b_vec[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        if (remaining >= VECTOR_WIDTH) {
                            a_vec[m] = vld1q_f32(&a[row * K + k_block]);
                        } else {
                            float tmp[VECTOR_WIDTH] = {0.0f};
                            memcpy(tmp, &a[row * K + k_block], remaining * sizeof(float));
                            a_vec[m] = vld1q_f32(tmp);
                        }
                    } else {
                        a_vec[m] = vdupq_n_f32(0.0f);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        if (remaining >= VECTOR_WIDTH) {
                            b_vec[n] = vld1q_f32(&b_transposed[col * K + k_block]);
                        } else {
                            float tmp[VECTOR_WIDTH] = {0.0f};
                            memcpy(tmp, &b_transposed[col * K + k_block], remaining * sizeof(float));
                            b_vec[n] = vld1q_f32(tmp);
                        }
                    } else {
                        b_vec[n] = vdupq_n_f32(0.0f);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n)
                        accumulators[m][n] = vfmaq_f32(accumulators[m][n], a_vec[m], b_vec[n]);
            }

            for (int m = 0; m < TILE_M; ++m) {
                size_t row = row_block + m;
                if (row >= M) continue;
                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col >= N) continue;
                    float sum = vaddvq_f32(accumulators[m][n]);
                    c[row * N + col] = sum;
                }
            }
        }
    }
}

void cactus_matmul_f32(
    const float* a,
    const float* b_transposed,
    float* c,
    size_t M,
    size_t K,
    size_t N
) {
    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;
    
    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);
                
                cactus_matmul_f32_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row
                );
            }
        });
}


#if !defined(__ARM_FEATURE_MATMUL_INT8)
static void cactus_matmul_int8_to_int32_worker(const int8_t* a, const int8_t* b_transposed, int32_t* c,
                                 size_t M, size_t K, size_t N, size_t start_row, size_t end_row) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int VECTOR_WIDTH = 16;
    constexpr int VECTOR_UNROLL = 2;
    const size_t K_aligned = (K / (VECTOR_WIDTH * VECTOR_UNROLL)) * (VECTOR_WIDTH * VECTOR_UNROLL);

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            int32x4_t accumulators[TILE_M][TILE_N];
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n)
                    accumulators[m][n] = vdupq_n_s32(0);

            for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * VECTOR_UNROLL) {
                int8x16_t a_vec[VECTOR_UNROLL][TILE_M];
                int8x16_t b_vec[VECTOR_UNROLL][TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        a_vec[0][m] = vld1q_s8(&a[row * K + k_block]);
                        a_vec[1][m] = vld1q_s8(&a[row * K + k_block + VECTOR_WIDTH]);
                    } else {
                        a_vec[0][m] = vdupq_n_s8(0);
                        a_vec[1][m] = vdupq_n_s8(0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        b_vec[0][n] = vld1q_s8(&b_transposed[col * K + k_block]);
                        b_vec[1][n] = vld1q_s8(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                    } else {
                        b_vec[0][n] = vdupq_n_s8(0);
                        b_vec[1][n] = vdupq_n_s8(0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m) {
                    for (int n = 0; n < TILE_N; ++n) {
                        accumulators[m][n] = accum_i8mm(accumulators[m][n], a_vec[0][m], b_vec[0][n]);
                        accumulators[m][n] = accum_i8mm(accumulators[m][n], a_vec[1][m], b_vec[1][n]);
                    }
                }
            }

            for (int m = 0; m < TILE_M; ++m) {
                size_t row = row_block + m;
                if (row >= M) continue;
                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col >= N) continue;

                    int32_t sum = vaddvq_s32(accumulators[m][n]);

                    for (size_t k = K_aligned; k < K; ++k) {
                        sum += static_cast<int32_t>(a[row * K + k]) *
                               static_cast<int32_t>(b_transposed[col * K + k]);
                    }

                    c[row * N + col] = sum;
                }
            }
        }
    }
}
#endif

#if !defined(__ARM_FEATURE_MATMUL_INT8)
void cactus_matmul_int8_to_int32(const int8_t* a, const int8_t* b_transposed, int32_t* c,
                                 size_t M, size_t K, size_t N) {
    if (M == 0) return;
    
    size_t num_threads = CactusThreading::compute_gemm_parallelism(M, K, N, sizeof(int8_t));
    
    if (num_threads == 1) {
        cactus_matmul_int8_to_int32_worker(a, b_transposed, c, M, K, N, 0, M);
        return;
    }
    
    size_t optimal_tile_m = std::min(CactusThreading::Thresholds::GEMM_TILE_M, (M + 1) / 2 * 2);
    size_t optimal_tile_n = std::min(CactusThreading::Thresholds::GEMM_TILE_N, (N + 1) / 2 * 2);

    size_t k_cache_footprint = K * sizeof(int8_t);
    if (k_cache_footprint > CactusThreading::Thresholds::L2_CACHE_SIZE) {
        optimal_tile_m = CactusThreading::Thresholds::GEMM_TILE_M_SMALL;
        optimal_tile_n = CactusThreading::Thresholds::GEMM_TILE_N_SMALL;
    }
    
    CactusThreading::parallel_for_2d_tiled(M, N, optimal_tile_m, optimal_tile_n,
        [=](size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
            
            constexpr int MICRO_TILE_M = 2;
            constexpr int MICRO_TILE_N = 2;
            constexpr int VECTOR_WIDTH = 16;
            constexpr int VECTOR_UNROLL = 4;
            const size_t K_aligned = (K / (VECTOR_WIDTH * VECTOR_UNROLL)) * (VECTOR_WIDTH * VECTOR_UNROLL);
            
            for (size_t row_block = row_start; row_block < row_end; row_block += MICRO_TILE_M) {
                for (size_t col_block = col_start; col_block < col_end; col_block += MICRO_TILE_N) {
                    int32x4_t accumulators[MICRO_TILE_M][MICRO_TILE_N];
                    for (int m = 0; m < MICRO_TILE_M; ++m)
                        for (int n = 0; n < MICRO_TILE_N; ++n)
                            accumulators[m][n] = vdupq_n_s32(0);

                    for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * VECTOR_UNROLL) {
                        int8x16_t a_vec[VECTOR_UNROLL][MICRO_TILE_M];
                        int8x16_t b_vec[VECTOR_UNROLL][MICRO_TILE_N];

                        for (int m = 0; m < MICRO_TILE_M; ++m) {
                            size_t row = row_block + m;
                            if (row < row_end) {
                                a_vec[0][m] = vld1q_s8(&a[row * K + k_block]);
                                a_vec[1][m] = vld1q_s8(&a[row * K + k_block + VECTOR_WIDTH]);
                                a_vec[2][m] = vld1q_s8(&a[row * K + k_block + VECTOR_WIDTH * 2]);
                                a_vec[3][m] = vld1q_s8(&a[row * K + k_block + VECTOR_WIDTH * 3]);
                            } else {
                                a_vec[0][m] = vdupq_n_s8(0);
                                a_vec[1][m] = vdupq_n_s8(0);
                                a_vec[2][m] = vdupq_n_s8(0);
                                a_vec[3][m] = vdupq_n_s8(0);
                            }
                        }

                        for (int n = 0; n < MICRO_TILE_N; ++n) {
                            size_t col = col_block + n;
                            if (col < col_end) {
                                b_vec[0][n] = vld1q_s8(&b_transposed[col * K + k_block]);
                                b_vec[1][n] = vld1q_s8(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                                b_vec[2][n] = vld1q_s8(&b_transposed[col * K + k_block + VECTOR_WIDTH * 2]);
                                b_vec[3][n] = vld1q_s8(&b_transposed[col * K + k_block + VECTOR_WIDTH * 3]);
                            } else {
                                b_vec[0][n] = vdupq_n_s8(0);
                                b_vec[1][n] = vdupq_n_s8(0);
                                b_vec[2][n] = vdupq_n_s8(0);
                                b_vec[3][n] = vdupq_n_s8(0);
                            }
                        }

                        accumulators[0][0] = accum_i8mm(accumulators[0][0], a_vec[0][0], b_vec[0][0]);
                        accumulators[0][1] = accum_i8mm(accumulators[0][1], a_vec[0][0], b_vec[0][1]);
                        accumulators[1][0] = accum_i8mm(accumulators[1][0], a_vec[0][1], b_vec[0][0]);
                        accumulators[1][1] = accum_i8mm(accumulators[1][1], a_vec[0][1], b_vec[0][1]);

                        accumulators[0][0] = accum_i8mm(accumulators[0][0], a_vec[1][0], b_vec[1][0]);
                        accumulators[0][1] = accum_i8mm(accumulators[0][1], a_vec[1][0], b_vec[1][1]);
                        accumulators[1][0] = accum_i8mm(accumulators[1][0], a_vec[1][1], b_vec[1][0]);
                        accumulators[1][1] = accum_i8mm(accumulators[1][1], a_vec[1][1], b_vec[1][1]);

                        accumulators[0][0] = accum_i8mm(accumulators[0][0], a_vec[2][0], b_vec[2][0]);
                        accumulators[0][1] = accum_i8mm(accumulators[0][1], a_vec[2][0], b_vec[2][1]);
                        accumulators[1][0] = accum_i8mm(accumulators[1][0], a_vec[2][1], b_vec[2][0]);
                        accumulators[1][1] = accum_i8mm(accumulators[1][1], a_vec[2][1], b_vec[2][1]);

                        accumulators[0][0] = accum_i8mm(accumulators[0][0], a_vec[3][0], b_vec[3][0]);
                        accumulators[0][1] = accum_i8mm(accumulators[0][1], a_vec[3][0], b_vec[3][1]);
                        accumulators[1][0] = accum_i8mm(accumulators[1][0], a_vec[3][1], b_vec[3][0]);
                        accumulators[1][1] = accum_i8mm(accumulators[1][1], a_vec[3][1], b_vec[3][1]);
                    }

                    for (size_t k_block = K_aligned; k_block < K; k_block += VECTOR_WIDTH) {
                        size_t remaining = K - k_block;
                        int8x16_t a_vec[MICRO_TILE_M], b_vec[MICRO_TILE_N];

                        for (int m = 0; m < MICRO_TILE_M; ++m) {
                            size_t row = row_block + m;
                            if (row < row_end) {
                                if (remaining >= VECTOR_WIDTH) {
                                    a_vec[m] = vld1q_s8(&a[row * K + k_block]);
                                } else {
                                    int8_t tmp[VECTOR_WIDTH] = {};
                                    memcpy(tmp, &a[row * K + k_block], remaining);
                                    a_vec[m] = vld1q_s8(tmp);
                                }
                            } else {
                                a_vec[m] = vdupq_n_s8(0);
                            }
                        }

                        for (int n = 0; n < MICRO_TILE_N; ++n) {
                            size_t col = col_block + n;
                            if (col < col_end) {
                                if (remaining >= VECTOR_WIDTH) {
                                    b_vec[n] = vld1q_s8(&b_transposed[col * K + k_block]);
                                } else {
                                    int8_t tmp[VECTOR_WIDTH] = {};
                                    memcpy(tmp, &b_transposed[col * K + k_block], remaining);
                                    b_vec[n] = vld1q_s8(tmp);
                                }
                            } else {
                                b_vec[n] = vdupq_n_s8(0);
                            }
                        }

                        for (int m = 0; m < MICRO_TILE_M; ++m)
                            for (int n = 0; n < MICRO_TILE_N; ++n)
                                accumulators[m][n] = accum_i8mm(accumulators[m][n], a_vec[m], b_vec[n]);
                    }
                    
                    for (int m = 0; m < MICRO_TILE_M; ++m) {
                        size_t row = row_block + m;
                        if (row >= row_end) continue;
                        for (int n = 0; n < MICRO_TILE_N; ++n) {
                            size_t col = col_block + n;
                            if (col >= col_end) continue;
                            int32_t sum = vaddvq_s32(accumulators[m][n]);
                            c[row * N + col] = sum;
                        }
                    }
                }
            }
        });
}
#endif


#if defined(__ARM_FEATURE_MATMUL_INT8)

static void cactus_matmul_int8_to_int32_smmla_worker(
    const int8_t* a, const int8_t* b_transposed, int32_t* output,
    size_t M, size_t K, size_t N, size_t start_row, size_t end_row,
    size_t start_col, size_t end_col
) {
    const size_t K_aligned = (K / 8) * 8;
    
    for (size_t row_block = start_row; row_block < end_row; row_block += 4) {
        for (size_t col_block = start_col; col_block < end_col; col_block += 4) {
            int32x4_t acc[2][2] = {{vdupq_n_s32(0), vdupq_n_s32(0)}, {vdupq_n_s32(0), vdupq_n_s32(0)}};
            
            for (size_t k_block = 0; k_block < K_aligned; k_block += 8) {
                if (row_block + 3 < M && col_block + 3 < N && k_block + 8 <= K) {
                    const size_t prefetch_distance = 64; 
                    if (k_block + prefetch_distance < K) {
                        __builtin_prefetch(&a[row_block * K + k_block + prefetch_distance], 0, 1);
                        __builtin_prefetch(&a[(row_block + 1) * K + k_block + prefetch_distance], 0, 1);
                        __builtin_prefetch(&a[(row_block + 2) * K + k_block + prefetch_distance], 0, 1);
                        __builtin_prefetch(&a[(row_block + 3) * K + k_block + prefetch_distance], 0, 1);
                    }
                    
                    const int8_t* a_base = &a[row_block * K + k_block];
                    int8x8_t a0 = vld1_s8(a_base);
                    int8x8_t a1 = vld1_s8(a_base + K);
                    int8x8_t a2 = vld1_s8(a_base + 2 * K);  
                    int8x8_t a3 = vld1_s8(a_base + 3 * K);
                    
                    const int8_t* b_base = &b_transposed[col_block * K + k_block];
                    int8x8_t b0 = vld1_s8(b_base);
                    int8x8_t b1 = vld1_s8(b_base + K);
                    int8x8_t b2 = vld1_s8(b_base + 2 * K);
                    int8x8_t b3 = vld1_s8(b_base + 3 * K);
                    
                    int8x16_t a_tiles[2] = {vcombine_s8(a0, a1), vcombine_s8(a2, a3)};
                    int8x16_t b_tiles[2] = {vcombine_s8(b0, b1), vcombine_s8(b2, b3)};
                    
                    asm volatile(
                        "smmla %0.4s, %4.16b, %6.16b\n"  // acc[0][0] += a_tiles[0] * b_tiles[0]
                        "smmla %1.4s, %4.16b, %7.16b\n"  // acc[0][1] += a_tiles[0] * b_tiles[1]
                        "smmla %2.4s, %5.16b, %6.16b\n"  // acc[1][0] += a_tiles[1] * b_tiles[0]
                        "smmla %3.4s, %5.16b, %7.16b"    // acc[1][1] += a_tiles[1] * b_tiles[1]
                        : "+w"(acc[0][0]), "+w"(acc[0][1]), "+w"(acc[1][0]), "+w"(acc[1][1])
                        : "w"(a_tiles[0]), "w"(a_tiles[1]), "w"(b_tiles[0]), "w"(b_tiles[1])
                    );
                } else {
                    for (size_t r = 0; r < 4; r += 2) {
                        for (size_t c = 0; c < 4; c += 2) {
                            if (row_block + r < M && col_block + c < N) {\
                                int8x16_t a_tile = vdupq_n_s8(0);
                                const int8_t* a_row_base = &a[(row_block + r) * K + k_block];
                                
                                if (row_block + r < M && k_block + 8 <= K) {
                                    int8x8_t a0 = vld1_s8(a_row_base);
                                    a_tile = vcombine_s8(a0, vdup_n_s8(0));

                                } else if (row_block + r < M) {

                                    int8_t temp_a[8] __attribute__((aligned(8))) = {0};
                                    size_t valid_k = std::min(8UL, K - k_block);
                                    memcpy(temp_a, a_row_base, valid_k);
                                    int8x8_t a0 = vld1_s8(temp_a);
                                    a_tile = vcombine_s8(a0, vdup_n_s8(0));
                                }
                                
                                if (row_block + r + 1 < M && k_block + 8 <= K) {
                                    int8x8_t a1 = vld1_s8(a_row_base + K);
                                    a_tile = vcombine_s8(vget_low_s8(a_tile), a1);

                                } else if (row_block + r + 1 < M) {
                                    int8_t temp_a[8] __attribute__((aligned(8))) = {0};
                                    size_t valid_k = std::min(8UL, K - k_block);
                                    memcpy(temp_a, a_row_base + K, valid_k);
                                    int8x8_t a1 = vld1_s8(temp_a);
                                    a_tile = vcombine_s8(vget_low_s8(a_tile), a1);
                                }
                                
                                int8x16_t b_tile = vdupq_n_s8(0);
                                const int8_t* b_col_base = &b_transposed[(col_block + c) * K + k_block];
                                
                                if (col_block + c < N && k_block + 8 <= K) {
                                    int8x8_t b0 = vld1_s8(b_col_base);
                                    b_tile = vcombine_s8(b0, vdup_n_s8(0));

                                } else if (col_block + c < N) {
                                    int8_t temp_b[8] __attribute__((aligned(8))) = {0};
                                    size_t valid_k = std::min(8UL, K - k_block);
                                    memcpy(temp_b, b_col_base, valid_k);
                                    int8x8_t b0 = vld1_s8(temp_b);
                                    b_tile = vcombine_s8(b0, vdup_n_s8(0));
                                }
                                
                                if (col_block + c + 1 < N && k_block + 8 <= K) {
                                    int8x8_t b1 = vld1_s8(b_col_base + K);
                                    b_tile = vcombine_s8(vget_low_s8(b_tile), b1);

                                } else if (col_block + c + 1 < N) {
                                    int8_t temp_b[8] __attribute__((aligned(8))) = {0};
                                    size_t valid_k = std::min(8UL, K - k_block);
                                    memcpy(temp_b, b_col_base + K, valid_k);
                                    int8x8_t b1 = vld1_s8(temp_b);
                                    b_tile = vcombine_s8(vget_low_s8(b_tile), b1);
                                }
                                
                                asm volatile(
                                    "smmla %0.4s, %1.16b, %2.16b"
                                    : "+w"(acc[r/2][c/2])
                                    : "w"(a_tile), "w"(b_tile)
                                );
                            }
                        }
                    }
                }
            }
            
            int32_t results[4][4];
            vst1q_s32(results[0], acc[0][0]);  // rows 0-1, cols 0-1
            vst1q_s32(results[1], acc[0][1]);  // rows 0-1, cols 2-3
            vst1q_s32(results[2], acc[1][0]);  // rows 2-3, cols 0-1
            vst1q_s32(results[3], acc[1][1]);  // rows 2-3, cols 2-3
            
            for (size_t k = K_aligned; k < K; k++) {
                for (size_t r = 0; r < 4 && row_block + r < M; r++) {
                    for (size_t c = 0; c < 4 && col_block + c < N; c++) {
                        size_t tile_idx = (r / 2) * 2 + (c / 2);
                        size_t elem_idx = (r % 2) * 2 + (c % 2);
                        results[tile_idx][elem_idx] += a[(row_block + r) * K + k] * b_transposed[(col_block + c) * K + k];
                    }
                }
            }
            
            for (size_t r = 0; r < 4 && row_block + r < M; r++) {
                for (size_t c = 0; c < 4 && col_block + c < N; c++) {
                    size_t tile_idx = (r / 2) * 2 + (c / 2);
                    size_t elem_idx = (r % 2) * 2 + (c % 2);
                    output[(row_block + r) * N + (col_block + c)] = results[tile_idx][elem_idx];
                }
            }
        }
    }
}

void cactus_matmul_int8_to_int32_i8mm(const int8_t* a, const int8_t* b_transposed, int32_t* c,
                                       size_t M, size_t K, size_t N) {
    if (M == 0) return;

    size_t total_ops = M * K * N;
    
    memset(c, 0, M * N * sizeof(int32_t));
    
    if (total_ops < CactusThreading::Thresholds::GEMM_SMALL) {
        cactus_matmul_int8_to_int32_smmla_worker(a, b_transposed, c, M, K, N, 0, M, 0, N);
        return;
    }
    
    size_t num_threads = CactusThreading::compute_gemm_parallelism(M, K, N, sizeof(int8_t));
    
    if (num_threads == 1) {
        cactus_matmul_int8_to_int32_smmla_worker(a, b_transposed, c, M, K, N, 0, M, 0, N);
        return;
    }
    
    size_t optimal_tile_m = std::min(CactusThreading::Thresholds::GEMM_TILE_M, (M + 3) / 4 * 4);
    size_t optimal_tile_n = std::min(CactusThreading::Thresholds::GEMM_TILE_N, (N + 3) / 4 * 4);

    size_t k_cache_footprint = K * sizeof(int8_t);
    if (k_cache_footprint > CactusThreading::Thresholds::L2_CACHE_SIZE) {
        optimal_tile_m = CactusThreading::Thresholds::GEMM_TILE_M_SMALL;
        optimal_tile_n = CactusThreading::Thresholds::GEMM_TILE_N_SMALL;
    }
    
    memset(c, 0, M * N * sizeof(int32_t));
    CactusThreading::parallel_for_2d_tiled(M, N, optimal_tile_m, optimal_tile_n,
        [=](size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
            cactus_matmul_int8_to_int32_smmla_worker(a, b_transposed, c, M, K, N, row_start, row_end, col_start, col_end);
        });
}

#endif // __ARM_FEATURE_MATMUL_INT8


static inline float quantize_row_fp16_to_int8(const __fp16* src, int8_t* dst, size_t K) {
    float32x4_t max_vec0 = vdupq_n_f32(0.0f);
    float32x4_t max_vec1 = vdupq_n_f32(0.0f);
    size_t k = 0;

    for (; k + 16 <= K; k += 16) {
        float16x8_t v0 = vld1q_f16(src + k);
        float16x8_t v1 = vld1q_f16(src + k + 8);
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_low_f16(v0))));
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_high_f16(v0))));
        max_vec1 = vmaxq_f32(max_vec1, vabsq_f32(vcvt_f32_f16(vget_low_f16(v1))));
        max_vec1 = vmaxq_f32(max_vec1, vabsq_f32(vcvt_f32_f16(vget_high_f16(v1))));
    }

    max_vec0 = vmaxq_f32(max_vec0, max_vec1);

    for (; k + 8 <= K; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_low_f16(vals))));
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_high_f16(vals))));
    }

    float max_abs = vmaxvq_f32(max_vec0);

    for (; k < K; k++) {
        float val = fabsf((float)src[k]);
        if (val > max_abs) max_abs = val;
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);

    k = 0;

    for (; k + 16 <= K; k += 16) {
        float16x8_t v0 = vld1q_f16(src + k);
        float16x8_t v1 = vld1q_f16(src + k + 8);

        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v0)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v0)), inv_scale_vec));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v1)), inv_scale_vec));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v1)), inv_scale_vec));

        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);
        int16x8_t s01 = vcombine_s16(s0, s1);
        int16x8_t s23 = vcombine_s16(s2, s3);
        int8x8_t r0 = vqmovn_s16(s01);
        int8x8_t r1 = vqmovn_s16(s23);
        vst1q_s8(dst + k, vcombine_s8(r0, r1));
    }

    for (; k + 8 <= K; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(vals)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(vals)), inv_scale_vec));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int8x8_t result = vqmovn_s16(vcombine_s16(s0, s1));
        vst1_s8(dst + k, result);
    }

    for (; k < K; k++) {
        float val = (float)src[k] * inv_scale;
        int32_t q = (int32_t)roundf(val);
        q = std::max(-128, std::min(127, q));
        dst[k] = (int8_t)q;
    }

    return scale;
}

static void matmul_int8_grouped_small_m(
    const __fp16* A,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size,
    size_t num_groups
) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t num_m_tiles = (M + TILE_M - 1) / TILE_M;

    const size_t K_aligned = ((K + 31) / 32) * 32;

    CactusThreading::parallel_for(num_m_tiles, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t tile_start, size_t tile_end) {
            std::vector<int8_t> a_quant_storage(TILE_M * K_aligned);
            int8_t* a_quant[TILE_M];
            
            for (size_t i = 0; i < TILE_M; i++) {
                a_quant[i] = a_quant_storage.data() + i * K_aligned;
            }
            float a_scales[TILE_M];

            for (size_t tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
                size_t m_start = tile_idx * TILE_M;
                size_t m_end = std::min(m_start + TILE_M, M);
                size_t actual_m = m_end - m_start;

                for (size_t mi = 0; mi < actual_m; mi++) {
                    a_scales[mi] = quantize_row_fp16_to_int8(
                        A + (m_start + mi) * K, a_quant[mi], K);
                }

                size_t n = 0;
                for (; n + TILE_N <= N; n += TILE_N) {
                    float acc[TILE_M][TILE_N] = {{0}};

                    for (size_t g = 0; g < num_groups; g++) {
                        size_t k_base = g * group_size;

                        int8x16_t b_vec0[TILE_N], b_vec1[TILE_N];
                        float b_scale[TILE_N];
                        for (size_t ni = 0; ni < TILE_N; ni++) {
                            b_vec0[ni] = vld1q_s8(B + (n + ni) * K + k_base);
                            b_vec1[ni] = vld1q_s8(B + (n + ni) * K + k_base + 16);
                            b_scale[ni] = (float)B_scales[(n + ni) * num_groups + g];
                        }

                        for (size_t mi = 0; mi < actual_m; mi++) {
                            int8x16_t a_vec0 = vld1q_s8(a_quant[mi] + k_base);
                            int8x16_t a_vec1 = vld1q_s8(a_quant[mi] + k_base + 16);
                            float combined_base = a_scales[mi];

                            for (size_t ni = 0; ni < TILE_N; ni++) {
                                int32x4_t sum = vdupq_n_s32(0);
                                sum = vdotq_s32(sum, a_vec0, b_vec0[ni]);
                                sum = vdotq_s32(sum, a_vec1, b_vec1[ni]);
                                acc[mi][ni] += (float)vaddvq_s32(sum) * (combined_base * b_scale[ni]);
                            }
                        }
                    }

                    for (size_t mi = 0; mi < actual_m; mi++) {
                        for (size_t ni = 0; ni < TILE_N; ni++) {
                            C[(m_start + mi) * N + n + ni] = (__fp16)acc[mi][ni];
                        }
                    }
                }

                for (; n < N; n++) {
                    for (size_t mi = 0; mi < actual_m; mi++) {
                        float acc = 0.0f;
                        for (size_t g = 0; g < num_groups; g++) {
                            size_t k_base = g * group_size;
                            float b_scale = (float)B_scales[n * num_groups + g];

                            int8x16_t a_vec0 = vld1q_s8(a_quant[mi] + k_base);
                            int8x16_t a_vec1 = vld1q_s8(a_quant[mi] + k_base + 16);
                            int8x16_t b_vec0 = vld1q_s8(B + n * K + k_base);
                            int8x16_t b_vec1 = vld1q_s8(B + n * K + k_base + 16);

                            int32x4_t sum = vdupq_n_s32(0);
                            sum = vdotq_s32(sum, a_vec0, b_vec0);
                            sum = vdotq_s32(sum, a_vec1, b_vec1);
                            acc += (float)vaddvq_s32(sum) * (a_scales[mi] * b_scale);
                        }
                        C[(m_start + mi) * N + n] = (__fp16)acc;
                    }
                }
            }
        });
}

static void matmul_int8_grouped_large_m(
    const __fp16* A,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size,
    size_t num_groups
) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t K_aligned = ((K + 31) / 32) * 32;

    std::vector<int8_t> A_quant(M * K_aligned);
    std::vector<float> A_scales(M);

    CactusThreading::parallel_for(M, CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t m_start, size_t m_end) {
            for (size_t m = m_start; m < m_end; m++) {
                A_scales[m] = quantize_row_fp16_to_int8(
                    A + m * K, A_quant.data() + m * K_aligned, K);
            }
        });

    CactusThreading::parallel_for_2d_tiled(M, N, TILE_M, TILE_N,
        [&](size_t m_start, size_t m_end, size_t n_start, size_t n_end) {
            size_t actual_m = m_end - m_start;
            size_t actual_n = n_end - n_start;

            float acc[TILE_M][TILE_N] = {{0}};

            for (size_t g = 0; g < num_groups; g++) {
                size_t k_base = g * group_size;

                int8x16_t b_vec0[TILE_N], b_vec1[TILE_N];
                float b_scale[TILE_N];
                for (size_t ni = 0; ni < actual_n; ni++) {
                    b_vec0[ni] = vld1q_s8(B + (n_start + ni) * K + k_base);
                    b_vec1[ni] = vld1q_s8(B + (n_start + ni) * K + k_base + 16);
                    b_scale[ni] = (float)B_scales[(n_start + ni) * num_groups + g];
                }

                for (size_t mi = 0; mi < actual_m; mi++) {
                    const int8_t* a_ptr = A_quant.data() + (m_start + mi) * K_aligned + k_base;
                    int8x16_t a_vec0 = vld1q_s8(a_ptr);
                    int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                    float a_scale = A_scales[m_start + mi];

                    for (size_t ni = 0; ni < actual_n; ni++) {
                        int32x4_t sum = vdupq_n_s32(0);
                        sum = vdotq_s32(sum, a_vec0, b_vec0[ni]);
                        sum = vdotq_s32(sum, a_vec1, b_vec1[ni]);
                        acc[mi][ni] += (float)vaddvq_s32(sum) * (a_scale * b_scale[ni]);
                    }
                }
            }

            for (size_t mi = 0; mi < actual_m; mi++) {
                for (size_t ni = 0; ni < actual_n; ni++) {
                    C[(m_start + mi) * N + (n_start + ni)] = (__fp16)acc[mi][ni];
                }
            }
        });
}

void cactus_matmul_int8_grouped(
    const __fp16* A,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    const size_t num_groups = K / group_size;

    constexpr size_t M_THRESHOLD = 16;

    if (M <= M_THRESHOLD) {
        matmul_int8_grouped_small_m(A, B, B_scales, C, M, K, N, group_size, num_groups);
    } else {
        matmul_int8_grouped_large_m(A, B, B_scales, C, M, K, N, group_size, num_groups);
    }
}