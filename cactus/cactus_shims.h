#ifndef CACTUS_SHIMS_H
#define CACTUS_SHIMS_H

// Core llama.cpp / ggml headers that shims depend on
#include "ggml.h"        // Provides lm_ggml_... types and functions
#include "gguf.h"        // Provides lm_gguf_... types and functions
#include "ggml-common.h" // For LM_GGML_ASSERT or GGML_ASSERT_IMPL, LM_GGML_PAD, LM_GGML_ABORT, LM_GGML_UNUSED
#include "ggml-cpp.h"    // For lm_ggml_context_ptr, lm_gguf_context_ptr
#include "ggml-backend.h" // For lm_ggml_backend_sched_new original definition

// --- Compatibility Shims for Vendored Headers (clip.h, mtmd.h, clip-impl.h, clip.cpp) ---

// 1. For ggml_log_level and its constants
//    Make `ggml_log_level` an alias for `lm_ggml_log_level` via define.
//    This way, `enum ggml_log_level` in vendored code becomes `enum lm_ggml_log_level`.
#define ggml_log_level lm_ggml_log_level
//    Define constants to map to LM_ versions.
#define GGML_LOG_LEVEL_NONE  LM_GGML_LOG_LEVEL_NONE
#define GGML_LOG_LEVEL_DEBUG LM_GGML_LOG_LEVEL_DEBUG
#define GGML_LOG_LEVEL_INFO  LM_GGML_LOG_LEVEL_INFO
#define GGML_LOG_LEVEL_WARN  LM_GGML_LOG_LEVEL_WARN
#define GGML_LOG_LEVEL_ERROR LM_GGML_LOG_LEVEL_ERROR
#define GGML_LOG_LEVEL_CONT  LM_GGML_LOG_LEVEL_CONT

// 2. For ggml_log_callback
//    This typedef uses `enum lm_ggml_log_level` in its definition.
//    The define for ggml_log_level above makes types compatible.
typedef lm_ggml_log_callback ggml_log_callback;

// 3. For GGUF_TYPE and its constants
//    Make `gguf_type` an alias for `lm_gguf_type`.
//    This way, `enum gguf_type` in vendored code becomes `enum lm_gguf_type`.
#define gguf_type lm_gguf_type
//    Define constants to map to LM_ versions.
#define GGUF_TYPE_UINT8   LM_GGUF_TYPE_UINT8
#define GGUF_TYPE_INT8    LM_GGUF_TYPE_INT8
#define GGUF_TYPE_UINT16  LM_GGUF_TYPE_UINT16
#define GGUF_TYPE_INT16   LM_GGUF_TYPE_INT16
#define GGUF_TYPE_UINT32  LM_GGUF_TYPE_UINT32
#define GGUF_TYPE_INT32   LM_GGUF_TYPE_INT32
#define GGUF_TYPE_FLOAT32 LM_GGUF_TYPE_FLOAT32
#define GGUF_TYPE_BOOL    LM_GGUF_TYPE_BOOL
#define GGUF_TYPE_STRING  LM_GGUF_TYPE_STRING
#define GGUF_TYPE_ARRAY   LM_GGUF_TYPE_ARRAY
#define GGUF_TYPE_UINT64  LM_GGUF_TYPE_UINT64
#define GGUF_TYPE_INT64   LM_GGUF_TYPE_INT64
#define GGUF_TYPE_FLOAT64 LM_GGUF_TYPE_FLOAT64

// Add defines for GGML types used in clip.cpp
#define GGML_TYPE_F32 LM_GGML_TYPE_F32
#define GGML_TYPE_I32 LM_GGML_TYPE_I32
// Add other GGML_TYPE_
#define ggml_type lm_ggml_type

// 4. For GGML_ASSERT, GGML_PAD, GGML_ABORT, GGML_UNUSED
#if defined(LM_GGML_ASSERT)
    #define GGML_ASSERT LM_GGML_ASSERT
#elif defined(GGML_ASSERT_IMPL) // ggml-common.h might define GGML_ASSERT_IMPL
    #define GGML_ASSERT GGML_ASSERT_IMPL
#else
    // Fallback if not found (should be in ggml-common.h)
    #include <cassert>
    #define GGML_ASSERT(x) assert(x)
#endif

#if defined(LM_GGML_PAD)
    #define GGML_PAD LM_GGML_PAD
#else
    // Fallback definition for GGML_PAD
    #define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))
#endif

#if defined(LM_GGML_ABORT)
    #define GGML_ABORT LM_GGML_ABORT
#else
    // Fallback definition for GGML_ABORT
    #include <cstdio>
    #include <cstdlib>
    #define GGML_ABORT(...) do { fprintf(stderr, "Abort: %s\n", #__VA_ARGS__); abort(); } while (0)
#endif

#if defined(LM_GGML_UNUSED)
    #define GGML_UNUSED LM_GGML_UNUSED
#else
    #define GGML_UNUSED(x) (void)(x) // Common fallback
#endif

// 5. Defines for struct names (so `struct ggml_tensor` becomes `struct lm_ggml_tensor`)
#define gguf_context lm_gguf_context
#define ggml_context lm_ggml_context
#define ggml_tensor lm_ggml_tensor

// For struct tags used like `struct ggml_cgraph` or `struct ggml_init_params`
#define ggml_cgraph lm_ggml_cgraph
#define ggml_init_params lm_ggml_init_params
#define gguf_init_params lm_gguf_init_params // For `struct gguf_init_params ...`

// For pointer types from ggml-cpp.h
#define gguf_context_ptr lm_gguf_context_ptr
#define ggml_context_ptr lm_ggml_context_ptr

// Backend related struct names
#define ggml_backend_t lm_ggml_backend_t
#define ggml_backend_buffer_type_t lm_ggml_backend_buffer_type_t
#define ggml_backend_sched_t lm_ggml_backend_sched_t
#define ggml_backend_sched_ptr lm_ggml_backend_sched_ptr
#define ggml_backend_buffer_ptr lm_ggml_backend_buffer_ptr
#define ggml_backend_dev_t lm_ggml_backend_dev_t
#define ggml_backend_reg_t lm_ggml_backend_reg_t
#define ggml_backend_set_n_threads_t lm_ggml_backend_set_n_threads_t 


// 6. Defines for GGUF functions (from clip-impl.h and clip.cpp errors)
#define gguf_get_kv_type lm_gguf_get_kv_type
#define gguf_get_val_str lm_gguf_get_val_str
#define gguf_get_arr_type lm_gguf_get_arr_type
#define gguf_get_arr_n lm_gguf_get_arr_n
#define gguf_get_arr_data lm_gguf_get_arr_data
#define gguf_get_arr_str lm_gguf_get_arr_str
#define gguf_get_val_data lm_gguf_get_val_data
#define gguf_init_from_file lm_gguf_init_from_file
#define gguf_get_n_tensors lm_gguf_get_n_tensors
#define gguf_get_n_kv lm_gguf_get_n_kv
#define gguf_get_tensor_name lm_gguf_get_tensor_name
#define gguf_get_tensor_offset lm_gguf_get_tensor_offset
#define gguf_get_tensor_type lm_gguf_get_tensor_type // lm_gguf_get_tensor_type returns lm_gguf_type, which is fine due to #define gguf_type lm_gguf_type
#define gguf_get_version lm_gguf_get_version
#define gguf_get_alignment lm_gguf_get_alignment
#define gguf_find_key lm_gguf_find_key
#define gguf_get_val_bool lm_gguf_get_val_bool
#define gguf_get_val_i32 lm_gguf_get_val_i32
#define gguf_get_val_u32 lm_gguf_get_val_u32
#define gguf_get_val_f32 lm_gguf_get_val_f32
#define gguf_get_data_offset lm_gguf_get_data_offset
#define ggml_n_dims lm_ggml_n_dims // Added for ggml_n_dims

// 7. Defines for GGML functions (primarily for clip.cpp model loading and graph building)
#define ggml_init lm_ggml_init
#define ggml_get_tensor lm_ggml_get_tensor
#define ggml_nbytes lm_ggml_nbytes
#define ggml_tensor_overhead lm_ggml_tensor_overhead
#define ggml_graph_overhead lm_ggml_graph_overhead 
#define ggml_dup_tensor lm_ggml_dup_tensor
#define ggml_set_name lm_ggml_set_name
#define ggml_new_graph lm_ggml_new_graph
#define ggml_new_tensor_1d lm_ggml_new_tensor_1d
#define ggml_new_tensor_2d lm_ggml_new_tensor_2d
#define ggml_new_tensor_3d lm_ggml_new_tensor_3d
#define ggml_set_input lm_ggml_set_input
#define ggml_build_forward_expand lm_ggml_build_forward_expand
#define ggml_conv_2d lm_ggml_conv_2d
#define ggml_reshape_2d lm_ggml_reshape_2d
#define ggml_reshape_3d lm_ggml_reshape_3d
#define ggml_reshape_4d lm_ggml_reshape_4d
#define ggml_transpose lm_ggml_transpose
#define ggml_cont lm_ggml_cont
#define ggml_add lm_ggml_add
#define ggml_mul_mat lm_ggml_mul_mat
#define ggml_norm lm_ggml_norm
#define ggml_rms_norm lm_ggml_rms_norm
#define ggml_mul lm_ggml_mul
#define ggml_gelu lm_ggml_gelu
#define ggml_gelu_quick lm_ggml_gelu_quick
#define ggml_silu lm_ggml_silu
#define ggml_permute lm_ggml_permute
#define ggml_get_rows lm_ggml_get_rows
#define ggml_concat lm_ggml_concat
#define ggml_soft_max_ext lm_ggml_soft_max_ext
#define ggml_cont_2d lm_ggml_cont_2d 
#define ggml_pool_2d lm_ggml_pool_2d
#define ggml_im2col lm_ggml_im2col
#define ggml_view_2d lm_ggml_view_2d
#define ggml_view_3d lm_ggml_view_3d
#define ggml_rope_ext lm_ggml_rope_ext
#define ggml_rope_multi lm_ggml_rope_multi
#define ggml_hardswish lm_ggml_hardswish
#define ggml_hardsigmoid lm_ggml_hardsigmoid
#define ggml_relu lm_ggml_relu
#define ggml_conv_2d_dw lm_ggml_conv_2d_dw
#define ggml_gelu_inplace lm_ggml_gelu_inplace
#define ggml_silu_inplace lm_ggml_silu_inplace
#define ggml_element_size lm_ggml_element_size
#define ggml_row_size lm_ggml_row_size
#define ggml_type_name lm_ggml_type_name
#define ggml_nelements lm_ggml_nelements
#define ggml_get_mem_size lm_ggml_get_mem_size // from clip_model_loader
#define ggml_time_ms lm_ggml_time_ms // Added shim for ggml_time_ms
#define ggml_scale lm_ggml_scale // Added for ggml_scale

// GGML Ops
#define GGML_OP_POOL_AVG LM_GGML_OP_POOL_AVG
#define GGML_OP_CONV_2D LM_GGML_OP_CONV_2D 
#define GGML_OP_RESHAPE LM_GGML_OP_RESHAPE
#define GGML_OP_TRANSPOSE LM_GGML_OP_TRANSPOSE
#define GGML_OP_ADD LM_GGML_OP_ADD
#define GGML_OP_MUL_MAT LM_GGML_OP_MUL_MAT
#define GGML_OP_NORM LM_GGML_OP_NORM
#define GGML_OP_RMS_NORM LM_GGML_OP_RMS_NORM
#define GGML_OP_MUL LM_GGML_OP_MUL
#define GGML_OP_GELU LM_GGML_OP_GELU
#define GGML_OP_SILU LM_GGML_OP_SILU
#define GGML_OP_PERMUTE LM_GGML_OP_PERMUTE
#define GGML_OP_GET_ROWS LM_GGML_OP_GET_ROWS
#define GGML_OP_CONCAT LM_GGML_OP_CONCAT
#define GGML_OP_SOFT_MAX LM_GGML_OP_SOFT_MAX 

// Backend related function defines
#define ggml_backend_init_by_type lm_ggml_backend_init_by_type
#define GGML_BACKEND_DEVICE_TYPE_CPU LM_GGML_BACKEND_DEVICE_TYPE_CPU
#define GGML_BACKEND_DEVICE_TYPE_GPU LM_GGML_BACKEND_DEVICE_TYPE_GPU
#define ggml_backend_name lm_ggml_backend_name
#define ggml_backend_get_default_buffer_type lm_ggml_backend_get_default_buffer_type

// Special shim for ggml_backend_sched_new to handle API mismatch (6 args in clip.cpp vs 5 in cactus's ggml-backend.h)
#undef ggml_backend_sched_new // Undefine previous simple alias if it exists
#define ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, ...) lm_ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel)

#define ggml_backend_free lm_ggml_backend_free
#define ggml_backend_alloc_ctx_tensors_from_buft lm_ggml_backend_alloc_ctx_tensors_from_buft
#define ggml_backend_buffer_set_usage lm_ggml_backend_buffer_set_usage
#define GGML_BACKEND_BUFFER_USAGE_WEIGHTS LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS
#define ggml_backend_buft_is_host lm_ggml_backend_buft_is_host
#define ggml_backend_tensor_set lm_ggml_backend_tensor_set
#define ggml_backend_sched_reserve lm_ggml_backend_sched_reserve
#define ggml_backend_sched_get_buffer_size lm_ggml_backend_sched_get_buffer_size
#define ggml_backend_buft_name lm_ggml_backend_buft_name
#define ggml_backend_sched_reset lm_ggml_backend_sched_reset
#define ggml_backend_sched_alloc_graph lm_ggml_backend_sched_alloc_graph
#define ggml_graph_get_tensor lm_ggml_graph_get_tensor
#define ggml_backend_get_device lm_ggml_backend_get_device
#define ggml_backend_dev_backend_reg lm_ggml_backend_dev_backend_reg
#define ggml_backend_reg_get_proc_address lm_ggml_backend_reg_get_proc_address
#define ggml_backend_sched_graph_compute lm_ggml_backend_sched_graph_compute
#define ggml_graph_node lm_ggml_graph_node
#define ggml_backend_tensor_get lm_ggml_backend_tensor_get
#define GGML_STATUS_SUCCESS LM_GGML_STATUS_SUCCESS
#define ggml_tensor_flags lm_ggml_tensor_flags 
#define GGML_TENSOR_FLAG_INPUT LM_GGML_TENSOR_FLAG_INPUT
#define GGML_ROPE_TYPE_VISION LM_GGML_ROPE_TYPE_VISION

#endif /* CACTUS_SHIMS_H */ 