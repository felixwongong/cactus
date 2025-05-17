cmake_minimum_required(VERSION 3.10)
project(cactus_test)

# Enable debugging symbols and AddressSanitizer
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-g -O0 -fsanitize=address -fno-omit-frame-pointer)
    link_libraries(-fsanitize=address)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define cactus source directory
set(CACTUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cactus)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CACTUS_SOURCE_DIR}
    ${CACTUS_SOURCE_DIR}/ggml-cpu
)

# Find Metal framework
find_library(METAL_LIBRARY Metal)
find_library(METALKIT_LIBRARY MetalKit)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(ACCELERATE_LIBRARY Accelerate)

# Add source files
add_executable(cactus_test
    test.cpp
    test_core_api.cpp
    test_ffi_api.cpp
    ${CACTUS_SOURCE_DIR}/cactus.cpp
    ${CACTUS_SOURCE_DIR}/cactus_context.cpp
    ${CACTUS_SOURCE_DIR}/cactus_loader.cpp
    ${CACTUS_SOURCE_DIR}/cactus_completion.cpp
    ${CACTUS_SOURCE_DIR}/cactus_utils.cpp
    ${CACTUS_SOURCE_DIR}/cactus_embedding.cpp
    ${CACTUS_SOURCE_DIR}/cactus_lora.cpp
    ${CACTUS_SOURCE_DIR}/cactus_ffi.cpp
    ${CACTUS_SOURCE_DIR}/llama.cpp
    ${CACTUS_SOURCE_DIR}/llama-context.cpp
    ${CACTUS_SOURCE_DIR}/llama-model.cpp
    ${CACTUS_SOURCE_DIR}/llama-chat.cpp
    ${CACTUS_SOURCE_DIR}/llama-sampling.cpp
    ${CACTUS_SOURCE_DIR}/llama-grammar.cpp
    ${CACTUS_SOURCE_DIR}/llama-vocab.cpp
    ${CACTUS_SOURCE_DIR}/llama-arch.cpp
    ${CACTUS_SOURCE_DIR}/llama-kv-cache.cpp
    ${CACTUS_SOURCE_DIR}/llama-model-loader.cpp
    ${CACTUS_SOURCE_DIR}/llama-impl.cpp
    ${CACTUS_SOURCE_DIR}/llama-mmap.cpp
    ${CACTUS_SOURCE_DIR}/llama-memory.cpp
    ${CACTUS_SOURCE_DIR}/llama-io.cpp
    ${CACTUS_SOURCE_DIR}/llama-cparams.cpp
    ${CACTUS_SOURCE_DIR}/llama-hparams.cpp
    ${CACTUS_SOURCE_DIR}/llama-batch.cpp
    ${CACTUS_SOURCE_DIR}/llama-adapter.cpp
    ${CACTUS_SOURCE_DIR}/llama-graph.cpp
    ${CACTUS_SOURCE_DIR}/ggml.c
    ${CACTUS_SOURCE_DIR}/ggml-alloc.c
    ${CACTUS_SOURCE_DIR}/ggml-backend.cpp
    ${CACTUS_SOURCE_DIR}/ggml-quants.c
    ${CACTUS_SOURCE_DIR}/ggml-opt.cpp
    ${CACTUS_SOURCE_DIR}/ggml-metal.m
    ${CACTUS_SOURCE_DIR}/ggml-cpu/amx/amx.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/amx/mmq.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/ggml-cpu.c
    ${CACTUS_SOURCE_DIR}/ggml-cpu/ggml-cpu.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/ggml-cpu-aarch64.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/ggml-cpu-quants.c
    ${CACTUS_SOURCE_DIR}/ggml-cpu/ggml-cpu-traits.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/unary-ops.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/binary-ops.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/sgemm.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/vec.cpp
    ${CACTUS_SOURCE_DIR}/ggml-cpu/ops.cpp
    ${CACTUS_SOURCE_DIR}/ggml-threading.cpp
    ${CACTUS_SOURCE_DIR}/ggml-backend-reg.cpp
    ${CACTUS_SOURCE_DIR}/gguf.cpp
    ${CACTUS_SOURCE_DIR}/common.cpp
    ${CACTUS_SOURCE_DIR}/chat.cpp
    ${CACTUS_SOURCE_DIR}/log.cpp
    ${CACTUS_SOURCE_DIR}/sampling.cpp
    ${CACTUS_SOURCE_DIR}/json-schema-to-grammar.cpp
    ${CACTUS_SOURCE_DIR}/unicode.cpp
    ${CACTUS_SOURCE_DIR}/unicode-data.cpp
    ${CACTUS_SOURCE_DIR}/cactus_bench.cpp
    ${CACTUS_SOURCE_DIR}/cactus_chat.cpp
)

# Link libraries
target_link_libraries(cactus_test
    PRIVATE
      ${METAL_LIBRARY}
      ${METALKIT_LIBRARY}
      ${FOUNDATION_LIBRARY}
      ${ACCELERATE_LIBRARY}
)

# Copy model file to build directory
add_custom_command(TARGET cactus_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/llm.gguf
    ${CMAKE_CURRENT_BINARY_DIR}/llm.gguf
)

# Ensure CPU backend is enabled for GGML
target_compile_definitions(cactus_test
  PRIVATE
    LM_GGML_USE_CPU
    LM_GGML_USE_ACCELERATE
    LM_GGML_USE_METAL
    LM_GGML_METAL_USE_BF16
)