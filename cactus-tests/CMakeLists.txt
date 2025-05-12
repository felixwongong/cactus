cmake_minimum_required(VERSION 3.10)
project(cactus_test)

# Enable debugging symbols and AddressSanitizer
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-g -O0 -fsanitize=address -fno-omit-frame-pointer)
    link_libraries(-fsanitize=address)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus
)

# Find Metal framework
find_library(METAL_LIBRARY Metal)
find_library(METALKIT_LIBRARY MetalKit)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(ACCELERATE_LIBRARY Accelerate)

# Add source files
add_executable(cactus_test
    test.cpp
    ../cactus/cactus.cpp
    ../cactus/cactus_context.cpp
    ../cactus/cactus_loader.cpp
    ../cactus/cactus_completion.cpp
    ../cactus/cactus_utils.cpp
    ../cactus/cactus_embedding.cpp
    ../cactus/cactus_lora.cpp
    ../cactus/llama.cpp
    ../cactus/llama-context.cpp
    ../cactus/llama-model.cpp
    ../cactus/llama-chat.cpp
    ../cactus/llama-sampling.cpp
    ../cactus/llama-grammar.cpp
    ../cactus/llama-vocab.cpp
    ../cactus/llama-arch.cpp
    ../cactus/llama-kv-cache.cpp
    ../cactus/llama-model-loader.cpp
    ../cactus/llama-impl.cpp
    ../cactus/llama-mmap.cpp
    ../cactus/llama-memory.cpp
    ../cactus/llama-io.cpp
    ../cactus/llama-cparams.cpp
    ../cactus/llama-hparams.cpp
    ../cactus/llama-batch.cpp
    ../cactus/llama-adapter.cpp
    ../cactus/llama-graph.cpp
    ../cactus/ggml.c
    ../cactus/ggml-alloc.c
    ../cactus/ggml-backend.cpp
    ../cactus/ggml-quants.c
    ../cactus/ggml-opt.cpp
    ../cactus/ggml-metal.m
    ../cactus/ggml-cpu.c
    ../cactus/ggml-cpu.cpp
    ../cactus/ggml-cpu-quants.c
    ../cactus/ggml-cpu-aarch64.cpp
    ../cactus/ggml-cpu-traits.cpp
    ../cactus/ggml-threading.cpp
    ../cactus/ggml-backend-reg.cpp
    ../cactus/gguf.cpp
    ../cactus/common.cpp
    ../cactus/chat.cpp
    ../cactus/log.cpp
    ../cactus/sampling.cpp
    ../cactus/json-schema-to-grammar.cpp
    ../cactus/unicode.cpp
    ../cactus/unicode-data.cpp
    ../cactus/sgemm.cpp
    ../cactus/amx/amx.cpp
    ../cactus/amx/mmq.cpp
    ../cactus/cactus_bench.cpp
    ../cactus/cactus_chat.cpp
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