cmake_minimum_required(VERSION 3.10)
project(cactus_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CACTUS_CORE_SOURCES
    cactus.cpp
    cactus_context.cpp
    cactus_loader.cpp
    cactus_completion.cpp
    cactus_utils.cpp
    cactus_embedding.cpp
    cactus_lora.cpp
    cactus_ffi.cpp
    llama.cpp
    llama-context.cpp
    llama-model.cpp
    llama-chat.cpp
    llama-sampling.cpp
    llama-grammar.cpp
    llama-vocab.cpp
    llama-arch.cpp
    llama-kv-cache.cpp
    llama-model-loader.cpp
    llama-impl.cpp
    llama-mmap.cpp
    llama-memory.cpp
    llama-io.cpp
    llama-cparams.cpp
    llama-hparams.cpp
    llama-batch.cpp
    llama-adapter.cpp
    llama-graph.cpp
    ggml.c
    ggml-alloc.c
    ggml-backend.cpp
    ggml-quants.c
    ggml-opt.cpp
    ggml-threading.cpp
    ggml-backend-reg.cpp
    gguf.cpp
    common.cpp
    chat.cpp
    log.cpp
    sampling.cpp
    json-schema-to-grammar.cpp
    unicode.cpp
    unicode-data.cpp
    cactus_bench.cpp
    cactus_chat.cpp
    # ggml-cpu sources
    ggml-cpu/amx/amx.cpp
    ggml-cpu/amx/mmq.cpp
    ggml-cpu/ggml-cpu.c
    ggml-cpu/ggml-cpu.cpp
    ggml-cpu/ggml-cpu-aarch64.cpp
    ggml-cpu/ggml-cpu-quants.c
    ggml-cpu/ggml-cpu-traits.cpp
    ggml-cpu/unary-ops.cpp
    ggml-cpu/binary-ops.cpp
    ggml-cpu/sgemm.cpp
    ggml-cpu/vec.cpp
    ggml-cpu/ops.cpp
)

# Conditionally add platform-specific sources
if(APPLE)
    list(APPEND CACTUS_CORE_SOURCES ggml-metal.m)
endif()

add_library(cactus_core_lib OBJECT ${CACTUS_CORE_SOURCES})

target_include_directories(cactus_core_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu>
    # For potential installation:
    # $<INSTALL_INTERFACE:include> 
)

# Find and link required Apple frameworks
# Assuming this library might be used on Apple platforms due to ggml-metal.m
if(APPLE)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(ACCELERATE_FRAMEWORK Accelerate) # Often found as a framework by this name

    if(METAL_LIBRARY AND METALKIT_LIBRARY AND FOUNDATION_LIBRARY AND ACCELERATE_FRAMEWORK)
        target_link_libraries(cactus_core_lib PUBLIC
            ${METAL_LIBRARY}
            ${METALKIT_LIBRARY}
            ${FOUNDATION_LIBRARY}
            ${ACCELERATE_FRAMEWORK}
        )
        target_compile_definitions(cactus_core_lib PUBLIC
            LM_GGML_USE_METAL
            LM_GGML_METAL_USE_BF16
            LM_GGML_USE_ACCELERATE
        )
    else()
        message(WARNING "One or more Apple frameworks (Metal, MetalKit, Foundation, Accelerate) not found during cactus_core_lib configuration. Metal/Accelerate support may be affected if targeting Apple platforms.")
    endif()
endif()

# Common compile definitions
target_compile_definitions(cactus_core_lib PUBLIC
    LM_GGML_USE_CPU
)

# Example for installation (optional)
# include(GNUInstallDirs)
# install(TARGETS cactus_core_lib
#     EXPORT cactus_core_libTargets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # If you list public headers
# )
# install(EXPORT cactus_core_libTargets
#   FILE cactus_core_libTargets.cmake
#   NAMESPACE cactus::
#   DESTINATION lib/cmake/cactus_core_lib
# )
# To list public headers for installation:
# target_sources(cactus_core_lib PUBLIC FILE_SET HEADERS FILES your_public_header.h)
# And then add PUBLIC_HEADER_DESTINATION to install(TARGETS ...) 