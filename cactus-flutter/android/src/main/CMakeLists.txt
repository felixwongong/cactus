cmake_minimum_required(VERSION 3.10)

project(cactus)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../../cactus)

# Add the cactus_core_lib defined in the main cactus directory
add_subdirectory(${SOURCE_DIR} ${CMAKE_BINARY_DIR}/cactus_core_build_flutter_android)

# Include directories for JNI or other Android-specific headers if they exist in this project.
# Core library headers are available via cactus_core_lib target_include_directories.
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # Example if JNI files were here
include_directories(${SOURCE_DIR}) # For top-level headers from core cactus if needed by JNI
include_directories(${SOURCE_DIR}/ggml-cpu) # If ggml-cpu headers are needed by JNI


# SOURCE_FILES list is removed as all core sources are in cactus_core_lib

find_library(LOG_LIB log)

function(build_library target_name cpu_flags)
    # Add a shared library using object files from cactus_core_lib
    add_library(
        ${target_name}
        SHARED
        $<TARGET_OBJECTS:cactus_core_lib>
    )

    # Link against Android log library and other system libs
    # cactus_core_lib objects are already included, so no need to link it as a separate library
    target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android)

    # Apply Android specific compile options (respecting user's re-addition of LM_GGML_USE_CPU)
    target_compile_options(${target_name} PRIVATE -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_AARCH64 -pthread ${cpu_flags})

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DCACTUS_ANDROID_ENABLE_LOGGING)
    endif ()

    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(${target_name} PRIVATE -flto)

endfunction()

# Default target (no specific CPU features)
build_library("cactus" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("cactus_v8" "-march=armv8-a")
    build_library("cactus_v8_2" "-march=armv8.2-a")
    build_library("cactus_v8_2_dotprod" "-march=armv8.2-a+dotprod")
    build_library("cactus_v8_2_i8mm" "-march=armv8.2-a+i8mm")
    build_library("cactus_v8_2_dotprod_i8mm" "-march=armv8.2-a+dotprod+i8mm")

elseif (${ANDROID_ABI} STREQUAL "x86_64")
    build_library("cactus_x86_64" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")

endif ()
