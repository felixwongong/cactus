use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let cactus_src = locate_cactus_source();
    set_rebuild_triggers(&cactus_src);

    if target_os == "linux" {
        apply_linux_compiler_workaround();
    }

    let build_dir = if target_os == "android" {
        build_native_library_android(&cactus_src)
    } else if target_arch == "x86_64" {
        build_native_library_x86(&cactus_src)
    } else {
        build_native_library(&cactus_src)
    };

    link_native_library(&build_dir, &target_os);
    link_platform_dependencies(&target_os);

    generate_bindings(&cactus_src, &target_os);
}

fn locate_cactus_source() -> PathBuf {
    let path = env::var("CACTUS_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root_fallback());

    assert!(
        path.exists(),
        "Cactus source not found at {path:?}. Set CACTUS_SOURCE_DIR or run: git submodule update --init --recursive"
    );

    path
}

fn repo_root_fallback() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // rust/cactus-sys -> rust -> repo root -> cactus/
    manifest_dir.ancestors().nth(2).unwrap().join("cactus")
}

/// The repo root (parent of cactus/ source).
fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir.ancestors().nth(2).unwrap().to_path_buf()
}

fn set_rebuild_triggers(cactus_src: &Path) {
    println!("cargo:rerun-if-env-changed=CACTUS_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=CACTUS_ANDROID_CMAKE_TOOLCHAIN");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        cactus_src.join("ffi/cactus_ffi.h").display()
    );
}

fn apply_linux_compiler_workaround() {
    // GCC requires explicit <iomanip>; upstream telemetry.cpp omits it.
    let existing = env::var("CXXFLAGS").unwrap_or_default();
    let cxxflags = if existing.is_empty() {
        "-include iomanip".to_string()
    } else {
        format!("-include iomanip {existing}")
    };
    unsafe { env::set_var("CXXFLAGS", cxxflags) };
}

/// The cofy-cactus crate root (4 levels up from cactus-sys manifest dir).
fn cofy_cactus_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // vendor/cactus/rust/cactus-sys → vendor/cactus/rust → vendor/cactus → vendor → cofy-cactus
    manifest_dir.ancestors().nth(4).unwrap().to_path_buf()
}

/// Build for x86_64 using NEON→SSE emulation via SIMDe (SIMD Everywhere).
///
/// Our `cofy-cactus/include/arm_neon.h` shim intercepts `#include <arm_neon.h>`
/// and routes through SIMDe, which provides the complete NEON API using native
/// x86 SSE/AVX/FMA instructions. Zero vendor C++ source modifications required.
fn build_native_library_x86(cactus_src: &Path) -> PathBuf {
    let include_dir = cofy_cactus_root().join("include");

    cmake::Config::new(cactus_src)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        // Our shim include dir must come first so #include <arm_neon.h> finds ours
        .cxxflag(&format!("-I{}", include_dir.display()))
        // x86 SIMD features required by SIMDe and F16C conversion
        .cxxflag("-msse4.2")
        .cxxflag("-mssse3")
        .cxxflag("-mfma")
        .cxxflag("-mf16c")
        // Trick code into thinking ARM NEON is available
        // (NOT defining __ARM_FEATURE_DOTPROD — use emulated dot product path)
        // (NOT defining __aarch64__ — avoid ARM stnp assembly in kernel_utils.h)
        .cxxflag("-D__ARM_NEON=1")
        .cxxflag("-D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1")
        // Define __fp16 globally — npu.h uses it without including arm_neon.h
        .cxxflag("-D__fp16=_Float16")
        .build_target("cactus")
        .build()
        .join("build")
}

fn build_native_library(cactus_src: &Path) -> PathBuf {
    cmake::Config::new(cactus_src)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("cactus")
        .build()
        .join("build")
}

fn build_native_library_android(cactus_src: &Path) -> PathBuf {
    let root = repo_root();
    let android_dir = root.join("android");
    let curl_root = root.join("libs").join("curl");

    // Resolve the NDK path — supports Docker (/android-ndk) and local SDK installs.
    let ndk_root = resolve_android_ndk();
    let ndk_toolchain = ndk_root
        .join("toolchains/llvm/prebuilt/linux-x86_64/bin");
    let ndk_sysroot = ndk_root.join("toolchains/llvm/prebuilt/linux-x86_64/sysroot");

    // Use the NDK's cmake toolchain file — it handles sysroot, compiler,
    // API level detection, and all Android-specific flags automatically.
    // NDK 28+ ships Clang 19 which has full C++20 support.
    // For older NDKs or Docker cross-rs, override compilers explicitly.
    let ndk_cmake_toolchain = ndk_root.join("build/cmake/android.toolchain.cmake");

    let mut config = cmake::Config::new(&android_dir);

    if ndk_cmake_toolchain.exists() {
        // Standard NDK build: use the toolchain file
        eprintln!("Using NDK cmake toolchain: {}", ndk_cmake_toolchain.display());
        config
            .define("CMAKE_TOOLCHAIN_FILE", ndk_cmake_toolchain.to_str().unwrap())
            .define("ANDROID_ABI", "arm64-v8a")
            .define("ANDROID_PLATFORM", "android-24")
            .define("ANDROID_STL", "c++_static");
    } else {
        // Fallback: manual compiler configuration (Docker cross-rs images)
        let cxx_compiler = env::var("CACTUS_ANDROID_CXX_COMPILER")
            .unwrap_or_else(|_| ndk_toolchain.join("clang++").to_string_lossy().to_string());
        let c_compiler = env::var("CACTUS_ANDROID_C_COMPILER")
            .unwrap_or_else(|_| ndk_toolchain.join("clang").to_string_lossy().to_string());
        config
            .define("CMAKE_C_COMPILER", &c_compiler)
            .define("CMAKE_CXX_COMPILER", &cxx_compiler)
            .define("CMAKE_C_COMPILER_TARGET", "aarch64-linux-android24")
            .define("CMAKE_CXX_COMPILER_TARGET", "aarch64-linux-android24")
            .define("CMAKE_SYSROOT", ndk_sysroot.to_str().unwrap())
            .define("ANDROID_ABI", "arm64-v8a")
            .define("ANDROID_PLATFORM", "android-24");
    }

    config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CACTUS_CURL_ROOT", curl_root.to_str().unwrap())
        // Override SOURCE_DIR so the android CMakeLists can find cactus sources
        .define("SOURCE_DIR", cactus_src.to_str().unwrap())
        // Satisfy find_library(LOG_LIB log) for the shared target we don't build.
        // Without this, cmake errors on NOTFOUND in target_link_libraries.
        .define("LOG_LIB", "log")
        .build_target("cactus_static")
        .build()
        .join("build")
}

/// Resolve the Android NDK root directory.
///
/// Search order:
/// 1. `ANDROID_NDK_HOME` env var (explicit override)
/// 2. `ANDROID_NDK_ROOT` env var
/// 3. Docker path `/android-ndk` (cross-rs images)
/// 4. `ANDROID_HOME`/ndk/<latest> (Android SDK Manager installs)
fn resolve_android_ndk() -> PathBuf {
    // Explicit env vars
    for var in ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME"] {
        if let Ok(path) = env::var(var) {
            let p = PathBuf::from(&path);
            if p.exists() {
                eprintln!("Using NDK from {var}={path}");
                return p;
            }
        }
    }

    // Docker cross-rs fallback
    let docker_ndk = PathBuf::from("/android-ndk");
    if docker_ndk.exists() {
        eprintln!("Using NDK from Docker path: /android-ndk");
        return docker_ndk;
    }

    // Android SDK Manager: find the latest NDK version
    if let Ok(sdk_home) = env::var("ANDROID_HOME") {
        let ndk_dir = PathBuf::from(&sdk_home).join("ndk");
        if ndk_dir.is_dir() {
            if let Some(latest) = find_latest_ndk_version(&ndk_dir) {
                eprintln!("Using NDK from ANDROID_HOME: {}", latest.display());
                return latest;
            }
        }
    }

    panic!(
        "Android NDK not found. Set ANDROID_NDK_HOME, ANDROID_NDK_ROOT, \
         or install via Android SDK Manager (ANDROID_HOME/ndk/)."
    );
}

/// Find the latest NDK version directory under `ndk_dir/`.
fn find_latest_ndk_version(ndk_dir: &Path) -> Option<PathBuf> {
    let mut versions: Vec<_> = std::fs::read_dir(ndk_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    // Sort by name descending to get the latest version
    versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    versions.first().map(|e| e.path())
}

fn link_native_library(build_dir: &Path, target_os: &str) {
    if target_os == "android" {
        // Android cmake puts static libs in lib/ subdirectory
        println!(
            "cargo:rustc-link-search=native={}",
            build_dir.join("lib").display()
        );
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=static=cactus_static");
    } else {
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        println!("cargo:rustc-link-lib=static=cactus");
    }
}

fn link_platform_dependencies(target_os: &str) {
    match target_os {
        "macos" => {
            for framework in [
                "Metal",
                "MetalPerformanceShaders",
                "Accelerate",
                "Foundation",
                "CoreML",
            ] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
            println!("cargo:rustc-link-lib=curl");
            println!("cargo:rustc-link-lib=c++");
        }
        "android" => {
            let root = repo_root();

            // Vendored libcurl (static)
            let curl_lib = root.join("libs/curl/android/arm64-v8a");
            println!("cargo:rustc-link-search=native={}", curl_lib.display());
            println!("cargo:rustc-link-lib=static=curl");

            // Vendored mbedTLS (static)
            let mbedtls_lib = root.join("android/mbedtls/arm64-v8a/lib");
            if mbedtls_lib.exists() {
                println!("cargo:rustc-link-search=native={}", mbedtls_lib.display());
            } else {
                // Fallback location
                let alt = root.join("libs/mbedtls/android/arm64-v8a/lib");
                println!("cargo:rustc-link-search=native={}", alt.display());
            }
            println!("cargo:rustc-link-lib=static=mbedtls");
            println!("cargo:rustc-link-lib=static=mbedx509");
            println!("cargo:rustc-link-lib=static=mbedcrypto");

            // Android: use libc++_static (NDK's C++ runtime), not libstdc++
            println!("cargo:rustc-link-lib=c++_static");
            println!("cargo:rustc-link-lib=c++abi");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=log");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=curl");
        }
        _ => {}
    }
}

fn generate_bindings(cactus_src: &Path, target_os: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let mut builder = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_str().unwrap())
        .clang_arg(format!("-I{}", cactus_src.display()))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++20")
        .allowlist_function("cactus_.*")
        .allowlist_type("cactus_.*")
        .allowlist_var("CACTUS_.*")
        .derive_debug(true)
        .derive_default(true);

    // For Android cross-compilation, set the target triple and NDK sysroot for clang
    if target_os == "android" {
        let ndk_root = resolve_android_ndk();
        let ndk_sysroot = ndk_root
            .join("toolchains/llvm/prebuilt/linux-x86_64/sysroot");
        builder = builder
            .clang_arg("--target=aarch64-linux-android24")
            .clang_arg(format!("--sysroot={}", ndk_sysroot.display()))
            .clang_arg("-DPLATFORM_CPU_ONLY=1");
    }

    builder
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings");
}
