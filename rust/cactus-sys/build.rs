use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cactus_src = locate_cactus_source();
    set_rebuild_triggers(&cactus_src);

    if target_os == "linux" {
        apply_linux_compiler_workaround();
    }

    let build_dir = if target_os == "android" {
        build_native_library_android(&cactus_src)
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

    // Use the android/ CMakeLists.txt which knows about vendored curl + mbedtls.
    // We need Clang 17+ for full C++20 support (P0960R3 parenthesized aggregate init).
    // The cross-rs image's NDK has Clang 14 which doesn't support this.
    // We override the compiler via CMAKE_CXX_COMPILER directly instead of using
    // a toolchain file, because cmake's built-in Android platform module overrides
    // the compiler from toolchain files.
    let cxx_compiler = env::var("CACTUS_ANDROID_CXX_COMPILER")
        .unwrap_or_else(|_| "/usr/bin/clang++-17".to_string());
    let c_compiler = env::var("CACTUS_ANDROID_C_COMPILER")
        .unwrap_or_else(|_| "/usr/bin/clang-17".to_string());

    let mut config = cmake::Config::new(&android_dir);
    config
        .define("CMAKE_C_COMPILER", &c_compiler)
        .define("CMAKE_CXX_COMPILER", &cxx_compiler)
        .define("CMAKE_C_COMPILER_TARGET", "aarch64-linux-android24")
        .define("CMAKE_CXX_COMPILER_TARGET", "aarch64-linux-android24")
        .define("CMAKE_SYSROOT", "/android-ndk/sysroot");
    config
        .define("ANDROID_ABI", "arm64-v8a")
        .define("ANDROID_PLATFORM", "android-24")
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

fn link_native_library(build_dir: &Path, target_os: &str) {
    if target_os == "android" {
        // Android cmake puts static libs in lib/ subdirectory
        println!("cargo:rustc-link-search=native={}", build_dir.join("lib").display());
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

    // For Android cross-compilation, set the target triple for clang
    if target_os == "android" {
        builder = builder
            .clang_arg("--target=aarch64-linux-android24")
            .clang_arg("-DPLATFORM_CPU_ONLY=1");
    }

    builder
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings");
}
