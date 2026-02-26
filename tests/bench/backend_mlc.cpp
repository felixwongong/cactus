#include "bench_driver.h"

#ifdef WITH_MLC

#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// MLC backend: loads a pre-compiled TVM module and dispatches quantized
// matmul through the TVM runtime, matching the code path MLC-LLM uses.
//
// The module must export "quantized_matmul_int4" and/or
// "quantized_matmul_int8" with signature: (A, B_packed, scales, C).
//
// Setup:
//   1. git clone --recursive https://github.com/mlc-ai/mlc-llm ../third_party/mlc
//   2. Build TVM runtime:
//        cd ../third_party/mlc/3rdparty/tvm && mkdir build && cd build
//        cmake .. -DUSE_LLVM=OFF && make tvm_runtime -j
//   3. Compile the benchmark kernels with TVM:
//        python tests/bench/compile_mlc_kernels.py
//   4. Set MLC_MATMUL_LIB=<path to compiled .so>
//   5. cmake -B build -DWITH_MLC=ON && cmake --build build

namespace {

static TVMModuleHandle s_mod = nullptr;
static TVMFunctionHandle s_q4_fn = nullptr;
static TVMFunctionHandle s_q8_fn = nullptr;

static bool load_module() {
    const char* path = std::getenv("MLC_MATMUL_LIB");
    if (!path) {
        fprintf(stderr, "[mlc] MLC_MATMUL_LIB not set; skipping MLC backend.\n"
                        "      Set it to a TVM-compiled .so with quantized matmul kernels.\n");
        return false;
    }

    if (TVMModLoadFromFile(path, "so", &s_mod) != 0) {
        fprintf(stderr, "[mlc] Failed to load %s: %s\n", path, TVMGetLastError());
        return false;
    }

    if (TVMModGetFunction(s_mod, "quantized_matmul_int4", 1, &s_q4_fn) != 0)
        s_q4_fn = nullptr;
    if (TVMModGetFunction(s_mod, "quantized_matmul_int8", 1, &s_q8_fn) != 0)
        s_q8_fn = nullptr;

    if (!s_q4_fn && !s_q8_fn) {
        fprintf(stderr, "[mlc] Module lacks quantized_matmul_int4/int8 functions\n");
        TVMModFree(s_mod);
        s_mod = nullptr;
        return false;
    }

    return true;
}

// ── Shared types and helpers ────────────────────────────────────────────

struct MlcWeights {
    size_t K, N;
    int bits;
    TVMFunctionHandle fn;
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;
    int64_t b_shape[2];
    int64_t s_shape[2];
};

struct MlcActivations {
    std::vector<__fp16> fp16;
};

static DLTensor make_dl(void* data, int64_t* shape, int ndim,
                         uint8_t type_code, uint8_t bits) {
    DLTensor t{};
    t.data = data;
    t.device = {kDLCPU, 0};
    t.ndim = ndim;
    t.dtype = {type_code, bits, 1};
    t.shape = shape;
    return t;
}

static void call_fn(TVMFunctionHandle fn, DLTensor* a, DLTensor* b,
                     DLTensor* s, DLTensor* c) {
    TVMValue args[4];
    int codes[4];
    for (int i = 0; i < 4; i++) {
        codes[i] = kTVMDLTensorHandle;
        args[i].v_handle = nullptr;
    }
    args[0].v_handle = a;
    args[1].v_handle = b;
    args[2].v_handle = s;
    args[3].v_handle = c;

    TVMValue ret;
    int ret_code;
    TVMFuncCall(fn, args, codes, 4, &ret, &ret_code);
}

// ── Prepare weights (shared, parameterized by bits) ─────────────────────

static void* prepare_impl(const float* fp32, size_t N, size_t K,
                            int bits, TVMFunctionHandle fn) {
    auto* w = new MlcWeights();
    w->K = K;
    w->N = N;
    w->bits = bits;
    w->fn = fn;

    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> quantized;
    std::vector<float> raw_scales;

    if (bits == 4) {
        bench::quantize_int4_per_group(src, N, K, quantized, raw_scales);
        w->packed.resize(N * K / 2);
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k += 2) {
                uint8_t lo = static_cast<uint8_t>(quantized[n * K + k] + 8);
                uint8_t hi = static_cast<uint8_t>(quantized[n * K + k + 1] + 8);
                w->packed[n * (K / 2) + k / 2] = lo | (hi << 4);
            }
        w->b_shape[0] = static_cast<int64_t>(N);
        w->b_shape[1] = static_cast<int64_t>(K / 2);
    } else {
        bench::quantize_int8_per_group(src, N, K, quantized, raw_scales);
        w->packed.resize(N * K);
        std::memcpy(w->packed.data(), quantized.data(), N * K);
        w->b_shape[0] = static_cast<int64_t>(N);
        w->b_shape[1] = static_cast<int64_t>(K);
    }

    const size_t num_groups = K / bench::kGroupSize;
    w->scales.resize(N * num_groups);
    for (size_t i = 0; i < N * num_groups; i++)
        w->scales[i] = static_cast<__fp16>(raw_scales[i]);
    w->s_shape[0] = static_cast<int64_t>(N);
    w->s_shape[1] = static_cast<int64_t>(num_groups);

    return w;
}

void* prepare_q4(const float* fp32, size_t N, size_t K) {
    return prepare_impl(fp32, N, K, 4, s_q4_fn);
}

void* prepare_q8(const float* fp32, size_t N, size_t K) {
    return prepare_impl(fp32, N, K, 8, s_q8_fn);
}

// ── Prepare activations ─────────────────────────────────────────────────

void* prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new MlcActivations();
    a->fp16.resize(M * K);
    for (size_t i = 0; i < M * K; i++)
        a->fp16[i] = static_cast<__fp16>(fp32[i]);
    return a;
}

// ── Run kernel ──────────────────────────────────────────────────────────

void run_kernel(size_t M, void* weights, void* activations,
                const int8_t*, const float*,
                float* output, float*) {
    auto* w = static_cast<MlcWeights*>(weights);
    auto* a = static_cast<MlcActivations*>(activations);
    w->output_buf.resize(M * w->N);

    int64_t a_shape[] = {static_cast<int64_t>(M), static_cast<int64_t>(w->K)};
    int64_t c_shape[] = {static_cast<int64_t>(M), static_cast<int64_t>(w->N)};

    uint8_t b_type = (w->bits == 4) ? kDLUInt : kDLInt;

    DLTensor a_dl = make_dl(a->fp16.data(), a_shape, 2, kDLFloat, 16);
    DLTensor b_dl = make_dl(w->packed.data(), w->b_shape, 2, b_type, 8);
    DLTensor s_dl = make_dl(w->scales.data(), w->s_shape, 2, kDLFloat, 16);
    DLTensor c_dl = make_dl(w->output_buf.data(), c_shape, 2, kDLFloat, 16);

    call_fn(w->fn, &a_dl, &b_dl, &s_dl, &c_dl);

    if (output)
        for (size_t i = 0; i < M * w->N; i++)
            output[i] = static_cast<float>(w->output_buf[i]);
}

// ── Cleanup ─────────────────────────────────────────────────────────────

void cleanup(void* weights, void* activations) {
    delete static_cast<MlcWeights*>(weights);
    if (activations) delete static_cast<MlcActivations*>(activations);
}

// ── Registration ────────────────────────────────────────────────────────

static int reg = [] {
    if (!load_module()) return 0;

    if (s_q4_fn)
        bench::register_backend({
            "mlc_int4", "mlc", bench::QuantCategory::INT4, 0,
            prepare_q4, prepare_act, run_kernel, cleanup
        });
    if (s_q8_fn)
        bench::register_backend({
            "mlc_int8", "mlc", bench::QuantCategory::INT8, 0,
            prepare_q8, prepare_act, run_kernel, cleanup
        });

    return 0;
}();

} // namespace

#else

namespace {
[[maybe_unused]] static int reg = [] { return 0; }();
} // namespace

#endif
