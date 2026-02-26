#include "bench_driver.h"

#ifdef WITH_ONNXRT

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

struct PBuf {
    std::vector<uint8_t> d;
    void varint(uint64_t v) {
        while (v > 0x7F) { d.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80)); v >>= 7; }
        d.push_back(static_cast<uint8_t>(v));
    }
    void fld_vi(int f, uint64_t v) {
        varint(static_cast<uint64_t>(f) << 3);
        varint(v);
    }
    void fld_ld(int f, const PBuf& sub) {
        varint(static_cast<uint64_t>(f) << 3 | 2);
        varint(sub.d.size());
        d.insert(d.end(), sub.d.begin(), sub.d.end());
    }
    void fld_str(int f, const char* s) {
        size_t n = std::strlen(s);
        varint(static_cast<uint64_t>(f) << 3 | 2);
        varint(n);
        d.insert(d.end(), s, s + n);
    }
};

static PBuf make_dim_param(const char* name) { PBuf d; d.fld_str(2, name); return d; }
static PBuf make_dim_value(size_t v) { PBuf d; d.fld_vi(1, v); return d; }

template<typename... Dims>
static PBuf make_value_info(const char* name, int elem_type, const Dims&... dims) {
    PBuf shape; (shape.fld_ld(1, dims), ...);
    PBuf tensor; tensor.fld_vi(1, elem_type); tensor.fld_ld(2, shape);
    PBuf type; type.fld_ld(1, tensor);
    PBuf vi; vi.fld_str(1, name); vi.fld_ld(2, type);
    return vi;
}

static PBuf make_attr_int(const char* name, int64_t val) {
    PBuf a;
    a.fld_str(1, name);
    a.fld_vi(3, static_cast<uint64_t>(val));
    a.fld_vi(20, 2); // AttributeType.INT = 2
    return a;
}

static Ort::Env& get_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bench");
    return env;
}

static Ort::MemoryInfo& get_cpu_mem() {
    static Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return mem;
}

static std::vector<uint8_t> build_model(size_t K, size_t N, int bits) {
    const size_t n_blocks = K / bench::kGroupSize;
    const size_t b_last_dim = (bits == 4) ? bench::kGroupSize / 2 : bench::kGroupSize;
    const size_t zp_last_dim = (bits == 4) ? (n_blocks + 1) / 2 : n_blocks;
    const char* graph_name = (bits == 4) ? "bench_int4" : "bench_int8";

    PBuf node;
    node.fld_str(1, "A");
    node.fld_str(1, "B");
    node.fld_str(1, "scales");
    node.fld_str(1, "zero_points");
    node.fld_str(2, "Y");
    node.fld_str(4, "MatMulNBits");
    node.fld_str(7, "com.microsoft");
    node.fld_ld(5, make_attr_int("K", static_cast<int64_t>(K)));
    node.fld_ld(5, make_attr_int("N", static_cast<int64_t>(N)));
    node.fld_ld(5, make_attr_int("bits", bits));
    node.fld_ld(5, make_attr_int("block_size", static_cast<int64_t>(bench::kGroupSize)));
    node.fld_ld(5, make_attr_int("accuracy_level", 4));

    auto a_vi  = make_value_info("A", 1, make_dim_param("M"), make_dim_value(K));
    auto b_vi  = make_value_info("B", 2, make_dim_value(N), make_dim_value(n_blocks),
                                 make_dim_value(b_last_dim));
    auto s_vi  = make_value_info("scales", 1, make_dim_value(N), make_dim_value(n_blocks));
    auto zp_vi = make_value_info("zero_points", 2,
                                 make_dim_value(N), make_dim_value(zp_last_dim));
    auto y_vi  = make_value_info("Y", 1, make_dim_param("M"), make_dim_value(N));

    PBuf graph;
    graph.fld_ld(1, node);
    graph.fld_str(2, graph_name);
    graph.fld_ld(11, a_vi);
    graph.fld_ld(11, b_vi);
    graph.fld_ld(11, s_vi);
    graph.fld_ld(11, zp_vi);
    graph.fld_ld(12, y_vi);

    PBuf opset1; opset1.fld_str(1, ""); opset1.fld_vi(2, 13);
    PBuf opset2; opset2.fld_str(1, "com.microsoft"); opset2.fld_vi(2, 1);

    PBuf model;
    model.fld_vi(1, 7);
    model.fld_ld(8, opset1);
    model.fld_ld(8, opset2);
    model.fld_ld(7, graph);
    return model.d;
}

static constexpr int kGemvThreads = 2;
static constexpr int kGemmThreads = 3;

struct OrtWeights {
    size_t K, N;
    int bits;
    std::vector<uint8_t> B_packed;
    std::vector<float> scales;
    std::vector<uint8_t> zero_points;
    std::unique_ptr<Ort::Session> session;
    Ort::RunOptions run_opts;
    std::vector<Ort::Value> inputs;

    void bind(float* act_data, size_t M) {
        int threads = bench::get_effective_threads((M == 1) ? kGemvThreads : kGemmThreads);
        auto bytes = build_model(K, N, bits);
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(get_env(), bytes.data(), bytes.size(), opts);

        auto& mem = get_cpu_mem();
        const size_t n_blocks = K / bench::kGroupSize;
        const size_t b_last_dim = (bits == 4) ? bench::kGroupSize / 2 : bench::kGroupSize;
        const size_t b_total = (bits == 4) ? N * K / 2 : N * K;
        const size_t zp_cols = (bits == 4) ? (n_blocks + 1) / 2 : n_blocks;

        int64_t a_shape[]  = {(int64_t)M, (int64_t)K};
        int64_t b_shape[]  = {(int64_t)N, (int64_t)n_blocks, (int64_t)b_last_dim};
        int64_t s_shape[]  = {(int64_t)N, (int64_t)n_blocks};
        int64_t zp_shape[] = {(int64_t)N, (int64_t)zp_cols};

        inputs.clear();
        inputs.reserve(4);
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, act_data, M * K, a_shape, 2));
        inputs.push_back(Ort::Value::CreateTensor<uint8_t>(mem, B_packed.data(), b_total, b_shape, 3));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, scales.data(), N * n_blocks, s_shape, 2));
        inputs.push_back(Ort::Value::CreateTensor<uint8_t>(mem, zero_points.data(), N * zp_cols, zp_shape, 2));
    }
};

struct OrtActivations {
    std::vector<float> fp32;
};

void* prepare_act(const float* fp32, size_t M, size_t K, void* raw_weights) {
    auto* w = static_cast<OrtWeights*>(raw_weights);
    auto* a = new OrtActivations();
    a->fp32.assign(fp32, fp32 + M * K);
    w->bind(a->fp32.data(), M);
    return a;
}

void run_kernel(size_t M, void* weights, void*,
                const int8_t*, const float*,
                float* output, float*) {
    auto* w = static_cast<OrtWeights*>(weights);
    if (!w->session || w->inputs.empty()) return;

    static const char* in_names[] = {"A", "B", "scales", "zero_points"};
    static const char* out_names[] = {"Y"};

    auto out = w->session->Run(w->run_opts, in_names, w->inputs.data(), 4, out_names, 1);
    if (output) {
        const float* y = out[0].GetTensorData<float>();
        std::memcpy(output, y, M * w->N * sizeof(float));
    }
}

void cleanup(void* weights, void* activations) {
    delete static_cast<OrtWeights*>(weights);
    if (activations) delete static_cast<OrtActivations*>(activations);
}

static void pack_int8(OrtWeights* w, const float* fp32) {
    size_t N = w->N, K = w->K;
    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> int8_vals;
    std::vector<float> raw_scales;
    bench::quantize_int8_per_group(src, N, K, int8_vals, raw_scales);

    w->B_packed.resize(N * K);
    for (size_t i = 0; i < N * K; i++)
        w->B_packed[i] = static_cast<uint8_t>(static_cast<int>(int8_vals[i]) + 128);

    w->scales = raw_scales;
    const size_t n_blocks = K / bench::kGroupSize;
    w->zero_points.resize(N * n_blocks, 128);
}

static void pack_int4(OrtWeights* w, const float* fp32) {
    size_t N = w->N, K = w->K;
    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> int4_vals;
    std::vector<float> raw_scales;
    bench::quantize_int4_per_group(src, N, K, int4_vals, raw_scales);

    w->B_packed.resize(N * K / 2);
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k += 2) {
            uint8_t lo = static_cast<uint8_t>(int4_vals[n * K + k] + 8);
            uint8_t hi = static_cast<uint8_t>(int4_vals[n * K + k + 1] + 8);
            w->B_packed[n * (K / 2) + k / 2] = (hi << 4) | lo;
        }
    }

    w->scales = raw_scales;
    const size_t n_blocks = K / bench::kGroupSize;
    const size_t zp_cols = (n_blocks + 1) / 2;
    w->zero_points.resize(N * zp_cols, 0x88);
}

void* i8_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new OrtWeights();
    w->K = K; w->N = N; w->bits = 8;
    pack_int8(w, fp32);
    return w;
}

void* i4_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new OrtWeights();
    w->K = K; w->N = N; w->bits = 4;
    pack_int4(w, fp32);
    return w;
}

static int reg = [] {
    bench::register_matmul_backend({"onnxrt_int8", "onnxrt", bench::QuantCategory::INT8, 0, i8_prepare, prepare_act, run_kernel, cleanup});
    bench::register_matmul_backend({"onnxrt_int4", "onnxrt", bench::QuantCategory::INT4, 0, i4_prepare, prepare_act, run_kernel, cleanup});
    return 0;
}();

} // namespace

#else

namespace {
[[maybe_unused]] static int reg = [] { return 0; }();
} // namespace

#endif
