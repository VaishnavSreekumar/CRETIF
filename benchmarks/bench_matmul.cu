// ── bench_matmul.cu — Matrix Multiplication Kernel Benchmarks ──────────────
//
// Benchmarks all three matmul kernel variants (naive, tiled, vectorized) across
// a sweep of matrix dimensions, reporting performance in GFLOPS and effective
// memory bandwidth, along with the percentage of theoretical peak achieved on
// the target GPU (RTX 3060).
//
// Methodology:
//   - Each (M, N, K, variant) configuration is run for a configurable number of
//     iterations (default: 100) after a warmup period (default: 10 iterations).
//   - Timing uses CUDA events (cudaEventRecord / cudaEventElapsedTime) which
//     measure GPU-side execution time with sub-microsecond precision, unaffected
//     by host-side overhead or CPU scheduling jitter.
//   - We report the MEDIAN time rather than the mean, because median is robust
//     to outliers from thermal throttling, OS interrupts, or occasional memory
//     management stalls that would skew a mean.
//   - GFLOPS is computed as (2 * M * N * K) / (median_time_seconds * 1e9),
//     where the factor of 2 counts both the multiply and add in each FMA.
//   - Effective memory bandwidth is computed from the minimum data movement
//     required: reading A (M*K floats) + reading B (K*N floats) + writing C
//     (M*N floats). This is a lower bound — actual traffic may be higher due
//     to cache misses and replayed loads.
//   - RTX 3060 theoretical peaks: 12.74 TFLOPS FP32, 336.5 GB/s memory BW.
//
// Usage:
//   ./bench_matmul                          # Run default dimension sweep
//   ./bench_matmul --m 1024 --n 1024 --k 1024   # Single custom dimension
//   ./bench_matmul --warmup 20 --iters 200       # Custom iteration count
//   ./bench_matmul --skip-naive                  # Skip naive (slow on large sizes)
//   ./bench_matmul --skip-cublas                 # Skip cuBLAS comparison

#include "cuinfer/cuda_utils.h"
#include "cuinfer/kernels/matmul.h"
#include "cuinfer/kernels/matmul_quant.h"
#include "cuinfer/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace cuinfer {

// ── RTX 3060 Theoretical Peaks ──────────────────────────────────────────
// These are used to compute the percentage of peak achieved. Source: NVIDIA
// spec sheet for GA106 (RTX 3060 12 GB): 3584 CUDA cores @ 1.78 GHz boost.
//
// FP32 peak: 3584 cores * 2 (FMA) * 1.78 GHz = 12.74 TFLOPS
// Memory BW: 192-bit bus * 15 Gbps (GDDR6) = 360 GB/s (effective ~336.5 GB/s)
//
// Note: actual achievable peaks depend on workload, occupancy, and thermal
// state. We use these as reference points, not hard limits.
constexpr double PEAK_TFLOPS_FP32 = 12.74;
constexpr double PEAK_BW_GBS = 336.5;

// ── Benchmark Configuration ─────────────────────────────────────────────

struct BenchConfig {
    int warmup_iters = 10;
    int bench_iters = 100;
    bool skip_naive = false;
    bool skip_cublas = false;

    // If non-empty, only run this single dimension instead of the full sweep
    int custom_m = 0;
    int custom_n = 0;
    int custom_k = 0;
};

// ── Result for a Single Benchmark Run ───────────────────────────────────

struct BenchResult {
    int M, N, K;
    std::string variant;
    double median_ms;       // Median kernel time in milliseconds
    double gflops;          // Achieved GFLOPS
    double pct_peak_gflops; // Percentage of RTX 3060 FP32 peak
    double eff_bw_gbs;      // Effective memory bandwidth in GB/s
    double pct_peak_bw;     // Percentage of RTX 3060 memory BW peak
};

// ── CLI Argument Parsing ────────────────────────────────────────────────

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iters = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.bench_iters = std::atoi(argv[++i]);
        } else if (arg == "--m" && i + 1 < argc) {
            cfg.custom_m = std::atoi(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            cfg.custom_n = std::atoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            cfg.custom_k = std::atoi(argv[++i]);
        } else if (arg == "--skip-naive") {
            cfg.skip_naive = true;
        } else if (arg == "--skip-cublas") {
            cfg.skip_cublas = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: bench_matmul [options]\n");
            printf("  --warmup N       Warmup iterations (default: 10)\n");
            printf("  --iters N        Benchmark iterations (default: 100)\n");
            printf("  --m M            Custom M dimension\n");
            printf("  --n N            Custom N dimension\n");
            printf("  --k K            Custom K dimension\n");
            printf("  --skip-naive     Skip naive variant (very slow at large sizes)\n");
            printf("  --skip-cublas    Skip cuBLAS comparison variant\n");
            printf("  -h, --help       Show this help\n");
            std::exit(0);
        }
    }

    return cfg;
}

// ── Compute Median of a Float Vector ────────────────────────────────────
//
// We use the median because GPU benchmark timings often have a long right
// tail due to thermal throttling, TLB misses on first runs, or background
// OS activity. The median gives us the "typical" execution time.

static double median(std::vector<float>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0;

    std::sort(v.begin(), v.end());

    if (n % 2 == 0) {
        return static_cast<double>(v[n / 2 - 1] + v[n / 2]) / 2.0;
    } else {
        return static_cast<double>(v[n / 2]);
    }
}

// ── Kernel Dispatch ─────────────────────────────────────────────────────
//
// Wraps the three matmul variants behind a common interface indexed by name,
// so the benchmarking loop can iterate over variants without duplication.

using MatmulFn = void (*)(const float*, const float*, float*,
                          int, int, int, cudaStream_t);

struct KernelVariant {
    std::string name;
    MatmulFn fn;
};

static std::vector<KernelVariant> get_variants(bool skip_naive, bool skip_cublas) {
    std::vector<KernelVariant> variants;

    if (!skip_naive) {
        variants.push_back({"naive", matmul_naive});
    }
    variants.push_back({"tiled", matmul_tiled});
    variants.push_back({"vectorized", matmul_vectorized});

    if (!skip_cublas) {
        variants.push_back({"cuBLAS", nullptr});  // sentinel — handled specially
    }

    return variants;
}

// ── cuBLAS Wrappers ─────────────────────────────────────────────────────
//
// cuBLAS uses column-major layout. For row-major C = A * B, we use the
// identity: C^T = B^T * A^T. Since column-major(X) == row-major(X^T),
// we pass B as first argument and A as second to cublasSgemm with no
// transpose flags, and cuBLAS writes row-major C directly.

static cublasHandle_t g_cublas_handle = nullptr;

static void init_cublas() {
    if (!g_cublas_handle) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS init failed: %d\n", static_cast<int>(status));
            std::exit(1);
        }
    }
}

static void destroy_cublas() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// FP32 cuBLAS matmul: C [M x N] = A [M x K] * B [K x N], all row-major.
static void matmul_cublas(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream) {
    cublasSetStream(g_cublas_handle, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major trick: compute C^T = B^T * A^T in column-major
    cublasSgemm(g_cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,    // B as column-major = B^T row-major
                A, K,    // A as column-major = A^T row-major
                &beta,
                C, N);   // C as column-major = C^T row-major (= C row-major)
}

// FP16 cuBLAS matmul: C [M x N] = A [M x K] * B^T [N x K], using GemmEx
// with FP32 accumulation for fair comparison with our FP16 kernel.
static void matmul_cublas_fp16(const float* A, const __half* B_transposed,
                               float* C, int M, int N, int K,
                               cudaStream_t stream) {
    cublasSetStream(g_cublas_handle, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // B_transposed is [N, K] row-major. In column-major view, this is [K, N].
    // We want C = A * B where B = B_transposed^T.
    // Row-major trick: C^T = B * A^T in column-major.
    // B_transposed in column-major = [K, N], but we need B^T for the matmul.
    // Using CUBLAS_OP_T on B_transposed gives us B_transposed^T = B = [K, N].
    // A in column-major = [K, M], so A^T = [M, K].
    // Actually: for row-major, we reverse the operation:
    //   C[M,N] = A[M,K] * B_transposed^T[K,N] in row-major
    //   = C_cm^T[N,M] = B_transposed_cm[K,N]^T * A_cm[K,M] in column-major
    // Wait, let's be precise:
    //   row-major A[M,K] is column-major A_cm[K,M]
    //   row-major B_transposed[N,K] is column-major B_cm[K,N]
    //   We want C = A * B_t^T => column-major: C_cm[N,M] = B_cm[K,N]^T * A_cm[K,M]
    //   CUBLAS_OP_T on B_cm[K,N] gives [N,K], so:
    //   cublasSgemmEx(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, B_cm, K, A_cm, K, beta, C_cm, N)
    cublasGemmEx(g_cublas_handle,
                 CUBLAS_OP_T,       // Transpose B (column-major [K,N] -> [N,K])
                 CUBLAS_OP_N,       // A as-is (column-major [K,M])
                 N, M, K,
                 &alpha,
                 B_transposed, CUDA_R_16F, K,   // B: FP16, ld=K
                 A, CUDA_R_32F, K,               // A: FP32, ld=K
                 &beta,
                 C, CUDA_R_32F, N,               // C: FP32, ld=N
                 CUBLAS_COMPUTE_32F,              // FP32 accumulation
                 CUBLAS_GEMM_DEFAULT);
}

// ── Run a Single Benchmark ──────────────────────────────────────────────
//
// Allocates A [M x K], B [K x N], C [M x N] on the GPU, fills A and B with
// deterministic data, then times the kernel over warmup + bench iterations.
// Returns a BenchResult with computed metrics.
//
// Memory note: we allocate and free per benchmark point rather than reusing
// a large pre-allocated buffer. This keeps the code simple and lets us report
// accurate VRAM usage at the end (after all temporary allocations are freed).

static BenchResult run_benchmark(int M, int N, int K,
                                 const KernelVariant& variant,
                                 const BenchConfig& cfg) {
    // Allocate device tensors. The Tensor class uses RAII so memory is freed
    // when these go out of scope.
    Tensor A({M, K}, DType::FP32);
    Tensor B({K, N}, DType::FP32);
    Tensor C({M, N}, DType::FP32);

    // Fill A and B with simple deterministic values on the host, then upload.
    // We use small values to avoid FP32 overflow in large matmuls.
    // The exact values don't matter for benchmarking — we just need non-zero
    // data so the compiler can't optimize away the computation.
    {
        std::vector<float> host_a(static_cast<size_t>(M) * K);
        std::vector<float> host_b(static_cast<size_t>(K) * N);

        for (size_t i = 0; i < host_a.size(); ++i) {
            host_a[i] = static_cast<float>((i % 7) - 3) * 0.01f;
        }
        for (size_t i = 0; i < host_b.size(); ++i) {
            host_b[i] = static_cast<float>((i % 11) - 5) * 0.01f;
        }

        A.copy_from_host(host_a.data(), host_a.size() * sizeof(float));
        B.copy_from_host(host_b.data(), host_b.size() * sizeof(float));
    }

    const float* d_A = A.data_as<float>();
    const float* d_B = B.data_as<float>();
    float* d_C = C.data_as<float>();

    // Select function: use cuBLAS wrapper if fn is null (sentinel)
    MatmulFn fn = variant.fn ? variant.fn : matmul_cublas;

    // Warmup runs: bring caches, TLBs, and GPU clock into steady state.
    // We don't record these timings.
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        fn(d_A, d_B, d_C, M, N, K, 0);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs: record each kernel invocation with CUDA events.
    // We collect all individual timings and compute the median afterwards.
    CudaTimer timer;
    std::vector<float> timings(cfg.bench_iters);

    for (int i = 0; i < cfg.bench_iters; ++i) {
        timer.start();
        fn(d_A, d_B, d_C, M, N, K, 0);
        timer.stop();
        timings[i] = timer.elapsed_ms();
    }

    // Compute median timing
    double med_ms = median(timings);

    // ── Compute performance metrics ──
    //
    // FLOPs: for C = A * B with A [M x K] and B [K x N]:
    //   Each of the M*N output elements requires K multiply-adds = 2K FLOPs.
    //   Total = 2 * M * N * K floating-point operations.
    double flops = 2.0 * M * N * K;
    double med_seconds = med_ms / 1000.0;
    double gflops = (med_seconds > 0.0) ? (flops / (med_seconds * 1e9)) : 0.0;
    double pct_peak = (gflops / (PEAK_TFLOPS_FP32 * 1000.0)) * 100.0;

    // Effective memory bandwidth: minimum bytes that must be transferred.
    //   Read A: M * K * 4 bytes
    //   Read B: K * N * 4 bytes
    //   Write C: M * N * 4 bytes
    // This is a lower bound — cache misses and replayed transactions increase
    // actual traffic. It's useful as a roofline model comparison point.
    double bytes_transferred = static_cast<double>(
        (static_cast<int64_t>(M) * K + static_cast<int64_t>(K) * N +
         static_cast<int64_t>(M) * N) * sizeof(float));
    double eff_bw = (med_seconds > 0.0) ? (bytes_transferred / (med_seconds * 1e9)) : 0.0;
    double pct_bw = (eff_bw / PEAK_BW_GBS) * 100.0;

    return BenchResult{M, N, K, variant.name, med_ms, gflops, pct_peak, eff_bw, pct_bw};
}

// ── Run FP16 Benchmark ──────────────────────────────────────────────────
//
// FP16 matmul uses B in __half format [N, K] (B-transposed) rather than
// FP32 [K, N]. Separate function to handle the different data setup.

static BenchResult run_benchmark_fp16(int M, int N, int K,
                                       const BenchConfig& cfg) {
    // A is FP32 [M, K], B is FP16 [N, K], C is FP32 [M, N]
    Tensor A({M, K}, DType::FP32);
    Tensor C({M, N}, DType::FP32);

    // Allocate FP16 B directly
    __half* d_B = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(N) * K * sizeof(__half)));

    // Fill A with FP32 data, B with FP16 data
    {
        std::vector<float> host_a(static_cast<size_t>(M) * K);
        std::vector<__half> host_b(static_cast<size_t>(N) * K);

        for (size_t i = 0; i < host_a.size(); ++i) {
            host_a[i] = static_cast<float>((i % 7) - 3) * 0.01f;
        }
        for (size_t i = 0; i < host_b.size(); ++i) {
            host_b[i] = __float2half(static_cast<float>((i % 11) - 5) * 0.01f);
        }

        A.copy_from_host(host_a.data(), host_a.size() * sizeof(float));
        CUINFER_CHECK_CUDA(cudaMemcpy(d_B, host_b.data(),
                                      host_b.size() * sizeof(__half),
                                      cudaMemcpyHostToDevice));
    }

    const float* d_A = A.data_as<float>();
    float* d_C = C.data_as<float>();

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        matmul_fp16(d_A, d_B, d_C, M, N, K, 0);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    CudaTimer timer;
    std::vector<float> timings(cfg.bench_iters);

    for (int i = 0; i < cfg.bench_iters; ++i) {
        timer.start();
        matmul_fp16(d_A, d_B, d_C, M, N, K, 0);
        timer.stop();
        timings[i] = timer.elapsed_ms();
    }

    cudaFree(d_B);

    double med_ms = median(timings);
    double flops = 2.0 * M * N * K;
    double med_seconds = med_ms / 1000.0;
    double gflops = (med_seconds > 0.0) ? (flops / (med_seconds * 1e9)) : 0.0;
    double pct_peak = (gflops / (PEAK_TFLOPS_FP32 * 1000.0)) * 100.0;

    // B is FP16 (2 bytes each), A and C are FP32 (4 bytes each)
    double bytes_transferred = static_cast<double>(
        static_cast<int64_t>(M) * K * 4 +  // A: FP32
        static_cast<int64_t>(K) * N * 2 +  // B: FP16
        static_cast<int64_t>(M) * N * 4);  // C: FP32
    double eff_bw = (med_seconds > 0.0) ? (bytes_transferred / (med_seconds * 1e9)) : 0.0;
    double pct_bw = (eff_bw / PEAK_BW_GBS) * 100.0;

    return BenchResult{M, N, K, "fp16", med_ms, gflops, pct_peak, eff_bw, pct_bw};
}

// ── Run cuBLAS FP16 Benchmark ───────────────────────────────────────────
//
// Uses cublasGemmEx with FP16 B and FP32 A/C, matching our FP16 kernel's
// data layout for a fair comparison.

static BenchResult run_benchmark_cublas_fp16(int M, int N, int K,
                                              const BenchConfig& cfg) {
    Tensor A({M, K}, DType::FP32);
    Tensor C({M, N}, DType::FP32);

    __half* d_B = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(N) * K * sizeof(__half)));

    {
        std::vector<float> host_a(static_cast<size_t>(M) * K);
        std::vector<__half> host_b(static_cast<size_t>(N) * K);

        for (size_t i = 0; i < host_a.size(); ++i) {
            host_a[i] = static_cast<float>((i % 7) - 3) * 0.01f;
        }
        for (size_t i = 0; i < host_b.size(); ++i) {
            host_b[i] = __float2half(static_cast<float>((i % 11) - 5) * 0.01f);
        }

        A.copy_from_host(host_a.data(), host_a.size() * sizeof(float));
        CUINFER_CHECK_CUDA(cudaMemcpy(d_B, host_b.data(),
                                      host_b.size() * sizeof(__half),
                                      cudaMemcpyHostToDevice));
    }

    const float* d_A = A.data_as<float>();
    float* d_C = C.data_as<float>();

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        matmul_cublas_fp16(d_A, d_B, d_C, M, N, K, 0);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    std::vector<float> timings(cfg.bench_iters);

    for (int i = 0; i < cfg.bench_iters; ++i) {
        timer.start();
        matmul_cublas_fp16(d_A, d_B, d_C, M, N, K, 0);
        timer.stop();
        timings[i] = timer.elapsed_ms();
    }

    cudaFree(d_B);

    double med_ms = median(timings);
    double flops = 2.0 * M * N * K;
    double med_seconds = med_ms / 1000.0;
    double gflops = (med_seconds > 0.0) ? (flops / (med_seconds * 1e9)) : 0.0;
    double pct_peak = (gflops / (PEAK_TFLOPS_FP32 * 1000.0)) * 100.0;

    double bytes_transferred = static_cast<double>(
        static_cast<int64_t>(M) * K * 4 +  // A: FP32
        static_cast<int64_t>(K) * N * 2 +  // B: FP16
        static_cast<int64_t>(M) * N * 4);  // C: FP32
    double eff_bw = (med_seconds > 0.0) ? (bytes_transferred / (med_seconds * 1e9)) : 0.0;
    double pct_bw = (eff_bw / PEAK_BW_GBS) * 100.0;

    return BenchResult{M, N, K, "cuBLAS-fp16", med_ms, gflops, pct_peak, eff_bw, pct_bw};
}

// ── Print Formatted Table ───────────────────────────────────────────────

static void print_table_header() {
    printf("\n");
    printf("%-6s %-6s %-6s  %-12s  %10s  %10s  %8s  %10s  %8s\n",
           "M", "N", "K", "Variant", "Time (ms)", "GFLOPS", "% Peak", "BW (GB/s)", "% BW");
    printf("%-6s %-6s %-6s  %-12s  %10s  %10s  %8s  %10s  %8s\n",
           "------", "------", "------", "------------", "----------",
           "----------", "--------", "----------", "--------");
}

static void print_result(const BenchResult& r) {
    printf("%-6d %-6d %-6d  %-12s  %10.4f  %10.2f  %7.2f%%  %10.2f  %7.2f%%\n",
           r.M, r.N, r.K, r.variant.c_str(),
           r.median_ms, r.gflops, r.pct_peak_gflops,
           r.eff_bw_gbs, r.pct_peak_bw);
}

// ── Default Dimension Sweep ─────────────────────────────────────────────
//
// The sweep includes both square and non-square dimensions to expose
// different performance characteristics:
//   - Small (128): tests kernel launch overhead relative to compute
//   - Medium (512, 1024): typical layer sizes in small transformer models
//   - Large (2048, 4096): representative of larger models; stresses memory
//
// Non-square cases test the boundary handling in tiled/vectorized kernels
// and reveal performance asymmetries (e.g., tall-skinny vs short-wide).

struct MatrixDims {
    int M, N, K;
};

static std::vector<MatrixDims> default_sweep() {
    return {
        // Square matrices: classic scaling test
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},

        // Non-square: tall-skinny (typical of MLP projections in transformers)
        // e.g., hidden_dim=768, intermediate=3072 -> (batch*seq, 3072, 768)
        {512, 768, 3072},
        {512, 3072, 768},

        // Non-square: batch matmul shapes (attention: seq x seq x head_dim)
        {1024, 64, 1024},
        {1024, 1024, 64},

        // Odd dimensions: stress-test boundary handling in tiled kernels.
        // Tiles are 32x32 (tiled) or 32x32 with K=8 strips (vectorized).
        // A dimension like 1000 is not a multiple of 32, forcing partial tiles.
        {1000, 1000, 1000},
    };
}

// ── Estimate VRAM needed for a single benchmark point ───────────────────
//
// Returns bytes needed for A + B + C. Used to skip configurations that
// would exceed VRAM (e.g., 4096^2 matrices on a 6 GB card with other
// allocations already present).

static size_t estimate_vram_bytes(int M, int N, int K) {
    return (static_cast<size_t>(M) * K +
            static_cast<size_t>(K) * N +
            static_cast<size_t>(M) * N) * sizeof(float);
}

// ── Main ────────────────────────────────────────────────────────────────

} // namespace cuinfer

int main(int argc, char** argv) {
    using namespace cuinfer;

    BenchConfig cfg = parse_args(argc, argv);

    // Initialize cuBLAS handle (needed even if --skip-cublas, to avoid
    // conditional logic — handle creation is cheap)
    if (!cfg.skip_cublas) {
        init_cublas();
    }

    // Print device info and initial VRAM state
    printf("========================================================\n");
    printf("  CuInfer — Matrix Multiplication Benchmark\n");
    printf("========================================================\n\n");
    print_device_info();
    printf("\n");
    print_vram_usage("Before benchmarks");

    // Query available VRAM for safety checks
    size_t free_bytes = 0, total_bytes = 0;
    CUINFER_CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));

    // Reserve 512 MB of headroom to avoid OOM from fragmentation or driver
    // overhead. On a 6 GB card this leaves ~5.5 GB for benchmark data.
    const size_t vram_headroom = 512ULL * 1024 * 1024;
    const size_t usable_vram = (free_bytes > vram_headroom)
                                   ? (free_bytes - vram_headroom) : 0;

    printf("[Config] Warmup: %d iters, Benchmark: %d iters, Skip naive: %s, Skip cuBLAS: %s\n",
           cfg.warmup_iters, cfg.bench_iters,
           cfg.skip_naive ? "yes" : "no",
           cfg.skip_cublas ? "yes" : "no");
    printf("[Config] Usable VRAM budget: %.2f MB\n\n",
           usable_vram / (1024.0 * 1024.0));

    printf("Reference peaks (RTX 3060):\n");
    printf("  FP32 compute: %.2f TFLOPS\n", PEAK_TFLOPS_FP32);
    printf("  Memory BW:    %.1f GB/s\n\n", PEAK_BW_GBS);

    // Build the dimension sweep
    std::vector<MatrixDims> dims;
    if (cfg.custom_m > 0 && cfg.custom_n > 0 && cfg.custom_k > 0) {
        dims.push_back({cfg.custom_m, cfg.custom_n, cfg.custom_k});
    } else {
        dims = default_sweep();
    }

    // Build kernel variant list
    auto variants = get_variants(cfg.skip_naive, cfg.skip_cublas);

    // Collect all results for potential post-processing
    std::vector<BenchResult> all_results;
    all_results.reserve(dims.size() * variants.size());

    // Run the sweep
    print_table_header();

    for (const auto& d : dims) {
        size_t needed = estimate_vram_bytes(d.M, d.N, d.K);

        for (const auto& variant : variants) {
            // Skip configurations that would exceed VRAM.
            // The naive kernel is also prohibitively slow at very large sizes,
            // so we skip it above 2048 by default to keep total benchmark
            // time reasonable.
            if (needed > usable_vram) {
                printf("%-6d %-6d %-6d  %-12s  %10s  %10s  %8s  %10s  %8s\n",
                       d.M, d.N, d.K, variant.name.c_str(),
                       "SKIP-OOM", "--", "--", "--", "--");
                continue;
            }

            // Optionally skip naive for very large dimensions where it would
            // take an unreasonable amount of time (minutes per run).
            if (variant.name == "naive" && d.M * d.N > 2048 * 2048) {
                printf("%-6d %-6d %-6d  %-12s  %10s  %10s  %8s  %10s  %8s\n",
                       d.M, d.N, d.K, variant.name.c_str(),
                       "SKIP-SLOW", "--", "--", "--", "--");
                continue;
            }

            BenchResult result = run_benchmark(d.M, d.N, d.K, variant, cfg);
            all_results.push_back(result);
            print_result(result);
        }

        // FP16 mixed-precision variant (B is FP16, A and C are FP32)
        // Only run if K is divisible by 32 (tile size requirement)
        if (needed <= usable_vram && d.K % 32 == 0) {
            BenchResult fp16_result = run_benchmark_fp16(d.M, d.N, d.K, cfg);
            all_results.push_back(fp16_result);
            print_result(fp16_result);

            // cuBLAS FP16 comparison
            if (!cfg.skip_cublas) {
                BenchResult cublas_fp16 = run_benchmark_cublas_fp16(d.M, d.N, d.K, cfg);
                all_results.push_back(cublas_fp16);
                print_result(cublas_fp16);
            }
        }

        // Visual separator between dimension groups
        printf("\n");
    }

    // ── Summary: best variant per dimension ─────────────────────────────
    printf("========================================================\n");
    printf("  Summary — Fastest Variant Per Dimension\n");
    printf("========================================================\n\n");
    printf("%-6s %-6s %-6s  %-12s  %10s  %10s\n",
           "M", "N", "K", "Best", "GFLOPS", "% Peak");
    printf("%-6s %-6s %-6s  %-12s  %10s  %10s\n",
           "------", "------", "------", "------------", "----------", "----------");

    // Group results by (M, N, K) and find the fastest
    for (const auto& d : dims) {
        double best_gflops = -1.0;
        std::string best_name;
        for (const auto& r : all_results) {
            if (r.M == d.M && r.N == d.N && r.K == d.K && r.gflops > best_gflops) {
                best_gflops = r.gflops;
                best_name = r.variant;
            }
        }
        if (best_gflops > 0.0) {
            double pct = (best_gflops / (PEAK_TFLOPS_FP32 * 1000.0)) * 100.0;
            printf("%-6d %-6d %-6d  %-12s  %10.2f  %9.2f%%\n",
                   d.M, d.N, d.K, best_name.c_str(), best_gflops, pct);
        }
    }

    // ── Speedup of vectorized over tiled ────────────────────────────────
    printf("\n========================================================\n");
    printf("  Speedup — Vectorized vs Tiled\n");
    printf("========================================================\n\n");
    printf("%-6s %-6s %-6s  %12s  %12s  %8s\n",
           "M", "N", "K", "Tiled (ms)", "Vec (ms)", "Speedup");
    printf("%-6s %-6s %-6s  %12s  %12s  %8s\n",
           "------", "------", "------", "------------", "------------", "--------");

    for (const auto& d : dims) {
        double tiled_ms = -1.0;
        double vec_ms = -1.0;

        for (const auto& r : all_results) {
            if (r.M == d.M && r.N == d.N && r.K == d.K) {
                if (r.variant == "tiled") tiled_ms = r.median_ms;
                if (r.variant == "vectorized") vec_ms = r.median_ms;
            }
        }

        if (tiled_ms > 0.0 && vec_ms > 0.0) {
            double speedup = tiled_ms / vec_ms;
            printf("%-6d %-6d %-6d  %12.4f  %12.4f  %7.2fx\n",
                   d.M, d.N, d.K, tiled_ms, vec_ms, speedup);
        }
    }

    // ── cuBLAS vs Custom comparison ─────────────────────────────────────
    if (!cfg.skip_cublas) {
        printf("\n========================================================\n");
        printf("  cuBLAS vs Custom — FP32\n");
        printf("========================================================\n\n");
        printf("%-6s %-6s %-6s  %12s  %12s  %12s  %8s\n",
               "M", "N", "K", "Custom (ms)", "cuBLAS (ms)", "cuBLAS GF", "Ratio");
        printf("%-6s %-6s %-6s  %12s  %12s  %12s  %8s\n",
               "------", "------", "------", "------------", "------------",
               "------------", "--------");

        for (const auto& d : dims) {
            double custom_ms = -1.0;
            double cublas_ms = -1.0;
            double cublas_gf = -1.0;

            for (const auto& r : all_results) {
                if (r.M == d.M && r.N == d.N && r.K == d.K) {
                    if (r.variant == "vectorized") custom_ms = r.median_ms;
                    if (r.variant == "cuBLAS") {
                        cublas_ms = r.median_ms;
                        cublas_gf = r.gflops;
                    }
                }
            }

            if (custom_ms > 0.0 && cublas_ms > 0.0) {
                double ratio = custom_ms / cublas_ms;
                printf("%-6d %-6d %-6d  %12.4f  %12.4f  %12.2f  %7.2fx\n",
                       d.M, d.N, d.K, custom_ms, cublas_ms, cublas_gf, ratio);
            }
        }

        printf("\n========================================================\n");
        printf("  cuBLAS vs Custom — FP16 (mixed precision)\n");
        printf("========================================================\n\n");
        printf("%-6s %-6s %-6s  %12s  %12s  %12s  %8s\n",
               "M", "N", "K", "Custom (ms)", "cuBLAS (ms)", "cuBLAS GF", "Ratio");
        printf("%-6s %-6s %-6s  %12s  %12s  %12s  %8s\n",
               "------", "------", "------", "------------", "------------",
               "------------", "--------");

        for (const auto& d : dims) {
            double custom_ms = -1.0;
            double cublas_ms = -1.0;
            double cublas_gf = -1.0;

            for (const auto& r : all_results) {
                if (r.M == d.M && r.N == d.N && r.K == d.K) {
                    if (r.variant == "fp16") custom_ms = r.median_ms;
                    if (r.variant == "cuBLAS-fp16") {
                        cublas_ms = r.median_ms;
                        cublas_gf = r.gflops;
                    }
                }
            }

            if (custom_ms > 0.0 && cublas_ms > 0.0) {
                double ratio = custom_ms / cublas_ms;
                printf("%-6d %-6d %-6d  %12.4f  %12.4f  %12.2f  %7.2fx\n",
                       d.M, d.N, d.K, custom_ms, cublas_ms, cublas_gf, ratio);
            }
        }
    }

    // ── Final VRAM report ───────────────────────────────────────────────
    printf("\n");
    print_vram_usage("After benchmarks (all temporaries freed)");

    printf("\n========================================================\n");
    printf("  Benchmark complete.\n");
    printf("========================================================\n");

    destroy_cublas();

    return 0;
}
