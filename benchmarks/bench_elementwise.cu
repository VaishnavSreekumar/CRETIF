// ============================================================================
// bench_elementwise.cu — CuInfer Elementwise Kernel Benchmarks
// ============================================================================
//
// Measures throughput (effective bandwidth in GB/s) for every elementwise
// kernel in src/kernels/elementwise.cu, then compares against the theoretical
// peak memory bandwidth of the RTX 3060 (336.5 GB/s).
//
// Methodology
// -----------
//   - Timing uses CUDA events (CudaTimer from cuda_utils.h) — no wall-clock.
//   - 10 warmup iterations are run to stabilize clocks / caches.
//   - 100 timed iterations are collected; we report the *median* to be robust
//     against outliers (thermal throttling spikes, OS interrupts, etc.).
//   - VRAM usage is printed at the beginning and end via print_vram_usage().
//   - Effective bandwidth is calculated as:
//         bytes_moved / median_time_seconds
//     where bytes_moved accounts for all reads *and* writes the kernel
//     performs, as documented per-kernel below.
//
// Build
// -----
//   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
//   cmake --build build --config Release -j$(nproc)
//   ./build/benchmarks/bench_elementwise [--warmup N] [--iters N]
//
// ============================================================================

#include "cuinfer/cuda_utils.h"
#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace cuinfer {

// ── Constants ────────────────────────────────────────────────────────────────

// RTX 3060 theoretical peak memory bandwidth (GB/s).
// Source: NVIDIA spec — 192-bit bus @ 15 Gbps GDDR6 = 360.0 GB/s rated,
// but effective benchmark ceiling is ~336.5 GB/s due to ECC overhead / bus
// efficiency.  We use 360.0 as the theoretical number and let the percentage
// speak for itself.
static constexpr double PEAK_BW_GBS = 360.0;

// Default benchmark parameters.
static constexpr int DEFAULT_WARMUP = 10;
static constexpr int DEFAULT_ITERS  = 100;

// ── CLI Argument Parsing ─────────────────────────────────────────────────────

struct BenchConfig {
    int warmup_iters = DEFAULT_WARMUP;
    int timed_iters  = DEFAULT_ITERS;
};

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if ((std::string(argv[i]) == "--warmup") && i + 1 < argc) {
            cfg.warmup_iters = std::atoi(argv[++i]);
        } else if ((std::string(argv[i]) == "--iters") && i + 1 < argc) {
            cfg.timed_iters = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printf("Usage: bench_elementwise [--warmup N] [--iters N]\n");
            printf("  --warmup N   Number of warmup iterations (default: %d)\n", DEFAULT_WARMUP);
            printf("  --iters  N   Number of timed iterations  (default: %d)\n", DEFAULT_ITERS);
            std::exit(0);
        }
    }
    return cfg;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the median of a sorted-in-place vector of floats.
static float median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

/// Fill a Tensor with pseudo-random values in [-1, 1].
/// Uses a simple LCG so results are reproducible without needing cuRAND.
static void fill_random(Tensor& t) {
    int64_t n = t.numel();
    std::vector<float> host(n);
    uint32_t state = 42u;  // seed
    for (int64_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;  // Numerical Recipes LCG
        host[i] = (static_cast<float>(state & 0x00FFFFFFu) / 0x00FFFFFFu) * 2.0f - 1.0f;
    }
    t.copy_from_host(host.data(), n * sizeof(float));
}

/// Print a formatted header row for a benchmark table.
static void print_table_header(const char* kernel_name) {
    printf("\n");
    printf("========================================================================\n");
    printf("  %s\n", kernel_name);
    printf("========================================================================\n");
    printf("  %-8s  %-8s  %-12s  %-12s  %-14s  %-8s\n",
           "Rows", "Cols", "Elements", "Median (us)", "BW (GB/s)", "% Peak");
    printf("------------------------------------------------------------------------\n");
}

/// Print one result row.
static void print_table_row(int rows, int cols, int64_t elements,
                            float median_us, double bw_gbs) {
    double pct = 100.0 * bw_gbs / PEAK_BW_GBS;
    printf("  %-8d  %-8d  %-12lld  %-12.2f  %-14.2f  %5.1f%%\n",
           rows, cols, static_cast<long long>(elements),
           median_us, bw_gbs, pct);
}

// ── Per-Kernel Benchmark Routines ────────────────────────────────────────────
//
// Each routine allocates the necessary tensors, runs warmup + timed
// iterations, collects timings, and prints the formatted result row.
//
// Effective bandwidth calculation per kernel:
//   - RMSNorm:  reads input (rows*n) + weight (n), writes output (rows*n)
//               → bytes = (2*rows*n + n) * 4
//   - SiLU:     reads input (n), writes output (n) → bytes = 2*n*4
//   - GeLU:     reads input (n), writes output (n) → bytes = 2*n*4
//   - Softmax:  reads input (rows*n), writes output (rows*n) → bytes = 2*rows*n*4
//               (ignores internal shared-mem traffic — we measure DRAM BW only)
//   - RoPE:     reads and writes qk in-place (n_heads*dim) → bytes = 2*n_heads*dim*4
//   - Add:      reads a (n) + b (n), writes output (n) → bytes = 3*n*4
//   - Multiply: reads a (n) + b (n), writes output (n) → bytes = 3*n*4

static void bench_rmsnorm(int rows, int cols, const BenchConfig& cfg) {
    // Allocate tensors: input [rows x cols], weight [cols], output [rows x cols]
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor weight({static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);
    fill_random(weight);

    // Effective bytes: read input + weight, write output
    // Weight is read once but cached — count it once per call.
    double bytes = (2.0 * rows * cols + cols) * sizeof(float);

    CudaTimer timer;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        rmsnorm(input.data_as<float>(), weight.data_as<float>(),
                output.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        rmsnorm(input.data_as<float>(), weight.data_as<float>(),
                output.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

static void bench_silu(int rows, int cols, const BenchConfig& cfg) {
    int64_t n = static_cast<int64_t>(rows) * cols;
    Tensor input({n});
    Tensor output({n});
    fill_random(input);

    // Read input + write output = 2 * n * 4 bytes
    double bytes = 2.0 * n * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        silu(input.data_as<float>(), output.data_as<float>(), static_cast<int>(n));
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        silu(input.data_as<float>(), output.data_as<float>(), static_cast<int>(n));
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    print_table_row(rows, cols, n, med_us, bw_gbs);
}

static void bench_gelu(int rows, int cols, const BenchConfig& cfg) {
    int64_t n = static_cast<int64_t>(rows) * cols;
    Tensor input({n});
    Tensor output({n});
    fill_random(input);

    double bytes = 2.0 * n * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        gelu(input.data_as<float>(), output.data_as<float>(), static_cast<int>(n));
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        gelu(input.data_as<float>(), output.data_as<float>(), static_cast<int>(n));
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    print_table_row(rows, cols, n, med_us, bw_gbs);
}

static void bench_softmax(int rows, int cols, const BenchConfig& cfg) {
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);

    // Reads input + writes output (DRAM perspective, ignoring shared mem).
    double bytes = 2.0 * rows * cols * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        softmax(input.data_as<float>(), output.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        softmax(input.data_as<float>(), output.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

/// RoPE benchmark.  Uses realistic head counts / head dimensions matching
/// common transformer configs:
///   - Small  (SmolLM-135M style):  12 heads x  64 dim
///   - Medium (LLaMA-7B style):     32 heads x 128 dim
///
/// The "rows" parameter here is repurposed as batch_tokens — the number of
/// independent position applications.  We run the kernel once per token
/// position inside the timed region to amortize launch overhead.
///
/// Effective bytes: each call reads + writes n_heads * dim floats in-place
/// → 2 * n_heads * dim * 4 bytes per call.
static void bench_rope(int n_heads, int head_dim, int n_positions,
                       const BenchConfig& cfg) {
    int64_t n = static_cast<int64_t>(n_heads) * head_dim;
    Tensor qk({n});
    fill_random(qk);

    // Bytes for one rope() call: read + write the full qk buffer in-place.
    double bytes_per_call = 2.0 * n * sizeof(float);
    // We will call rope() n_positions times in one timed iteration.
    double total_bytes = bytes_per_call * n_positions;

    CudaTimer timer;

    // Warmup
    for (int w = 0; w < cfg.warmup_iters; ++w) {
        for (int p = 0; p < n_positions; ++p) {
            rope(qk.data_as<float>(), p, head_dim, n_heads);
        }
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        // Re-fill between iterations so the data is realistic.
        // (RoPE is in-place; repeated application would diverge.)
        fill_random(qk);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        timer.start();
        for (int p = 0; p < n_positions; ++p) {
            rope(qk.data_as<float>(), p, head_dim, n_heads);
        }
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (total_bytes / (med_ms * 1e-3)) / 1e9;

    // Print with heads*dim as "cols" and n_positions as "rows" for clarity.
    int64_t total_elements = n * n_positions;
    print_table_row(n_positions, static_cast<int>(n), total_elements, med_us, bw_gbs);
}

static void bench_add(int rows, int cols, const BenchConfig& cfg) {
    int64_t n = static_cast<int64_t>(rows) * cols;
    Tensor a({n});
    Tensor b({n});
    Tensor output({n});
    fill_random(a);
    fill_random(b);

    // Reads a + b, writes output = 3 * n * 4 bytes
    double bytes = 3.0 * n * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        add(a.data_as<float>(), b.data_as<float>(),
            output.data_as<float>(), static_cast<int>(n));
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        add(a.data_as<float>(), b.data_as<float>(),
            output.data_as<float>(), static_cast<int>(n));
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    print_table_row(rows, cols, n, med_us, bw_gbs);
}

static void bench_multiply(int rows, int cols, const BenchConfig& cfg) {
    int64_t n = static_cast<int64_t>(rows) * cols;
    Tensor a({n});
    Tensor b({n});
    Tensor output({n});
    fill_random(a);
    fill_random(b);

    // Reads a + b, writes output = 3 * n * 4 bytes
    double bytes = 3.0 * n * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        multiply(a.data_as<float>(), b.data_as<float>(),
                 output.data_as<float>(), static_cast<int>(n));
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        multiply(a.data_as<float>(), b.data_as<float>(),
                 output.data_as<float>(), static_cast<int>(n));
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    print_table_row(rows, cols, n, med_us, bw_gbs);
}

// ── CUB-Accelerated Benchmark Routines ────────────────────────────────────────
//
// Side-by-side comparison of hand-written vs CUB reductions.

static void bench_rmsnorm_cub(int rows, int cols, const BenchConfig& cfg) {
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor weight({static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);
    fill_random(weight);

    double bytes = (2.0 * rows * cols + cols) * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        rmsnorm_cub(input.data_as<float>(), weight.data_as<float>(),
                    output.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        rmsnorm_cub(input.data_as<float>(), weight.data_as<float>(),
                    output.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

static void bench_softmax_cub(int rows, int cols, const BenchConfig& cfg) {
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);

    double bytes = 2.0 * rows * cols * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        softmax_cub(input.data_as<float>(), output.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        softmax_cub(input.data_as<float>(), output.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

// ── Fused RMSNorm + Residual Add ────────────────────────────────────────────
//
// Benchmarks the fused kernel that combines residual addition with RMSNorm
// into a single kernel launch. Compared against separate add() + rmsnorm().
//
// Effective bytes:
//   reads input [rows*n] + residual [rows*n] + weight [n]
//   writes output [rows*n] + normed [rows*n]
//   → (4*rows*n + n) * sizeof(float)
//
// Separate (unfused) would read/write the intermediate hidden state an
// extra time: (5*rows*n + 2*n) bytes. The fused kernel saves rows*n reads.

static void bench_rmsnorm_residual(int rows, int cols, const BenchConfig& cfg) {
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor residual({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor weight({static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor normed({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);
    fill_random(residual);
    fill_random(weight);

    // Effective bytes: read input + residual + weight, write output + normed
    double bytes = (4.0 * rows * cols + cols) * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        rmsnorm_residual(input.data_as<float>(), residual.data_as<float>(),
                         weight.data_as<float>(), output.data_as<float>(),
                         normed.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        rmsnorm_residual(input.data_as<float>(), residual.data_as<float>(),
                         weight.data_as<float>(), output.data_as<float>(),
                         normed.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

static void bench_rmsnorm_residual_cub(int rows, int cols, const BenchConfig& cfg) {
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor residual({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor weight({static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor normed({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);
    fill_random(residual);
    fill_random(weight);

    double bytes = (4.0 * rows * cols + cols) * sizeof(float);

    CudaTimer timer;

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        rmsnorm_residual_cub(input.data_as<float>(), residual.data_as<float>(),
                             weight.data_as<float>(), output.data_as<float>(),
                             normed.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        rmsnorm_residual_cub(input.data_as<float>(), residual.data_as<float>(),
                             weight.data_as<float>(), output.data_as<float>(),
                             normed.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

/// Benchmark for unfused add() + rmsnorm() as a comparison baseline.
static void bench_add_then_rmsnorm(int rows, int cols, const BenchConfig& cfg) {
    Tensor input({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor residual({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor weight({static_cast<int64_t>(cols)});
    Tensor output({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    Tensor normed({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    fill_random(input);
    fill_random(residual);
    fill_random(weight);

    // Same logical bytes moved as fused (for fair comparison)
    double bytes = (4.0 * rows * cols + cols) * sizeof(float);

    CudaTimer timer;

    int n = rows * cols;
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        add(input.data_as<float>(), residual.data_as<float>(),
            output.data_as<float>(), n);
        rmsnorm(output.data_as<float>(), weight.data_as<float>(),
                normed.data_as<float>(), rows, cols);
    }
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(cfg.timed_iters);
    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start();
        add(input.data_as<float>(), residual.data_as<float>(),
            output.data_as<float>(), n);
        rmsnorm(output.data_as<float>(), weight.data_as<float>(),
                normed.data_as<float>(), rows, cols);
        timer.stop();
        times[i] = timer.elapsed_ms();
    }

    float med_ms = median(times);
    float med_us = med_ms * 1000.0f;
    double bw_gbs = (bytes / (med_ms * 1e-3)) / 1e9;

    int64_t elements = static_cast<int64_t>(rows) * cols;
    print_table_row(rows, cols, elements, med_us, bw_gbs);
}

// ── Size Sweep Configuration ─────────────────────────────────────────────────
//
// We sweep across typical transformer hidden dimensions (cols) and batch/sequence
// sizes (rows).  These cover the range from tiny (SmolLM-135M, hidden=768) to
// moderate (LLaMA-7B, hidden=4096) to stress-test sizes (hidden=8192).

static const std::vector<int> COL_SIZES = {768, 2048, 4096, 8192};
static const std::vector<int> ROW_SIZES = {1, 32, 128, 512};

// RoPE configs: {n_heads, head_dim, label}
struct RopeConfig {
    int n_heads;
    int head_dim;
    const char* label;
};

static const std::vector<RopeConfig> ROPE_CONFIGS = {
    {12,  64,  "Small  (12 heads x  64 dim)"},
    {32, 128,  "Medium (32 heads x 128 dim)"},
};

// Number of positions to process per RoPE timed iteration.
static const std::vector<int> ROPE_POSITIONS = {1, 32, 128, 512};

} // namespace cuinfer


// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    using namespace cuinfer;

    BenchConfig cfg = parse_args(argc, argv);

    // Print device info and initial VRAM state.
    printf("\n");
    print_device_info();
    printf("\n");
    printf("Benchmark config: %d warmup, %d timed iterations (median reported)\n",
           cfg.warmup_iters, cfg.timed_iters);
    printf("Peak memory bandwidth (theoretical): %.1f GB/s\n", PEAK_BW_GBS);
    print_vram_usage("Before allocation");

    // ── RMSNorm (hand-written) ─────────────────────────────────────────────
    // Bandwidth: reads input [rows*cols] + weight [cols], writes output [rows*cols]
    print_table_header("RMSNorm (hand-written)  —  (2*rows*cols + cols) * 4 bytes moved");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_rmsnorm(rows, cols, cfg);
        }
    }

    // ── RMSNorm (CUB) ────────────────────────────────────────────────────
    print_table_header("RMSNorm (CUB BlockReduce)  —  (2*rows*cols + cols) * 4 bytes moved");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_rmsnorm_cub(rows, cols, cfg);
        }
    }

    // ── SiLU ─────────────────────────────────────────────────────────────────
    // Bandwidth: reads input [n], writes output [n]
    print_table_header("SiLU  —  2 * n * 4 bytes moved");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_silu(rows, cols, cfg);
        }
    }

    // ── GeLU ─────────────────────────────────────────────────────────────────
    // Bandwidth: reads input [n], writes output [n]
    print_table_header("GeLU  —  2 * n * 4 bytes moved");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_gelu(rows, cols, cfg);
        }
    }

    // ── Softmax (hand-written) ─────────────────────────────────────────────
    // Bandwidth: reads input [rows*cols], writes output [rows*cols]
    // Note: the kernel does 3 passes over the data, but from a DRAM perspective
    // the L2 cache should absorb the re-reads for moderate row lengths.  We report
    // the logical read+write to DRAM (i.e., as if data is only transferred once).
    print_table_header("Softmax (hand-written)  —  2 * rows * cols * 4 bytes moved (DRAM view)");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_softmax(rows, cols, cfg);
        }
    }

    // ── Softmax (CUB) ────────────────────────────────────────────────────
    print_table_header("Softmax (CUB BlockReduce)  —  2 * rows * cols * 4 bytes moved (DRAM view)");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_softmax_cub(rows, cols, cfg);
        }
    }

    // ── RoPE ─────────────────────────────────────────────────────────────────
    // In-place operation: reads + writes n_heads * dim floats per position.
    // We sweep across head configs and number of positions.
    for (const auto& rc : ROPE_CONFIGS) {
        char label_buf[128];
        snprintf(label_buf, sizeof(label_buf),
                 "RoPE %s  —  2 * %d * %d * 4 * n_pos bytes moved",
                 rc.label, rc.n_heads, rc.head_dim);
        print_table_header(label_buf);

        // Print column headers adjusted for RoPE: "Rows" = positions, "Cols" = n_heads*dim
        printf("  (Rows = positions, Cols = n_heads * head_dim)\n\n");

        for (int n_pos : ROPE_POSITIONS) {
            bench_rope(rc.n_heads, rc.head_dim, n_pos, cfg);
        }
    }

    // ── Add ──────────────────────────────────────────────────────────────────
    // Bandwidth: reads a [n] + b [n], writes output [n]
    print_table_header("Add  —  3 * n * 4 bytes moved");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_add(rows, cols, cfg);
        }
    }

    // ── Multiply ─────────────────────────────────────────────────────────────
    // Bandwidth: reads a [n] + b [n], writes output [n]
    print_table_header("Multiply  —  3 * n * 4 bytes moved");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_multiply(rows, cols, cfg);
        }
    }

    // ── Fused RMSNorm + Residual Add (hand-written) ──────────────────────────
    print_table_header("Fused RMSNorm+Residual (hand-written)  —  (4*rows*cols + cols) * 4 bytes");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_rmsnorm_residual(rows, cols, cfg);
        }
    }

    // ── Fused RMSNorm + Residual Add (CUB) ────────────────────────────────
    print_table_header("Fused RMSNorm+Residual (CUB)  —  (4*rows*cols + cols) * 4 bytes");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_rmsnorm_residual_cub(rows, cols, cfg);
        }
    }

    // ── Unfused add() + rmsnorm() baseline ────────────────────────────────
    print_table_header("Unfused add()+rmsnorm() baseline  —  comparison target");
    for (int cols : COL_SIZES) {
        for (int rows : ROW_SIZES) {
            bench_add_then_rmsnorm(rows, cols, cfg);
        }
    }

    // ── VRAM Summary ─────────────────────────────────────────────────────────
    printf("\n");
    printf("========================================================================\n");
    printf("  VRAM Summary\n");
    printf("========================================================================\n");
    print_vram_usage("After benchmarks");
    printf("\n");

    return 0;
}
