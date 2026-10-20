// ── test_matmul.cu — Unit Tests for MatMul Kernels ─────────────────────
//
// Tests all three matmul variants (naive, tiled, vectorized) against a
// CPU reference implementation. Each test:
//   1. Generates random FP32 input matrices (seeded for reproducibility)
//   2. Computes the reference result on CPU
//   3. Uploads inputs to GPU, runs the kernel, downloads the result
//   4. Compares element-wise with tolerance 1e-4

#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>

#include "cuinfer/kernels/matmul.h"
#include "cuinfer/cuda_utils.h"

// ── CPU Reference Implementation ───────────────────────────────────────

static void cpu_matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ── RAII GPU Buffer ────────────────────────────────────────────────────
// Simple wrapper to ensure cudaFree is always called, even on test failure.

struct GpuBuffer {
    float* ptr = nullptr;
    GpuBuffer(size_t n) { CUINFER_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float))); }
    ~GpuBuffer() { if (ptr) cudaFree(ptr); }
    GpuBuffer(GpuBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) { if (ptr) cudaFree(ptr); ptr = other.ptr; other.ptr = nullptr; }
        return *this;
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
};

// ── Test Helper ────────────────────────────────────────────────────────
// Fills a host vector with random floats in [0, 1].

static void fill_random(std::vector<float>& v) {
    for (auto& x : v) {
        x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Tolerance for FP32 comparison.
// Matmul accumulates K products; with K up to 1024 and random inputs
// in [0,1], the accumulated rounding error stays well under 1e-4.
constexpr float TOL = 1e-4f;

// ── Helper: run a matmul kernel and compare against CPU reference ──────
// The `kernel_fn` parameter accepts any of the three wrapper signatures.

using MatmulFn = void(*)(const float*, const float*, float*, int, int, int, cudaStream_t);

static void run_and_verify(MatmulFn kernel_fn, int M, int N, int K) {
    srand(42);

    // Host allocations
    std::vector<float> h_A(M * K), h_B(K * N), h_C_gpu(M * N), h_C_ref(M * N);
    fill_random(h_A);
    fill_random(h_B);

    // CPU reference
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // GPU allocations
    GpuBuffer d_A(M * K), d_B(K * N), d_C(M * N);

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A.ptr, h_A.data(),
                                  M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B.ptr, h_B.data(),
                                  K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemset(d_C.ptr, 0, M * N * sizeof(float)));

    // Run kernel
    kernel_fn(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K, 0);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Download result
    CUINFER_CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C.ptr,
                                  M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify element-wise
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(h_C_gpu[i], h_C_ref[i], TOL)
            << "Mismatch at flat index " << i
            << " (row=" << i / N << ", col=" << i % N << ")"
            << ": gpu=" << h_C_gpu[i] << " cpu=" << h_C_ref[i];
    }
}

// ════════════════════════════════════════════════════════════════════════
// Naive MatMul Tests
// ════════════════════════════════════════════════════════════════════════

TEST(MatmulNaive, Square256) {
    run_and_verify(cuinfer::matmul_naive, 256, 256, 256);
}

TEST(MatmulNaive, NonSquare) {
    run_and_verify(cuinfer::matmul_naive, 128, 256, 64);
}

TEST(MatmulNaive, Small) {
    // Edge case: 1x1 matrices — single element
    run_and_verify(cuinfer::matmul_naive, 1, 1, 1);
}

// ════════════════════════════════════════════════════════════════════════
// Tiled MatMul Tests
// ════════════════════════════════════════════════════════════════════════

TEST(MatmulTiled, Square256) {
    run_and_verify(cuinfer::matmul_tiled, 256, 256, 256);
}

TEST(MatmulTiled, Square1024) {
    run_and_verify(cuinfer::matmul_tiled, 1024, 1024, 1024);
}

TEST(MatmulTiled, NonSquare) {
    run_and_verify(cuinfer::matmul_tiled, 128, 256, 64);
}

TEST(MatmulTiled, NonAligned) {
    // M=100, N=200, K=150 — not multiples of TILE_SIZE (32)
    // Tests boundary handling in shared memory tile loads
    run_and_verify(cuinfer::matmul_tiled, 100, 200, 150);
}

// ════════════════════════════════════════════════════════════════════════
// Vectorized MatMul Tests
// ════════════════════════════════════════════════════════════════════════

TEST(MatmulVectorized, Square256) {
    run_and_verify(cuinfer::matmul_vectorized, 256, 256, 256);
}

TEST(MatmulVectorized, Square1024) {
    run_and_verify(cuinfer::matmul_vectorized, 1024, 1024, 1024);
}

TEST(MatmulVectorized, NonSquare) {
    run_and_verify(cuinfer::matmul_vectorized, 128, 256, 64);
}
