#pragma once

#include <cuda_runtime.h>

namespace cuinfer {

// ── Matrix Multiplication Host Wrappers ────────────────────────────────
//
// All functions compute C = A * B where:
//   A is [M x K], B is [K x N], C is [M x N]
//   Row-major layout: A[i*K+k], B[k*N+j], C[i*N+j]
//   All pointers must reside in device (GPU) memory.
//
// Each wrapper configures grid/block dimensions and launches the
// corresponding CUDA kernel, followed by an error check.

/// Naive matmul: one thread per output element, no shared memory.
/// Simple but memory-bound — every thread reads an entire row of A
/// and an entire column of B from global memory.
void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream = 0);

/// Tiled matmul: loads TILE_SIZE x TILE_SIZE tiles into shared memory
/// to exploit data reuse. Each element of A and B is loaded from global
/// memory ~(K/TILE_SIZE) times fewer than the naive approach.
void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream = 0);

/// Vectorized matmul: extends the tiled approach with float4 loads
/// for coalesced 128-bit memory transactions on B. Each thread computes
/// multiple output elements to increase arithmetic intensity.
void matmul_vectorized(const float* A, const float* B, float* C,
                       int M, int N, int K, cudaStream_t stream = 0);

} // namespace cuinfer
