// ── matmul.cu — Matrix Multiplication CUDA Kernels ─────────────────────
//
// Three variants of C = A * B (FP32, row-major):
//   1. Naive     — one thread per output element, global memory only
//   2. Tiled     — shared memory tiles to reduce global memory traffic
//   3. Vectorized — float4 loads for coalesced 128-bit reads on B
//
// All kernels handle boundary conditions for non-tile-aligned dimensions.

#include "cuinfer/kernels/matmul.h"
#include "cuinfer/cuda_utils.h"

namespace cuinfer {

// ════════════════════════════════════════════════════════════════════════
// Kernel 1: Naive MatMul
// ════════════════════════════════════════════════════════════════════════
//
// Each thread computes exactly one element C[row][col] by walking the
// entire dot product A[row,:] . B[:,col] in global memory.
//
// Performance analysis:
//   - Each output element requires 2K global memory reads (K from A, K
//     from B), giving O(M*N*K) total reads for O(M*N*K) FLOPs.
//   - Arithmetic intensity ~0.25 FLOP/byte — deeply memory-bound.
//   - Column accesses on B are strided (stride = N), causing poor
//     coalescing within a warp when N is large.
//
// Launch config:
//   Block: 16x16 = 256 threads (good occupancy on SM 8.6)
//   Grid:  ceil(N/16) x ceil(M/16)

__global__ void kernel_matmul_naive(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ════════════════════════════════════════════════════════════════════════
// Kernel 2: Tiled MatMul (Shared Memory)
// ════════════════════════════════════════════════════════════════════════
//
// Key idea: instead of each thread independently reading from global
// memory, a block of threads cooperatively loads a TILE_SIZE x TILE_SIZE
// sub-matrix of A and B into shared memory. All threads in the block
// then read from shared memory (100x lower latency than global memory)
// to compute their partial dot products.
//
// Data reuse: each element loaded into shared memory is reused by
// TILE_SIZE threads, reducing global memory traffic by ~TILE_SIZE x
// compared to the naive kernel.
//
// Shared memory usage per block:
//   2 tiles * TILE_SIZE^2 * sizeof(float) = 2 * 32^2 * 4 = 8,192 bytes (8 KB)
//   RTX 3060 has 100 KB shared memory per SM -> up to 12 blocks per SM
//   (though register usage will be the practical limiter).
//
// Launch config:
//   Block: TILE_SIZE x TILE_SIZE = 32x32 = 1024 threads (max per block on SM 8.6)
//   Grid:  ceil(N/TILE_SIZE) x ceil(M/TILE_SIZE)

constexpr int TILE_SIZE = 32;

__global__ void kernel_matmul_tiled(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    // Shared memory tiles for A and B sub-matrices
    // Each tile is TILE_SIZE x TILE_SIZE floats = 4 KB, total 8 KB per block
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Sweep across the K dimension in TILE_SIZE-wide strips
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Cooperative load: each thread loads one element of tile_a and tile_b.
        // Boundary check: threads outside the matrix write 0 (neutral for addition).
        const int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            tile_b[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Ensure all threads have finished loading before computing
        __syncthreads();

        // Accumulate partial dot product from this tile
        // Fully unrolled since TILE_SIZE is a compile-time constant
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        // Ensure all threads have finished computing before the next
        // tile load overwrites shared memory
        __syncthreads();
    }

    // Write result (only threads within the output matrix bounds)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Kernel 3: Vectorized MatMul (float4 Loads)
// ════════════════════════════════════════════════════════════════════════
//
// Extends the tiled approach with two optimizations:
//
// 1. float4 loads on B: when reading B from global memory, we use
//    128-bit (float4) loads instead of 32-bit (float) loads. This
//    matches the GPU's 128-byte cache line and memory transaction size,
//    achieving full coalescing. A single float4 load fetches 4 floats
//    in one memory transaction — 4x fewer transactions for the same data.
//
// 2. Each thread computes a 1x4 micro-tile of C: thread (ty, tx)
//    computes C[row][col*4 + 0..3]. This increases arithmetic intensity
//    (more FLOPs per byte loaded) and amortizes the tile-loading overhead
//    across more useful work per thread.
//
// Memory access pattern for B:
//   - B is [K x N], row-major. Consecutive columns in B are contiguous
//     in memory: B[k*N + j], B[k*N + j+1], B[k*N + j+2], B[k*N + j+3].
//   - Threads within a warp have consecutive threadIdx.x values, so
//     adjacent threads access adjacent float4 groups — perfectly coalesced.
//
// Tile dimensions:
//   TILE_M = 32: number of rows processed per block
//   TILE_N = 32: number of columns processed per block (each thread handles 4)
//   TILE_K = 8:  K-dimension strip width per iteration
//
// Block: (TILE_N/4, TILE_M) = (8, 32) = 256 threads
// Each thread computes 4 output elements -> 32*32 = 1024 outputs per block
//
// Shared memory usage per block:
//   tile_a: TILE_M * TILE_K * 4 = 32 * 8 * 4 = 1,024 bytes
//   tile_b: TILE_K * TILE_N * 4 = 8 * 32 * 4 = 1,024 bytes
//   Total: 2,048 bytes (2 KB) — very light, enabling high occupancy.

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 8;

__global__ void kernel_matmul_vectorized(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    // Shared memory tiles
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N];

    // Block dimensions: blockDim.x = TILE_N/4 = 8, blockDim.y = TILE_M = 32
    // Each thread computes 4 consecutive columns in C
    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col_base = blockIdx.x * TILE_N + threadIdx.x * 4;

    // Accumulators for the 4 output elements this thread computes
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_offset = t * TILE_K;

        // ── Load tile_a ──
        // Block has 256 threads (8 x 32). tile_a is 32 x 8 = 256 elements.
        // Each thread loads exactly one element.
        // Thread (tx, ty) loads tile_a[ty][tx] (tx ranges 0..7, ty ranges 0..31)
        {
            const int a_k = k_offset + threadIdx.x;  // threadIdx.x in [0,7]
            if (threadIdx.x < TILE_K) {
                if (row < M && a_k < K) {
                    tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_k];
                } else {
                    tile_a[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
        }

        // ── Load tile_b ──
        // tile_b is 8 x 32 = 256 elements. 256 threads -> one element each.
        // Map thread linear ID to (tile_row, tile_col) in tile_b.
        {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..255
            const int b_tile_row = tid / TILE_N;  // 0..7
            const int b_tile_col = tid % TILE_N;  // 0..31

            const int b_k = k_offset + b_tile_row;
            const int b_n = blockIdx.x * TILE_N + b_tile_col;

            if (b_k < K && b_n < N) {
                tile_b[b_tile_row][b_tile_col] = B[b_k * N + b_n];
            } else {
                tile_b[b_tile_row][b_tile_col] = 0.0f;
            }
        }

        __syncthreads();

        // ── Compute partial dot products ──
        // Each thread accumulates across the TILE_K dimension for its 4 outputs.
        // Reading tile_a[threadIdx.y][k] is broadcast across threads in the same row.
        // Reading tile_b[k][threadIdx.x*4 + 0..3] is sequential in shared memory.
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_val = tile_a[threadIdx.y][k];
            sum0 += a_val * tile_b[k][threadIdx.x * 4 + 0];
            sum1 += a_val * tile_b[k][threadIdx.x * 4 + 1];
            sum2 += a_val * tile_b[k][threadIdx.x * 4 + 2];
            sum3 += a_val * tile_b[k][threadIdx.x * 4 + 3];
        }

        __syncthreads();
    }

    // ── Write results ──
    // Use float4 store when all 4 columns are within bounds and the address is
    // 16-byte aligned. Otherwise, fall back to scalar stores with bounds checks.
    if (row < M) {
        if (col_base + 3 < N) {
            // All 4 elements are within bounds.
            // float4 store: single 128-bit write — coalesced across the warp.
            // Alignment: col_base is threadIdx.x*4, and blockIdx.x*TILE_N is a
            // multiple of 32 (TILE_N=32). So col_base is always a multiple of 4,
            // meaning the address &C[row*N + col_base] is 16-byte aligned when
            // N is a multiple of 4. For non-aligned N we fall through to scalar.
            if ((col_base % 4 == 0) && (N % 4 == 0)) {
                float4* out_ptr = reinterpret_cast<float4*>(&C[row * N + col_base]);
                *out_ptr = make_float4(sum0, sum1, sum2, sum3);
            } else {
                C[row * N + col_base + 0] = sum0;
                C[row * N + col_base + 1] = sum1;
                C[row * N + col_base + 2] = sum2;
                C[row * N + col_base + 3] = sum3;
            }
        } else {
            // Boundary: fewer than 4 columns remaining. Write individually.
            if (col_base + 0 < N) C[row * N + col_base + 0] = sum0;
            if (col_base + 1 < N) C[row * N + col_base + 1] = sum1;
            if (col_base + 2 < N) C[row * N + col_base + 2] = sum2;
            if (col_base + 3 < N) C[row * N + col_base + 3] = sum3;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Host Wrappers
// ════════════════════════════════════════════════════════════════════════

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream) {
    // Block: 16x16 = 256 threads per block
    // Grid:  enough blocks to cover all M rows and N columns
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    kernel_matmul_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream) {
    // Block: TILE_SIZE x TILE_SIZE = 32x32 = 1024 threads per block
    // Grid:  enough blocks to cover the output matrix
    // Shared memory: 2 * 32^2 * 4 = 8 KB per block (statically allocated)
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    kernel_matmul_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

void matmul_vectorized(const float* A, const float* B, float* C,
                       int M, int N, int K, cudaStream_t stream) {
    // Block: (TILE_N/4, TILE_M) = (8, 32) = 256 threads per block
    //   - Each thread handles 4 output columns via float4 operations
    //   - 256 threads * 4 outputs = 1024 C elements per block
    // Grid: enough blocks to tile the output matrix
    // Shared memory: (32*8 + 8*32) * 4 = 2 KB per block (statically allocated)
    constexpr int BLOCK_X = TILE_N / 4;  // 8
    constexpr int BLOCK_Y = TILE_M;      // 32
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    kernel_matmul_vectorized<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

} // namespace cuinfer
