// ── matmul_quant.cu — Fused Dequant + MatMul CUDA Kernels ──────────────
//
// Computes C = A * dequant(B)^T where B is stored in GGUF-native [N,K]
// quantized layout. Dequantization happens on-the-fly in shared memory:
//
//   1. Load A tile from global → shared (standard FP32 tile load)
//   2. Load quantized B blocks from global → dequant to FP32 in shared
//   3. Standard FP32 dot-product accumulation in registers
//
// Tile dimensions: TILE_M=32, TILE_N=32, TILE_K=32
// TILE_K=32 matches Q8_0/Q4_0 block size, so each tile iteration
// processes exactly one quantization block per B row.
//
// For K-quant types (Q4_K, Q6_K) with 256-element super-blocks,
// TILE_K=32 processes 1/8 of a super-block per tile iteration. The kernel
// reads sub-block data by computing offsets into the super-block structure.
//
// Memory layout (B-transposed):
//   B_quant[n][k] — row n contains the quantized representation of
//   output feature n's weight vector. Each row is a sequence of
//   quantization blocks covering K elements.
//
// Shared memory per block:
//   tile_a[32][32] = 4 KB (FP32)
//   tile_b[32][32] = 4 KB (FP32, after dequant)
//   Total: 8 KB per block

#include "cuinfer/kernels/matmul_quant.h"
#include "cuinfer/quantization.h"
#include "cuinfer/cuda_utils.h"

#include <cuda_fp16.h>

namespace cuinfer {

// ── Tile dimensions ─────────────────────────────────────────
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
// Padded column dimension for tile_b shared memory tile to eliminate 32-way bank conflicts.
// Stride of 33 floats (132 bytes) ensures that (threadIdx.x * 33 + threadIdx.y) % 32
// maps every thread in a warp to a distinct shared memory bank.
constexpr int TILE_N_PAD = TILE_N + 1;

// ════════════════════════════════════════════════════════════════
// Q8_0 Quantized MatMul
// ════════════════════════════════════════════════════════════════
//
// Each tile-K iteration loads one Q8_0 block per B row (32 elements).
// Block layout: 2 bytes FP16 scale + 32 int8 values = 34 bytes.
//
// B is [N, K] where each row has K/32 BlockQ8_0 blocks.
// For tile at (tile_n, tile_k):
//   B row index = tile_n * TILE_N + threadIdx.x (or threadIdx.y)
//   Block index within row = tile_k (since TILE_K = block_size = 32)
//
// Thread mapping: block is 32x32 = 1024 threads.
// Each thread (ty, tx) dequantizes one element of the B tile:
//   row_in_tile = ty, col_in_tile = tx
//   B row = blockIdx.x * TILE_N + ty
//   B block = tile_iter (the current k-tile)
//   element within block = tx

__global__ void kernel_matmul_q8_0(
    const float* __restrict__ A,
    const BlockQ8_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N_PAD];

    const int row = blockIdx.y * TILE_M + threadIdx.y;  // output row (M dim)
    const int col = blockIdx.x * TILE_N + threadIdx.x;  // output col (N dim)

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;  // K must be divisible by 32
    const int blocks_per_row = K / Q8_0_BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile_a[ty][tx] = A[row, t*TILE_K + tx]
        const int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load and dequantize tile_b[tx][ty]:
        // We need tile_b[k][n] = dequant(B[col_base + ty][t*TILE_K + tx])
        // But B is stored as blocks. For row (blockIdx.x * TILE_N + ty),
        // the t-th block contains elements [t*32 .. t*32+31].
        // Element tx within that block = B_block.qs[tx] * B_block.scale
        const int b_row = blockIdx.x * TILE_N + threadIdx.y;
        if (b_row < N) {
            // Index into the flat BlockQ8_0 array:
            // row b_row has blocks_per_row blocks starting at B[b_row * blocks_per_row]
            const BlockQ8_0& blk = B[b_row * blocks_per_row + t];
            float scale = __half2float(blk.scale);
            tile_b[threadIdx.x][threadIdx.y] = scale * static_cast<float>(blk.qs[threadIdx.x]);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Accumulate: sum += tile_a[ty][k] * tile_b[k][tx]
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_q8_0(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);  // 32x32 = 1024 threads
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    kernel_matmul_q8_0<<<grid, block, 0, stream>>>(
        A, static_cast<const BlockQ8_0*>(B_quant), C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q4_0 Quantized MatMul
// ════════════════════════════════════════════════════════════════
//
// Same structure as Q8_0 but each block has 16 packed bytes (2 nibbles each).
// Thread tx dequantizes element tx from the Q4_0 block:
//   byte = qs[tx / 2]
//   nibble = (tx % 2 == 0) ? (byte & 0x0F) : (byte >> 4)
//   value = scale * (nibble - 8)

__global__ void kernel_matmul_q4_0(
    const float* __restrict__ A,
    const BlockQ4_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N_PAD];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;
    const int blocks_per_row = K / Q4_0_BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile_a
        const int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load and dequantize tile_b from Q4_0
        const int b_row = blockIdx.x * TILE_N + threadIdx.y;
        if (b_row < N) {
            const BlockQ4_0& blk = B[b_row * blocks_per_row + t];
            float scale = __half2float(blk.scale);
            int byte_idx = threadIdx.x / 2;
            uint8_t packed = blk.qs[byte_idx];
            int nibble = (threadIdx.x % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
            tile_b[threadIdx.x][threadIdx.y] = scale * static_cast<float>(nibble - 8);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_q4_0(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    kernel_matmul_q4_0<<<grid, block, 0, stream>>>(
        A, static_cast<const BlockQ4_0*>(B_quant), C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q5_0 Quantized MatMul
// ════════════════════════════════════════════════════════════════
//
// Same structure as Q4_0 but each element is 5 bits: low 4 bits from qs
// (packed nibbles) + 1 high bit from qh (bit-addressed over 4 bytes).
// q5 = low_nibble | (high_bit << 4), range [0, 31], offset by -16.
// value = scale * (q5 - 16)

__global__ void kernel_matmul_q5_0(
    const float* __restrict__ A,
    const BlockQ5_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N_PAD];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;
    const int blocks_per_row = K / Q5_0_BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile_a
        const int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load and dequantize tile_b from Q5_0
        // Q5_0 layout: elements 0..15 use lower nibble of qs[0..15],
        //              elements 16..31 use upper nibble of qs[0..15].
        const int b_row = blockIdx.x * TILE_N + threadIdx.y;
        if (b_row < N) {
            const BlockQ5_0& blk = B[b_row * blocks_per_row + t];
            float scale = __half2float(blk.scale);

            const int half = threadIdx.x / 16;
            const int j = threadIdx.x % 16;
            int low4 = (half == 0) ? (blk.qs[j] & 0x0F) : (blk.qs[j] >> 4);

            // Construct uint32 byte-by-byte to avoid misaligned access
            uint32_t qh_u32 = static_cast<uint32_t>(blk.qh[0])
                             | (static_cast<uint32_t>(blk.qh[1]) << 8)
                             | (static_cast<uint32_t>(blk.qh[2]) << 16)
                             | (static_cast<uint32_t>(blk.qh[3]) << 24);
            int high_bit = (qh_u32 >> threadIdx.x) & 1;

            int q5_val = low4 | (high_bit << 4);
            tile_b[threadIdx.x][threadIdx.y] = scale * static_cast<float>(q5_val - 16);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_q5_0(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    kernel_matmul_q5_0<<<grid, block, 0, stream>>>(
        A, static_cast<const BlockQ5_0*>(B_quant), C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q4_K Quantized MatMul
// ════════════════════════════════════════════════════════════════
//
// Q4_K has 256-element super-blocks. With TILE_K=32, each tile iteration
// processes one sub-block (32 elements) of the 256-element super-block.
// We need 8 tile iterations to cover one super-block.
//
// For tile iteration t:
//   super_block_idx = t / 8  (which super-block in the row)
//   sub_block_idx   = t % 8  (which 32-element sub-block within)
//
// Within sub-block j, element i (0..31):
//   Uses GGML half-interleaved nibble layout: each 64-element chunk
//   uses 32 qs bytes — lower nibbles for first 32, upper for next 32.
//
// Scale/min decoding uses the same 6-bit packing as the standalone dequant.

__global__ void kernel_matmul_q4_k(
    const float* __restrict__ A,
    const BlockQ4_K* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N_PAD];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;
    const int super_blocks_per_row = K / Q4_K_BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile_a
        const int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load and dequantize from Q4_K super-block
        const int b_row = blockIdx.x * TILE_N + threadIdx.y;
        if (b_row < N) {
            const int sb_idx = t / 8;   // super-block index within row
            const int sub_block = t % 8; // sub-block within super-block

            const BlockQ4_K& blk = B[b_row * super_blocks_per_row + sb_idx];
            const float d    = __half2float(blk.d);
            const float dmin = __half2float(blk.dmin);

            // Decode 6-bit scale and min for this sub-block
            uint8_t sc, mn;
            if (sub_block < 4) {
                sc = blk.scales[sub_block] & 0x3F;
                mn = blk.scales[sub_block + 4] & 0x3F;
            } else {
                sc = (blk.scales[sub_block + 4] & 0x0F) | ((blk.scales[sub_block - 4] >> 6) << 4);
                mn = (blk.scales[sub_block + 4] >> 4)    | ((blk.scales[sub_block]     >> 6) << 4);
            }

            // Element index within the 256-element super-block
            const int elem_in_sb = sub_block * 32 + threadIdx.x;

            // GGML half-interleaved nibble layout: each 64-element chunk
            // uses 32 bytes — lower nibbles for first 32, upper for next 32.
            const int chunk64 = elem_in_sb / 64;
            const int within_64 = elem_in_sb % 64;
            const int l = within_64 % 32;
            const int qs_byte = chunk64 * 32 + l;
            int nibble = (within_64 < 32) ? (blk.qs[qs_byte] & 0x0F)
                                          : (blk.qs[qs_byte] >> 4);

            tile_b[threadIdx.x][threadIdx.y] =
                d * static_cast<float>(sc) * static_cast<float>(nibble) -
                dmin * static_cast<float>(mn);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_q4_k(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    kernel_matmul_q4_k<<<grid, block, 0, stream>>>(
        A, static_cast<const BlockQ4_K*>(B_quant), C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q6_K Quantized MatMul
// ════════════════════════════════════════════════════════════════
//
// Q6_K: 256-element super-blocks, 210 bytes each.
// Similar tile strategy to Q4_K: 8 tile iterations per super-block.
// Each thread dequantizes one 6-bit value from ql/qh arrays.
// Uses GGML interleaved nibble layout (NOT simple consecutive pairs).

__global__ void kernel_matmul_q6_k(
    const float* __restrict__ A,
    const BlockQ6_K* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N_PAD];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;
    const int super_blocks_per_row = K / Q6_K_BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int b_row = blockIdx.x * TILE_N + threadIdx.y;
        if (b_row < N) {
            const int sb_idx = t / 8;
            const int sub_block_32 = t % 8;

            const BlockQ6_K& blk = B[b_row * super_blocks_per_row + sb_idx];
            const float d = __half2float(blk.d);

            // Element index within the 256-element super-block
            const int elem_in_sb = sub_block_32 * 32 + threadIdx.x;

            // GGML interleaved Q6_K layout (same as dequantize kernel)
            const int chunk = elem_in_sb / 128;
            const int within_chunk = elem_in_sb % 128;
            const int quarter = within_chunk / 32;
            const int l = within_chunk % 32;

            // Scale: 16 groups of 16 elements, ALL scale indices used.
            const int scale_idx = chunk * 8 + quarter * 2 + l / 16;
            const float sc = static_cast<float>(blk.scales[scale_idx]);

            // ql: interleaved layout
            const int ql_offset = (quarter & 1) * 32 + l + chunk * 64;
            const int ql_nibble = (quarter < 2) ? (blk.ql[ql_offset] & 0x0F)
                                                : (blk.ql[ql_offset] >> 4);

            // qh: each byte holds 4 elements (2 bits each)
            const int qh_byte = l + chunk * 32;
            const int qh_shift = quarter * 2;
            const int qh_bits = (blk.qh[qh_byte] >> qh_shift) & 0x03;

            const int q6_val = (ql_nibble | (qh_bits << 4)) - 32;

            tile_b[threadIdx.x][threadIdx.y] = d * sc * static_cast<float>(q6_val);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_q6_k(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    kernel_matmul_q6_k<<<grid, block, 0, stream>>>(
        A, static_cast<const BlockQ6_K*>(B_quant), C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// FP16 Mixed-Precision MatMul
// ════════════════════════════════════════════════════════════════
//
// Weights stored as native __half in GGUF [N, K] layout.
// Same tiled structure as the quantized kernels: load A tile as FP32,
// load B tile as __half → convert to FP32 in shared memory, then
// standard FP32 dot-product accumulation.
//
// Thread mapping identical to quantized kernels:
//   Each thread (ty, tx) dequantizes one element of the B tile.
//   B row = blockIdx.x * TILE_N + ty
//   B column = t * TILE_K + tx

__global__ void kernel_matmul_fp16(
    const float* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N_PAD];

    const int row = blockIdx.y * TILE_M + threadIdx.y;  // output row (M dim)
    const int col = blockIdx.x * TILE_N + threadIdx.x;  // output col (N dim)

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile_a[ty][tx] = A[row, t*TILE_K + tx]
        const int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load and dequantize tile_b: B is [N, K] with __half elements.
        // tile_b[tx][ty] = __half2float(B[b_row * K + t*TILE_K + tx])
        const int b_row = blockIdx.x * TILE_N + threadIdx.y;
        if (b_row < N) {
            const int b_col = t * TILE_K + threadIdx.x;
            tile_b[threadIdx.x][threadIdx.y] = __half2float(B[b_row * K + b_col]);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Accumulate: sum += tile_a[ty][k] * tile_b[k][tx]
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_fp16(const float* A, const void* B_fp16, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);  // 32x32 = 1024 threads
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    kernel_matmul_fp16<<<grid, block, 0, stream>>>(
        A, static_cast<const __half*>(B_fp16), C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

} // namespace cuinfer
