// ── embedding.cu — Embedding Layer, LM Head, and Sampling Kernels ────────
//
// Implements the first and last layers of a transformer LLM:
//   1. Token embedding lookup (discrete token IDs -> dense vectors)
//   2. LM head projection (hidden states -> vocabulary logits)
//   3. Sampling utilities (argmax, top-k)
//
// The embedding and LM head often share the same weight matrix (tied weights),
// which saves significant VRAM — critical on our 6 GB RTX 3060 budget.

#include "cuinfer/embedding.h"
#include "cuinfer/cuda_utils.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <vector>

namespace cuinfer {

// ════════════════════════════════════════════════════════════════════════
// Kernel: Embedding Lookup
// ════════════════════════════════════════════════════════════════════════
//
// Each thread handles one element of the output tensor.
// Thread at position (i, j) reads: output[i * dim + j] = weight[token_ids[i] * dim + j]
//
// Memory access pattern:
//   - token_ids: each warp reads the same token_id[i] (broadcast, cached in L1)
//   - weight: consecutive threads read consecutive addresses within a row
//     (coalesced 128-byte transactions since j varies across threads in a warp)
//   - output: consecutive threads write consecutive addresses (coalesced)
//
// Launch config:
//   Block: 256 threads (1D)
//   Grid:  ceil(seq_len * dim / 256) blocks
//   Each thread computes one output element using a flat index.

__global__ void kernel_embedding_lookup(
    const int32_t* __restrict__ token_ids,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int seq_len,
    int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * dim;

    if (idx >= total) return;

    // Decompose flat index into (token_position, embedding_dimension)
    const int token_pos = idx / dim;
    const int embed_dim = idx % dim;

    // Look up the token ID for this position and fetch the corresponding
    // embedding vector element. No bounds check on token_ids[token_pos]
    // needed in the kernel — caller must ensure IDs are in [0, vocab_size).
    const int32_t token_id = token_ids[token_pos];
    output[idx] = weight[token_id * dim + embed_dim];
}

// ════════════════════════════════════════════════════════════════════════
// Kernel: MatMul with Transposed B (C = A * B^T)
// ════════════════════════════════════════════════════════════════════════
//
// Computes C[i,j] = sum_k A[i,k] * B[j,k]
// where A is [M, K], B is [N, K] (row-major), C is [M, N].
//
// This is the core operation for the LM head. The weight matrix is stored
// as [vocab_size, dim], and we need:
//   logits[i,j] = sum_k hidden[i,k] * weight[j,k]
//
// We use shared memory tiling for performance, similar to the standard
// tiled matmul but with B indexed as B[j, k] instead of B[k, j].
//
// Shared memory usage per block:
//   2 tiles * TILE^2 * sizeof(float) = 2 * 32^2 * 4 = 8,192 bytes (8 KB)
//
// Launch config:
//   Block: TILE x TILE = 32x32 = 1024 threads
//   Grid:  ceil(N/TILE) x ceil(M/TILE)

constexpr int LM_TILE = 32;

__global__ void kernel_matmul_transposed_b(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tile_a[LM_TILE][LM_TILE];
    __shared__ float tile_b[LM_TILE][LM_TILE];

    const int row = blockIdx.y * LM_TILE + threadIdx.y;  // output row (seq_len)
    const int col = blockIdx.x * LM_TILE + threadIdx.x;  // output col (vocab_size)

    float sum = 0.0f;
    const int num_tiles = (K + LM_TILE - 1) / LM_TILE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A: A[row, t*TILE + threadIdx.x]
        const int a_col = t * LM_TILE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B^T: we need B[col, t*TILE + threadIdx.y]
        // This reads from B's row `col`, column `t*TILE + threadIdx.y`.
        // Since B is [N, K] row-major, the address is B[col * K + k].
        const int b_col = t * LM_TILE + threadIdx.y;
        if (col < N && b_col < K) {
            tile_b[threadIdx.y][threadIdx.x] = B[col * K + b_col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate: sum += tile_a[ty][k] * tile_b[k][tx]
        // tile_b[k][tx] holds B[col, t*TILE + k], which is what we need
        // for the transposed multiply.
        #pragma unroll
        for (int k = 0; k < LM_TILE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Kernel: Argmax Reduction
// ════════════════════════════════════════════════════════════════════════
//
// Two-phase parallel reduction to find the index of the maximum value
// in a 1D array. Each block reduces a portion of the input, writing
// its local maximum (value + index) to a small intermediate buffer.
// A second kernel (or host-side finish) picks the global winner.
//
// For typical LLM vocab sizes (32K-128K), a single block of 256 threads
// with stride loop handles everything, so we use a single-block approach
// for simplicity and launch with 1 block when n <= 256*256.
//
// Shared memory per block:
//   256 floats (values) + 256 ints (indices) = 2 KB

constexpr int ARGMAX_BLOCK = 256;

__global__ void kernel_argmax(
    const float* __restrict__ input,
    int n,
    float* __restrict__ out_val,
    int32_t* __restrict__ out_idx)
{
    __shared__ float s_vals[ARGMAX_BLOCK];
    __shared__ int32_t s_idxs[ARGMAX_BLOCK];

    const int tid = threadIdx.x;

    // Phase 1: Each thread finds the max in its strided portion of the input.
    // For n=50000 and 256 threads, each thread examines ~195 elements.
    float local_max = -FLT_MAX;
    int32_t local_idx = 0;

    for (int i = tid; i < n; i += ARGMAX_BLOCK) {
        float val = input[i];
        if (val > local_max || (val == local_max && i < local_idx)) {
            local_max = val;
            local_idx = i;
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    // Phase 2: Tree reduction within the block.
    // At each step, thread tid compares its value with the value at tid + stride.
    // After log2(ARGMAX_BLOCK) steps, thread 0 holds the global max.
    for (int stride = ARGMAX_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_vals[tid + stride];
            int32_t other_idx = s_idxs[tid + stride];
            if (other_val > s_vals[tid] ||
                (other_val == s_vals[tid] && other_idx < s_idxs[tid])) {
                s_vals[tid] = other_val;
                s_idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the result.
    if (tid == 0) {
        *out_val = s_vals[0];
        *out_idx = s_idxs[0];
    }
}

// ════════════════════════════════════════════════════════════════════════
// Host Wrappers
// ════════════════════════════════════════════════════════════════════════

void embedding_lookup(const int32_t* token_ids, const float* weight,
                      float* output, int seq_len, int dim,
                      cudaStream_t stream) {
    const int total = seq_len * dim;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;

    kernel_embedding_lookup<<<grid, BLOCK, 0, stream>>>(
        token_ids, weight, output, seq_len, dim);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

void matmul_transposed_b(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream) {
    dim3 block(LM_TILE, LM_TILE);
    dim3 grid((N + LM_TILE - 1) / LM_TILE,
              (M + LM_TILE - 1) / LM_TILE);

    kernel_matmul_transposed_b<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ── Embedding Class ──────────────────────────────────────────────────

Embedding::Embedding(int vocab_size, int dim)
    : vocab_size_(vocab_size), dim_(dim) {}

Tensor Embedding::forward(const int32_t* token_ids, int seq_len,
                          const Tensor& weight, cudaStream_t stream) {
    // Validate weight shape: should be [vocab_size, dim].
    // With 4D right-alignment, a 2D [V, D] tensor has shape {1, 1, V, D}.
    // We check the last two dimensions.
    const int64_t w_rows = weight.shape(2);
    const int64_t w_cols = weight.shape(3);

    if (w_rows != vocab_size_ || w_cols != dim_) {
        throw std::runtime_error(
            "Embedding::forward: weight shape mismatch. Expected [" +
            std::to_string(vocab_size_) + ", " + std::to_string(dim_) +
            "], got [" + std::to_string(w_rows) + ", " +
            std::to_string(w_cols) + "]");
    }

    // Allocate output: [seq_len, dim]
    Tensor output({static_cast<int64_t>(seq_len),
                   static_cast<int64_t>(dim_)});

    embedding_lookup(token_ids, weight.data_as<float>(),
                     output.data_as<float>(), seq_len, dim_, stream);

    return output;
}

// ── LMHead Class ─────────────────────────────────────────────────────

LMHead::LMHead(int vocab_size, int dim)
    : vocab_size_(vocab_size), dim_(dim) {}

Tensor LMHead::forward(const Tensor& hidden, const Tensor& weight,
                       cudaStream_t stream) {
    // hidden: [seq_len, dim] -> 4D shape {1, 1, seq_len, dim}
    // weight: [vocab_size, dim] -> 4D shape {1, 1, vocab_size, dim}
    const int seq_len = static_cast<int>(hidden.shape(2));
    const int h_dim = static_cast<int>(hidden.shape(3));
    const int w_vocab = static_cast<int>(weight.shape(2));
    const int w_dim = static_cast<int>(weight.shape(3));

    if (h_dim != dim_) {
        throw std::runtime_error(
            "LMHead::forward: hidden dim mismatch. Expected " +
            std::to_string(dim_) + ", got " + std::to_string(h_dim));
    }
    if (w_vocab != vocab_size_ || w_dim != dim_) {
        throw std::runtime_error(
            "LMHead::forward: weight shape mismatch. Expected [" +
            std::to_string(vocab_size_) + ", " + std::to_string(dim_) +
            "], got [" + std::to_string(w_vocab) + ", " +
            std::to_string(w_dim) + "]");
    }

    // logits = hidden [seq_len, dim] * weight^T [dim, vocab_size]
    //        = [seq_len, vocab_size]
    // M = seq_len, N = vocab_size, K = dim
    Tensor logits({static_cast<int64_t>(seq_len),
                   static_cast<int64_t>(vocab_size_)});

    matmul_transposed_b(hidden.data_as<float>(), weight.data_as<float>(),
                        logits.data_as<float>(),
                        seq_len, vocab_size_, dim_, stream);

    return logits;
}

// ── Sampling: Argmax ─────────────────────────────────────────────────

int32_t argmax_sample(const float* logits, int n, cudaStream_t stream) {
    // Allocate tiny device buffers for the reduction output.
    float* d_val = nullptr;
    int32_t* d_idx = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_val, sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_idx, sizeof(int32_t)));

    // Single block handles the entire vocab (up to ~256K elements
    // with the stride loop). This is efficient because argmax is
    // memory-bound and we only need one pass through the data.
    kernel_argmax<<<1, ARGMAX_BLOCK, 0, stream>>>(logits, n, d_val, d_idx);
    CUINFER_CHECK_CUDA(cudaGetLastError());

    int32_t result;
    CUINFER_CHECK_CUDA(cudaMemcpy(&result, d_idx, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));

    cudaFree(d_val);
    cudaFree(d_idx);

    return result;
}

// ── Sampling: Top-K ──────────────────────────────────────────────────
//
// Implementation strategy: CPU-side top-k for simplicity and correctness.
// During autoregressive generation, we call this once per token — the
// bottleneck is the LM head matmul, not sampling. A GPU-accelerated
// version (e.g., using radix select) would only matter for very large
// batch sizes, which we don't target on 6 GB VRAM.
//
// Steps:
//   1. Copy logits from device to host
//   2. Find top-k values using partial_sort
//   3. Apply temperature scaling
//   4. Softmax over top-k candidates
//   5. Sample from the categorical distribution

int32_t top_k_sample(const float* logits, int n, int k,
                     float temperature, unsigned int seed,
                     cudaStream_t stream) {
    if (k <= 0) k = 1;
    if (k > n) k = n;
    if (temperature <= 0.0f) temperature = 1e-8f;  // Prevent division by zero

    // Synchronize before copying (ensure logits are computed)
    CUINFER_CHECK_CUDA(cudaStreamSynchronize(stream));

    // Step 1: Copy logits to host
    std::vector<float> h_logits(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(h_logits.data(), logits,
                                  n * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    // Step 2: Build index array and partial sort to find top-k
    // We sort indices by their corresponding logit values (descending).
    std::vector<int32_t> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&h_logits](int32_t a, int32_t b) {
                          return h_logits[a] > h_logits[b];
                      });

    // Step 3: Apply temperature scaling to top-k logits
    std::vector<float> top_k_logits(k);
    for (int i = 0; i < k; ++i) {
        top_k_logits[i] = h_logits[indices[i]] / temperature;
    }

    // Step 4: Softmax over top-k
    // Numerically stable: subtract max before exp
    float max_val = *std::max_element(top_k_logits.begin(), top_k_logits.end());
    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        top_k_logits[i] = std::exp(top_k_logits[i] - max_val);
        sum_exp += top_k_logits[i];
    }
    for (int i = 0; i < k; ++i) {
        top_k_logits[i] /= sum_exp;
    }

    // Step 5: Sample from the categorical distribution
    // Simple linear scan CDF sampling with a deterministic seed
    // for reproducible results in tests.
    // Using a basic LCG (Linear Congruential Generator) for portability.
    unsigned int state = seed;
    state = state * 1103515245u + 12345u;
    float r = static_cast<float>(state & 0x7FFFFFFFu) /
              static_cast<float>(0x7FFFFFFFu);

    float cumulative = 0.0f;
    for (int i = 0; i < k; ++i) {
        cumulative += top_k_logits[i];
        if (r <= cumulative) {
            return indices[i];
        }
    }

    // Fallback: return last candidate (handles floating-point rounding)
    return indices[k - 1];
}

} // namespace cuinfer
