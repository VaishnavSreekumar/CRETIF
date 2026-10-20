// ── attention.cu — Multi-Head / Grouped-Query Attention Block ────────────
//
// Implements the full attention mechanism used in transformer inference:
//   1. QKV projection via matmul
//   2. RoPE positional embeddings
//   3. KV-cache update
//   4. Scaled dot-product attention with causal mask and softmax
//   5. Output projection via matmul
//
// Custom CUDA kernels in this file:
//   - kernel_causal_mask: applies causal mask to attention scores
//   - kernel_scale: scales attention scores by 1/sqrt(head_dim)
//   - kernel_batched_matmul_naive: batched matmul for small per-head matrices
//   - kernel_batched_matmul_gqa: batched matmul with GQA K/V head broadcast
//
// The existing matmul_tiled kernel is used for the large projection matmuls
// (input x W_q, input x W_k, etc.), while the batched kernels handle the
// per-head attention computation where matrices are small (seq_len x head_dim).

#include "cuinfer/attention.h"
#include "cuinfer/cuda_utils.h"
#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/kernels/matmul.h"
#include "cuinfer/kernels/matmul_quant.h"
#include "cuinfer/model.h"  // for WeightFormat

#include <cub/cub.cuh>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <string>

namespace cuinfer {

// ── Constants ──────────────────────────────────────────────────
constexpr int BLOCK_SIZE = 256;

// ════════════════════════════════════════════════════════════════
// Kernel: Causal Mask
// ════════════════════════════════════════════════════════════════
//
// Applies a causal (autoregressive) mask to attention scores in-place.
// For each query position q (0-indexed within the current chunk) at
// absolute position (start_pos + q), the mask prevents attending to
// any key position k where k > start_pos + q.
//
// scores layout: [n_heads, seq_len, kv_len] — contiguous row-major.
// Each thread handles one element at index (head, q, k):
//   head = flat_idx / (seq_len * kv_len)
//   q    = (flat_idx / kv_len) % seq_len
//   k    = flat_idx % kv_len
//
// If k > start_pos + q, set scores[flat_idx] = -infinity.
// This ensures each token can only attend to itself and predecessors.
//
// Grid: ceil(total_elements / 256), 256 threads per block.
// No shared memory needed — pure element-wise operation.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_causal_mask(float* scores, int n_heads,
                                   int seq_len, int kv_len,
                                   int start_pos) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_heads * seq_len * kv_len;

    if (idx < total) {
        // Decompose flat index into (head, q, k).
        const int q = (idx / kv_len) % seq_len;
        const int k = idx % kv_len;

        // The absolute position of query token q is start_pos + q.
        // It may attend to key positions 0 through start_pos + q (inclusive).
        if (k > start_pos + q) {
            scores[idx] = -INFINITY;
        }
    }
}

void causal_mask(float* scores, int n_heads, int seq_len, int kv_len,
                 int start_pos, cudaStream_t stream) {
    const int total = n_heads * seq_len * kv_len;
    const int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_causal_mask<<<grid, BLOCK_SIZE, 0, stream>>>(
        scores, n_heads, seq_len, kv_len, start_pos);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Scale
// ════════════════════════════════════════════════════════════════
//
// Multiplies every element by a constant factor. Used to scale
// attention scores by 1/sqrt(head_dim) before softmax.
//
// Pure element-wise — no shared memory, no reductions.
// Grid: ceil(n / 256), 256 threads per block.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_scale(float* data, int n, float factor) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

void scale(float* data, int n, float factor, cudaStream_t stream) {
    const int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_scale<<<grid, BLOCK_SIZE, 0, stream>>>(data, n, factor);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Batched MatMul (Naive, for small per-head matrices)
// ════════════════════════════════════════════════════════════════
//
// Computes C[b] = A[b] * B[b] for b in [0, batch), where each matrix
// is small (typical sizes: M=1..seq_len, K=head_dim, N=kv_len or head_dim).
//
// Each batch element is a standard matmul:
//   A[b]: [M, K], B[b]: [K, N], C[b]: [M, N]
//
// Layout in memory:
//   A is contiguous: batch * M * K floats
//   B is contiguous: batch * K * N floats
//   C is contiguous: batch * M * N floats
//
// Launch strategy: 3D grid (ceil(N/16), ceil(M/16), batch).
// Block: 16x16 = 256 threads. Each thread computes one element of C[b].
// For the small matrices in attention (M<=128, N<=128), this gives
// reasonable occupancy without the overhead of shared memory tiling.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_batched_matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int M, int N, int K) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int b   = blockIdx.z;

    if (b >= batch || row >= M || col >= N) return;

    const float* a_ptr = A + b * M * K;
    const float* b_ptr = B + b * K * N;
    float* c_ptr       = C + b * M * N;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a_ptr[row * K + k] * b_ptr[k * N + col];
    }
    c_ptr[row * N + col] = sum;
}

void batched_matmul(const float* A, const float* B, float* C,
                    int batch, int M, int N, int K,
                    cudaStream_t stream) {
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK,
              (M + BLOCK - 1) / BLOCK,
              batch);

    kernel_batched_matmul_naive<<<grid, block, 0, stream>>>(
        A, B, C, batch, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Batched MatMul with GQA Broadcast
// ════════════════════════════════════════════════════════════════
//
// For Grouped-Query Attention (GQA), we have n_heads query heads but
// only n_kv_heads key/value heads. Each group of (n_heads/n_kv_heads)
// query heads shares the same K/V head.
//
// Instead of physically replicating the K/V heads, this kernel indexes
// into B using the KV head index: kv_head = head / n_groups.
//
// A: [n_heads, M, K]      — one matrix per query head
// B: [n_kv_heads, K, N]   — one matrix per KV head
// C: [n_heads, M, N]      — one output per query head
//
// The mapping head -> kv_head handles the broadcast implicitly.
// This saves both memory (no K/V replication) and bandwidth.
//
// Launch: 3D grid (ceil(N/16), ceil(M/16), n_heads).
// Block: 16x16 = 256 threads.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_batched_matmul_gqa(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n_heads, int n_kv_heads,
    int M, int N, int K) {

    const int col  = blockIdx.x * blockDim.x + threadIdx.x;
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int head = blockIdx.z;

    if (head >= n_heads || row >= M || col >= N) return;

    // GQA mapping: multiple query heads share one KV head.
    const int n_groups = n_heads / n_kv_heads;
    const int kv_head  = head / n_groups;

    const float* a_ptr = A + head    * M * K;
    const float* b_ptr = B + kv_head * K * N;
    float* c_ptr       = C + head    * M * N;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a_ptr[row * K + k] * b_ptr[k * N + col];
    }
    c_ptr[row * N + col] = sum;
}

void batched_matmul_gqa(const float* A, const float* B, float* C,
                        int n_heads, int n_kv_heads,
                        int M, int N, int K,
                        cudaStream_t stream) {
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK,
              (M + BLOCK - 1) / BLOCK,
              n_heads);

    kernel_batched_matmul_gqa<<<grid, block, 0, stream>>>(
        A, B, C, n_heads, n_kv_heads, M, N, K);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Transpose [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
// ════════════════════════════════════════════════════════════════
//
// After QKV projection, the Q tensor has shape [seq_len, n_heads * head_dim].
// We reshape it logically to [seq_len, n_heads, head_dim] and then transpose
// the first two dims to get [n_heads, seq_len, head_dim], which is the layout
// needed for per-head batched matmul.
//
// Each thread moves one element:
//   src index: (s * n_heads + h) * head_dim + d  (seq_len-major)
//   dst index: (h * seq_len + s) * head_dim + d  (n_heads-major)
//
// Grid: ceil(total_elements / 256), 256 threads per block.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_transpose_seq_heads(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int seq_len, int n_heads, int head_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * n_heads * head_dim;

    if (idx < total) {
        // Decompose flat index in the source layout [seq_len, n_heads, head_dim].
        const int d = idx % head_dim;
        const int h = (idx / head_dim) % n_heads;
        const int s = idx / (n_heads * head_dim);

        // Compute flat index in the destination layout [n_heads, seq_len, head_dim].
        const int dst_idx = (h * seq_len + s) * head_dim + d;
        dst[dst_idx] = src[idx];
    }
}

// Host wrapper for the transpose kernel.
static void transpose_seq_heads(const float* src, float* dst,
                                int seq_len, int n_heads, int head_dim,
                                cudaStream_t stream = 0) {
    const int total = seq_len * n_heads * head_dim;
    const int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_transpose_seq_heads<<<grid, BLOCK_SIZE, 0, stream>>>(
        src, dst, seq_len, n_heads, head_dim);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Transpose [n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
// ════════════════════════════════════════════════════════════════
//
// Inverse of kernel_transpose_seq_heads. Used after the attention computation
// to convert from the per-head layout back to the interleaved layout for
// the output projection.
//
// Each thread moves one element:
//   src index: (h * seq_len + s) * head_dim + d  (n_heads-major)
//   dst index: (s * n_heads + h) * head_dim + d  (seq_len-major)
// ────────────────────────────────────────────────────────────────

__global__ void kernel_transpose_heads_seq(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int seq_len, int n_heads, int head_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * n_heads * head_dim;

    if (idx < total) {
        // Decompose flat index in the source layout [n_heads, seq_len, head_dim].
        const int d = idx % head_dim;
        const int s = (idx / head_dim) % seq_len;
        const int h = idx / (seq_len * head_dim);

        // Compute flat index in the destination layout [seq_len, n_heads, head_dim].
        const int dst_idx = (s * n_heads + h) * head_dim + d;
        dst[dst_idx] = src[idx];
    }
}

static void transpose_heads_seq(const float* src, float* dst,
                                int seq_len, int n_heads, int head_dim,
                                cudaStream_t stream = 0) {
    const int total = seq_len * n_heads * head_dim;
    const int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_transpose_heads_seq<<<grid, BLOCK_SIZE, 0, stream>>>(
        src, dst, seq_len, n_heads, head_dim);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Transpose K from [n_kv_heads, kv_len, head_dim]
//                       to [n_kv_heads, head_dim, kv_len]
// ════════════════════════════════════════════════════════════════
//
// For computing Q * K^T, we need K transposed within each head:
// the last two dimensions are swapped from [kv_len, head_dim] to
// [head_dim, kv_len].
//
// Each thread handles one element:
//   src: [h, s, d] -> flat = (h * kv_len + s) * head_dim + d
//   dst: [h, d, s] -> flat = (h * head_dim + d) * kv_len + s
// ────────────────────────────────────────────────────────────────

__global__ void kernel_transpose_k(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_kv_heads, int kv_len, int head_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv_heads * kv_len * head_dim;

    if (idx < total) {
        const int d = idx % head_dim;
        const int s = (idx / head_dim) % kv_len;
        const int h = idx / (kv_len * head_dim);

        const int dst_idx = (h * head_dim + d) * kv_len + s;
        dst[dst_idx] = src[idx];
    }
}

static void transpose_k(const float* src, float* dst,
                         int n_kv_heads, int kv_len, int head_dim,
                         cudaStream_t stream = 0) {
    const int total = n_kv_heads * kv_len * head_dim;
    const int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_transpose_k<<<grid, BLOCK_SIZE, 0, stream>>>(
        src, dst, n_kv_heads, kv_len, head_dim);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel: Fused Decode Attention (seq_len=1, Online Softmax)
// ════════════════════════════════════════════════════════════════
//
// For single-token decode, the standard multi-kernel attention pipeline
// (transpose → Q*K^T → scale → mask → softmax → weights*V → transpose)
// launches 9+ kernels and materializes intermediate buffers. This fused
// kernel does everything in one launch by reading K/V directly from cache.
//
// Algorithm (one block per query head):
//   1. Load Q[h, :] into shared memory (head_dim floats).
//   2. Initialize accumulators in shared memory: acc[head_dim] = 0.
//   3. For each cached position k = 0..kv_len-1:
//      a. Cooperatively compute dot(Q, K[k, kv_h, :]) via partial sums
//         + CUB BlockReduce.
//      b. Apply scale: score = dot * scale_factor.
//      c. Thread 0: online softmax update:
//         - old_max = max_score
//         - max_score = max(max_score, score)
//         - correction = exp(old_max - max_score)
//         - exp_score = exp(score - max_score)
//         - sum_exp = sum_exp * correction + exp_score
//         - Broadcast correction + exp_score to all threads via shared mem.
//      d. All threads: acc[d] = acc[d] * correction + exp_score * V[k, kv_h, d]
//   4. Normalize: output[h*head_dim + d] = acc[d] / sum_exp.
//
// Online softmax avoids materializing the full score vector and handles
// causal masking implicitly (since we only iterate over valid positions).
//
// Shared memory layout:
//   [0..head_dim-1]:       Q vector for this head
//   [head_dim..2*hd-1]:    accumulator
//   [2*hd]:                max_score
//   [2*hd+1]:              sum_exp
//   [2*hd+2]:              correction (broadcast)
//   [2*hd+3]:              exp_score (broadcast)
//   [aligned]:             CUB BlockReduce TempStorage
//
// Grid: n_heads blocks, 256 threads per block.
// ────────────────────────────────────────────────────────────────

constexpr int FUSED_ATTN_BLOCK = 256;

__global__ void kernel_fused_decode_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ output,
    int n_heads, int n_kv_heads, int head_dim, int kv_len,
    float scale_factor) {

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int n_groups = n_heads / n_kv_heads;
    const int kv_h = h / n_groups;
    const int kv_dim = n_kv_heads * head_dim;

    const float* q_ptr = Q + h * head_dim;

    using BlockReduce = cub::BlockReduce<float, FUSED_ATTN_BLOCK>;

    // Shared memory: Q[head_dim] + acc[head_dim] + {max_score, sum_exp, correction, exp_score}
    extern __shared__ char smem[];
    float* s_q   = reinterpret_cast<float*>(smem);
    float* s_acc = s_q + head_dim;
    // Control: [0]=max_score, [1]=sum_exp, [2]=correction, [3]=exp_score
    float* s_ctl = s_acc + head_dim;
    constexpr int N_CTL = 4;

    typename BlockReduce::TempStorage& temp_storage =
        *reinterpret_cast<typename BlockReduce::TempStorage*>(s_ctl + N_CTL);

    // Load Q into shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_q[d] = q_ptr[d];
        s_acc[d] = 0.0f;
    }
    if (threadIdx.x == 0) {
        s_ctl[0] = -INFINITY;  // max_score
        s_ctl[1] = 0.0f;       // sum_exp
    }
    __syncthreads();

    for (int k = 0; k < kv_len; ++k) {
        const float* k_ptr = K_cache + k * kv_dim + kv_h * head_dim;

        // Dot product: Q . K[k]
        float partial = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += s_q[d] * k_ptr[d];
        }

        __syncthreads();
        float dot = BlockReduce(temp_storage).Sum(partial);

        // Thread 0: online softmax update, broadcast correction + exp_score
        if (threadIdx.x == 0) {
            float score = dot * scale_factor;
            float old_max = s_ctl[0];
            float new_max = fmaxf(old_max, score);

            float correction = expf(old_max - new_max);
            float exp_score  = expf(score - new_max);

            s_ctl[0] = new_max;
            s_ctl[1] = s_ctl[1] * correction + exp_score;
            s_ctl[2] = correction;
            s_ctl[3] = exp_score;
        }
        __syncthreads();

        float corr = s_ctl[2];
        float es   = s_ctl[3];

        const float* v_ptr = V_cache + k * kv_dim + kv_h * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            s_acc[d] = s_acc[d] * corr + es * v_ptr[d];
        }
        __syncthreads();
    }

    // Normalize and write output
    float inv_sum = 1.0f / s_ctl[1];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        output[h * head_dim + d] = s_acc[d] * inv_sum;
    }
}

void fused_decode_attention(
    const float* Q, const float* K_cache, const float* V_cache,
    float* output,
    int n_heads, int n_kv_heads, int head_dim, int kv_len,
    float scale, cudaStream_t stream) {

    // Shared memory: Q[head_dim] + acc[head_dim] + 4 control floats + CUB TempStorage
    constexpr int N_CTL = 4;
    size_t smem_bytes = (2 * head_dim + N_CTL) * sizeof(float)
                      + sizeof(typename cub::BlockReduce<float, FUSED_ATTN_BLOCK>::TempStorage);

    kernel_fused_decode_attention<<<n_heads, FUSED_ATTN_BLOCK, smem_bytes, stream>>>(
        Q, K_cache, V_cache, output,
        n_heads, n_kv_heads, head_dim, kv_len, scale);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// AttentionBlock Implementation
// ════════════════════════════════════════════════════════════════

AttentionBlock::AttentionBlock(const AttentionConfig& config)
    : config_(config) {
    if (config_.n_heads <= 0) {
        throw std::invalid_argument(
            "AttentionBlock: n_heads must be > 0, got " +
            std::to_string(config_.n_heads));
    }
    if (config_.n_kv_heads <= 0) {
        throw std::invalid_argument(
            "AttentionBlock: n_kv_heads must be > 0, got " +
            std::to_string(config_.n_kv_heads));
    }
    if (config_.head_dim <= 0 || config_.head_dim % 2 != 0) {
        throw std::invalid_argument(
            "AttentionBlock: head_dim must be > 0 and even (for RoPE), got " +
            std::to_string(config_.head_dim));
    }
    if (config_.n_heads % config_.n_kv_heads != 0) {
        throw std::invalid_argument(
            "AttentionBlock: n_heads (" + std::to_string(config_.n_heads) +
            ") must be divisible by n_kv_heads (" +
            std::to_string(config_.n_kv_heads) + ")");
    }
}

Tensor AttentionBlock::forward(
    const Tensor& input,
    const Tensor& w_q, const Tensor& w_k,
    const Tensor& w_v, const Tensor& w_o,
    KVCache& kv_cache, int layer_idx, int pos) {

    const int n_heads   = config_.n_heads;
    const int n_kv_heads = config_.n_kv_heads;
    const int head_dim  = config_.head_dim;
    const int dim       = config_.dim();
    const int kv_dim    = config_.kv_dim();

    // input is [seq_len, dim]. Extract seq_len from the tensor shape.
    // Tensor shape is right-aligned in a 4D array: {1, 1, seq_len, dim}.
    const int seq_len = static_cast<int>(input.shape(2));

    // ── Step 1: QKV Projections ─────────────────────────────────
    //
    // Q = input * W_q  =>  [seq_len, dim]   x [dim, dim]    = [seq_len, dim]
    // K = input * W_k  =>  [seq_len, dim]   x [dim, kv_dim] = [seq_len, kv_dim]
    // V = input * W_v  =>  [seq_len, dim]   x [dim, kv_dim] = [seq_len, kv_dim]
    //
    // We use the tiled matmul for these projections since they are the largest
    // matmuls in the attention block (dim can be 576+ for SmolLM2-135M).

    Tensor q_proj = Tensor::zeros({seq_len, (int64_t)dim});
    Tensor k_proj = Tensor::zeros({seq_len, (int64_t)kv_dim});
    Tensor v_proj = Tensor::zeros({seq_len, (int64_t)kv_dim});

    matmul_tiled(input.data_as<float>(), w_q.data_as<float>(),
                 q_proj.data_as<float>(), seq_len, dim, dim);
    matmul_tiled(input.data_as<float>(), w_k.data_as<float>(),
                 k_proj.data_as<float>(), seq_len, kv_dim, dim);
    matmul_tiled(input.data_as<float>(), w_v.data_as<float>(),
                 v_proj.data_as<float>(), seq_len, kv_dim, dim);

    // ── Step 2: Apply RoPE to Q and K ───────────────────────────
    //
    // RoPE expects layout [n_heads, head_dim] for a single position.
    // For multi-token input, we apply RoPE position-by-position.
    //
    // q_proj has layout [seq_len, n_heads * head_dim] which is equivalent
    // to [seq_len, n_heads, head_dim] — each row is already in the correct
    // layout for the RoPE kernel (n_heads consecutive head vectors).
    //
    // Similarly, k_proj is [seq_len, n_kv_heads * head_dim].

    for (int t = 0; t < seq_len; ++t) {
        // Pointer to the t-th row of Q: offset = t * dim floats.
        float* q_row = q_proj.data_as<float>() + t * dim;
        rope(q_row, pos + t, head_dim, n_heads, config_.rope_theta);

        // Pointer to the t-th row of K: offset = t * kv_dim floats.
        float* k_row = k_proj.data_as<float>() + t * kv_dim;
        rope(k_row, pos + t, head_dim, n_kv_heads, config_.rope_theta);
    }

    // ── Step 3: Update KV-Cache ─────────────────────────────────
    //
    // k_proj: [seq_len, n_kv_heads * head_dim] is equivalent to
    //         [seq_len, n_kv_heads, head_dim] which matches the cache layout.
    // v_proj: same layout.
    //
    // The KVCache::append copies these into the cache at position `pos`.

    kv_cache.append(layer_idx, k_proj.data_as<float>(),
                    v_proj.data_as<float>(), seq_len);

    // After append, the cache holds positions 0..pos+seq_len-1.
    const int kv_len = pos + seq_len;

    // ── Fused decode fast path (seq_len=1) ──────────────────────
    //
    // For single-token decode, use the fused kernel that reads K/V
    // directly from the cache layout — replaces 9+ kernel launches.
    if (seq_len == 1) {
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        Tensor attn_concat = Tensor::zeros({1, (int64_t)dim});
        fused_decode_attention(
            q_proj.data_as<float>(),
            kv_cache.k_cache(layer_idx), kv_cache.v_cache(layer_idx),
            attn_concat.data_as<float>(),
            n_heads, n_kv_heads, head_dim, kv_len, attn_scale);

        // Output projection: Wo
        Tensor output = Tensor::zeros({1, (int64_t)dim});
        matmul_tiled(attn_concat.data_as<float>(), w_o.data_as<float>(),
                     output.data_as<float>(), 1, dim, dim);
        return output;
    }

    // ── Step 4: Scaled Dot-Product Attention (prefill path) ─────
    //
    // For each query head h:
    //   scores[h] = Q_h * K_{kv_h}^T          [seq_len, kv_len]
    //   scores[h] *= 1/sqrt(head_dim)
    //   apply causal mask
    //   attn_weights[h] = softmax(scores[h])   [seq_len, kv_len]
    //   output[h] = attn_weights[h] * V_{kv_h} [seq_len, head_dim]
    //
    // where kv_h = h / n_groups for GQA.

    // 4a. Transpose Q from [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim]
    Tensor q_heads = Tensor::empty({(int64_t)n_heads, (int64_t)seq_len, (int64_t)head_dim});
    transpose_seq_heads(q_proj.data_as<float>(), q_heads.data_as<float>(),
                        seq_len, n_heads, head_dim);

    // 4b. Gather K and V from cache: layout [kv_len, n_kv_heads, head_dim].
    //     Transpose K to [n_kv_heads, head_dim, kv_len] for Q * K^T.
    //     Transpose V to [n_kv_heads, kv_len, head_dim] for attn_weights * V.
    const float* k_cache_ptr = kv_cache.k_cache(layer_idx);
    const float* v_cache_ptr = kv_cache.v_cache(layer_idx);

    // k_cache_ptr points to [kv_len, n_kv_heads, head_dim] (only first kv_len positions valid).
    // We need [n_kv_heads, head_dim, kv_len] for Q * K^T.
    // First transpose to [n_kv_heads, kv_len, head_dim], then transpose last two dims.

    // Intermediate: [n_kv_heads, kv_len, head_dim]
    Tensor k_transposed_nh = Tensor::empty(
        {(int64_t)n_kv_heads, (int64_t)kv_len, (int64_t)head_dim});
    transpose_seq_heads(k_cache_ptr, k_transposed_nh.data_as<float>(),
                        kv_len, n_kv_heads, head_dim);

    // Final K^T: [n_kv_heads, head_dim, kv_len]
    Tensor k_t = Tensor::empty(
        {(int64_t)n_kv_heads, (int64_t)head_dim, (int64_t)kv_len});
    transpose_k(k_transposed_nh.data_as<float>(), k_t.data_as<float>(),
                n_kv_heads, kv_len, head_dim);

    // V: transpose from [kv_len, n_kv_heads, head_dim] to [n_kv_heads, kv_len, head_dim]
    Tensor v_heads = Tensor::empty(
        {(int64_t)n_kv_heads, (int64_t)kv_len, (int64_t)head_dim});
    transpose_seq_heads(v_cache_ptr, v_heads.data_as<float>(),
                        kv_len, n_kv_heads, head_dim);

    // 4c. Compute attention scores: Q * K^T -> [n_heads, seq_len, kv_len]
    //     Q: [n_heads, seq_len, head_dim], K^T: [n_kv_heads, head_dim, kv_len]
    //     Using GQA batched matmul to handle the head broadcast.
    Tensor scores = Tensor::zeros(
        {(int64_t)n_heads, (int64_t)seq_len, (int64_t)kv_len});

    batched_matmul_gqa(q_heads.data_as<float>(),
                       k_t.data_as<float>(),
                       scores.data_as<float>(),
                       n_heads, n_kv_heads,
                       seq_len, kv_len, head_dim);

    // 4d. Scale by 1/sqrt(head_dim)
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total_scores = n_heads * seq_len * kv_len;
    scale(scores.data_as<float>(), total_scores, scale_factor);

    // 4e. Apply causal mask
    causal_mask(scores.data_as<float>(), n_heads, seq_len, kv_len, pos);

    // 4f. Softmax over the kv_len dimension (per row).
    //     scores is [n_heads, seq_len, kv_len] = [n_heads * seq_len] rows of length kv_len.
    Tensor attn_weights = Tensor::empty(
        {(int64_t)n_heads, (int64_t)seq_len, (int64_t)kv_len});
    softmax(scores.data_as<float>(), attn_weights.data_as<float>(),
            n_heads * seq_len, kv_len);

    // 4g. Compute attention output: attn_weights * V -> [n_heads, seq_len, head_dim]
    //     attn_weights: [n_heads, seq_len, kv_len]
    //     V:            [n_kv_heads, kv_len, head_dim]
    Tensor attn_output = Tensor::zeros(
        {(int64_t)n_heads, (int64_t)seq_len, (int64_t)head_dim});

    batched_matmul_gqa(attn_weights.data_as<float>(),
                       v_heads.data_as<float>(),
                       attn_output.data_as<float>(),
                       n_heads, n_kv_heads,
                       seq_len, head_dim, kv_len);

    // ── Step 5: Concatenate heads and output projection ─────────
    //
    // Transpose attn_output from [n_heads, seq_len, head_dim]
    // back to [seq_len, n_heads * head_dim] = [seq_len, dim].
    Tensor concat = Tensor::empty({(int64_t)seq_len, (int64_t)dim});
    transpose_heads_seq(attn_output.data_as<float>(),
                        concat.data_as<float>(),
                        seq_len, n_heads, head_dim);

    // output = concat * W_o  =>  [seq_len, dim] x [dim, dim] = [seq_len, dim]
    Tensor output = Tensor::zeros({(int64_t)seq_len, (int64_t)dim});
    matmul_tiled(concat.data_as<float>(), w_o.data_as<float>(),
                 output.data_as<float>(), seq_len, dim, dim);

    return output;
}

// ════════════════════════════════════════════════════════════════
// forward_from_qkv — Attention with pre-projected Q/K/V
// ════════════════════════════════════════════════════════════════
//
// Used for quantized inference: QKV projections are done externally
// via quantized matmul, and RoPE is already applied. This method
// handles the rest: KV-cache update, scaled dot-product attention,
// and output projection.

/// Matmul dispatch for output projection (duplicated from model.cu to avoid
/// circular dependency — the function is trivial).
static void attn_matmul_dispatch(const float* A, const Tensor& weight,
                                  float* C, int M, int N, int K,
                                  WeightFormat format,
                                  cudaStream_t stream = 0) {
    switch (format) {
        case WeightFormat::FP32:
            matmul_tiled(A, weight.data_as<float>(), C, M, N, K, stream);
            break;
        case WeightFormat::FP16:
            matmul_fp16(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q8_0:
            matmul_q8_0(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q4_0:
            matmul_q4_0(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q5_0:
            matmul_q5_0(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q4_K:
            matmul_q4_k(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q6_K:
            matmul_q6_k(A, weight.data(), C, M, N, K, stream);
            break;
    }
}

Tensor AttentionBlock::forward_from_qkv(
    const Tensor& q_proj, const Tensor& k_proj, const Tensor& v_proj,
    const Tensor& w_o, WeightFormat w_o_format,
    KVCache& kv_cache, int layer_idx, int pos) {

    const int n_heads    = config_.n_heads;
    const int n_kv_heads = config_.n_kv_heads;
    const int head_dim   = config_.head_dim;
    const int dim        = config_.dim();
    const int kv_dim     = config_.kv_dim();
    const int seq_len    = static_cast<int>(q_proj.shape(2));

    // KV-cache update: k/v projections already have RoPE applied
    kv_cache.append(layer_idx, k_proj.data_as<float>(),
                    v_proj.data_as<float>(), seq_len);
    const int kv_len = pos + seq_len;

    // ── Fused decode fast path (seq_len=1) ──────────────────────
    if (seq_len == 1) {
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        Tensor attn_concat = Tensor::zeros({1, (int64_t)dim});
        fused_decode_attention(
            q_proj.data_as<float>(),
            kv_cache.k_cache(layer_idx), kv_cache.v_cache(layer_idx),
            attn_concat.data_as<float>(),
            n_heads, n_kv_heads, head_dim, kv_len, attn_scale);

        // Output projection with matmul dispatch
        Tensor output = Tensor::zeros({1, (int64_t)dim});
        attn_matmul_dispatch(attn_concat.data_as<float>(), w_o,
                              output.data_as<float>(), 1, dim, dim,
                              w_o_format);
        return output;
    }

    // ── Scaled Dot-Product Attention (prefill path) ─────────────

    // Transpose Q: [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
    Tensor q_heads = Tensor::empty({(int64_t)n_heads, (int64_t)seq_len, (int64_t)head_dim});
    transpose_seq_heads(q_proj.data_as<float>(), q_heads.data_as<float>(),
                        seq_len, n_heads, head_dim);

    // K cache: [kv_len, n_kv_heads, head_dim] -> [n_kv_heads, head_dim, kv_len]
    const float* k_cache_ptr = kv_cache.k_cache(layer_idx);
    Tensor k_transposed_nh = Tensor::empty(
        {(int64_t)n_kv_heads, (int64_t)kv_len, (int64_t)head_dim});
    transpose_seq_heads(k_cache_ptr, k_transposed_nh.data_as<float>(),
                        kv_len, n_kv_heads, head_dim);

    Tensor k_t = Tensor::empty(
        {(int64_t)n_kv_heads, (int64_t)head_dim, (int64_t)kv_len});
    transpose_k(k_transposed_nh.data_as<float>(), k_t.data_as<float>(),
                n_kv_heads, kv_len, head_dim);

    // V cache: [kv_len, n_kv_heads, head_dim] -> [n_kv_heads, kv_len, head_dim]
    const float* v_cache_ptr = kv_cache.v_cache(layer_idx);
    Tensor v_heads = Tensor::empty(
        {(int64_t)n_kv_heads, (int64_t)kv_len, (int64_t)head_dim});
    transpose_seq_heads(v_cache_ptr, v_heads.data_as<float>(),
                        kv_len, n_kv_heads, head_dim);

    // Compute attention scores: Q * K^T -> [n_heads, seq_len, kv_len]
    Tensor scores = Tensor::zeros(
        {(int64_t)n_heads, (int64_t)seq_len, (int64_t)kv_len});
    batched_matmul_gqa(q_heads.data_as<float>(), k_t.data_as<float>(),
                       scores.data_as<float>(),
                       n_heads, n_kv_heads, seq_len, kv_len, head_dim);

    // Scale by 1/sqrt(head_dim)
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total_scores = n_heads * seq_len * kv_len;
    scale(scores.data_as<float>(), total_scores, scale_factor);

    // Causal mask
    causal_mask(scores.data_as<float>(), n_heads, seq_len, kv_len, pos);

    // Softmax
    Tensor attn_weights = Tensor::empty(
        {(int64_t)n_heads, (int64_t)seq_len, (int64_t)kv_len});
    softmax(scores.data_as<float>(), attn_weights.data_as<float>(),
            n_heads * seq_len, kv_len);

    // Attention output: attn_weights * V -> [n_heads, seq_len, head_dim]
    Tensor attn_output = Tensor::zeros(
        {(int64_t)n_heads, (int64_t)seq_len, (int64_t)head_dim});
    batched_matmul_gqa(attn_weights.data_as<float>(),
                       v_heads.data_as<float>(),
                       attn_output.data_as<float>(),
                       n_heads, n_kv_heads, seq_len, head_dim, kv_len);

    // Transpose back: [n_heads, seq_len, head_dim] -> [seq_len, dim]
    Tensor concat = Tensor::empty({(int64_t)seq_len, (int64_t)dim});
    transpose_heads_seq(attn_output.data_as<float>(), concat.data_as<float>(),
                        seq_len, n_heads, head_dim);

    // Output projection with matmul dispatch
    Tensor output = Tensor::zeros({(int64_t)seq_len, (int64_t)dim});
    attn_matmul_dispatch(concat.data_as<float>(), w_o,
                          output.data_as<float>(), seq_len, dim, dim,
                          w_o_format);

    return output;
}

} // namespace cuinfer
