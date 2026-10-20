#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/cuda_utils.h"

#include <cfloat>   // FLT_MAX
#include <cub/cub.cuh>

namespace cuinfer {

// ── Constants ──────────────────────────────────────────────────

constexpr int BLOCK_SIZE = 256;

// ════════════════════════════════════════════════════════════════
// Kernel 1: RMSNorm
// ════════════════════════════════════════════════════════════════
//
// Each block processes one row of length n.
// Shared memory is used for a parallel reduction to compute sum(x[i]^2).
//
// Algorithm:
//   1. Each thread accumulates x[i]^2 for its stride-loop elements.
//   2. Store partial sums in shared memory.
//   3. Tree-based reduction to get total sum.
//   4. Thread 0 computes the normalization factor: rsqrt(mean_sq + eps).
//   5. Each thread writes: output[i] = input[i] * weight[i] * norm_factor.
//
// Shared memory: BLOCK_SIZE floats = 1024 bytes per block.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_rmsnorm(const float* input, const float* weight,
                               float* output, int n, float eps) {
    // Each block handles one row; blockIdx.x = row index.
    const int row = blockIdx.x;
    const float* row_in = input + row * n;
    float* row_out = output + row * n;

    extern __shared__ float sdata[];  // BLOCK_SIZE floats

    // ── Pass 1: Accumulate sum of squares ──
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = row_in[i];
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // ── Parallel tree reduction ──
    // Halve the active threads each step. This avoids bank conflicts
    // by keeping stride powers-of-two and using sequential addressing.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // sdata[0] now holds sum(x[i]^2) for this row.
    // Broadcast the normalization factor to all threads via shared memory.
    if (threadIdx.x == 0) {
        float mean_sq = sdata[0] / static_cast<float>(n);
        sdata[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float norm_factor = sdata[0];

    // ── Pass 2: Normalize and write output ──
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] = row_in[i] * weight[i] * norm_factor;
    }
}

void rmsnorm(const float* input, const float* weight, float* output,
             int rows, int n, float eps, cudaStream_t stream) {
    // One block per row. Shared memory for the parallel reduction.
    int smem_bytes = BLOCK_SIZE * sizeof(float);
    kernel_rmsnorm<<<rows, BLOCK_SIZE, smem_bytes, stream>>>(
        input, weight, output, n, eps);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 2: SiLU (Swish Activation)
// ════════════════════════════════════════════════════════════════
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Pure element-wise — no shared memory, no reductions.
// Each thread processes one element via a grid-stride loop.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_silu(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

void silu(const float* input, float* output, int n, cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_silu<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 3: GeLU (Approximate, Tanh Variant)
// ════════════════════════════════════════════════════════════════
//
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Uses the fast tanh approximation from the GPT-2 / LLaMA papers.
// The constant sqrt(2/pi) ~ 0.7978845608 is precomputed.
//
// Pure element-wise — no shared memory.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_gelu(const float* input, float* output, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void gelu(const float* input, float* output, int n, cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_gelu<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 4: Softmax (Numerically Stable)
// ════════════════════════════════════════════════════════════════
//
// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//
// Subtracting max(x) before exponentiation prevents overflow — this is
// the standard numerically stable softmax trick (log-sum-exp).
//
// Three-pass algorithm within one kernel:
//   Pass 1: Parallel reduction to find max_val.
//   Pass 2: Parallel reduction to compute sum(exp(x[i] - max_val)).
//   Pass 3: Normalize each element.
//
// One block per row. Shared memory: BLOCK_SIZE floats = 1024 bytes.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_softmax(const float* input, float* output, int n) {
    const int row = blockIdx.x;
    const float* row_in = input + row * n;
    float* row_out = output + row * n;

    extern __shared__ float sdata[];  // BLOCK_SIZE floats

    // ── Pass 1: Find max value (for numerical stability) ──
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[i]);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduction: max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x],
                                        sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();  // Ensure all threads read max_val before reusing sdata

    // ── Pass 2: Compute sum(exp(x[i] - max_val)) ──
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += expf(row_in[i] - max_val);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduction: sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    // ── Pass 3: Normalize ──
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - max_val) / sum_exp;
    }
}

void softmax(const float* input, float* output, int rows, int n,
             cudaStream_t stream) {
    int smem_bytes = BLOCK_SIZE * sizeof(float);
    kernel_softmax<<<rows, BLOCK_SIZE, smem_bytes, stream>>>(
        input, output, n);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 5: RoPE (Rotary Positional Embeddings)
// ════════════════════════════════════════════════════════════════
//
// Rotary embeddings encode positional information by rotating pairs of
// dimensions in the Q/K vectors. For dimension pair (2i, 2i+1):
//
//   freq_i = 1.0 / (theta_base ^ (2i / dim))
//   theta  = pos * freq_i
//
//   q_out[2i]   = q[2i] * cos(theta) - q[2i+1] * sin(theta)
//   q_out[2i+1] = q[2i] * sin(theta) + q[2i+1] * cos(theta)
//
// This is applied in-place. Each thread handles one pair across all heads.
// The qk buffer is laid out as [n_heads x dim], so pair index maps to:
//   head = pair_idx / (dim/2)
//   local_pair = pair_idx % (dim/2)
//   offset = head * dim + 2 * local_pair
// ────────────────────────────────────────────────────────────────

__global__ void kernel_rope(float* qk, const int pos, int dim,
                            int n_heads, float theta_base) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    int total_pairs = n_heads * half_dim;

    if (pair_idx < total_pairs) {
        int head = pair_idx / half_dim;
        int local_pair = pair_idx % half_dim;

        // Compute the rotation angle for this dimension pair.
        // freq = 1 / (theta_base ^ (2 * local_pair / dim))
        float freq = 1.0f / powf(theta_base,
                                  (2.0f * local_pair) / static_cast<float>(dim));
        float theta = static_cast<float>(pos) * freq;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        int offset = head * dim + 2 * local_pair;
        float x0 = qk[offset];
        float x1 = qk[offset + 1];

        // Apply 2D rotation to the pair.
        qk[offset]     = x0 * cos_theta - x1 * sin_theta;
        qk[offset + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

void rope(float* qk, int pos, int dim, int n_heads,
          float theta_base, cudaStream_t stream) {
    int total_pairs = n_heads * (dim / 2);
    int grid = (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_rope<<<grid, BLOCK_SIZE, 0, stream>>>(
        qk, pos, dim, n_heads, theta_base);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 6: Element-wise Add
// ════════════════════════════════════════════════════════════════
//
// output[i] = a[i] + b[i]
// Straightforward element-wise operation, no shared memory.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_add(const float* a, const float* b,
                           float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

void add(const float* a, const float* b, float* output, int n,
         cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add<<<grid, BLOCK_SIZE, 0, stream>>>(a, b, output, n);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 7: Element-wise Multiply
// ════════════════════════════════════════════════════════════════
//
// output[i] = a[i] * b[i]
// Straightforward element-wise operation, no shared memory.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_multiply(const float* a, const float* b,
                                float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx];
    }
}

void multiply(const float* a, const float* b, float* output, int n,
              cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_multiply<<<grid, BLOCK_SIZE, 0, stream>>>(a, b, output, n);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 8: RMSNorm (CUB-Accelerated)
// ════════════════════════════════════════════════════════════════
//
// Same algorithm as kernel_rmsnorm, but replaces the hand-written
// tree reduction with cub::BlockReduce<float, 256>::Sum().
// CUB uses warp shuffles for the final 32 elements instead of
// __syncthreads(), which should be measurably faster.
//
// No dynamic shared memory needed — CUB's TempStorage is a
// statically-sized union declared in-kernel.
// ────────────────────────────────────────────────────────────────

constexpr int CUB_BLOCK_SIZE = 256;

__global__ void kernel_rmsnorm_cub(const float* input, const float* weight,
                                    float* output, int n, float eps) {
    const int row = blockIdx.x;
    const float* row_in = input + row * n;
    float* row_out = output + row * n;

    using BlockReduce = cub::BlockReduce<float, CUB_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ── Pass 1: Accumulate sum of squares ──
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        float val = row_in[i];
        thread_sum += val * val;
    }

    // CUB block-wide sum reduction (uses warp shuffles internally)
    float total_sum = BlockReduce(temp_storage).Sum(thread_sum);

    // Broadcast normalization factor via shared memory
    __shared__ float norm_factor;
    if (threadIdx.x == 0) {
        float mean_sq = total_sum / static_cast<float>(n);
        norm_factor = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // ── Pass 2: Normalize and write output ──
    float nf = norm_factor;
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        row_out[i] = row_in[i] * weight[i] * nf;
    }
}

void rmsnorm_cub(const float* input, const float* weight, float* output,
                 int rows, int n, float eps, cudaStream_t stream) {
    kernel_rmsnorm_cub<<<rows, CUB_BLOCK_SIZE, 0, stream>>>(
        input, weight, output, n, eps);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 9: Softmax (CUB-Accelerated)
// ════════════════════════════════════════════════════════════════
//
// Same three-pass algorithm as kernel_softmax, but uses
// cub::BlockReduce for both the max-reduction and sum-reduction.
//
// TempStorage is reused between the two reductions with a
// __syncthreads() barrier in between, per CUB documentation.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_softmax_cub(const float* input, float* output, int n) {
    const int row = blockIdx.x;
    const float* row_in = input + row * n;
    float* row_out = output + row * n;

    using BlockReduce = cub::BlockReduce<float, CUB_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ── Pass 1: Find max value ──
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_in[i]);
    }

    float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    // Broadcast max to all threads
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        max_val = block_max;
    }
    __syncthreads();

    // ── Pass 2: Compute sum(exp(x[i] - max_val)) ──
    float thread_sum = 0.0f;
    float mv = max_val;
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        thread_sum += expf(row_in[i] - mv);
    }

    // Must sync before reusing TempStorage
    __syncthreads();
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    __shared__ float sum_exp;
    if (threadIdx.x == 0) {
        sum_exp = block_sum;
    }
    __syncthreads();

    // ── Pass 3: Normalize ──
    float se = sum_exp;
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        row_out[i] = expf(row_in[i] - mv) / se;
    }
}

void softmax_cub(const float* input, float* output, int rows, int n,
                 cudaStream_t stream) {
    kernel_softmax_cub<<<rows, CUB_BLOCK_SIZE, 0, stream>>>(
        input, output, n);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 10: Fused RMSNorm + Residual Add (Hand-Written)
// ════════════════════════════════════════════════════════════════
//
// Fuses two operations into a single kernel:
//   1. Residual addition: output[i] = input[i] + residual[i]
//   2. RMSNorm: normed[i] = output[i] * weight[i] * rsqrt(mean(output^2) + eps)
//
// This saves one kernel launch and one full read/write of the hidden
// state compared to separate add() + rmsnorm() calls. During decode
// (seq_len=1), this eliminates ~50% of the elementwise kernel overhead
// per transformer layer.
//
// Algorithm (1 block per row, 256 threads):
//   Pass 1: Each thread loops over stride elements. For each:
//           val = input[i] + residual[i]; write to output[i]; accumulate val*val.
//           Store partial sum in shared memory.
//   Reduction: Tree-based sum of squares in shared memory.
//   Thread 0: Compute norm_factor = rsqrt(sum/n + eps), broadcast via sdata[0].
//   Pass 2: Each thread: normed[i] = output[i] * weight[i] * norm_factor.
//
// Note: output may alias input (in-place residual update). This is safe
// because pass 1 reads input[i] before writing output[i] at the same index.
//
// Shared memory: BLOCK_SIZE floats = 1024 bytes per block.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_rmsnorm_residual(
    const float* input, const float* residual, const float* weight,
    float* output, float* normed, int n, float eps) {

    const int row = blockIdx.x;
    const float* row_in  = input    + row * n;
    const float* row_res = residual + row * n;
    float* row_out    = output + row * n;
    float* row_normed = normed + row * n;

    extern __shared__ float sdata[];

    // ── Pass 1: Add residual and accumulate sum of squares ──
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = row_in[i] + row_res[i];
        row_out[i] = val;
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // ── Parallel tree reduction ──
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Broadcast normalization factor
    if (threadIdx.x == 0) {
        float mean_sq = sdata[0] / static_cast<float>(n);
        sdata[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float norm_factor = sdata[0];

    // ── Pass 2: Normalize and write normed output ──
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_normed[i] = row_out[i] * weight[i] * norm_factor;
    }
}

void rmsnorm_residual(const float* input, const float* residual,
                      const float* weight, float* output, float* normed,
                      int rows, int n, float eps, cudaStream_t stream) {
    int smem_bytes = BLOCK_SIZE * sizeof(float);
    kernel_rmsnorm_residual<<<rows, BLOCK_SIZE, smem_bytes, stream>>>(
        input, residual, weight, output, normed, n, eps);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Kernel 11: Fused RMSNorm + Residual Add (CUB-Accelerated)
// ════════════════════════════════════════════════════════════════
//
// Same algorithm as kernel_rmsnorm_residual but uses cub::BlockReduce
// for the sum-of-squares reduction. The CUB variant uses warp shuffles
// for the final 32 elements, avoiding extra __syncthreads() calls.
// ────────────────────────────────────────────────────────────────

__global__ void kernel_rmsnorm_residual_cub(
    const float* input, const float* residual, const float* weight,
    float* output, float* normed, int n, float eps) {

    const int row = blockIdx.x;
    const float* row_in  = input    + row * n;
    const float* row_res = residual + row * n;
    float* row_out    = output + row * n;
    float* row_normed = normed + row * n;

    using BlockReduce = cub::BlockReduce<float, CUB_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ── Pass 1: Add residual and accumulate sum of squares ──
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        float val = row_in[i] + row_res[i];
        row_out[i] = val;
        thread_sum += val * val;
    }

    // CUB block-wide sum reduction
    float total_sum = BlockReduce(temp_storage).Sum(thread_sum);

    // Broadcast normalization factor
    __shared__ float norm_factor;
    if (threadIdx.x == 0) {
        float mean_sq = total_sum / static_cast<float>(n);
        norm_factor = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float nf = norm_factor;

    // ── Pass 2: Normalize and write normed output ──
    for (int i = threadIdx.x; i < n; i += CUB_BLOCK_SIZE) {
        row_normed[i] = row_out[i] * weight[i] * nf;
    }
}

void rmsnorm_residual_cub(const float* input, const float* residual,
                          const float* weight, float* output, float* normed,
                          int rows, int n, float eps, cudaStream_t stream) {
    kernel_rmsnorm_residual_cub<<<rows, CUB_BLOCK_SIZE, 0, stream>>>(
        input, residual, weight, output, normed, n, eps);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

} // namespace cuinfer
