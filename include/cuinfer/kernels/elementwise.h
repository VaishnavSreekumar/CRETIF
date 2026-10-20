#pragma once

#include <cuda_runtime.h>

namespace cuinfer {

// ── RMSNorm ────────────────────────────────────────────────────
// Applies Root Mean Square Layer Normalization per row:
//   output[i] = input[i] * weight[i] * rsqrt(mean(input^2) + eps)
//
// input:  [rows x n]  — contiguous row-major
// weight: [n]          — shared across all rows
// output: [rows x n]  — same layout as input
// Grid: rows blocks (one block per row), 256 threads per block.
void rmsnorm(const float* input, const float* weight, float* output,
             int rows, int n, float eps = 1e-5f, cudaStream_t stream = 0);

// ── SiLU (Swish Activation) ────────────────────────────────────
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Pure element-wise operation on n elements.
// Grid: ceil(n / 256) blocks, 256 threads per block.
void silu(const float* input, float* output, int n,
          cudaStream_t stream = 0);

// ── GeLU (Approximate, Tanh Variant) ───────────────────────────
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Pure element-wise operation on n elements.
// Grid: ceil(n / 256) blocks, 256 threads per block.
void gelu(const float* input, float* output, int n,
          cudaStream_t stream = 0);

// ── Softmax (Numerically Stable) ───────────────────────────────
// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//
// Per-row softmax. Three-pass within a single kernel:
//   1. Parallel reduction to find max
//   2. Parallel reduction to compute sum(exp(x[i] - max))
//   3. Normalize: output[i] = exp(input[i] - max) / sum_exp
//
// input:  [rows x n]  — contiguous row-major
// output: [rows x n]  — same layout
// Grid: rows blocks, 256 threads per block.
void softmax(const float* input, float* output, int rows, int n,
             cudaStream_t stream = 0);

// ── RoPE (Rotary Positional Embeddings) ────────────────────────
// Applies rotary embeddings in-place to Q or K vectors.
// For each dimension pair (2i, 2i+1) at position pos:
//   theta = pos / (theta_base ^ (2i / dim))
//   qk[2i]   = qk[2i] * cos(theta) - qk[2i+1] * sin(theta)
//   qk[2i+1] = qk[2i] * sin(theta) + qk[2i+1] * cos(theta)
//
// qk:         [n_heads x dim] — modified in-place
// pos:        token position in the sequence
// dim:        dimension per head (must be even)
// n_heads:    number of attention heads
// theta_base: base for frequency computation (default 10000)
// Grid: ceil(n_heads * dim/2 / 256) blocks, 256 threads per block.
void rope(float* qk, int pos, int dim, int n_heads,
          float theta_base = 10000.0f, cudaStream_t stream = 0);

// ── Element-wise Add ───────────────────────────────────────────
// output[i] = a[i] + b[i]
//
// Grid: ceil(n / 256) blocks, 256 threads per block.
void add(const float* a, const float* b, float* output, int n,
         cudaStream_t stream = 0);

// ── Element-wise Multiply ──────────────────────────────────────
// output[i] = a[i] * b[i]
//
// Grid: ceil(n / 256) blocks, 256 threads per block.
void multiply(const float* a, const float* b, float* output, int n,
              cudaStream_t stream = 0);

// ── RMSNorm (CUB-accelerated) ─────────────────────────────────
// Same interface as rmsnorm(), but uses cub::BlockReduce for the
// parallel sum reduction. Warp shuffles in the final 32 elements
// replace the hand-written __syncthreads() tree reduction.
// Side-by-side comparison with the hand-written version for benchmarking.
void rmsnorm_cub(const float* input, const float* weight, float* output,
                 int rows, int n, float eps = 1e-5f, cudaStream_t stream = 0);

// ── Softmax (CUB-accelerated) ─────────────────────────────────
// Same interface as softmax(), but uses cub::BlockReduce for the
// max-reduction and sum-reduction passes. Same three-pass structure.
void softmax_cub(const float* input, float* output, int rows, int n,
                 cudaStream_t stream = 0);

// ── Fused RMSNorm + Residual Add ────────────────────────────
// Combines residual addition and RMSNorm into a single kernel launch:
//   output[i] = input[i] + residual[i]         (residual add)
//   normed[i] = output[i] * weight[i] * rsqrt(mean(output^2) + eps)
//
// output may alias input for in-place residual update.
// This saves one kernel launch and one global memory round-trip vs
// separate add() + rmsnorm() calls.
//
// input:    [rows x n] — current hidden state
// residual: [rows x n] — value to add (e.g., attention output)
// weight:   [n]        — RMSNorm weight vector
// output:   [rows x n] — updated hidden state (input + residual)
// normed:   [rows x n] — normalized output for next sublayer
// Grid: rows blocks, 256 threads per block.
void rmsnorm_residual(const float* input, const float* residual,
                      const float* weight, float* output, float* normed,
                      int rows, int n, float eps = 1e-5f,
                      cudaStream_t stream = 0);

// ── Fused RMSNorm + Residual Add (CUB-accelerated) ─────────
// Same as rmsnorm_residual() but uses cub::BlockReduce for the
// sum-of-squares reduction.
void rmsnorm_residual_cub(const float* input, const float* residual,
                          const float* weight, float* output, float* normed,
                          int rows, int n, float eps = 1e-5f,
                          cudaStream_t stream = 0);

} // namespace cuinfer
