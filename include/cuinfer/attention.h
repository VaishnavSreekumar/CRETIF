#pragma once

#include "cuinfer/kv_cache.h"
#include "cuinfer/tensor.h"

#include <cuda_runtime.h>

// Forward-declare WeightFormat to avoid circular include
namespace cuinfer { enum class WeightFormat; }

namespace cuinfer {

// ── Attention Configuration ──────────────────────────────────
//
// Parameters that define the multi-head (or grouped-query) attention block.
// For standard multi-head attention (MHA), n_heads == n_kv_heads.
// For grouped-query attention (GQA), n_kv_heads < n_heads, and each group
// of (n_heads / n_kv_heads) query heads shares one K/V head.

struct AttentionConfig {
    int n_heads;       ///< Number of query heads
    int n_kv_heads;    ///< Number of key/value heads (< n_heads for GQA)
    int head_dim;      ///< Dimension per head
    int max_seq_len;   ///< Maximum sequence length
    float rope_theta = 10000.0f;  ///< RoPE frequency base

    /// Full model dimension: n_heads * head_dim.
    int dim() const { return n_heads * head_dim; }

    /// Key/value projection dimension: n_kv_heads * head_dim.
    int kv_dim() const { return n_kv_heads * head_dim; }

    /// Number of query heads per K/V head group (for GQA broadcast).
    int n_groups() const { return n_heads / n_kv_heads; }
};

// ── Attention CUDA Kernels (host wrappers) ──────────────────
//
// These are the low-level building blocks used by AttentionBlock.
// Exposed in the header so tests can exercise them independently.

/// Applies a causal mask to attention scores in-place.
/// scores: [n_heads, seq_len, kv_len] — row-major
/// Sets scores[h][q][k] = -inf for k > (pos - seq_len + 1 + q),
/// i.e., each query position can only attend to itself and earlier positions.
///
/// Grid: ceil(n_heads * seq_len * kv_len / 256) blocks, 256 threads/block.
void causal_mask(float* scores, int n_heads, int seq_len, int kv_len,
                 int start_pos, cudaStream_t stream = 0);

/// Scales attention scores in-place by a constant factor.
/// scores: pointer to n elements on the GPU.
/// factor: the multiplicative scale (typically 1/sqrt(head_dim)).
///
/// Grid: ceil(n / 256) blocks, 256 threads/block.
void scale(float* data, int n, float factor, cudaStream_t stream = 0);

/// Computes batched matmul: C[b] = A[b] * B[b] for b in [0, batch).
/// All matrices within a batch are contiguous in memory.
///   A: [batch, M, K] row-major
///   B: [batch, K, N] row-major
///   C: [batch, M, N] row-major
///
/// Each batch is dispatched as an independent matmul to reuse the tiled kernel.
void batched_matmul(const float* A, const float* B, float* C,
                    int batch, int M, int N, int K,
                    cudaStream_t stream = 0);

/// Computes batched matmul with GQA broadcast on B:
///   A: [n_heads, M, K]         — one matrix per query head
///   B: [n_kv_heads, K, N]      — one matrix per KV head (fewer than n_heads)
///   C: [n_heads, M, N]         — one output per query head
///
/// Query head h uses B[h / n_groups] where n_groups = n_heads / n_kv_heads.
/// This avoids physically expanding the K/V tensors for GQA.
///
/// Grid: one 2D block per (head, output_tile). Each thread computes one
/// element of C using a naive dot product (sufficient for small M, N).
void batched_matmul_gqa(const float* A, const float* B, float* C,
                        int n_heads, int n_kv_heads,
                        int M, int N, int K,
                        cudaStream_t stream = 0);

// ── AttentionBlock ──────────────────────────────────────────
//
// Implements a complete multi-head / grouped-query attention layer:
//   1. QKV projection (input x W_q/W_k/W_v)
//   2. RoPE on Q and K
//   3. KV-cache update (append new K, V)
//   4. Scaled dot-product attention with causal mask
//   5. Output projection (attn_output x W_o)
//
// The forward() method expects single-token decode (seq_len=1) or
// multi-token prefill (seq_len>1) inputs.

class AttentionBlock {
public:
    explicit AttentionBlock(const AttentionConfig& config);

    /// Runs the full attention forward pass.
    ///
    /// @param input   Device tensor [seq_len, dim] — input activations.
    /// @param w_q     Device tensor [dim, dim]     — query projection weights.
    /// @param w_k     Device tensor [dim, kv_dim]  — key projection weights.
    /// @param w_v     Device tensor [dim, kv_dim]  — value projection weights.
    /// @param w_o     Device tensor [dim, dim]      — output projection weights.
    /// @param kv_cache KV-cache (modified in place).
    /// @param layer_idx Layer index in the transformer stack [0, n_layers).
    /// @param pos     Current sequence position (= kv_cache.current_length()
    ///                before this call; provided explicitly for clarity).
    /// @return        Output tensor [seq_len, dim] on GPU.
    Tensor forward(const Tensor& input,
                   const Tensor& w_q, const Tensor& w_k,
                   const Tensor& w_v, const Tensor& w_o,
                   KVCache& kv_cache, int layer_idx, int pos);

    /// Quantized attention forward: takes pre-projected Q/K/V tensors.
    /// Q/K/V have already been computed via quantized matmul and RoPE applied.
    /// This method handles KV-cache update and the attention computation,
    /// then applies the output projection using matmul_dispatch.
    ///
    /// @param q_proj   [seq_len, dim] — projected queries with RoPE.
    /// @param k_proj   [seq_len, kv_dim] — projected keys with RoPE.
    /// @param v_proj   [seq_len, kv_dim] — projected values.
    /// @param w_o      Output projection weight (quantized or FP32).
    /// @param w_o_format  Weight format of w_o.
    Tensor forward_from_qkv(const Tensor& q_proj, const Tensor& k_proj,
                            const Tensor& v_proj,
                            const Tensor& w_o, WeightFormat w_o_format,
                            KVCache& kv_cache, int layer_idx, int pos);

    /// Returns the attention configuration.
    const AttentionConfig& config() const { return config_; }

private:
    AttentionConfig config_;
};

// ── Fused Decode Attention (seq_len=1) ──────────────────────
//
// Single-kernel attention for autoregressive decode. Replaces the
// 9+ kernel launches (3 transposes, Q*K^T matmul, scale, causal_mask,
// softmax, weights*V matmul, output transpose) with one fused kernel
// that reads K/V directly from the cache layout.
//
// Uses online softmax (Milakov & Gimelshein, 2018) to compute the
// attention output in a single pass over the KV-cache positions,
// without materializing the full attention score matrix.
//
// Grid: n_heads blocks (one block per query head).
// Block: 256 threads.
// Shared memory: head_dim floats (Q vector) + head_dim floats (accum)
//                + 2 control floats (max_score, sum_exp) + CUB TempStorage.
//
// Q:        [n_heads * head_dim] — single query token, all heads contiguous.
// K_cache:  [kv_len, n_kv_heads, head_dim] — cache layout directly.
// V_cache:  [kv_len, n_kv_heads, head_dim] — cache layout directly.
// output:   [n_heads * head_dim] — attention output for this token.
//
// GQA is handled via kv_h = h / (n_heads / n_kv_heads).
void fused_decode_attention(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* output,
    int n_heads, int n_kv_heads, int head_dim, int kv_len,
    float scale, cudaStream_t stream = 0);

} // namespace cuinfer
