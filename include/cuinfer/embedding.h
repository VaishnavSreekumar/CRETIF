#pragma once

#include "cuinfer/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace cuinfer {

// ── Embedding Layer ──────────────────────────────────────────────────
//
// Token embedding lookup: maps integer token IDs to dense vectors.
// Given a weight matrix W [vocab_size, dim] and token IDs [seq_len],
// produces output [seq_len, dim] where output[i] = W[token_ids[i]].
//
// In LLM architectures (GPT, Llama, SmolLM2), this is the first
// layer — converting discrete token indices into continuous vectors
// that the transformer can process.

class Embedding {
public:
    /// Create an embedding layer descriptor.
    /// @param vocab_size  Number of tokens in the vocabulary
    /// @param dim         Embedding dimension per token
    Embedding(int vocab_size, int dim);

    /// Forward pass: look up embedding vectors for each token ID.
    ///
    /// @param token_ids  Device tensor of shape [seq_len] containing int32
    ///                   token IDs. Despite being stored in a Tensor, the
    ///                   data is interpreted as int32_t (packed into FP32
    ///                   storage, or passed via raw pointer internally).
    ///                   Each value must be in [0, vocab_size).
    /// @param weight     Device tensor of shape [vocab_size, dim] (FP32),
    ///                   the embedding weight matrix. Can be shared with
    ///                   LMHead for weight tying.
    /// @return           Newly allocated tensor of shape [seq_len, dim]
    ///                   containing the looked-up embedding vectors.
    Tensor forward(const int32_t* token_ids, int seq_len,
                   const Tensor& weight, cudaStream_t stream = 0);

private:
    int vocab_size_;
    int dim_;
};

// ── LM Head (Language Model Head) ────────────────────────────────────
//
// Final linear projection: hidden_states [seq_len, dim] -> logits [seq_len, vocab_size]
//
// Computes: logits = hidden_states * weight^T
//
// where weight is [vocab_size, dim]. This is equivalent to a matmul
// with the weight transposed:
//   C[i,j] = sum_k hidden[i,k] * weight[j,k]
//
// In many models (Llama, SmolLM2), the LM head weight is tied
// (shared) with the token embedding weight — the same [vocab_size, dim]
// matrix is used for both embedding lookup and final projection.

class LMHead {
public:
    /// Create an LM head descriptor.
    /// @param vocab_size  Output vocabulary size (logits dimension)
    /// @param dim         Input hidden state dimension
    LMHead(int vocab_size, int dim);

    /// Forward pass: project hidden states to vocabulary logits.
    ///
    /// @param hidden   Device tensor of shape [seq_len, dim] (FP32)
    /// @param weight   Device tensor of shape [vocab_size, dim] (FP32).
    ///                 Can be the same tensor as the embedding weight
    ///                 (tied weights).
    /// @return         Newly allocated tensor of shape [seq_len, vocab_size]
    Tensor forward(const Tensor& hidden, const Tensor& weight,
                   cudaStream_t stream = 0);

private:
    int vocab_size_;
    int dim_;
};

// ── Sampling Utilities ───────────────────────────────────────────────
//
// Given a logits vector [vocab_size] (the last row of the LM head
// output during autoregressive generation), select the next token.

/// Greedy decoding: return the index of the maximum logit.
/// @param logits  Device pointer to [vocab_size] floats
/// @param n       Number of elements (vocab_size)
/// @return        Token ID with the highest logit value
int32_t argmax_sample(const float* logits, int n, cudaStream_t stream = 0);

/// Top-k sampling with temperature scaling.
///
/// 1. Copy logits to a working buffer
/// 2. Find the top-k logits via partial sort
/// 3. Apply temperature scaling: logit_i /= temperature
/// 4. Softmax over the top-k candidates
/// 5. Sample from the resulting distribution
///
/// @param logits      Device pointer to [vocab_size] floats
/// @param n           Number of elements (vocab_size)
/// @param k           Number of top candidates to consider (default: 40)
/// @param temperature Temperature for scaling (default: 1.0). Lower = more
///                    deterministic, higher = more random. Must be > 0.
/// @param seed        Random seed for reproducible sampling
/// @return            Sampled token ID
int32_t top_k_sample(const float* logits, int n, int k = 40,
                     float temperature = 1.0f, unsigned int seed = 42,
                     cudaStream_t stream = 0);

// ── Low-Level Kernel Wrappers ────────────────────────────────────────
// These are exposed for testing and benchmarking. Most callers should
// use the Embedding/LMHead classes instead.

/// Embedding lookup kernel wrapper.
/// Launches kernel_embedding_lookup: each thread copies one float
/// from the embedding table to the output.
///
/// @param token_ids  Device pointer to [seq_len] int32 token IDs
/// @param weight     Device pointer to [vocab_size * dim] float weight matrix
/// @param output     Device pointer to [seq_len * dim] float output
/// @param seq_len    Number of tokens
/// @param dim        Embedding dimension
void embedding_lookup(const int32_t* token_ids, const float* weight,
                      float* output, int seq_len, int dim,
                      cudaStream_t stream = 0);

/// Matrix multiplication with transposed B: C = A * B^T
/// where A is [M, K] and B is [N, K] (note: B stored row-major as [N, K]).
/// Output C is [M, N].
///
/// This is needed for the LM head projection because the weight matrix
/// is stored as [vocab_size, dim] (same as embedding table), so we need
/// hidden [seq_len, dim] * weight^T [dim, vocab_size] = logits [seq_len, vocab_size].
void matmul_transposed_b(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream = 0);

} // namespace cuinfer
