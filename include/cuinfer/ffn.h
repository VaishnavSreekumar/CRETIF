#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuinfer {

// ── FFN Configuration ──────────────────────────────────────
//
// Holds the hyperparameters for a feed-forward network block.
// Two FFN variants are supported:
//
// 1. Llama-style Gated FFN (SwiGLU):
//      hidden = SiLU(input * gate_proj) * (input * up_proj)
//      output = hidden * down_proj
//    Used by Llama, SmolLM2, Mistral, and most modern LLMs.
//    Requires three weight matrices: gate_proj, up_proj, down_proj.
//
// 2. Standard FFN (GPT-style):
//      hidden = GeLU(input * W1 + b1)
//      output = hidden * W2 + b2
//    Used by GPT-2, BERT, and older transformer models.
//    Requires two weight matrices and two bias vectors.

struct FFNConfig {
    int dim;                ///< Model dimension (input/output size)
    int intermediate_size;  ///< FFN hidden dimension
                            ///< Typically 4x dim for standard FFN, or ~2.67x for Llama-style
                            ///< (e.g., SmolLM2-135M: dim=576, intermediate=1536)
};

// ── Llama-style Gated FFN (SwiGLU) ─────────────────────────
//
// Implements the gated feed-forward network used by Llama and derivatives:
//
//   hidden = SiLU(input * gate_proj) * (input * up_proj)
//   output = hidden * down_proj
//
// This is the "SwiGLU" variant from the Llama paper, which gates the
// up-projection with a SiLU-activated gate projection. The gating
// mechanism allows the network to learn which features to amplify vs
// suppress, improving expressiveness compared to a simple nonlinearity.
//
// Weight layouts (row-major):
//   gate_proj: [dim, intermediate_size]
//   up_proj:   [dim, intermediate_size]
//   down_proj: [intermediate_size, dim]
//
// Input/output:
//   input:  [seq_len, dim]
//   output: [seq_len, dim]
//
// Internal temporaries (allocated once, reused per forward call):
//   gate_buf:   [seq_len, intermediate_size]  — result of input * gate_proj
//   up_buf:     [seq_len, intermediate_size]  — result of input * up_proj
//   hidden_buf: [seq_len, intermediate_size]  — SiLU(gate_buf) * up_buf
//
// Note: gate_proj and up_proj could be fused into a single matmul
// if the weights are concatenated along the output dimension:
//   [gate; up] = input * [gate_proj | up_proj]  ([dim, 2*intermediate_size])
// This would halve the number of matmul launches at the cost of
// requiring the weight layout to be contiguous. Deferred to a future
// optimization pass — the separate matmuls are clearer for now.

class GatedFFN {
public:
    /// Constructs the gated FFN block and pre-allocates intermediate buffers
    /// for the given maximum sequence length.
    ///
    /// @param config       FFN dimensions (dim, intermediate_size).
    /// @param max_seq_len  Maximum sequence length to support. Determines the
    ///                     size of internal temporary buffers. Must be >= 1.
    GatedFFN(const FFNConfig& config, int max_seq_len);
    ~GatedFFN();

    // No copy — owns GPU temporaries.
    GatedFFN(const GatedFFN&) = delete;
    GatedFFN& operator=(const GatedFFN&) = delete;

    // Move semantics transfer buffer ownership.
    GatedFFN(GatedFFN&& other) noexcept;
    GatedFFN& operator=(GatedFFN&& other) noexcept;

    /// Forward pass: computes the gated FFN output.
    ///
    /// @param input    Device pointer to input, shape [seq_len, dim].
    /// @param w_gate   Device pointer to gate_proj weights, shape [dim, intermediate_size].
    /// @param w_up     Device pointer to up_proj weights, shape [dim, intermediate_size].
    /// @param w_down   Device pointer to down_proj weights, shape [intermediate_size, dim].
    /// @param output   Device pointer to output buffer, shape [seq_len, dim].
    /// @param seq_len  Number of tokens in the sequence (must be <= max_seq_len).
    /// @param stream   CUDA stream for async execution.
    void forward(const float* input,
                 const float* w_gate, const float* w_up, const float* w_down,
                 float* output, int seq_len, cudaStream_t stream = nullptr);

    const FFNConfig& config() const { return config_; }
    int max_seq_len() const { return max_seq_len_; }

    /// Total GPU memory used by internal temporary buffers (in bytes).
    size_t temp_buffer_bytes() const;

    /// Direct access to internal buffers for quantized forward pass.
    /// The model's forward loop uses these to do quantized matmul externally,
    /// then SiLU/multiply, then another quantized matmul — reusing the same
    /// pre-allocated buffers that the FP32 forward() would use internally.
    float* gate_buf() { return gate_buf_; }
    float* up_buf() { return up_buf_; }
    float* hidden_buf() { return hidden_buf_; }

private:
    FFNConfig config_;
    int max_seq_len_;

    // Pre-allocated GPU temporaries for the intermediate activations.
    // Sized for [max_seq_len, intermediate_size] each.
    float* gate_buf_ = nullptr;   // input * gate_proj result
    float* up_buf_   = nullptr;   // input * up_proj result
    float* hidden_buf_ = nullptr; // SiLU(gate_buf) * up_buf result

    void free_buffers();
};

// ── Standard FFN (GPT-style) ───────────────────────────────
//
// Implements the classic two-layer feed-forward network:
//
//   hidden = GeLU(input * W1 + b1)
//   output = hidden * W2 + b2
//
// Used by GPT-2, BERT, and similar architectures.
//
// Weight layouts (row-major):
//   w1: [dim, intermediate_size]
//   b1: [intermediate_size]           (optional, may be nullptr)
//   w2: [intermediate_size, dim]
//   b2: [dim]                         (optional, may be nullptr)
//
// Input/output:
//   input:  [seq_len, dim]
//   output: [seq_len, dim]

class StandardFFN {
public:
    /// Constructs the standard FFN block and pre-allocates temporaries.
    ///
    /// @param config       FFN dimensions (dim, intermediate_size).
    /// @param max_seq_len  Maximum sequence length to support.
    StandardFFN(const FFNConfig& config, int max_seq_len);
    ~StandardFFN();

    // No copy — owns GPU temporaries.
    StandardFFN(const StandardFFN&) = delete;
    StandardFFN& operator=(const StandardFFN&) = delete;

    // Move semantics.
    StandardFFN(StandardFFN&& other) noexcept;
    StandardFFN& operator=(StandardFFN&& other) noexcept;

    /// Forward pass: computes the standard FFN output.
    ///
    /// @param input    Device pointer to input, shape [seq_len, dim].
    /// @param w1       Device pointer to first linear weights, shape [dim, intermediate_size].
    /// @param b1       Device pointer to first bias, shape [intermediate_size]. May be nullptr.
    /// @param w2       Device pointer to second linear weights, shape [intermediate_size, dim].
    /// @param b2       Device pointer to second bias, shape [dim]. May be nullptr.
    /// @param output   Device pointer to output buffer, shape [seq_len, dim].
    /// @param seq_len  Number of tokens (must be <= max_seq_len).
    /// @param stream   CUDA stream for async execution.
    void forward(const float* input,
                 const float* w1, const float* b1,
                 const float* w2, const float* b2,
                 float* output, int seq_len, cudaStream_t stream = nullptr);

    const FFNConfig& config() const { return config_; }
    int max_seq_len() const { return max_seq_len_; }

    /// Total GPU memory used by internal temporary buffers (in bytes).
    size_t temp_buffer_bytes() const;

private:
    FFNConfig config_;
    int max_seq_len_;

    // Pre-allocated GPU temporaries.
    float* hidden_buf_ = nullptr;  // [max_seq_len, intermediate_size]

    void free_buffers();
};

// ── Bias-Add Kernel ─────────────────────────────────────────
// Adds a row-broadcast bias to a 2D matrix:
//   output[row, col] = input[row, col] + bias[col]
//
// input/output: [rows x cols], row-major
// bias:         [cols]
// Grid: ceil(rows * cols / 256) blocks, 256 threads per block.
void bias_add(const float* input, const float* bias, float* output,
              int rows, int cols, cudaStream_t stream = 0);

} // namespace cuinfer
