#pragma once

// TransformerModel — Orchestrates the full transformer forward pass.
//
// Loads a GGUF model file, extracts weights and tokenizer, and provides
// forward() for computing logits and generate() for autoregressive text
// generation. This is the top-level class that ties together all the
// individual building blocks: Embedding, AttentionBlock, GatedFFN,
// RMSNorm, LMHead, KVCache, and Tokenizer.
//
// Usage:
//   TransformerModel model("smollm2-135m.gguf");
//   std::string output = model.generate("Once upon a time", 50);

#include "cuinfer/attention.h"
#include "cuinfer/embedding.h"
#include "cuinfer/ffn.h"
#include "cuinfer/gguf_parser.h"
#include "cuinfer/kv_cache.h"
#include "cuinfer/tensor.h"
#include "cuinfer/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace cuinfer {

// ── Model Configuration ──────────────────────────────────────
// Extracted from GGUF metadata during model loading.

/// Weight storage format for the model.
/// Determines which matmul kernel is used for linear projections.
enum class WeightFormat {
    FP32,   // Standard FP32 weights (Phase 2 default)
    FP16,   // Native FP16 weights (Phase 3 — halves weight VRAM)
    Q8_0,   // GGML 8-bit quantization
    Q4_0,   // GGML 4-bit quantization
    Q5_0,   // GGML 5-bit quantization
    Q4_K,   // GGML 4-bit K-quant (may also contain Q6_K tensors)
    Q6_K,   // GGML 6-bit K-quant
};

/// Returns a human-readable name for a weight format (e.g., "Q4_K").
inline const char* weight_format_name(WeightFormat fmt) {
    switch (fmt) {
        case WeightFormat::FP32: return "FP32";
        case WeightFormat::FP16: return "FP16";
        case WeightFormat::Q8_0: return "Q8_0";
        case WeightFormat::Q4_0: return "Q4_0";
        case WeightFormat::Q5_0: return "Q5_0";
        case WeightFormat::Q4_K: return "Q4_K";
        case WeightFormat::Q6_K: return "Q6_K";
    }
    return "UNKNOWN";
}

/// Holds all hyperparameters extracted from GGUF metadata that define
/// the transformer model architecture. Populated during model loading.
struct ModelConfig {
    std::string architecture;    ///< Architecture family, e.g., "llama"
    int n_layers;                ///< Number of transformer blocks
    int dim;                     ///< Embedding / hidden dimension
    int n_heads;                 ///< Number of query attention heads
    int n_kv_heads;              ///< Number of KV heads (< n_heads for GQA)
    int head_dim;                ///< Dimension per head (= dim / n_heads)
    int intermediate_size;       ///< FFN hidden dimension
    int vocab_size;              ///< Vocabulary size
    int max_seq_len;             ///< Maximum context length
    float rms_norm_eps;          ///< RMSNorm epsilon
    float rope_theta;            ///< RoPE frequency base
    bool tie_embeddings;         ///< True if LM head shares embedding weights
    WeightFormat weight_format = WeightFormat::FP32;  ///< Detected from GGUF
};

// ── TransformerModel ─────────────────────────────────────────

class TransformerModel {
public:
    /// Load a model from a GGUF file.
    ///
    /// @param gguf_path   Path to the GGUF model file.
    /// @param max_seq_len Maximum sequence length to support. Clamped to the
    ///                    model's context_length from GGUF metadata. Lower
    ///                    values save VRAM (smaller KV-cache).
    explicit TransformerModel(const std::string& gguf_path,
                              int max_seq_len = 2048);
    ~TransformerModel();

    // Non-copyable (owns large GPU allocations)
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    // Movable
    TransformerModel(TransformerModel&&) noexcept;
    TransformerModel& operator=(TransformerModel&&) noexcept;

    // ── Forward Pass ────────────────────────────────────────

    /// Forward pass for a sequence of token IDs.
    /// Returns logits tensor [seq_len, vocab_size] on GPU.
    /// Updates the KV-cache with the new positions.
    ///
    /// Handles both prefill (seq_len > 1) and single-token decode (seq_len = 1).
    Tensor forward(const std::vector<int32_t>& token_ids);

    // ── Text Generation ─────────────────────────────────────

    /// Generate text from a prompt string.
    ///
    /// @param prompt       Input text to continue from.
    /// @param max_tokens   Maximum number of new tokens to generate.
    /// @param temperature  Sampling temperature (0 = greedy argmax).
    /// @param top_k        Top-k filtering (0 = greedy argmax).
    /// @return             Generated text (decoded from all tokens including prompt).
    std::string generate(const std::string& prompt,
                         int max_tokens = 100,
                         float temperature = 0.0f,
                         int top_k = 0);

    // ── State Management ────────────────────────────────────

    /// Reset the KV-cache for a new conversation/prompt.
    void reset();

    /// Current sequence position (number of tokens in KV-cache).
    int current_position() const;

    // ── Info / Debug ────────────────────────────────────────

    /// Returns the model configuration (architecture, dims, etc.).
    const ModelConfig& config() const { return config_; }

    /// Returns a reference to the model's tokenizer.
    const Tokenizer& tokenizer() const { return *tokenizer_; }

    /// Print VRAM breakdown: weights, KV-cache, buffers.
    void print_memory_report() const;

private:
    // ── Weight Loading Helpers ──────────────────────────────

    /// Load a 2D weight, transposing from GGUF [out, in] to [in, out]
    /// for use with matmul_tiled (which expects B as [K, N]).
    /// Used for FP32/FP16 weights only.
    Tensor load_weight_transposed(const GGUFParser& parser,
                                  const std::string& name);

    /// Load a 1D weight (norm weights) — always dequantized to FP32.
    /// Norm weights are small and used in FP32 elementwise ops.
    Tensor load_weight_1d(const GGUFParser& parser,
                          const std::string& name);

    /// Load a 2D weight as-is (for embedding table, used via row lookup).
    /// Always dequantized to FP32 for row-lookup access pattern.
    Tensor load_weight_2d(const GGUFParser& parser,
                          const std::string& name);

    /// Load a 2D weight in its native quantized format (no transposition).
    /// Uploads raw quantized bytes directly to GPU. The quantized matmul
    /// kernels handle the [N,K] B-transposed layout natively.
    /// @param out_format  Set to the tensor's quantization format.
    Tensor load_weight_quantized(const GGUFParser& parser,
                                 const std::string& name,
                                 WeightFormat& out_format);

    /// Detect the weight format from the first layer's attention Q weight.
    WeightFormat detect_weight_format(const GGUFParser& parser);

    /// Build Tokenizer from GGUF metadata arrays.
    std::unique_ptr<Tokenizer> build_tokenizer(const GGUFParser& parser);

    // ── Model Configuration ─────────────────────────────────
    ModelConfig config_;

    // ── Components ──────────────────────────────────────────
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<KVCache> kv_cache_;
    Embedding embedding_;
    LMHead lm_head_;
    AttentionBlock attention_;
    GatedFFN ffn_;

    // ── Weights ─────────────────────────────────────────────
    // Global
    Tensor token_embd_weight_;   // [vocab_size, dim]
    Tensor output_norm_weight_;  // [dim]
    Tensor lm_head_weight_;      // [vocab_size, dim] or empty if tied

    // Per-layer (indexed by layer number)
    std::vector<Tensor> attn_norm_weight_;   // [dim] each (always FP32)
    std::vector<Tensor> attn_q_weight_;      // [dim, dim] each (transposed or quantized)
    std::vector<Tensor> attn_k_weight_;      // [dim, kv_dim] each
    std::vector<Tensor> attn_v_weight_;      // [dim, kv_dim] each
    std::vector<Tensor> attn_o_weight_;      // [dim, dim] each
    std::vector<Tensor> ffn_norm_weight_;     // [dim] each (always FP32)
    std::vector<Tensor> ffn_gate_weight_;     // [dim, intermediate] each
    std::vector<Tensor> ffn_up_weight_;       // [dim, intermediate] each
    std::vector<Tensor> ffn_down_weight_;     // [intermediate, dim] each

    // Per-tensor weight format (for Q4_K_M files where some tensors are Q6_K)
    std::vector<WeightFormat> attn_q_format_;
    std::vector<WeightFormat> attn_k_format_;
    std::vector<WeightFormat> attn_v_format_;
    std::vector<WeightFormat> attn_o_format_;
    std::vector<WeightFormat> ffn_gate_format_;
    std::vector<WeightFormat> ffn_up_format_;
    std::vector<WeightFormat> ffn_down_format_;
    WeightFormat lm_head_format_ = WeightFormat::FP32;

    // ── Activation Buffers (pre-allocated, reused per forward) ──
    Tensor normed_buf_;          // [max_seq_len, dim]
    Tensor ffn_output_buf_;      // [max_seq_len, dim]
    int32_t* token_ids_gpu_ = nullptr;  // [max_seq_len] on device
};

} // namespace cuinfer
