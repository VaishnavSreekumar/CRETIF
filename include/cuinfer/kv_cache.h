#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

namespace cuinfer {

// ── KV-Cache Configuration ──────────────────────────────────
//
// Holds the hyperparameters that determine KV-cache sizing.
// These come from the model config (e.g. GGUF metadata):
//   - n_layers:     number of transformer layers
//   - max_seq_len:  maximum sequence length the cache supports
//   - n_kv_heads:   number of key/value heads (may be < n_query_heads for GQA)
//   - head_dim:     dimension of each attention head
//
// For Grouped-Query Attention (GQA), n_kv_heads < n_query_heads.
// The cache only stores n_kv_heads entries per position; during
// attention each group of (n_query_heads / n_kv_heads) query heads
// shares the same K/V head.

struct KVCacheConfig {
    int32_t n_layers    = 0;  ///< Number of transformer layers
    int32_t max_seq_len = 0;  ///< Maximum sequence length the cache supports
    int32_t n_kv_heads  = 0;  ///< Number of key/value heads (may be < n_query_heads for GQA)
    int32_t head_dim    = 0;  ///< Dimension of each attention head

    /// Number of float elements in one layer's K (or V) cache.
    /// Layout: [max_seq_len, n_kv_heads, head_dim]
    size_t layer_elements() const {
        return static_cast<size_t>(max_seq_len) *
               static_cast<size_t>(n_kv_heads) *
               static_cast<size_t>(head_dim);
    }

    /// Bytes for one layer's K cache (FP32).
    size_t layer_k_bytes() const { return layer_elements() * sizeof(float); }

    /// Bytes for one layer's V cache (FP32).
    size_t layer_v_bytes() const { return layer_elements() * sizeof(float); }

    /// Total bytes for all layers' K and V caches combined.
    size_t total_bytes() const {
        return static_cast<size_t>(n_layers) *
               (layer_k_bytes() + layer_v_bytes());
    }
};

// ── KV-Cache ────────────────────────────────────────────────
//
// Pre-allocated GPU buffer that stores Key and Value projections
// from all transformer layers across the generated sequence.
//
// Memory layout (two separate allocations, one for all K, one for all V):
//
//   K data: [n_layers, max_seq_len, n_kv_heads, head_dim]  (contiguous FP32)
//   V data: [n_layers, max_seq_len, n_kv_heads, head_dim]  (contiguous FP32)
//
// Within each allocation, layers are laid out contiguously:
//
//   k_data_ + layer * (max_seq_len * n_kv_heads * head_dim)
//            ^--- start of layer's K cache
//
// Positions 0..current_length_-1 hold valid cached data.
// The cache is shared across all layers: a single position counter
// tracks how many tokens have been processed (all layers process the
// same tokens in lockstep during generation).
//
// Usage pattern:
//   1. Prefill:  append(layer, k, v, prompt_len, stream) for each layer
//   2. Decode:   append(layer, k, v, 1, stream)          for each layer
//   3. Reset:    clear() to start a new sequence (no reallocation)

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& config);
    ~KVCache();

    // No copy -- the cache owns large GPU allocations.
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    // Move semantics transfer ownership of GPU memory.
    KVCache(KVCache&& other) noexcept;
    KVCache& operator=(KVCache&& other) noexcept;

    // ── Per-layer accessors ─────────────────────────────────

    /// Returns a device pointer to the start of layer's K cache.
    /// The returned buffer has layout [max_seq_len, n_kv_heads, head_dim].
    float* k_cache(int32_t layer);

    /// Returns a device pointer to the start of layer's V cache.
    /// The returned buffer has layout [max_seq_len, n_kv_heads, head_dim].
    float* v_cache(int32_t layer);

    const float* k_cache(int32_t layer) const;
    const float* v_cache(int32_t layer) const;

    // ── Append (core cache operation) ───────────────────────

    /// Copies new K/V data into the cache at the current write position.
    ///
    /// @param layer        Transformer layer index [0, n_layers).
    /// @param k_new        Device pointer to new K data,
    ///                     shape [n_new_tokens, n_kv_heads, head_dim].
    /// @param v_new        Device pointer to new V data, same shape.
    /// @param n_new_tokens Number of new tokens to append (1 for decode,
    ///                     prompt_len for prefill).
    /// @param stream       CUDA stream for async execution (nullptr = default).
    ///
    /// The position counter (current_length_) is incremented only when
    /// the LAST layer (n_layers - 1) is appended. This matches the typical
    /// transformer loop: append to layer 0, 1, ..., n_layers-1 for the
    /// same token(s), and the position advances after the final layer.
    ///
    /// Throws std::overflow_error if the append would exceed max_seq_len.
    void append(int32_t layer, const float* k_new, const float* v_new,
                int32_t n_new_tokens, cudaStream_t stream = nullptr);

    // ── Valid-range accessors ───────────────────────────────

    /// Returns a device pointer to the valid (filled) portion of layer's
    /// K cache. The caller should read current_length() * n_kv_heads *
    /// head_dim elements from this pointer.
    float* k_cache_valid(int32_t layer);
    float* v_cache_valid(int32_t layer);
    const float* k_cache_valid(int32_t layer) const;
    const float* v_cache_valid(int32_t layer) const;

    // ── State queries ───────────────────────────────────────

    /// Number of tokens currently stored in the cache.
    int32_t current_length() const { return current_length_; }

    /// Resets the position counter to zero for a new generation.
    /// Does NOT free or zero GPU memory -- just resets the write head.
    /// Old data will be overwritten by subsequent appends.
    void clear();

    // ── Memory reporting ────────────────────────────────────

    /// Total bytes allocated on GPU (K + V, all layers).
    size_t total_bytes() const;

    /// Bytes of cache actually holding valid data (based on current_length).
    size_t used_bytes() const;

    /// Prints a human-readable summary of cache dimensions and VRAM usage.
    void print_memory_report() const;

    // ── Config access ───────────────────────────────────────

    const KVCacheConfig& config() const { return config_; }

private:
    KVCacheConfig config_;
    int32_t current_length_ = 0;

    // Single contiguous GPU allocation per cache (K and V separate).
    // Layout: [n_layers][max_seq_len][n_kv_heads][head_dim]
    // Total elements per allocation: n_layers * max_seq_len * n_kv_heads * head_dim
    float* k_data_ = nullptr;
    float* v_data_ = nullptr;

    /// Helper: byte offset from k_data_ (or v_data_) to the start of
    /// a given layer's cache region.
    size_t layer_offset_bytes(int32_t layer) const;

    /// Helper: byte offset from a layer's start to a specific position
    /// within that layer.
    size_t position_offset_bytes(int32_t position) const;

    /// Validates that `layer` is in [0, n_layers). Throws on failure.
    void check_layer(int32_t layer) const;
};

} // namespace cuinfer
