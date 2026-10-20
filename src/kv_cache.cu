#include "cuinfer/kv_cache.h"
#include "cuinfer/cuda_utils.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace cuinfer {

// ── Construction / Destruction ──────────────────────────────

KVCache::KVCache(const KVCacheConfig& config)
    : config_(config) {
    // Validate configuration -- catch misuse early rather than silently
    // allocating zero-byte buffers.
    if (config_.n_layers <= 0) {
        throw std::invalid_argument(
            "KVCache: n_layers must be > 0, got " +
            std::to_string(config_.n_layers));
    }
    if (config_.max_seq_len <= 0) {
        throw std::invalid_argument(
            "KVCache: max_seq_len must be > 0, got " +
            std::to_string(config_.max_seq_len));
    }
    if (config_.n_kv_heads <= 0) {
        throw std::invalid_argument(
            "KVCache: n_kv_heads must be > 0, got " +
            std::to_string(config_.n_kv_heads));
    }
    if (config_.head_dim <= 0) {
        throw std::invalid_argument(
            "KVCache: head_dim must be > 0, got " +
            std::to_string(config_.head_dim));
    }

    // Allocate two large contiguous GPU buffers -- one for all layers' K
    // caches and one for all layers' V caches. A single allocation per
    // cache avoids VRAM fragmentation and simplifies pointer arithmetic.
    //
    // Each buffer: n_layers * max_seq_len * n_kv_heads * head_dim floats
    const size_t alloc_bytes = static_cast<size_t>(config_.n_layers) *
                               config_.layer_k_bytes();

    CUINFER_CHECK_CUDA(cudaMalloc(&k_data_, alloc_bytes));
    CUINFER_CHECK_CUDA(cudaMalloc(&v_data_, alloc_bytes));

    // Zero-initialize so that unused positions contain deterministic values.
    // This is not strictly required for correctness (the attention mask
    // should ignore positions beyond current_length_), but it helps with
    // debugging and avoids exposing stale GPU memory.
    CUINFER_CHECK_CUDA(cudaMemset(k_data_, 0, alloc_bytes));
    CUINFER_CHECK_CUDA(cudaMemset(v_data_, 0, alloc_bytes));
}

KVCache::~KVCache() {
    // cudaFree is safe even if the pointer is null or the CUDA context
    // has already been torn down. We intentionally ignore errors here
    // to avoid throwing in a destructor.
    if (k_data_ != nullptr) {
        cudaFree(k_data_);
        k_data_ = nullptr;
    }
    if (v_data_ != nullptr) {
        cudaFree(v_data_);
        v_data_ = nullptr;
    }
}

// ── Move semantics ──────────────────────────────────────────

KVCache::KVCache(KVCache&& other) noexcept
    : config_(other.config_),
      current_length_(other.current_length_),
      k_data_(other.k_data_),
      v_data_(other.v_data_) {
    // Leave the moved-from object in a safe empty state.
    other.k_data_ = nullptr;
    other.v_data_ = nullptr;
    other.current_length_ = 0;
    other.config_ = {};
}

KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        // Free our current allocations.
        if (k_data_ != nullptr) cudaFree(k_data_);
        if (v_data_ != nullptr) cudaFree(v_data_);

        config_ = other.config_;
        current_length_ = other.current_length_;
        k_data_ = other.k_data_;
        v_data_ = other.v_data_;

        other.k_data_ = nullptr;
        other.v_data_ = nullptr;
        other.current_length_ = 0;
        other.config_ = {};
    }
    return *this;
}

// ── Private helpers ─────────────────────────────────────────

size_t KVCache::layer_offset_bytes(int32_t layer) const {
    // Each layer occupies (max_seq_len * n_kv_heads * head_dim) floats.
    return static_cast<size_t>(layer) * config_.layer_k_bytes();
}

size_t KVCache::position_offset_bytes(int32_t position) const {
    // Within a layer, each position occupies (n_kv_heads * head_dim) floats.
    return static_cast<size_t>(position) *
           static_cast<size_t>(config_.n_kv_heads) *
           static_cast<size_t>(config_.head_dim) *
           sizeof(float);
}

void KVCache::check_layer(int32_t layer) const {
    if (layer < 0 || layer >= config_.n_layers) {
        throw std::out_of_range(
            "KVCache: layer index " + std::to_string(layer) +
            " out of range [0, " + std::to_string(config_.n_layers) + ")");
    }
}

// ── Per-layer accessors ─────────────────────────────────────

float* KVCache::k_cache(int32_t layer) {
    check_layer(layer);
    // Pointer arithmetic: k_data_ points to the start of the contiguous
    // K buffer. Advance by layer * (max_seq_len * n_kv_heads * head_dim)
    // elements to reach this layer's region.
    return reinterpret_cast<float*>(
        reinterpret_cast<char*>(k_data_) + layer_offset_bytes(layer));
}

float* KVCache::v_cache(int32_t layer) {
    check_layer(layer);
    return reinterpret_cast<float*>(
        reinterpret_cast<char*>(v_data_) + layer_offset_bytes(layer));
}

const float* KVCache::k_cache(int32_t layer) const {
    check_layer(layer);
    return reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(k_data_) + layer_offset_bytes(layer));
}

const float* KVCache::v_cache(int32_t layer) const {
    check_layer(layer);
    return reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(v_data_) + layer_offset_bytes(layer));
}

// ── Append ──────────────────────────────────────────────────

void KVCache::append(int32_t layer, const float* k_new, const float* v_new,
                     int32_t n_new_tokens, cudaStream_t stream) {
    check_layer(layer);

    if (n_new_tokens <= 0) {
        throw std::invalid_argument(
            "KVCache::append: n_new_tokens must be > 0, got " +
            std::to_string(n_new_tokens));
    }

    // Overflow check: would this append push past max_seq_len?
    if (current_length_ + n_new_tokens > config_.max_seq_len) {
        throw std::overflow_error(
            "KVCache::append: appending " + std::to_string(n_new_tokens) +
            " tokens at position " + std::to_string(current_length_) +
            " would exceed max_seq_len=" +
            std::to_string(config_.max_seq_len));
    }

    // Compute the destination address within this layer's cache.
    // Destination = layer_start + current_length_ * n_kv_heads * head_dim floats
    const size_t dst_offset = layer_offset_bytes(layer) +
                              position_offset_bytes(current_length_);
    float* k_dst = reinterpret_cast<float*>(
        reinterpret_cast<char*>(k_data_) + dst_offset);
    float* v_dst = reinterpret_cast<float*>(
        reinterpret_cast<char*>(v_data_) + dst_offset);

    // Number of bytes to copy: n_new_tokens * n_kv_heads * head_dim * sizeof(float)
    const size_t copy_bytes =
        static_cast<size_t>(n_new_tokens) *
        static_cast<size_t>(config_.n_kv_heads) *
        static_cast<size_t>(config_.head_dim) *
        sizeof(float);

    // Device-to-device async copy. Using cudaMemcpyAsync allows the copy
    // to overlap with host-side work when a non-default stream is used.
    CUINFER_CHECK_CUDA(
        cudaMemcpyAsync(k_dst, k_new, copy_bytes,
                        cudaMemcpyDeviceToDevice, stream));
    CUINFER_CHECK_CUDA(
        cudaMemcpyAsync(v_dst, v_new, copy_bytes,
                        cudaMemcpyDeviceToDevice, stream));

    // The position counter tracks the global sequence position shared by
    // all layers. During generation, the caller processes all layers for
    // the same token(s) in order: append(layer=0), append(layer=1), ...
    // We increment current_length_ only when the LAST layer is appended,
    // so intermediate layers see the correct write position.
    //
    // The GPU copies may still be in flight when this returns (if a
    // non-default stream was used). The caller must synchronize the stream
    // before issuing attention kernels that read from the updated cache.
    if (layer == config_.n_layers - 1) {
        current_length_ += n_new_tokens;
    }
}

// ── Valid-range accessors ───────────────────────────────────

float* KVCache::k_cache_valid(int32_t layer) {
    // Valid data starts at position 0 of this layer's cache.
    // The caller reads current_length() * n_kv_heads * head_dim elements.
    return k_cache(layer);
}

float* KVCache::v_cache_valid(int32_t layer) {
    return v_cache(layer);
}

const float* KVCache::k_cache_valid(int32_t layer) const {
    return k_cache(layer);
}

const float* KVCache::v_cache_valid(int32_t layer) const {
    return v_cache(layer);
}

// ── State management ────────────────────────────────────────

void KVCache::clear() {
    // Just reset the write head. The GPU memory is retained for reuse.
    // Old data will be overwritten by future appends. We do not zero the
    // memory here to avoid an unnecessary GPU operation -- the attention
    // kernel should use current_length() to bound its reads.
    current_length_ = 0;
}

// ── Memory reporting ────────────────────────────────────────

size_t KVCache::total_bytes() const {
    return config_.total_bytes();
}

size_t KVCache::used_bytes() const {
    // Proportion of the cache that holds valid data, across all layers.
    // Each layer has current_length_ positions filled out of max_seq_len.
    if (config_.max_seq_len == 0) return 0;
    const size_t per_layer_used =
        static_cast<size_t>(current_length_) *
        static_cast<size_t>(config_.n_kv_heads) *
        static_cast<size_t>(config_.head_dim) *
        sizeof(float);
    // K + V for each layer
    return static_cast<size_t>(config_.n_layers) * 2 * per_layer_used;
}

void KVCache::print_memory_report() const {
    const double total_mb = total_bytes() / (1024.0 * 1024.0);
    const double used_mb  = used_bytes()  / (1024.0 * 1024.0);
    const double pct = (total_bytes() > 0)
        ? (100.0 * used_bytes() / total_bytes())
        : 0.0;

    printf("┌─────────────────────────────────────────┐\n");
    printf("│ KV-Cache Memory Report                  │\n");
    printf("├─────────────────────────────────────────┤\n");
    printf("│ Layers:       %-4d                       │\n", config_.n_layers);
    printf("│ Max seq len:  %-5d                      │\n", config_.max_seq_len);
    printf("│ KV heads:     %-3d                        │\n", config_.n_kv_heads);
    printf("│ Head dim:     %-3d                        │\n", config_.head_dim);
    printf("│ Cur length:   %-5d / %-5d              │\n",
           current_length_, config_.max_seq_len);
    printf("│ Allocated:    %-7.2f MB                 │\n", total_mb);
    printf("│ Used:         %-7.2f MB (%.1f%%)         │\n", used_mb, pct);
    printf("└─────────────────────────────────────────┘\n");
}

} // namespace cuinfer
