#pragma once

// GGUF Binary Format Parser
//
// Parses files in the GGUF format (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md),
// the de facto standard for distributing quantized LLM weights. The parser memory-maps the file
// for zero-copy access to tensor data, which avoids duplicating multi-GB weight files in RAM.
//
// GGUF file layout:
//   [Header]              magic (4B) + version (4B) + tensor_count (8B) + metadata_kv_count (8B)
//   [Metadata KV pairs]   variable-length key-value entries
//   [Tensor Info]          name + shape + type + offset for each tensor
//   [Padding]              aligned to 32-byte boundary
//   [Tensor Data]          raw weight data, each tensor at its declared offset

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cuinfer {

// ── GGUF Constants ──────────────────────────────────────────

constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION_3 = 3;
constexpr size_t   GGUF_DEFAULT_ALIGNMENT = 32;

// ── GGUF Value Types ────────────────────────────────────────
// Type tags for metadata values in the key-value section.

enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

/// Returns a human-readable name for a GGUF value type.
const char* gguf_value_type_name(GGUFValueType type);

// ── GGML Tensor Types ───────────────────────────────────────
// Data type tags for tensor weight storage. Quantized types pack
// multiple elements into fixed-size blocks (e.g., Q4_0 stores 32
// values in 18 bytes: 1 FP16 scale + 16 bytes of 4-bit values).

enum class GGMLType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    // 4, 5 are deprecated/unused
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    // K-quant types (more complex block structures)
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
};

/// Returns a human-readable name for a GGML type.
const char* ggml_type_name(GGMLType type);

/// Returns the byte size of one block for the given GGML type.
/// For non-quantized types (F32, F16) the "block" is one element.
size_t ggml_type_size(GGMLType type);

/// Returns the number of elements per block for the given GGML type.
/// Non-quantized types have block_size = 1.
size_t ggml_block_size(GGMLType type);

// ── GGUFValue ───────────────────────────────────────────────
// A tagged union for metadata values. Uses std::variant for type-safe
// storage of the 13 GGUF value types (scalars, string, and arrays).

struct GGUFValue {
    GGUFValueType type;  ///< Tag identifying which variant member is active.

    /// The actual payload. Array elements are stored recursively.
    using ValueVariant = std::variant<
        uint8_t,                    // UINT8
        int8_t,                     // INT8
        uint16_t,                   // UINT16
        int16_t,                    // INT16
        uint32_t,                   // UINT32
        int32_t,                    // INT32
        float,                      // FLOAT32
        bool,                       // BOOL
        std::string,                // STRING
        std::vector<GGUFValue>,     // ARRAY
        uint64_t,                   // UINT64
        int64_t,                    // INT64
        double                      // FLOAT64
    >;

    ValueVariant value;

    /// Convenience typed accessors. Each throws std::runtime_error if the
    /// stored value type does not match the requested type.
    uint8_t     as_uint8()   const;
    int8_t      as_int8()    const;
    uint16_t    as_uint16()  const;
    int16_t     as_int16()   const;
    uint32_t    as_uint32()  const;
    int32_t     as_int32()   const;
    float       as_float32() const;
    bool        as_bool()    const;
    const std::string& as_string() const;
    const std::vector<GGUFValue>& as_array() const;
    uint64_t    as_uint64()  const;
    int64_t     as_int64()   const;
    double      as_float64() const;
};

// ── GGUFTensorInfo ──────────────────────────────────────────
// Descriptor for a single tensor in the file. Does not hold the data itself;
// the data is accessed via the memory-mapped region.

struct GGUFTensorInfo {
    std::string name;                 ///< Tensor name, e.g., "blk.0.attn_q.weight"
    std::vector<uint64_t> dimensions; ///< Shape, e.g. {4096, 4096} for a weight matrix
    GGMLType type;                    ///< Data type (F32, F16, Q8_0, Q4_K, etc.)
    uint64_t offset;                  ///< Offset within the data section (as stored in file)
    uint64_t data_offset;             ///< Absolute byte offset from the start of the file

    /// Calculates the total byte size of this tensor's data, accounting for
    /// quantized block packing. For quantized types, the total element count
    /// must be divisible by the block size.
    size_t byte_size() const;

    /// Total number of elements (product of all dimensions).
    uint64_t element_count() const;
};

// ── GGUFParser ──────────────────────────────────────────────
// Main entry point: parses a GGUF file and provides access to metadata
// and tensor data via memory mapping.
//
// Usage:
//   GGUFParser parser("model.gguf");
//   parser.print_info();
//   auto arch = parser.architecture();
//   const void* data = parser.tensor_data("token_embd.weight");

class GGUFParser {
public:
    /// Opens and parses a GGUF file. Throws std::runtime_error if the file
    /// cannot be opened, has an invalid magic number, or is malformed.
    explicit GGUFParser(const std::string& file_path);
    ~GGUFParser();

    // Non-copyable (owns memory-mapped region)
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

    // Movable
    GGUFParser(GGUFParser&&) noexcept;
    GGUFParser& operator=(GGUFParser&&) noexcept;

    // ── Header Info ─────────────────────────────────────────

    /// GGUF format version (typically 3).
    uint32_t version() const;

    /// Number of tensors stored in the file.
    uint64_t tensor_count() const;

    /// Number of metadata key-value pairs.
    uint64_t metadata_count() const;

    // ── Metadata Access ─────────────────────────────────────

    /// Returns the full metadata map.
    const std::unordered_map<std::string, GGUFValue>& metadata() const;

    /// Returns true if the given metadata key exists.
    bool has_metadata(const std::string& key) const;

    /// Typed getters. Each throws std::runtime_error if the key is missing
    /// or the stored value type does not match.
    std::string get_string(const std::string& key) const;
    uint32_t    get_uint32(const std::string& key) const;
    int32_t     get_int32(const std::string& key) const;
    uint64_t    get_uint64(const std::string& key) const;
    float       get_float32(const std::string& key) const;

    // ── Model Architecture Helpers ──────────────────────────
    // These read well-known GGUF metadata keys used by LLM model files.
    // The architecture name (e.g., "llama") is used as a prefix for
    // architecture-specific keys like "{arch}.context_length".

    std::string architecture() const;         // "general.architecture"
    uint32_t    context_length() const;        // "{arch}.context_length"
    uint32_t    embedding_length() const;      // "{arch}.embedding_length"
    uint32_t    block_count() const;           // "{arch}.block_count"
    uint32_t    head_count() const;            // "{arch}.attention.head_count"
    uint32_t    head_count_kv() const;         // "{arch}.attention.head_count_kv"
    uint32_t    feed_forward_length() const;   // "{arch}.feed_forward_length"
    float       rope_freq_base() const;        // "{arch}.rope.freq_base" (default 10000.0)

    /// RMSNorm epsilon value for layer normalization. Returns the default 1e-5
    /// if the key is absent from metadata.
    float       rms_norm_eps() const;          // "{arch}.attention.layer_norm_rms_epsilon"

    /// Number of dimensions used for RoPE positional encoding (typically = head_dim).
    uint32_t    rope_dimension_count() const;  // "{arch}.rope.dimension_count"

    /// Vocabulary size. Reads "{arch}.vocab_size" if present, otherwise falls
    /// back to the length of the "tokenizer.ggml.tokens" array.
    uint32_t    vocab_size() const;

    /// Returns true if the LM head reuses the token embedding weights.
    /// Determined by checking whether "output.weight" tensor is absent.
    bool        tie_word_embeddings() const;

    // ── Tensor Access ───────────────────────────────────────
    const std::vector<GGUFTensorInfo>& tensors() const;

    /// Finds a tensor by name, returns nullptr if not found.
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    /// Returns a raw pointer into the memory-mapped file at the tensor's data offset.
    /// The pointer remains valid for the lifetime of this GGUFParser instance.
    const void* tensor_data(const GGUFTensorInfo& info) const;
    const void* tensor_data(const std::string& name) const;

    // ── Debug ───────────────────────────────────────────────
    /// Prints a summary of the file to stdout: header, metadata keys, tensor list.
    void print_info() const;

private:
    // Pimpl pattern keeps memory-mapping details out of the public header
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuinfer
