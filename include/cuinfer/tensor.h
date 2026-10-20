#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuinfer {

// ── Data Type ───────────────────────────────────────────────

enum class DType {
    FP32,   // 4 bytes per element
    FP16,   // 2 bytes per element
    INT8,   // 1 byte per element
    Q8_0,   // 34 bytes per block of 32 elements (GGML Q8_0)
    Q4_0,   // 18 bytes per block of 32 elements (GGML Q4_0)
    Q5_0,   // 22 bytes per block of 32 elements (GGML Q5_0)
    Q4_K,   // 144 bytes per block of 256 elements (GGML Q4_K)
    Q6_K,   // 210 bytes per block of 256 elements (GGML Q6_K)
};

/// Returns the byte size of a single element for non-quantized dtypes.
/// For quantized types, use dtype_block_bytes() and dtype_block_elements() instead.
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::INT8: return 1;
        // Quantized types don't have a per-element size; callers should use
        // dtype_block_bytes() / dtype_block_elements() for these.
        case DType::Q8_0: return 0;
        case DType::Q4_0: return 0;
        case DType::Q5_0: return 0;
        case DType::Q4_K: return 0;
        case DType::Q6_K: return 0;
    }
    return 0; // unreachable — silences compiler warning
}

/// Returns true if this dtype uses block-quantized storage.
inline bool dtype_is_quantized(DType dtype) {
    return dtype == DType::Q8_0 || dtype == DType::Q4_0 ||
           dtype == DType::Q5_0 || dtype == DType::Q4_K ||
           dtype == DType::Q6_K;
}

/// Returns the number of elements per quantization block.
/// For non-quantized types, returns 1 (each "block" is one element).
inline int dtype_block_elements(DType dtype) {
    switch (dtype) {
        case DType::FP32: return 1;
        case DType::FP16: return 1;
        case DType::INT8: return 1;
        case DType::Q8_0: return 32;
        case DType::Q4_0: return 32;
        case DType::Q5_0: return 32;
        case DType::Q4_K: return 256;
        case DType::Q6_K: return 256;
    }
    return 1;
}

/// Returns the byte size of one quantization block.
/// For non-quantized types, this equals the per-element size.
inline size_t dtype_block_bytes(DType dtype) {
    switch (dtype) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::INT8: return 1;
        case DType::Q8_0: return 34;
        case DType::Q4_0: return 18;
        case DType::Q5_0: return 22;
        case DType::Q4_K: return 144;
        case DType::Q6_K: return 210;
    }
    return 0;
}

/// Returns a human-readable name for a dtype.
inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::FP32: return "FP32";
        case DType::FP16: return "FP16";
        case DType::INT8: return "INT8";
        case DType::Q8_0: return "Q8_0";
        case DType::Q4_0: return "Q4_0";
        case DType::Q5_0: return "Q5_0";
        case DType::Q4_K: return "Q4_K";
        case DType::Q6_K: return "Q6_K";
    }
    return "UNKNOWN";
}

// ── Tensor ──────────────────────────────────────────────────

/// Device-resident tensor with RAII memory management.
///
/// Owns GPU memory when created via constructors/factory methods.
/// Views (from view() / reshape()) are non-owning and share the
/// underlying buffer — the caller must ensure the source outlives
/// any views.
class Tensor {
public:
    // -- Constructors / Destructor ------------------------------------------

    /// Constructs an empty (null) tensor.
    Tensor();

    /// Allocates device memory for the given shape and dtype.
    /// Shape vector can have 1-4 elements; dimensions are right-aligned
    /// (missing leading dims are filled with 1).
    explicit Tensor(std::vector<int64_t> shape, DType dtype = DType::FP32);

    ~Tensor();

    // Move semantics — transfers ownership.
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Copying is prohibited to prevent accidental GPU allocations.
    // Use clone() for explicit deep copies.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // -- Static Factory Methods ---------------------------------------------

    /// Allocates device memory and copies host data to the GPU.
    static Tensor from_host(const void* host_data,
                            std::vector<int64_t> shape,
                            DType dtype = DType::FP32);

    /// Allocates device memory initialized to zero.
    static Tensor zeros(std::vector<int64_t> shape,
                        DType dtype = DType::FP32);

    /// Allocates device memory without initialization (alias for constructor).
    static Tensor empty(std::vector<int64_t> shape,
                        DType dtype = DType::FP32);

    // -- Core Methods -------------------------------------------------------

    /// Returns a non-owning view that slices dimension `dim` from `start`
    /// (inclusive) to `end` (exclusive). The returned tensor shares the
    /// underlying data pointer (offset appropriately).
    Tensor view(int64_t dim, int64_t start, int64_t end) const;

    /// Returns a non-owning view with a different logical shape.
    /// Total element count must match. Tensor must be contiguous.
    Tensor reshape(std::vector<int64_t> new_shape) const;

    /// Deep copy: allocates new device memory and copies data device-to-device.
    Tensor clone() const;

    /// Copies `bytes` from host memory into the tensor's device buffer.
    void copy_from_host(const void* host_data, size_t bytes);

    /// Copies `bytes` from the tensor's device buffer to host memory.
    void copy_to_host(void* host_data, size_t bytes) const;

    /// Total number of elements.
    int64_t numel() const;

    /// Total size in bytes (numel * element size).
    size_t size_bytes() const;

    /// Number of meaningful (non-trivial) dimensions.
    /// Trailing-1 dimensions from padding are ignored, minimum 1.
    int ndim() const;

    /// True if the strides correspond to a contiguous row-major layout.
    bool is_contiguous() const;

    // -- Accessors ----------------------------------------------------------

    /// Returns the raw device pointer to the tensor data.
    void* data() { return data_; }
    const void* data() const { return data_; }

    /// Returns the device pointer cast to the specified type.
    template <typename T>
    T* data_as() { return static_cast<T*>(data_); }

    template <typename T>
    const T* data_as() const { return static_cast<const T*>(data_); }

    /// Returns the full 4D shape array (right-aligned, padded with leading 1s).
    const std::array<int64_t, 4>& shape() const { return shape_; }

    /// Returns the full 4D stride array.
    const std::array<int64_t, 4>& stride() const { return stride_; }

    /// Access a single dimension of the shape (0-indexed). Bounds-checked.
    int64_t shape(int dim) const;

    /// Access a single dimension of the stride (0-indexed). Bounds-checked.
    int64_t stride(int dim) const;

    /// Returns the data type of this tensor's elements.
    DType dtype() const { return dtype_; }

    /// Returns true if this tensor owns its GPU memory (false for views).
    bool is_owning() const { return owning_; }

    // -- Debug --------------------------------------------------------------

    /// Prints the first `max_elems` elements (FP32 only) to stdout.
    /// Useful for quick inspection during development.
    void print(const std::string& name = "", int max_elems = 10) const;

private:
    // Private constructor for building views (non-owning tensors).
    Tensor(void* data,
           std::array<int64_t, 4> shape,
           std::array<int64_t, 4> stride,
           DType dtype);

    /// Converts a user-provided shape (1-4 dims) into a right-aligned
    /// 4D shape array padded with leading 1s, and computes row-major strides.
    static void compute_shape_and_stride(const std::vector<int64_t>& shape,
                                         std::array<int64_t, 4>& out_shape,
                                         std::array<int64_t, 4>& out_stride);

    void* data_ = nullptr;
    std::array<int64_t, 4> shape_ = {1, 1, 1, 1};
    std::array<int64_t, 4> stride_ = {1, 1, 1, 1};
    DType dtype_ = DType::FP32;
    bool owning_ = false;
};

} // namespace cuinfer
