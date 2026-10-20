#include "cuinfer/tensor.h"
#include "cuinfer/cuda_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cuinfer {

// ── Helper: shape/stride computation ────────────────────────

void Tensor::compute_shape_and_stride(const std::vector<int64_t>& shape,
                                      std::array<int64_t, 4>& out_shape,
                                      std::array<int64_t, 4>& out_stride) {
    assert(!shape.empty() && shape.size() <= 4);

    // Right-align: prepend 1s so that the user-supplied dims occupy
    // the rightmost positions of the 4D array.
    out_shape.fill(1);
    const int offset = 4 - static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        assert(shape[i] > 0);
        out_shape[offset + i] = shape[i];
    }

    // Row-major strides: stride[i] = product of shape[i+1 .. 3].
    out_stride[3] = 1;
    for (int i = 2; i >= 0; --i) {
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
}

// ── Constructors / Destructor ───────────────────────────────

Tensor::Tensor() = default;

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : dtype_(dtype), owning_(true) {
    if (shape.empty() || shape.size() > 4) {
        throw std::invalid_argument(
            "Tensor shape must have 1-4 dimensions, got " +
            std::to_string(shape.size()));
    }

    compute_shape_and_stride(shape, shape_, stride_);

    const size_t bytes = size_bytes();
    if (bytes > 0) {
        CUINFER_CHECK_CUDA(cudaMalloc(&data_, bytes));
    }
}

Tensor::~Tensor() {
    if (owning_ && data_ != nullptr) {
        // cudaFree is safe to call even after context teardown; we
        // intentionally ignore errors in the destructor to avoid
        // throwing during stack unwinding.
        cudaFree(data_);
        data_ = nullptr;
    }
}

// ── Move semantics ──────────────────────────────────────────

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(other.shape_),
      stride_(other.stride_),
      dtype_(other.dtype_),
      owning_(other.owning_) {
    // Leave `other` in a safe empty state.
    other.data_ = nullptr;
    other.shape_.fill(1);
    other.stride_.fill(1);
    other.owning_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Release our current allocation, if any.
        if (owning_ && data_ != nullptr) {
            cudaFree(data_);
        }

        data_ = other.data_;
        shape_ = other.shape_;
        stride_ = other.stride_;
        dtype_ = other.dtype_;
        owning_ = other.owning_;

        other.data_ = nullptr;
        other.shape_.fill(1);
        other.stride_.fill(1);
        other.owning_ = false;
    }
    return *this;
}

// ── Private view constructor ────────────────────────────────

Tensor::Tensor(void* data,
               std::array<int64_t, 4> shape,
               std::array<int64_t, 4> stride,
               DType dtype)
    : data_(data),
      shape_(shape),
      stride_(stride),
      dtype_(dtype),
      owning_(false) {}

// ── Static Factory Methods ──────────────────────────────────

Tensor Tensor::from_host(const void* host_data,
                         std::vector<int64_t> shape,
                         DType dtype) {
    Tensor t(std::move(shape), dtype);
    CUINFER_CHECK_CUDA(cudaMemcpy(t.data_, host_data, t.size_bytes(),
                                  cudaMemcpyHostToDevice));
    return t;
}

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype) {
    Tensor t(std::move(shape), dtype);
    CUINFER_CHECK_CUDA(cudaMemset(t.data_, 0, t.size_bytes()));
    return t;
}

Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype) {
    return Tensor(std::move(shape), dtype);
}

// ── Core Methods ────────────────────────────────────────────

Tensor Tensor::view(int64_t dim, int64_t start, int64_t end) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("view(): dim must be in [0, 3], got " +
                                std::to_string(dim));
    }
    if (start < 0 || end > shape_[dim] || start >= end) {
        throw std::out_of_range(
            "view(): invalid range [" + std::to_string(start) + ", " +
            std::to_string(end) + ") for dim " + std::to_string(dim) +
            " with size " + std::to_string(shape_[dim]));
    }

    // The view points to the element at `start` along `dim`.
    // Offset in *elements* (not bytes) = start * stride[dim].
    const size_t byte_offset =
        static_cast<size_t>(start) *
        static_cast<size_t>(stride_[dim]) *
        dtype_size(dtype_);

    auto new_shape = shape_;
    new_shape[dim] = end - start;

    // Strides remain the same — we just shift the base pointer.
    auto* offset_ptr = static_cast<char*>(data_) + byte_offset;
    return Tensor(static_cast<void*>(offset_ptr), new_shape, stride_, dtype_);
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error(
            "reshape(): only contiguous tensors can be reshaped");
    }
    if (new_shape.empty() || new_shape.size() > 4) {
        throw std::invalid_argument(
            "reshape(): shape must have 1-4 dimensions");
    }

    std::array<int64_t, 4> out_shape{};
    std::array<int64_t, 4> out_stride{};
    compute_shape_and_stride(new_shape, out_shape, out_stride);

    // Verify element count is preserved.
    int64_t new_numel = out_shape[0] * out_shape[1] *
                        out_shape[2] * out_shape[3];
    if (new_numel != numel()) {
        throw std::invalid_argument(
            "reshape(): new shape has " + std::to_string(new_numel) +
            " elements but tensor has " + std::to_string(numel()));
    }

    return Tensor(data_, out_shape, out_stride, dtype_);
}

Tensor Tensor::clone() const {
    // Build an owning tensor with the same shape and dtype.
    // We construct via the vector-based constructor so it allocates fresh memory.
    // To recover the original user-facing shape we could just pass all 4 dims.
    Tensor t({shape_[0], shape_[1], shape_[2], shape_[3]}, dtype_);

    if (data_ != nullptr && t.data_ != nullptr) {
        CUINFER_CHECK_CUDA(cudaMemcpy(t.data_, data_, size_bytes(),
                                      cudaMemcpyDeviceToDevice));
    }
    return t;
}

void Tensor::copy_from_host(const void* host_data, size_t bytes) {
    if (bytes > size_bytes()) {
        throw std::invalid_argument(
            "copy_from_host(): requested " + std::to_string(bytes) +
            " bytes but tensor only holds " + std::to_string(size_bytes()));
    }
    CUINFER_CHECK_CUDA(
        cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(void* host_data, size_t bytes) const {
    if (bytes > size_bytes()) {
        throw std::invalid_argument(
            "copy_to_host(): requested " + std::to_string(bytes) +
            " bytes but tensor only holds " + std::to_string(size_bytes()));
    }
    CUINFER_CHECK_CUDA(
        cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost));
}

int64_t Tensor::numel() const {
    return shape_[0] * shape_[1] * shape_[2] * shape_[3];
}

size_t Tensor::size_bytes() const {
    if (dtype_is_quantized(dtype_)) {
        // Block-quantized: total_bytes = (numel / block_elements) * block_bytes
        const int64_t n = numel();
        const int block_elems = dtype_block_elements(dtype_);
        const size_t block_bytes = dtype_block_bytes(dtype_);
        return static_cast<size_t>(n / block_elems) * block_bytes;
    }
    return static_cast<size_t>(numel()) * dtype_size(dtype_);
}

int Tensor::ndim() const {
    // Count leading 1s — those are padding, not real dimensions.
    // E.g. {1,1,32,768} has ndim=2, {1,1,1,768} has ndim=1.
    int dims = 4;
    for (int i = 0; i < 3; ++i) { // check first 3 dims; always keep at least 1
        if (shape_[i] == 1) {
            --dims;
        } else {
            break;
        }
    }
    return dims;
}

bool Tensor::is_contiguous() const {
    // A contiguous (row-major) tensor has:
    //   stride[3] == 1
    //   stride[i] == stride[i+1] * shape[i+1]  for i in {0,1,2}
    if (stride_[3] != 1) return false;
    for (int i = 2; i >= 0; --i) {
        if (stride_[i] != stride_[i + 1] * shape_[i + 1]) {
            return false;
        }
    }
    return true;
}

// ── Per-dim accessors (bounds-checked) ──────────────────────

int64_t Tensor::shape(int dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("shape(): dim must be in [0, 3], got " +
                                std::to_string(dim));
    }
    return shape_[dim];
}

int64_t Tensor::stride(int dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("stride(): dim must be in [0, 3], got " +
                                std::to_string(dim));
    }
    return stride_[dim];
}

// ── Debug ───────────────────────────────────────────────────

void Tensor::print(const std::string& name, int max_elems) const {
    if (!name.empty()) {
        printf("Tensor \"%s\"  ", name.c_str());
    } else {
        printf("Tensor  ");
    }
    printf("shape=[%lld,%lld,%lld,%lld]  dtype=%s  numel=%lld  %s\n",
           static_cast<long long>(shape_[0]),
           static_cast<long long>(shape_[1]),
           static_cast<long long>(shape_[2]),
           static_cast<long long>(shape_[3]),
           dtype_name(dtype_),
           static_cast<long long>(numel()),
           owning_ ? "owning" : "view");

    if (data_ == nullptr) {
        printf("  (null data)\n");
        return;
    }
    if (dtype_ != DType::FP32) {
        printf("  (print only supports FP32)\n");
        return;
    }

    const int64_t n = std::min(static_cast<int64_t>(max_elems), numel());
    std::vector<float> buf(static_cast<size_t>(n));
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.data(), data_,
                                  static_cast<size_t>(n) * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    printf("  data[0:%lld] = [", static_cast<long long>(n));
    for (int64_t i = 0; i < n; ++i) {
        if (i > 0) printf(", ");
        printf("%.6f", buf[static_cast<size_t>(i)]);
    }
    if (numel() > n) printf(", ...");
    printf("]\n");
}

} // namespace cuinfer
