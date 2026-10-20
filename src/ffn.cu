// ── ffn.cu — Feed-Forward Network Implementations ───────────────────
//
// Two FFN variants for transformer inference:
//   1. GatedFFN  — Llama-style SwiGLU:  SiLU(x*gate) * (x*up), then down_proj
//   2. StandardFFN — GPT-style:          GeLU(x*W1 + b1), then W2 + b2
//
// Both classes pre-allocate intermediate GPU buffers at construction time
// to avoid dynamic allocation during the generation loop (critical for
// low-latency token-by-token decoding).
//
// All matrix multiplications delegate to the project's tiled matmul kernel.
// SiLU, GeLU, and element-wise multiply use the existing elementwise kernels.

#include "cuinfer/ffn.h"
#include "cuinfer/cuda_utils.h"
#include "cuinfer/kernels/matmul.h"
#include "cuinfer/kernels/elementwise.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace cuinfer {

// ── Bias-Add Kernel ─────────────────────────────────────────
//
// Adds a 1D bias vector to every row of a 2D matrix.
// bias[col] is broadcast across all rows.
//
// This is needed for the standard FFN where linear layers have bias terms:
//   output[row][col] = input[row][col] + bias[col]
//
// Grid: ceil(total_elements / 256) blocks, 256 threads per block.
// No shared memory required — pure element-wise with modular indexing.

constexpr int BLOCK_SIZE = 256;

__global__ void kernel_bias_add(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

void bias_add(const float* input, const float* bias, float* output,
              int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_bias_add<<<grid, BLOCK_SIZE, 0, stream>>>(
        input, bias, output, rows, cols);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// GatedFFN — Llama-style SwiGLU Feed-Forward Network
// ════════════════════════════════════════════════════════════════

GatedFFN::GatedFFN(const FFNConfig& config, int max_seq_len)
    : config_(config), max_seq_len_(max_seq_len) {

    if (config_.dim <= 0) {
        throw std::invalid_argument(
            "GatedFFN: dim must be > 0, got " + std::to_string(config_.dim));
    }
    if (config_.intermediate_size <= 0) {
        throw std::invalid_argument(
            "GatedFFN: intermediate_size must be > 0, got " +
            std::to_string(config_.intermediate_size));
    }
    if (max_seq_len_ <= 0) {
        throw std::invalid_argument(
            "GatedFFN: max_seq_len must be > 0, got " +
            std::to_string(max_seq_len_));
    }

    // Pre-allocate intermediate buffers for the maximum sequence length.
    // Each buffer holds [max_seq_len, intermediate_size] floats.
    //
    // Memory budget for SmolLM2-135M (dim=576, intermediate=1536, max_seq=2048):
    //   3 buffers * 2048 * 1536 * 4 bytes = 37.7 MB
    // Well within our 5.5 GB VRAM budget.
    size_t buf_elements = static_cast<size_t>(max_seq_len_) *
                          static_cast<size_t>(config_.intermediate_size);
    size_t buf_bytes = buf_elements * sizeof(float);

    CUINFER_CHECK_CUDA(cudaMalloc(&gate_buf_, buf_bytes));
    CUINFER_CHECK_CUDA(cudaMalloc(&up_buf_, buf_bytes));
    CUINFER_CHECK_CUDA(cudaMalloc(&hidden_buf_, buf_bytes));
}

GatedFFN::~GatedFFN() {
    free_buffers();
}

GatedFFN::GatedFFN(GatedFFN&& other) noexcept
    : config_(other.config_),
      max_seq_len_(other.max_seq_len_),
      gate_buf_(other.gate_buf_),
      up_buf_(other.up_buf_),
      hidden_buf_(other.hidden_buf_) {
    other.gate_buf_ = nullptr;
    other.up_buf_ = nullptr;
    other.hidden_buf_ = nullptr;
}

GatedFFN& GatedFFN::operator=(GatedFFN&& other) noexcept {
    if (this != &other) {
        free_buffers();
        config_ = other.config_;
        max_seq_len_ = other.max_seq_len_;
        gate_buf_ = other.gate_buf_;
        up_buf_ = other.up_buf_;
        hidden_buf_ = other.hidden_buf_;
        other.gate_buf_ = nullptr;
        other.up_buf_ = nullptr;
        other.hidden_buf_ = nullptr;
    }
    return *this;
}

void GatedFFN::free_buffers() {
    if (gate_buf_) { cudaFree(gate_buf_); gate_buf_ = nullptr; }
    if (up_buf_) { cudaFree(up_buf_); up_buf_ = nullptr; }
    if (hidden_buf_) { cudaFree(hidden_buf_); hidden_buf_ = nullptr; }
}

void GatedFFN::forward(const float* input,
                       const float* w_gate, const float* w_up, const float* w_down,
                       float* output, int seq_len, cudaStream_t stream) {
    // Validate seq_len against the pre-allocated buffer capacity.
    if (seq_len <= 0 || seq_len > max_seq_len_) {
        throw std::invalid_argument(
            "GatedFFN::forward: seq_len must be in [1, " +
            std::to_string(max_seq_len_) + "], got " +
            std::to_string(seq_len));
    }

    const int dim = config_.dim;
    const int inter = config_.intermediate_size;

    // ── Step 1: gate = input * gate_proj ───────────────────────
    // input:     [seq_len, dim]
    // gate_proj: [dim, intermediate_size]
    // gate_buf:  [seq_len, intermediate_size]
    //
    // Matmul: M=seq_len, N=intermediate_size, K=dim
    matmul_tiled(input, w_gate, gate_buf_, seq_len, inter, dim, stream);

    // ── Step 2: up = input * up_proj ──────────────────────────
    // Same dimensions as step 1 but with up_proj weights.
    //
    // NOTE: Steps 1 and 2 are independent and could run on separate
    // streams for overlap. Keeping them sequential on the same stream
    // for simplicity; the stream parameter allows the caller to pipeline
    // with other work.
    matmul_tiled(input, w_up, up_buf_, seq_len, inter, dim, stream);

    // ── Step 3: hidden = SiLU(gate) * up ─────────────────────
    // Apply SiLU activation to the gate projection in-place (gate_buf),
    // then element-wise multiply with the up projection.
    //
    // SiLU(x) = x * sigmoid(x) — smooth gating that allows gradient flow.
    // The element-wise multiply with up_buf implements the "gated" part
    // of SwiGLU: the gate controls how much of each hidden dimension
    // passes through.
    int hidden_elements = seq_len * inter;

    // SiLU(gate_buf) -> hidden_buf (use hidden_buf as scratch for SiLU output)
    silu(gate_buf_, hidden_buf_, hidden_elements, stream);

    // hidden_buf = SiLU(gate) * up -> gate_buf (reuse gate_buf for final hidden)
    // We write back into gate_buf since hidden_buf holds SiLU output and
    // we need a third buffer for the product. Instead, we can write the
    // product into hidden_buf by using it as both input and output of multiply:
    //   Actually, multiply reads from hidden_buf and up_buf, writes to gate_buf.
    //   Then gate_buf holds the final hidden representation.
    multiply(hidden_buf_, up_buf_, gate_buf_, hidden_elements, stream);

    // ── Step 4: output = hidden * down_proj ──────────────────
    // gate_buf (hidden): [seq_len, intermediate_size]
    // down_proj:         [intermediate_size, dim]
    // output:            [seq_len, dim]
    //
    // Matmul: M=seq_len, N=dim, K=intermediate_size
    matmul_tiled(gate_buf_, w_down, output, seq_len, dim, inter, stream);
}

size_t GatedFFN::temp_buffer_bytes() const {
    size_t buf_elements = static_cast<size_t>(max_seq_len_) *
                          static_cast<size_t>(config_.intermediate_size);
    // Three buffers: gate_buf, up_buf, hidden_buf
    return 3 * buf_elements * sizeof(float);
}

// ════════════════════════════════════════════════════════════════
// StandardFFN — GPT-style Feed-Forward Network
// ════════════════════════════════════════════════════════════════

StandardFFN::StandardFFN(const FFNConfig& config, int max_seq_len)
    : config_(config), max_seq_len_(max_seq_len) {

    if (config_.dim <= 0) {
        throw std::invalid_argument(
            "StandardFFN: dim must be > 0, got " +
            std::to_string(config_.dim));
    }
    if (config_.intermediate_size <= 0) {
        throw std::invalid_argument(
            "StandardFFN: intermediate_size must be > 0, got " +
            std::to_string(config_.intermediate_size));
    }
    if (max_seq_len_ <= 0) {
        throw std::invalid_argument(
            "StandardFFN: max_seq_len must be > 0, got " +
            std::to_string(max_seq_len_));
    }

    // Pre-allocate the hidden-layer buffer: [max_seq_len, intermediate_size].
    // Only one intermediate buffer needed (compared to three for GatedFFN).
    size_t buf_elements = static_cast<size_t>(max_seq_len_) *
                          static_cast<size_t>(config_.intermediate_size);
    size_t buf_bytes = buf_elements * sizeof(float);

    CUINFER_CHECK_CUDA(cudaMalloc(&hidden_buf_, buf_bytes));
}

StandardFFN::~StandardFFN() {
    free_buffers();
}

StandardFFN::StandardFFN(StandardFFN&& other) noexcept
    : config_(other.config_),
      max_seq_len_(other.max_seq_len_),
      hidden_buf_(other.hidden_buf_) {
    other.hidden_buf_ = nullptr;
}

StandardFFN& StandardFFN::operator=(StandardFFN&& other) noexcept {
    if (this != &other) {
        free_buffers();
        config_ = other.config_;
        max_seq_len_ = other.max_seq_len_;
        hidden_buf_ = other.hidden_buf_;
        other.hidden_buf_ = nullptr;
    }
    return *this;
}

void StandardFFN::free_buffers() {
    if (hidden_buf_) { cudaFree(hidden_buf_); hidden_buf_ = nullptr; }
}

void StandardFFN::forward(const float* input,
                          const float* w1, const float* b1,
                          const float* w2, const float* b2,
                          float* output, int seq_len, cudaStream_t stream) {
    if (seq_len <= 0 || seq_len > max_seq_len_) {
        throw std::invalid_argument(
            "StandardFFN::forward: seq_len must be in [1, " +
            std::to_string(max_seq_len_) + "], got " +
            std::to_string(seq_len));
    }

    const int dim = config_.dim;
    const int inter = config_.intermediate_size;
    int hidden_elements = seq_len * inter;

    // ── Step 1: hidden = input * W1 ─────────────────────────
    // input: [seq_len, dim]
    // W1:    [dim, intermediate_size]
    // hidden_buf: [seq_len, intermediate_size]
    matmul_tiled(input, w1, hidden_buf_, seq_len, inter, dim, stream);

    // ── Step 2: hidden += b1 (optional bias) ────────────────
    // Broadcasts b1 [intermediate_size] across all rows.
    if (b1 != nullptr) {
        bias_add(hidden_buf_, b1, hidden_buf_, seq_len, inter, stream);
    }

    // ── Step 3: hidden = GeLU(hidden) ───────────────────────
    // Apply GeLU activation in-place.
    // GeLU is the standard activation for GPT-style FFN blocks.
    gelu(hidden_buf_, hidden_buf_, hidden_elements, stream);

    // ── Step 4: output = hidden * W2 ────────────────────────
    // hidden_buf: [seq_len, intermediate_size]
    // W2:         [intermediate_size, dim]
    // output:     [seq_len, dim]
    matmul_tiled(hidden_buf_, w2, output, seq_len, dim, inter, stream);

    // ── Step 5: output += b2 (optional bias) ────────────────
    if (b2 != nullptr) {
        bias_add(output, b2, output, seq_len, dim, stream);
    }
}

size_t StandardFFN::temp_buffer_bytes() const {
    size_t buf_elements = static_cast<size_t>(max_seq_len_) *
                          static_cast<size_t>(config_.intermediate_size);
    // One buffer: hidden_buf
    return buf_elements * sizeof(float);
}

} // namespace cuinfer
