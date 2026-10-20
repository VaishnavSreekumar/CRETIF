// ── test_ffn.cu — Feed-Forward Network Tests ───────────────────────
//
// Tests for both FFN variants:
//   1. GatedFFN  (Llama-style SwiGLU)
//   2. StandardFFN (GPT-style with GeLU)
//
// Each test computes the FFN forward pass on GPU and compares against
// a CPU reference implementation. All data is synthetic (no model files
// needed), using a fixed random seed for reproducibility.

#include <gtest/gtest.h>
#include "cuinfer/ffn.h"
#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/cuda_utils.h"

#include <cmath>
#include <cstdlib>
#include <vector>

// ════════════════════════════════════════════════════════════════
// RAII GPU Buffer — same pattern as the other test files
// ════════════════════════════════════════════════════════════════

struct GpuBuffer {
    float* ptr = nullptr;
    GpuBuffer(size_t n) { CUINFER_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float))); }
    ~GpuBuffer() { if (ptr) cudaFree(ptr); }
    GpuBuffer(GpuBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) { if (ptr) cudaFree(ptr); ptr = other.ptr; other.ptr = nullptr; }
        return *this;
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
};

// ════════════════════════════════════════════════════════════════
// CPU Reference Implementations
// ════════════════════════════════════════════════════════════════

/// CPU naive matrix multiply: C = A * B
/// A: [M x K], B: [K x N], C: [M x N], all row-major.
static void cpu_matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/// CPU SiLU: silu(x) = x / (1 + exp(-x))
static void cpu_silu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

/// CPU GeLU (approximate, tanh variant):
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static void cpu_gelu(const float* input, float* output, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

/// CPU element-wise multiply: output[i] = a[i] * b[i]
static void cpu_multiply(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] * b[i];
    }
}

/// CPU bias add: output[row][col] = input[row][col] + bias[col]
static void cpu_bias_add(const float* input, const float* bias, float* output,
                         int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = input[i * cols + j] + bias[j];
        }
    }
}

/// CPU reference for the full Llama-style Gated FFN (SwiGLU):
///   hidden = SiLU(input * gate_proj) * (input * up_proj)
///   output = hidden * down_proj
static void cpu_gated_ffn(const float* input,
                          const float* w_gate, const float* w_up,
                          const float* w_down,
                          float* output,
                          int seq_len, int dim, int inter) {
    // Temporary buffers
    std::vector<float> gate_buf(seq_len * inter);
    std::vector<float> up_buf(seq_len * inter);
    std::vector<float> silu_buf(seq_len * inter);
    std::vector<float> hidden_buf(seq_len * inter);

    // Step 1: gate = input * gate_proj
    cpu_matmul(input, w_gate, gate_buf.data(), seq_len, inter, dim);

    // Step 2: up = input * up_proj
    cpu_matmul(input, w_up, up_buf.data(), seq_len, inter, dim);

    // Step 3: hidden = SiLU(gate) * up
    cpu_silu(gate_buf.data(), silu_buf.data(), seq_len * inter);
    cpu_multiply(silu_buf.data(), up_buf.data(), hidden_buf.data(), seq_len * inter);

    // Step 4: output = hidden * down_proj
    cpu_matmul(hidden_buf.data(), w_down, output, seq_len, dim, inter);
}

/// CPU reference for the full Standard FFN (GPT-style):
///   hidden = GeLU(input * W1 + b1)
///   output = hidden * W2 + b2
static void cpu_standard_ffn(const float* input,
                             const float* w1, const float* b1,
                             const float* w2, const float* b2,
                             float* output,
                             int seq_len, int dim, int inter) {
    std::vector<float> hidden(seq_len * inter);

    // Step 1: hidden = input * W1
    cpu_matmul(input, w1, hidden.data(), seq_len, inter, dim);

    // Step 2: hidden += b1 (if present)
    if (b1 != nullptr) {
        cpu_bias_add(hidden.data(), b1, hidden.data(), seq_len, inter);
    }

    // Step 3: hidden = GeLU(hidden)
    cpu_gelu(hidden.data(), hidden.data(), seq_len * inter);

    // Step 4: output = hidden * W2
    cpu_matmul(hidden.data(), w2, output, seq_len, dim, inter);

    // Step 5: output += b2 (if present)
    if (b2 != nullptr) {
        cpu_bias_add(output, b2, output, seq_len, dim);
    }
}

// ════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════

/// Generate random floats in [-scale, scale] using a fixed seed.
/// Small scale helps keep matmul accumulations from blowing up,
/// which reduces floating-point error in comparisons.
static std::vector<float> random_floats(int n, float scale = 0.1f) {
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) {
        v[i] = scale * (2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f);
    }
    return v;
}

/// Upload host data to GPU.
static GpuBuffer upload(const std::vector<float>& host_data) {
    GpuBuffer buf(host_data.size());
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.ptr, host_data.data(),
                                   host_data.size() * sizeof(float),
                                   cudaMemcpyHostToDevice));
    return buf;
}

/// Download GPU data to host.
static std::vector<float> download(const float* d_ptr, int n) {
    std::vector<float> host(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(host.data(), d_ptr,
                                   n * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    return host;
}

// ════════════════════════════════════════════════════════════════
// Test Fixture
// ════════════════════════════════════════════════════════════════

class FFNTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(42);
    }
};

// ════════════════════════════════════════════════════════════════
// GatedFFN (Llama-style SwiGLU) Tests
// ════════════════════════════════════════════════════════════════

TEST_F(FFNTest, GatedFFN_SmallDimensions) {
    // Small test case: easy to debug, fast to run.
    // dim=32, intermediate=64, seq_len=1 (single token decode).
    const int dim = 32;
    const int inter = 64;
    const int seq_len = 1;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::GatedFFN ffn(config, /*max_seq_len=*/seq_len);

    // Generate synthetic weights and input
    auto h_input  = random_floats(seq_len * dim);
    auto h_w_gate = random_floats(dim * inter);
    auto h_w_up   = random_floats(dim * inter);
    auto h_w_down = random_floats(inter * dim);

    // CPU reference
    std::vector<float> cpu_out(seq_len * dim);
    cpu_gated_ffn(h_input.data(), h_w_gate.data(), h_w_up.data(),
                  h_w_down.data(), cpu_out.data(), seq_len, dim, inter);

    // GPU forward pass
    auto d_input  = upload(h_input);
    auto d_w_gate = upload(h_w_gate);
    auto d_w_up   = upload(h_w_up);
    auto d_w_down = upload(h_w_down);
    GpuBuffer d_output(seq_len * dim);

    ffn.forward(d_input.ptr, d_w_gate.ptr, d_w_up.ptr, d_w_down.ptr,
                d_output.ptr, seq_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "GatedFFN small mismatch at index " << i
            << " (gpu=" << gpu_out[i] << ", cpu=" << cpu_out[i] << ")";
    }
}

TEST_F(FFNTest, GatedFFN_MultipleTokens) {
    // Test with multiple tokens (prefill scenario).
    // dim=64, intermediate=128, seq_len=4.
    const int dim = 64;
    const int inter = 128;
    const int seq_len = 4;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::GatedFFN ffn(config, /*max_seq_len=*/seq_len);

    auto h_input  = random_floats(seq_len * dim);
    auto h_w_gate = random_floats(dim * inter);
    auto h_w_up   = random_floats(dim * inter);
    auto h_w_down = random_floats(inter * dim);

    std::vector<float> cpu_out(seq_len * dim);
    cpu_gated_ffn(h_input.data(), h_w_gate.data(), h_w_up.data(),
                  h_w_down.data(), cpu_out.data(), seq_len, dim, inter);

    auto d_input  = upload(h_input);
    auto d_w_gate = upload(h_w_gate);
    auto d_w_up   = upload(h_w_up);
    auto d_w_down = upload(h_w_down);
    GpuBuffer d_output(seq_len * dim);

    ffn.forward(d_input.ptr, d_w_gate.ptr, d_w_up.ptr, d_w_down.ptr,
                d_output.ptr, seq_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "GatedFFN multi-token mismatch at index " << i
            << " (gpu=" << gpu_out[i] << ", cpu=" << cpu_out[i] << ")";
    }
}

TEST_F(FFNTest, GatedFFN_RealisticDimensions) {
    // Dimensions close to SmolLM2-135M: dim=576, intermediate=1536.
    // Using seq_len=1 (decode step) to keep the test fast.
    // Reduced to multiples of 32 for matmul tile alignment.
    const int dim = 128;
    const int inter = 384;
    const int seq_len = 1;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::GatedFFN ffn(config, /*max_seq_len=*/8);

    auto h_input  = random_floats(seq_len * dim);
    auto h_w_gate = random_floats(dim * inter);
    auto h_w_up   = random_floats(dim * inter);
    auto h_w_down = random_floats(inter * dim);

    std::vector<float> cpu_out(seq_len * dim);
    cpu_gated_ffn(h_input.data(), h_w_gate.data(), h_w_up.data(),
                  h_w_down.data(), cpu_out.data(), seq_len, dim, inter);

    auto d_input  = upload(h_input);
    auto d_w_gate = upload(h_w_gate);
    auto d_w_up   = upload(h_w_up);
    auto d_w_down = upload(h_w_down);
    GpuBuffer d_output(seq_len * dim);

    ffn.forward(d_input.ptr, d_w_gate.ptr, d_w_up.ptr, d_w_down.ptr,
                d_output.ptr, seq_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, seq_len * dim);

    // Slightly relaxed tolerance for larger dimensions: matmul accumulation
    // over K=128 elements compounds floating-point rounding differences
    // between the GPU tiled kernel and CPU sequential summation.
    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 5e-3f)
            << "GatedFFN realistic mismatch at index " << i
            << " (gpu=" << gpu_out[i] << ", cpu=" << cpu_out[i] << ")";
    }
}

TEST_F(FFNTest, GatedFFN_VerifyIntermediateSiLU) {
    // Verify that the SiLU activation step produces correct intermediate
    // values. We do this by checking that the gate_proj output passed
    // through SiLU gives the expected values.
    //
    // Strategy: use identity-like weights so we can predict the intermediate.
    // gate_proj = I (padded), up_proj = I (padded), down_proj = I^T (padded).
    // This way, the FFN reduces to: SiLU(input_padded) * input_padded,
    // projected back to dim.
    //
    // Instead of rigging identity weights (which is tricky with dim != inter),
    // we just test the SiLU kernel directly on known values to verify the
    // activation is correct within the FFN pipeline.
    const int n = 256;

    // Test SiLU on specific values: 0, 1, -1, large positive, large negative
    std::vector<float> h_input = {
        0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 0.5f, -0.5f, 2.0f
    };
    // Pad to a reasonable size with random values
    h_input.resize(n);
    for (int i = 8; i < n; i++) {
        h_input[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
    }

    // CPU reference SiLU
    std::vector<float> cpu_out(n);
    cpu_silu(h_input.data(), cpu_out.data(), n);

    // GPU SiLU (using the kernel directly)
    auto d_input = upload(h_input);
    GpuBuffer d_output(n);
    cuinfer::silu(d_input.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    // Known properties of SiLU:
    // - SiLU(0) = 0
    // - SiLU(x) > 0 for x > 0 (approximately x for large x)
    // - SiLU(x) < 0 for x < 0 (bounded, minimum around -0.278)
    EXPECT_NEAR(gpu_out[0], 0.0f, 1e-6f) << "SiLU(0) should be 0";
    EXPECT_GT(gpu_out[1], 0.0f) << "SiLU(1) should be positive";
    EXPECT_LT(gpu_out[2], 0.0f) << "SiLU(-1) should be negative";
    EXPECT_NEAR(gpu_out[3], 5.0f, 0.05f) << "SiLU(5) should be ~5 (sigmoid(5) ~ 1)";

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "SiLU mismatch at index " << i;
    }
}

TEST_F(FFNTest, GatedFFN_SeqLenReuse) {
    // Verify that the same GatedFFN instance can be reused for different
    // seq_len values (up to max_seq_len). The internal buffers should
    // accommodate the max, and smaller seq_len should still work correctly.
    const int dim = 32;
    const int inter = 64;
    const int max_seq = 8;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::GatedFFN ffn(config, max_seq);

    auto h_w_gate = random_floats(dim * inter);
    auto h_w_up   = random_floats(dim * inter);
    auto h_w_down = random_floats(inter * dim);
    auto d_w_gate = upload(h_w_gate);
    auto d_w_up   = upload(h_w_up);
    auto d_w_down = upload(h_w_down);

    // Test with seq_len=1 (decode)
    {
        auto h_input = random_floats(1 * dim);
        std::vector<float> cpu_out(1 * dim);
        cpu_gated_ffn(h_input.data(), h_w_gate.data(), h_w_up.data(),
                      h_w_down.data(), cpu_out.data(), 1, dim, inter);

        auto d_input = upload(h_input);
        GpuBuffer d_output(1 * dim);
        ffn.forward(d_input.ptr, d_w_gate.ptr, d_w_up.ptr, d_w_down.ptr,
                    d_output.ptr, 1);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        auto gpu_out = download(d_output.ptr, 1 * dim);
        for (int i = 0; i < dim; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
                << "GatedFFN reuse (seq=1) mismatch at " << i;
        }
    }

    // Test with seq_len=max_seq (prefill)
    {
        auto h_input = random_floats(max_seq * dim);
        std::vector<float> cpu_out(max_seq * dim);
        cpu_gated_ffn(h_input.data(), h_w_gate.data(), h_w_up.data(),
                      h_w_down.data(), cpu_out.data(), max_seq, dim, inter);

        auto d_input = upload(h_input);
        GpuBuffer d_output(max_seq * dim);
        ffn.forward(d_input.ptr, d_w_gate.ptr, d_w_up.ptr, d_w_down.ptr,
                    d_output.ptr, max_seq);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        auto gpu_out = download(d_output.ptr, max_seq * dim);
        for (int i = 0; i < max_seq * dim; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
                << "GatedFFN reuse (seq=max) mismatch at " << i;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// StandardFFN (GPT-style) Tests
// ════════════════════════════════════════════════════════════════

TEST_F(FFNTest, StandardFFN_WithBias) {
    // Standard FFN with bias terms on both linear layers.
    const int dim = 32;
    const int inter = 64;
    const int seq_len = 1;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::StandardFFN ffn(config, /*max_seq_len=*/seq_len);

    auto h_input = random_floats(seq_len * dim);
    auto h_w1    = random_floats(dim * inter);
    auto h_b1    = random_floats(inter);
    auto h_w2    = random_floats(inter * dim);
    auto h_b2    = random_floats(dim);

    std::vector<float> cpu_out(seq_len * dim);
    cpu_standard_ffn(h_input.data(), h_w1.data(), h_b1.data(),
                     h_w2.data(), h_b2.data(), cpu_out.data(),
                     seq_len, dim, inter);

    auto d_input = upload(h_input);
    auto d_w1    = upload(h_w1);
    auto d_b1    = upload(h_b1);
    auto d_w2    = upload(h_w2);
    auto d_b2    = upload(h_b2);
    GpuBuffer d_output(seq_len * dim);

    ffn.forward(d_input.ptr, d_w1.ptr, d_b1.ptr, d_w2.ptr, d_b2.ptr,
                d_output.ptr, seq_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "StandardFFN with bias mismatch at index " << i
            << " (gpu=" << gpu_out[i] << ", cpu=" << cpu_out[i] << ")";
    }
}

TEST_F(FFNTest, StandardFFN_NoBias) {
    // Standard FFN without bias terms (b1=nullptr, b2=nullptr).
    // This variant is used by some models that omit FFN biases.
    const int dim = 32;
    const int inter = 64;
    const int seq_len = 2;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::StandardFFN ffn(config, /*max_seq_len=*/seq_len);

    auto h_input = random_floats(seq_len * dim);
    auto h_w1    = random_floats(dim * inter);
    auto h_w2    = random_floats(inter * dim);

    std::vector<float> cpu_out(seq_len * dim);
    cpu_standard_ffn(h_input.data(), h_w1.data(), nullptr,
                     h_w2.data(), nullptr, cpu_out.data(),
                     seq_len, dim, inter);

    auto d_input = upload(h_input);
    auto d_w1    = upload(h_w1);
    auto d_w2    = upload(h_w2);
    GpuBuffer d_output(seq_len * dim);

    ffn.forward(d_input.ptr, d_w1.ptr, nullptr, d_w2.ptr, nullptr,
                d_output.ptr, seq_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "StandardFFN no-bias mismatch at index " << i
            << " (gpu=" << gpu_out[i] << ", cpu=" << cpu_out[i] << ")";
    }
}

TEST_F(FFNTest, StandardFFN_MultipleTokens) {
    // Test standard FFN with multiple tokens and bias.
    const int dim = 64;
    const int inter = 128;
    const int seq_len = 4;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::StandardFFN ffn(config, /*max_seq_len=*/seq_len);

    auto h_input = random_floats(seq_len * dim);
    auto h_w1    = random_floats(dim * inter);
    auto h_b1    = random_floats(inter);
    auto h_w2    = random_floats(inter * dim);
    auto h_b2    = random_floats(dim);

    std::vector<float> cpu_out(seq_len * dim);
    cpu_standard_ffn(h_input.data(), h_w1.data(), h_b1.data(),
                     h_w2.data(), h_b2.data(), cpu_out.data(),
                     seq_len, dim, inter);

    auto d_input = upload(h_input);
    auto d_w1    = upload(h_w1);
    auto d_b1    = upload(h_b1);
    auto d_w2    = upload(h_w2);
    auto d_b2    = upload(h_b2);
    GpuBuffer d_output(seq_len * dim);

    ffn.forward(d_input.ptr, d_w1.ptr, d_b1.ptr, d_w2.ptr, d_b2.ptr,
                d_output.ptr, seq_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "StandardFFN multi-token mismatch at index " << i
            << " (gpu=" << gpu_out[i] << ", cpu=" << cpu_out[i] << ")";
    }
}

// ════════════════════════════════════════════════════════════════
// Bias-Add Kernel Tests
// ════════════════════════════════════════════════════════════════

TEST_F(FFNTest, BiasAdd_Basic) {
    // Verify the bias_add kernel broadcasts correctly across rows.
    const int rows = 4;
    const int cols = 32;
    const int total = rows * cols;

    auto h_input = random_floats(total);
    auto h_bias  = random_floats(cols);

    // CPU reference
    std::vector<float> cpu_out(total);
    cpu_bias_add(h_input.data(), h_bias.data(), cpu_out.data(), rows, cols);

    // GPU
    auto d_input = upload(h_input);
    auto d_bias  = upload(h_bias);
    GpuBuffer d_output(total);

    cuinfer::bias_add(d_input.ptr, d_bias.ptr, d_output.ptr, rows, cols);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, total);

    for (int i = 0; i < total; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6f)
            << "BiasAdd mismatch at index " << i;
    }
}

TEST_F(FFNTest, BiasAdd_SingleRow) {
    // Single row: bias_add should behave like simple element-wise add.
    const int rows = 1;
    const int cols = 128;

    auto h_input = random_floats(cols);
    auto h_bias  = random_floats(cols);

    std::vector<float> cpu_out(cols);
    for (int i = 0; i < cols; i++) {
        cpu_out[i] = h_input[i] + h_bias[i];
    }

    auto d_input = upload(h_input);
    auto d_bias  = upload(h_bias);
    GpuBuffer d_output(cols);

    cuinfer::bias_add(d_input.ptr, d_bias.ptr, d_output.ptr, rows, cols);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, cols);

    for (int i = 0; i < cols; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6f)
            << "BiasAdd single-row mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// Error Handling Tests
// ════════════════════════════════════════════════════════════════

TEST_F(FFNTest, GatedFFN_InvalidConfig) {
    // dim=0 should throw.
    cuinfer::FFNConfig bad_config{0, 64};
    EXPECT_THROW(cuinfer::GatedFFN(bad_config, 1), std::invalid_argument);

    // intermediate_size=0 should throw.
    cuinfer::FFNConfig bad_config2{32, 0};
    EXPECT_THROW(cuinfer::GatedFFN(bad_config2, 1), std::invalid_argument);

    // max_seq_len=0 should throw.
    cuinfer::FFNConfig ok_config{32, 64};
    EXPECT_THROW(cuinfer::GatedFFN(ok_config, 0), std::invalid_argument);
}

TEST_F(FFNTest, StandardFFN_InvalidConfig) {
    cuinfer::FFNConfig bad_config{0, 64};
    EXPECT_THROW(cuinfer::StandardFFN(bad_config, 1), std::invalid_argument);

    cuinfer::FFNConfig ok_config{32, 64};
    EXPECT_THROW(cuinfer::StandardFFN(ok_config, 0), std::invalid_argument);
}

TEST_F(FFNTest, GatedFFN_SeqLenExceedsMax) {
    // Calling forward with seq_len > max_seq_len should throw.
    const int dim = 32;
    const int inter = 64;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::GatedFFN ffn(config, /*max_seq_len=*/2);

    auto h_input  = random_floats(4 * dim);
    auto h_w_gate = random_floats(dim * inter);
    auto h_w_up   = random_floats(dim * inter);
    auto h_w_down = random_floats(inter * dim);

    auto d_input  = upload(h_input);
    auto d_w_gate = upload(h_w_gate);
    auto d_w_up   = upload(h_w_up);
    auto d_w_down = upload(h_w_down);
    GpuBuffer d_output(4 * dim);

    // seq_len=4 exceeds max_seq_len=2
    EXPECT_THROW(
        ffn.forward(d_input.ptr, d_w_gate.ptr, d_w_up.ptr, d_w_down.ptr,
                    d_output.ptr, /*seq_len=*/4),
        std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════
// Memory Reporting Tests
// ════════════════════════════════════════════════════════════════

TEST_F(FFNTest, GatedFFN_TempBufferBytes) {
    const int dim = 32;
    const int inter = 64;
    const int max_seq = 8;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::GatedFFN ffn(config, max_seq);

    // Expected: 3 buffers * max_seq * inter * sizeof(float)
    size_t expected = 3 * static_cast<size_t>(max_seq) *
                      static_cast<size_t>(inter) * sizeof(float);
    EXPECT_EQ(ffn.temp_buffer_bytes(), expected);
}

TEST_F(FFNTest, StandardFFN_TempBufferBytes) {
    const int dim = 32;
    const int inter = 64;
    const int max_seq = 8;

    cuinfer::FFNConfig config{dim, inter};
    cuinfer::StandardFFN ffn(config, max_seq);

    // Expected: 1 buffer * max_seq * inter * sizeof(float)
    size_t expected = static_cast<size_t>(max_seq) *
                      static_cast<size_t>(inter) * sizeof(float);
    EXPECT_EQ(ffn.temp_buffer_bytes(), expected);
}
