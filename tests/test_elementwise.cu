#include <gtest/gtest.h>
#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/cuda_utils.h"

#include <cmath>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <cfloat>

// ════════════════════════════════════════════════════════════════
// RAII GPU Buffer — allocates on construction, frees on destruction
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

static void cpu_rmsnorm(const float* input, const float* weight,
                        float* output, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float mean_sq = sum_sq / static_cast<float>(n);
    float norm_factor = 1.0f / std::sqrt(mean_sq + eps);
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * weight[i] * norm_factor;
    }
}

static void cpu_silu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

static void cpu_gelu(const float* input, float* output, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

static void cpu_softmax(const float* input, float* output, int n) {
    // Numerically stable: subtract max before exp
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += std::exp(input[i] - max_val);
    }
    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val) / sum_exp;
    }
}

static void cpu_rope(float* qk, int pos, int dim, int n_heads,
                     float theta_base) {
    int half_dim = dim / 2;
    for (int head = 0; head < n_heads; head++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(theta_base,
                                          (2.0f * i) / static_cast<float>(dim));
            float theta = static_cast<float>(pos) * freq;
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);

            int offset = head * dim + 2 * i;
            float x0 = qk[offset];
            float x1 = qk[offset + 1];

            qk[offset]     = x0 * cos_theta - x1 * sin_theta;
            qk[offset + 1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

static void cpu_add(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}

static void cpu_multiply(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] * b[i];
    }
}

// ════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════

// Generate random floats in [-1, 1] using a fixed seed for reproducibility.
static std::vector<float> random_floats(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) {
        v[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
    }
    return v;
}

// Upload host data to a GpuBuffer and return the buffer.
static GpuBuffer upload(const std::vector<float>& host_data) {
    GpuBuffer buf(host_data.size());
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.ptr, host_data.data(),
                                   host_data.size() * sizeof(float),
                                   cudaMemcpyHostToDevice));
    return buf;
}

// Download GPU data to a host vector.
static std::vector<float> download(const float* d_ptr, int n) {
    std::vector<float> host(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(host.data(), d_ptr,
                                   n * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    return host;
}

// ════════════════════════════════════════════════════════════════
// Test Fixture — seeds RNG once before each test
// ════════════════════════════════════════════════════════════════

class ElementwiseTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(42);
    }
};

// ════════════════════════════════════════════════════════════════
// RMSNorm Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, RMSNorm_Basic) {
    const int n = 768;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_weight = random_floats(n);

    // CPU reference
    std::vector<float> cpu_out(rows * n);
    cpu_rmsnorm(h_input.data(), h_weight.data(), cpu_out.data(), n, eps);

    // GPU
    auto d_input = upload(h_input);
    auto d_weight = upload(h_weight);
    GpuBuffer d_output(rows * n);

    cuinfer::rmsnorm(d_input.ptr, d_weight.ptr, d_output.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNorm_MultiRow) {
    const int n = 768;
    const int rows = 4;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_weight = random_floats(n);

    // CPU reference: process each row independently
    std::vector<float> cpu_out(rows * n);
    for (int r = 0; r < rows; r++) {
        cpu_rmsnorm(h_input.data() + r * n, h_weight.data(),
                    cpu_out.data() + r * n, n, eps);
    }

    // GPU
    auto d_input = upload(h_input);
    auto d_weight = upload(h_weight);
    GpuBuffer d_output(rows * n);

    cuinfer::rmsnorm(d_input.ptr, d_weight.ptr, d_output.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNorm_LargeRow) {
    // n=4096 exceeds block size (256), testing the stride loop path.
    const int n = 4096;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_weight = random_floats(n);

    std::vector<float> cpu_out(rows * n);
    cpu_rmsnorm(h_input.data(), h_weight.data(), cpu_out.data(), n, eps);

    auto d_input = upload(h_input);
    auto d_weight = upload(h_weight);
    GpuBuffer d_output(rows * n);

    cuinfer::rmsnorm(d_input.ptr, d_weight.ptr, d_output.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// SiLU Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, SiLU_Basic) {
    const int n = 1024;

    auto h_input = random_floats(n);
    std::vector<float> cpu_out(n);
    cpu_silu(h_input.data(), cpu_out.data(), n);

    auto d_input = upload(h_input);
    GpuBuffer d_output(n);

    cuinfer::silu(d_input.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, SiLU_NegativeInputs) {
    const int n = 1024;

    // All negative values: SiLU should return values close to 0 but slightly
    // negative (since sigmoid(x) < 0.5 for x < 0, and x < 0).
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = -static_cast<float>(rand()) / RAND_MAX * 5.0f - 0.01f;
    }

    std::vector<float> cpu_out(n);
    cpu_silu(h_input.data(), cpu_out.data(), n);

    auto d_input = upload(h_input);
    GpuBuffer d_output(n);

    cuinfer::silu(d_input.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Mismatch at index " << i;
        // SiLU of negative input should be negative
        EXPECT_LE(gpu_out[i], 0.0f);
    }
}

// ════════════════════════════════════════════════════════════════
// GeLU Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, GeLU_Basic) {
    const int n = 1024;

    auto h_input = random_floats(n);
    std::vector<float> cpu_out(n);
    cpu_gelu(h_input.data(), cpu_out.data(), n);

    auto d_input = upload(h_input);
    GpuBuffer d_output(n);

    cuinfer::gelu(d_input.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, GeLU_NearZero) {
    // Near zero, GeLU(x) ~ 0.5 * x (since tanh(0) = 0, so the factor is ~0.5).
    const int n = 256;
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.01f;
    }

    std::vector<float> cpu_out(n);
    cpu_gelu(h_input.data(), cpu_out.data(), n);

    auto d_input = upload(h_input);
    GpuBuffer d_output(n);

    cuinfer::gelu(d_input.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
        // Near zero, GeLU should be approximately 0.5 * x
        EXPECT_NEAR(gpu_out[i], 0.5f * h_input[i], 1e-4f)
            << "GeLU near-zero approximation failed at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// Softmax Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, Softmax_Basic) {
    const int n = 256;
    const int rows = 1;

    auto h_input = random_floats(rows * n);
    std::vector<float> cpu_out(rows * n);
    cpu_softmax(h_input.data(), cpu_out.data(), n);

    auto d_input = upload(h_input);
    GpuBuffer d_output(rows * n);

    cuinfer::softmax(d_input.ptr, d_output.ptr, rows, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);

    // Check that output matches CPU reference
    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    // Check that softmax sums to 1.0
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += gpu_out[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax output does not sum to 1.0";
}

TEST_F(ElementwiseTest, Softmax_MultiRow) {
    const int n = 256;
    const int rows = 4;

    auto h_input = random_floats(rows * n);
    std::vector<float> cpu_out(rows * n);
    for (int r = 0; r < rows; r++) {
        cpu_softmax(h_input.data() + r * n, cpu_out.data() + r * n, n);
    }

    auto d_input = upload(h_input);
    GpuBuffer d_output(rows * n);

    cuinfer::softmax(d_input.ptr, d_output.ptr, rows, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    // Verify each row sums to 1.0
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += gpu_out[r * n + i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f)
            << "Row " << r << " softmax does not sum to 1.0";
    }
}

TEST_F(ElementwiseTest, Softmax_LargeRow) {
    // n=4096 exceeds block size (256), testing the stride loop path.
    const int n = 4096;
    const int rows = 1;

    auto h_input = random_floats(rows * n);
    std::vector<float> cpu_out(rows * n);
    cpu_softmax(h_input.data(), cpu_out.data(), n);

    auto d_input = upload(h_input);
    GpuBuffer d_output(rows * n);

    cuinfer::softmax(d_input.ptr, d_output.ptr, rows, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += gpu_out[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax output does not sum to 1.0";
}

// ════════════════════════════════════════════════════════════════
// RoPE Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, RoPE_Basic) {
    // At pos=0, theta = 0 for all dimensions.
    // cos(0) = 1, sin(0) = 0, so RoPE should be the identity transform.
    const int dim = 64;
    const int n_heads = 8;
    const int total = n_heads * dim;
    const float theta_base = 10000.0f;

    auto h_input = random_floats(total);
    auto h_expected = h_input;  // pos=0 => identity

    auto d_qk = upload(h_input);

    cuinfer::rope(d_qk.ptr, /*pos=*/0, dim, n_heads, theta_base);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_qk.ptr, total);

    for (int i = 0; i < total; i++) {
        EXPECT_NEAR(gpu_out[i], h_expected[i], 1e-4f)
            << "RoPE at pos=0 should be identity, mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RoPE_NonZeroPos) {
    const int dim = 64;
    const int n_heads = 8;
    const int total = n_heads * dim;
    const float theta_base = 10000.0f;
    const int pos = 10;

    auto h_input = random_floats(total);

    // CPU reference: apply RoPE in-place on a copy
    auto cpu_qk = h_input;
    cpu_rope(cpu_qk.data(), pos, dim, n_heads, theta_base);

    // GPU
    auto d_qk = upload(h_input);
    cuinfer::rope(d_qk.ptr, pos, dim, n_heads, theta_base);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_qk.ptr, total);

    for (int i = 0; i < total; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_qk[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// Add Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, Add_Basic) {
    const int n = 1024;

    auto h_a = random_floats(n);
    auto h_b = random_floats(n);
    std::vector<float> cpu_out(n);
    cpu_add(h_a.data(), h_b.data(), cpu_out.data(), n);

    auto d_a = upload(h_a);
    auto d_b = upload(h_b);
    GpuBuffer d_output(n);

    cuinfer::add(d_a.ptr, d_b.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// Multiply Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, Multiply_Basic) {
    const int n = 1024;

    auto h_a = random_floats(n);
    auto h_b = random_floats(n);
    std::vector<float> cpu_out(n);
    cpu_multiply(h_a.data(), h_b.data(), cpu_out.data(), n);

    auto d_a = upload(h_a);
    auto d_b = upload(h_b);
    GpuBuffer d_output(n);

    cuinfer::multiply(d_a.ptr, d_b.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// CUB-Accelerated RMSNorm Tests
// ════════════════════════════════════════════════════════════════
//
// Verify CUB variant matches hand-written variant within 1e-5
// (near bit-identical since both compute the same FP32 math).

TEST_F(ElementwiseTest, RMSNormCUB_Basic) {
    const int n = 768;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_weight = random_floats(n);

    auto d_input = upload(h_input);
    auto d_weight = upload(h_weight);
    GpuBuffer d_handwritten(rows * n);
    GpuBuffer d_cub(rows * n);

    // Run hand-written
    cuinfer::rmsnorm(d_input.ptr, d_weight.ptr, d_handwritten.ptr, rows, n, eps);
    // Run CUB
    cuinfer::rmsnorm_cub(d_input.ptr, d_weight.ptr, d_cub.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_handwritten.ptr, rows * n);
    auto cub_out = download(d_cub.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs hand-written mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormCUB_MultiRow) {
    const int n = 768;
    const int rows = 4;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_weight = random_floats(n);

    auto d_input = upload(h_input);
    auto d_weight = upload(h_weight);
    GpuBuffer d_handwritten(rows * n);
    GpuBuffer d_cub(rows * n);

    cuinfer::rmsnorm(d_input.ptr, d_weight.ptr, d_handwritten.ptr, rows, n, eps);
    cuinfer::rmsnorm_cub(d_input.ptr, d_weight.ptr, d_cub.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_handwritten.ptr, rows * n);
    auto cub_out = download(d_cub.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs hand-written mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormCUB_LargeRow) {
    const int n = 4096;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_weight = random_floats(n);

    auto d_input = upload(h_input);
    auto d_weight = upload(h_weight);
    GpuBuffer d_handwritten(rows * n);
    GpuBuffer d_cub(rows * n);

    cuinfer::rmsnorm(d_input.ptr, d_weight.ptr, d_handwritten.ptr, rows, n, eps);
    cuinfer::rmsnorm_cub(d_input.ptr, d_weight.ptr, d_cub.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_handwritten.ptr, rows * n);
    auto cub_out = download(d_cub.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs hand-written mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// CUB-Accelerated Softmax Tests
// ════════════════════════════════════════════════════════════════

TEST_F(ElementwiseTest, SoftmaxCUB_Basic) {
    const int n = 256;
    const int rows = 1;

    auto h_input = random_floats(rows * n);

    auto d_input = upload(h_input);
    GpuBuffer d_handwritten(rows * n);
    GpuBuffer d_cub(rows * n);

    cuinfer::softmax(d_input.ptr, d_handwritten.ptr, rows, n);
    cuinfer::softmax_cub(d_input.ptr, d_cub.ptr, rows, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_handwritten.ptr, rows * n);
    auto cub_out = download(d_cub.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs hand-written mismatch at index " << i;
    }

    // Verify sums to 1.0
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += cub_out[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "CUB softmax does not sum to 1.0";
}

TEST_F(ElementwiseTest, SoftmaxCUB_MultiRow) {
    const int n = 256;
    const int rows = 4;

    auto h_input = random_floats(rows * n);

    auto d_input = upload(h_input);
    GpuBuffer d_handwritten(rows * n);
    GpuBuffer d_cub(rows * n);

    cuinfer::softmax(d_input.ptr, d_handwritten.ptr, rows, n);
    cuinfer::softmax_cub(d_input.ptr, d_cub.ptr, rows, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_handwritten.ptr, rows * n);
    auto cub_out = download(d_cub.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs hand-written mismatch at index " << i;
    }

    // Verify each row sums to 1.0
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += cub_out[r * n + i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f)
            << "Row " << r << " CUB softmax does not sum to 1.0";
    }
}

TEST_F(ElementwiseTest, SoftmaxCUB_LargeRow) {
    const int n = 4096;
    const int rows = 1;

    auto h_input = random_floats(rows * n);

    auto d_input = upload(h_input);
    GpuBuffer d_handwritten(rows * n);
    GpuBuffer d_cub(rows * n);

    cuinfer::softmax(d_input.ptr, d_handwritten.ptr, rows, n);
    cuinfer::softmax_cub(d_input.ptr, d_cub.ptr, rows, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_handwritten.ptr, rows * n);
    auto cub_out = download(d_cub.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs hand-written mismatch at index " << i;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += cub_out[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "CUB softmax does not sum to 1.0";
}

// ════════════════════════════════════════════════════════════════
// Fused RMSNorm + Residual Add Tests
// ════════════════════════════════════════════════════════════════
//
// CPU reference: add + rmsnorm as two separate steps, then compare.

static void cpu_rmsnorm_residual(const float* input, const float* residual,
                                 const float* weight,
                                 float* output, float* normed,
                                 int n, float eps) {
    // Step 1: residual add
    for (int i = 0; i < n; i++) {
        output[i] = input[i] + residual[i];
    }
    // Step 2: rmsnorm on the sum
    cpu_rmsnorm(output, weight, normed, n, eps);
}

TEST_F(ElementwiseTest, RMSNormResidual_Basic) {
    const int n = 768;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    // CPU reference
    std::vector<float> cpu_out(rows * n), cpu_normed(rows * n);
    cpu_rmsnorm_residual(h_input.data(), h_residual.data(), h_weight.data(),
                         cpu_out.data(), cpu_normed.data(), n, eps);

    // GPU
    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_output(rows * n);
    GpuBuffer d_normed(rows * n);

    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_output.ptr, d_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);
    auto gpu_normed = download(d_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Output mismatch at index " << i;
        EXPECT_NEAR(gpu_normed[i], cpu_normed[i], 1e-4f)
            << "Normed mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormResidual_MultiRow) {
    const int n = 768;
    const int rows = 4;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    std::vector<float> cpu_out(rows * n), cpu_normed(rows * n);
    for (int r = 0; r < rows; r++) {
        cpu_rmsnorm_residual(h_input.data() + r * n, h_residual.data() + r * n,
                             h_weight.data(),
                             cpu_out.data() + r * n, cpu_normed.data() + r * n,
                             n, eps);
    }

    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_output(rows * n);
    GpuBuffer d_normed(rows * n);

    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_output.ptr, d_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);
    auto gpu_normed = download(d_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Output mismatch at index " << i;
        EXPECT_NEAR(gpu_normed[i], cpu_normed[i], 1e-4f)
            << "Normed mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormResidual_LargeRow) {
    const int n = 4096;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    std::vector<float> cpu_out(rows * n), cpu_normed(rows * n);
    cpu_rmsnorm_residual(h_input.data(), h_residual.data(), h_weight.data(),
                         cpu_out.data(), cpu_normed.data(), n, eps);

    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_output(rows * n);
    GpuBuffer d_normed(rows * n);

    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_output.ptr, d_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_output.ptr, rows * n);
    auto gpu_normed = download(d_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Output mismatch at index " << i;
        EXPECT_NEAR(gpu_normed[i], cpu_normed[i], 1e-4f)
            << "Normed mismatch at index " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormResidual_InPlace) {
    // Test that output can alias input (in-place residual update).
    const int n = 768;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    std::vector<float> cpu_out(rows * n), cpu_normed(rows * n);
    cpu_rmsnorm_residual(h_input.data(), h_residual.data(), h_weight.data(),
                         cpu_out.data(), cpu_normed.data(), n, eps);

    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_normed(rows * n);

    // output == input (in-place)
    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_input.ptr, d_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_input.ptr, rows * n);
    auto gpu_normed = download(d_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "In-place output mismatch at index " << i;
        EXPECT_NEAR(gpu_normed[i], cpu_normed[i], 1e-4f)
            << "In-place normed mismatch at index " << i;
    }
}

// ── CUB variants ──

TEST_F(ElementwiseTest, RMSNormResidualCUB_Basic) {
    const int n = 768;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    // Use hand-written fused kernel as reference
    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_hw_out(rows * n), d_hw_normed(rows * n);
    GpuBuffer d_cub_out(rows * n), d_cub_normed(rows * n);

    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_hw_out.ptr, d_hw_normed.ptr, rows, n, eps);
    cuinfer::rmsnorm_residual_cub(d_input.ptr, d_residual.ptr, d_weight.ptr,
                                  d_cub_out.ptr, d_cub_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_hw_out.ptr, rows * n);
    auto cub_out = download(d_cub_out.ptr, rows * n);
    auto hw_normed = download(d_hw_normed.ptr, rows * n);
    auto cub_normed = download(d_cub_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs HW output mismatch at " << i;
        EXPECT_NEAR(cub_normed[i], hw_normed[i], 1e-5f)
            << "CUB vs HW normed mismatch at " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormResidualCUB_MultiRow) {
    const int n = 768;
    const int rows = 4;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_hw_out(rows * n), d_hw_normed(rows * n);
    GpuBuffer d_cub_out(rows * n), d_cub_normed(rows * n);

    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_hw_out.ptr, d_hw_normed.ptr, rows, n, eps);
    cuinfer::rmsnorm_residual_cub(d_input.ptr, d_residual.ptr, d_weight.ptr,
                                  d_cub_out.ptr, d_cub_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_hw_out.ptr, rows * n);
    auto cub_out = download(d_cub_out.ptr, rows * n);
    auto hw_normed = download(d_hw_normed.ptr, rows * n);
    auto cub_normed = download(d_cub_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs HW output mismatch at " << i;
        EXPECT_NEAR(cub_normed[i], hw_normed[i], 1e-5f)
            << "CUB vs HW normed mismatch at " << i;
    }
}

TEST_F(ElementwiseTest, RMSNormResidualCUB_LargeRow) {
    const int n = 4096;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(rows * n);
    auto h_residual = random_floats(rows * n);
    auto h_weight = random_floats(n);

    auto d_input = upload(h_input);
    auto d_residual = upload(h_residual);
    auto d_weight = upload(h_weight);
    GpuBuffer d_hw_out(rows * n), d_hw_normed(rows * n);
    GpuBuffer d_cub_out(rows * n), d_cub_normed(rows * n);

    cuinfer::rmsnorm_residual(d_input.ptr, d_residual.ptr, d_weight.ptr,
                              d_hw_out.ptr, d_hw_normed.ptr, rows, n, eps);
    cuinfer::rmsnorm_residual_cub(d_input.ptr, d_residual.ptr, d_weight.ptr,
                                  d_cub_out.ptr, d_cub_normed.ptr, rows, n, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto hw_out = download(d_hw_out.ptr, rows * n);
    auto cub_out = download(d_cub_out.ptr, rows * n);
    auto hw_normed = download(d_hw_normed.ptr, rows * n);
    auto cub_normed = download(d_cub_normed.ptr, rows * n);

    for (int i = 0; i < rows * n; i++) {
        EXPECT_NEAR(cub_out[i], hw_out[i], 1e-5f)
            << "CUB vs HW output mismatch at " << i;
        EXPECT_NEAR(cub_normed[i], hw_normed[i], 1e-5f)
            << "CUB vs HW normed mismatch at " << i;
    }
}
