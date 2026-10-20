// ── test_embedding.cu — Unit Tests for Embedding, LM Head, and Sampling ──
//
// Tests cover:
//   1. Embedding lookup correctness (known token IDs -> correct rows)
//   2. LM head correctness (matmul_transposed_b against CPU reference)
//   3. Argmax correctness (finds index of maximum value)
//   4. Top-k sampling (sampled token is within top-k candidates)
//   5. Edge cases: token ID 0, last token (vocab_size-1), single token,
//      multiple tokens, tied weights (same matrix for embed + LM head)

#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <numeric>

#include "cuinfer/embedding.h"
#include "cuinfer/cuda_utils.h"

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

// RAII GPU buffer for float data
struct GpuFloatBuffer {
    float* ptr = nullptr;
    GpuFloatBuffer(size_t n) {
        CUINFER_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float)));
    }
    ~GpuFloatBuffer() { if (ptr) cudaFree(ptr); }
    GpuFloatBuffer(const GpuFloatBuffer&) = delete;
    GpuFloatBuffer& operator=(const GpuFloatBuffer&) = delete;
};

// RAII GPU buffer for int32 data (token IDs)
struct GpuIntBuffer {
    int32_t* ptr = nullptr;
    GpuIntBuffer(size_t n) {
        CUINFER_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(int32_t)));
    }
    ~GpuIntBuffer() { if (ptr) cudaFree(ptr); }
    GpuIntBuffer(const GpuIntBuffer&) = delete;
    GpuIntBuffer& operator=(const GpuIntBuffer&) = delete;
    GpuIntBuffer(GpuIntBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    GpuIntBuffer& operator=(GpuIntBuffer&& other) noexcept {
        if (this != &other) { if (ptr) cudaFree(ptr); ptr = other.ptr; other.ptr = nullptr; }
        return *this;
    }
};

// Generate random floats in [-1, 1] with a fixed seed for reproducibility.
static std::vector<float> random_floats(int n, unsigned int seed = 42) {
    srand(seed);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
    }
    return v;
}

// Upload int32 data to GPU
static GpuIntBuffer upload_ints(const std::vector<int32_t>& data) {
    GpuIntBuffer buf(data.size());
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.ptr, data.data(),
                                  data.size() * sizeof(int32_t),
                                  cudaMemcpyHostToDevice));
    return buf;
}

// Download float data from GPU
static std::vector<float> download_floats(const float* d_ptr, int n) {
    std::vector<float> host(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(host.data(), d_ptr,
                                  n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return host;
}

// CPU reference: embedding lookup
static void cpu_embedding_lookup(const int32_t* token_ids,
                                 const float* weight,
                                 float* output,
                                 int seq_len, int dim) {
    for (int i = 0; i < seq_len; ++i) {
        int32_t tid = token_ids[i];
        for (int j = 0; j < dim; ++j) {
            output[i * dim + j] = weight[tid * dim + j];
        }
    }
}

// CPU reference: matmul with transposed B
// C[i,j] = sum_k A[i,k] * B[j,k]
// A is [M, K], B is [N, K], C is [M, N]
static void cpu_matmul_transposed_b(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU reference: argmax
static int32_t cpu_argmax(const float* data, int n) {
    int32_t best_idx = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    return best_idx;
}

// Tolerance for FP32 comparisons.
constexpr float TOL = 1e-5f;

// Larger tolerance for matmul (accumulation over K values introduces rounding).
constexpr float MATMUL_TOL = 1e-3f;

// ════════════════════════════════════════════════════════════════════════
// Test Fixture
// ════════════════════════════════════════════════════════════════════════

class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(42);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Embedding Lookup Kernel Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, LookupKernel_SingleToken) {
    // Simplest case: look up a single token.
    const int vocab_size = 100;
    const int dim = 64;
    const int seq_len = 1;

    auto h_weight = random_floats(vocab_size * dim);
    std::vector<int32_t> h_ids = {42};  // Look up token 42

    // CPU reference
    std::vector<float> cpu_out(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_out.data(),
                         seq_len, dim);

    // GPU
    auto d_ids = upload_ints(h_ids);
    GpuFloatBuffer d_weight(vocab_size * dim);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_weight.ptr, h_weight.data(),
                                  vocab_size * dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
    GpuFloatBuffer d_output(seq_len * dim);

    cuinfer::embedding_lookup(d_ids.ptr, d_weight.ptr, d_output.ptr,
                              seq_len, dim);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], TOL)
            << "Mismatch at index " << i;
    }
}

TEST_F(EmbeddingTest, LookupKernel_MultipleTokens) {
    // Look up several tokens in sequence.
    const int vocab_size = 256;
    const int dim = 128;
    const int seq_len = 8;

    auto h_weight = random_floats(vocab_size * dim);
    std::vector<int32_t> h_ids = {0, 10, 42, 100, 200, 255, 1, 127};

    // CPU reference
    std::vector<float> cpu_out(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_out.data(),
                         seq_len, dim);

    // GPU
    auto d_ids = upload_ints(h_ids);
    GpuFloatBuffer d_weight(vocab_size * dim);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_weight.ptr, h_weight.data(),
                                  vocab_size * dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
    GpuFloatBuffer d_output(seq_len * dim);

    cuinfer::embedding_lookup(d_ids.ptr, d_weight.ptr, d_output.ptr,
                              seq_len, dim);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], TOL)
            << "Mismatch at index " << i;
    }
}

TEST_F(EmbeddingTest, LookupKernel_TokenIdZero) {
    // Edge case: token ID 0 (first row of embedding table).
    const int vocab_size = 50;
    const int dim = 32;
    const int seq_len = 1;

    auto h_weight = random_floats(vocab_size * dim);
    std::vector<int32_t> h_ids = {0};

    std::vector<float> cpu_out(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_out.data(),
                         seq_len, dim);

    auto d_ids = upload_ints(h_ids);
    GpuFloatBuffer d_weight(vocab_size * dim);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_weight.ptr, h_weight.data(),
                                  vocab_size * dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
    GpuFloatBuffer d_output(seq_len * dim);

    cuinfer::embedding_lookup(d_ids.ptr, d_weight.ptr, d_output.ptr,
                              seq_len, dim);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_output.ptr, seq_len * dim);

    // Verify output matches first row of weight table exactly
    for (int j = 0; j < dim; ++j) {
        EXPECT_NEAR(gpu_out[j], h_weight[j], TOL)
            << "Token ID 0: mismatch at dim " << j;
    }
}

TEST_F(EmbeddingTest, LookupKernel_LastTokenId) {
    // Edge case: last valid token ID (vocab_size - 1).
    const int vocab_size = 100;
    const int dim = 64;
    const int seq_len = 1;

    auto h_weight = random_floats(vocab_size * dim);
    std::vector<int32_t> h_ids = {vocab_size - 1};

    std::vector<float> cpu_out(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_out.data(),
                         seq_len, dim);

    auto d_ids = upload_ints(h_ids);
    GpuFloatBuffer d_weight(vocab_size * dim);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_weight.ptr, h_weight.data(),
                                  vocab_size * dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
    GpuFloatBuffer d_output(seq_len * dim);

    cuinfer::embedding_lookup(d_ids.ptr, d_weight.ptr, d_output.ptr,
                              seq_len, dim);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_output.ptr, seq_len * dim);

    // Verify output matches last row of weight table exactly
    for (int j = 0; j < dim; ++j) {
        EXPECT_NEAR(gpu_out[j], h_weight[(vocab_size - 1) * dim + j], TOL)
            << "Last token ID: mismatch at dim " << j;
    }
}

TEST_F(EmbeddingTest, LookupKernel_RealisticSize) {
    // SmolLM2-135M style: vocab_size=49152, dim=576, seq_len=64
    // This tests with a larger but still manageable size.
    const int vocab_size = 1024;  // Scaled down to avoid huge allocs in test
    const int dim = 576;
    const int seq_len = 64;

    auto h_weight = random_floats(vocab_size * dim);

    // Generate random token IDs in [0, vocab_size)
    std::vector<int32_t> h_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        h_ids[i] = rand() % vocab_size;
    }

    std::vector<float> cpu_out(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_out.data(),
                         seq_len, dim);

    auto d_ids = upload_ints(h_ids);
    GpuFloatBuffer d_weight(vocab_size * dim);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_weight.ptr, h_weight.data(),
                                  vocab_size * dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
    GpuFloatBuffer d_output(seq_len * dim);

    cuinfer::embedding_lookup(d_ids.ptr, d_weight.ptr, d_output.ptr,
                              seq_len, dim);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_output.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], TOL)
            << "Mismatch at index " << i
            << " (token=" << i / dim << ", dim=" << i % dim << ")";
    }
}

// ════════════════════════════════════════════════════════════════════════
// Embedding Class Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, EmbeddingClass_Forward) {
    // Test the Embedding class wrapper using the Tensor API.
    const int vocab_size = 256;
    const int dim = 64;
    const int seq_len = 4;

    auto h_weight = random_floats(vocab_size * dim);
    std::vector<int32_t> h_ids = {5, 50, 100, 255};

    // CPU reference
    std::vector<float> cpu_out(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_out.data(),
                         seq_len, dim);

    // Create weight tensor on GPU
    cuinfer::Tensor weight = cuinfer::Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    // Upload token IDs
    auto d_ids = upload_ints(h_ids);

    // Forward
    cuinfer::Embedding emb(vocab_size, dim);
    cuinfer::Tensor output = emb.forward(d_ids.ptr, seq_len, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Verify shape
    EXPECT_EQ(output.shape(2), seq_len);
    EXPECT_EQ(output.shape(3), dim);

    // Verify values
    auto gpu_out = download_floats(output.data_as<float>(), seq_len * dim);
    for (int i = 0; i < seq_len * dim; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], TOL)
            << "Mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════════════
// MatMul Transposed-B Kernel Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, MatmulTransposedB_Small) {
    // Small case: easy to debug.
    const int M = 2, N = 4, K = 3;

    auto h_A = random_floats(M * K);
    auto h_B = random_floats(N * K);

    std::vector<float> cpu_out(M * N);
    cpu_matmul_transposed_b(h_A.data(), h_B.data(), cpu_out.data(), M, N, K);

    GpuFloatBuffer d_A(M * K), d_B(N * K), d_C(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_A.ptr, h_A.data(),
                                  M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B.ptr, h_B.data(),
                                  N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemset(d_C.ptr, 0, M * N * sizeof(float)));

    cuinfer::matmul_transposed_b(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_C.ptr, M * N);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], MATMUL_TOL)
            << "Mismatch at flat index " << i
            << " (row=" << i / N << ", col=" << i % N << ")";
    }
}

TEST_F(EmbeddingTest, MatmulTransposedB_Square) {
    const int M = 64, N = 64, K = 64;

    auto h_A = random_floats(M * K);
    auto h_B = random_floats(N * K);

    std::vector<float> cpu_out(M * N);
    cpu_matmul_transposed_b(h_A.data(), h_B.data(), cpu_out.data(), M, N, K);

    GpuFloatBuffer d_A(M * K), d_B(N * K), d_C(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_A.ptr, h_A.data(),
                                  M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B.ptr, h_B.data(),
                                  N * K * sizeof(float), cudaMemcpyHostToDevice));

    cuinfer::matmul_transposed_b(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_C.ptr, M * N);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], MATMUL_TOL)
            << "Mismatch at flat index " << i;
    }
}

TEST_F(EmbeddingTest, MatmulTransposedB_NonAligned) {
    // Non-tile-aligned dimensions to test boundary handling.
    const int M = 5, N = 100, K = 37;

    auto h_A = random_floats(M * K);
    auto h_B = random_floats(N * K);

    std::vector<float> cpu_out(M * N);
    cpu_matmul_transposed_b(h_A.data(), h_B.data(), cpu_out.data(), M, N, K);

    GpuFloatBuffer d_A(M * K), d_B(N * K), d_C(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_A.ptr, h_A.data(),
                                  M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B.ptr, h_B.data(),
                                  N * K * sizeof(float), cudaMemcpyHostToDevice));

    cuinfer::matmul_transposed_b(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download_floats(d_C.ptr, M * N);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], MATMUL_TOL)
            << "Mismatch at flat index " << i;
    }
}

// ════════════════════════════════════════════════════════════════════════
// LM Head Class Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, LMHead_SingleToken) {
    // Typical generation scenario: project a single hidden state to logits.
    const int vocab_size = 128;
    const int dim = 64;
    const int seq_len = 1;

    auto h_hidden = random_floats(seq_len * dim);
    auto h_weight = random_floats(vocab_size * dim);

    // CPU reference: logits = hidden * weight^T
    std::vector<float> cpu_logits(seq_len * vocab_size);
    cpu_matmul_transposed_b(h_hidden.data(), h_weight.data(),
                            cpu_logits.data(), seq_len, vocab_size, dim);

    // GPU via LMHead class
    cuinfer::Tensor hidden = cuinfer::Tensor::from_host(
        h_hidden.data(),
        {static_cast<int64_t>(seq_len), static_cast<int64_t>(dim)});
    cuinfer::Tensor weight = cuinfer::Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    cuinfer::LMHead lm_head(vocab_size, dim);
    cuinfer::Tensor logits = lm_head.forward(hidden, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Verify shape
    EXPECT_EQ(logits.shape(2), seq_len);
    EXPECT_EQ(logits.shape(3), vocab_size);

    // Verify values
    auto gpu_logits = download_floats(logits.data_as<float>(),
                                      seq_len * vocab_size);
    for (int i = 0; i < seq_len * vocab_size; ++i) {
        EXPECT_NEAR(gpu_logits[i], cpu_logits[i], MATMUL_TOL)
            << "Mismatch at index " << i;
    }
}

TEST_F(EmbeddingTest, LMHead_MultiToken) {
    // Prefill scenario: project multiple hidden states at once.
    const int vocab_size = 128;
    const int dim = 64;
    const int seq_len = 16;

    auto h_hidden = random_floats(seq_len * dim);
    auto h_weight = random_floats(vocab_size * dim);

    std::vector<float> cpu_logits(seq_len * vocab_size);
    cpu_matmul_transposed_b(h_hidden.data(), h_weight.data(),
                            cpu_logits.data(), seq_len, vocab_size, dim);

    cuinfer::Tensor hidden = cuinfer::Tensor::from_host(
        h_hidden.data(),
        {static_cast<int64_t>(seq_len), static_cast<int64_t>(dim)});
    cuinfer::Tensor weight = cuinfer::Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    cuinfer::LMHead lm_head(vocab_size, dim);
    cuinfer::Tensor logits = lm_head.forward(hidden, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    EXPECT_EQ(logits.shape(2), seq_len);
    EXPECT_EQ(logits.shape(3), vocab_size);

    auto gpu_logits = download_floats(logits.data_as<float>(),
                                      seq_len * vocab_size);
    for (int i = 0; i < seq_len * vocab_size; ++i) {
        EXPECT_NEAR(gpu_logits[i], cpu_logits[i], MATMUL_TOL)
            << "Mismatch at index " << i;
    }
}

TEST_F(EmbeddingTest, LMHead_TiedWeights) {
    // Verify that the same weight matrix can be used for both
    // embedding lookup and LM head projection (weight tying).
    const int vocab_size = 128;
    const int dim = 64;

    auto h_weight = random_floats(vocab_size * dim);
    cuinfer::Tensor weight = cuinfer::Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    // Step 1: Embed a token
    std::vector<int32_t> h_ids = {42};
    auto d_ids = upload_ints(h_ids);

    cuinfer::Embedding emb(vocab_size, dim);
    cuinfer::Tensor hidden = emb.forward(d_ids.ptr, 1, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: Project back through LM head using the SAME weight
    cuinfer::LMHead lm_head(vocab_size, dim);
    cuinfer::Tensor logits = lm_head.forward(hidden, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // The logit for token 42 should be the highest (or close to it),
    // because hidden = weight[42], and logits[j] = dot(weight[42], weight[j]).
    // The dot product of a row with itself (j=42) gives ||weight[42]||^2,
    // which should be the maximum (by Cauchy-Schwarz, unless another row
    // is a scalar multiple with larger norm). We verify token 42 has
    // a high logit — not necessarily the absolute max due to random weights,
    // but it should be among the top few.
    auto gpu_logits = download_floats(logits.data_as<float>(), vocab_size);

    // Compute the CPU reference for the self-dot product
    float self_dot = 0.0f;
    for (int k = 0; k < dim; ++k) {
        float w = h_weight[42 * dim + k];
        self_dot += w * w;
    }

    // The logit at position 42 should match the self-dot product
    EXPECT_NEAR(gpu_logits[42], self_dot, MATMUL_TOL)
        << "Tied weight self-dot product mismatch for token 42";
}

// ════════════════════════════════════════════════════════════════════════
// Argmax Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, Argmax_Basic) {
    // Known maximum at a specific position.
    const int n = 100;
    std::vector<float> h_data(n, 0.0f);
    h_data[73] = 10.0f;  // Maximum at index 73

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t result = cuinfer::argmax_sample(d_data.ptr, n);
    EXPECT_EQ(result, 73) << "Argmax should find index 73";
}

TEST_F(EmbeddingTest, Argmax_FirstElement) {
    // Maximum at the first position.
    const int n = 256;
    std::vector<float> h_data(n, -1.0f);
    h_data[0] = 5.0f;

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t result = cuinfer::argmax_sample(d_data.ptr, n);
    EXPECT_EQ(result, 0) << "Argmax should find index 0";
}

TEST_F(EmbeddingTest, Argmax_LastElement) {
    // Maximum at the last position.
    const int n = 256;
    std::vector<float> h_data(n, -1.0f);
    h_data[n - 1] = 5.0f;

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t result = cuinfer::argmax_sample(d_data.ptr, n);
    EXPECT_EQ(result, n - 1) << "Argmax should find last index";
}

TEST_F(EmbeddingTest, Argmax_NegativeValues) {
    // All negative values — argmax should still find the least negative.
    const int n = 512;
    std::vector<float> h_data(n);
    for (int i = 0; i < n; ++i) {
        h_data[i] = -static_cast<float>(n - i);  // -512, -511, ..., -1
    }
    // Maximum is h_data[511] = -1.0

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t result = cuinfer::argmax_sample(d_data.ptr, n);
    EXPECT_EQ(result, n - 1);
}

TEST_F(EmbeddingTest, Argmax_LargeArray) {
    // Vocab-sized array: tests the stride loop in the kernel.
    const int n = 50000;
    auto h_data = random_floats(n);

    // Plant a known maximum
    h_data[31337] = 1000.0f;

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t result = cuinfer::argmax_sample(d_data.ptr, n);
    EXPECT_EQ(result, 31337) << "Argmax should find index 31337 in large array";
}

TEST_F(EmbeddingTest, Argmax_RandomData) {
    // Compare GPU argmax against CPU argmax on random data.
    const int n = 10000;
    auto h_data = random_floats(n, 123);

    int32_t cpu_result = cpu_argmax(h_data.data(), n);

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t gpu_result = cuinfer::argmax_sample(d_data.ptr, n);
    EXPECT_EQ(gpu_result, cpu_result)
        << "GPU argmax disagrees with CPU argmax";
}

// ════════════════════════════════════════════════════════════════════════
// Top-K Sampling Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, TopK_ResultInTopK) {
    // Verify that the sampled token is actually within the top-k candidates.
    const int n = 1000;
    auto h_data = random_floats(n);

    // Upload to GPU
    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    const int k = 10;
    int32_t sampled = cuinfer::top_k_sample(d_data.ptr, n, k, 1.0f, 42);

    // Find actual top-k indices on CPU
    std::vector<int32_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&h_data](int32_t a, int32_t b) {
                          return h_data[a] > h_data[b];
                      });

    // Check that sampled token is in the top-k set
    bool found = false;
    for (int i = 0; i < k; ++i) {
        if (indices[i] == sampled) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found)
        << "Sampled token " << sampled << " is not in the top-" << k;
}

TEST_F(EmbeddingTest, TopK_LowTemperature) {
    // Very low temperature should behave like argmax.
    const int n = 500;
    auto h_data = random_floats(n, 77);

    // Plant a clear maximum
    h_data[42] = 100.0f;

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    // With very low temperature, softmax becomes very peaked at the max.
    // Multiple runs should all return the argmax.
    for (unsigned int seed = 0; seed < 10; ++seed) {
        int32_t sampled = cuinfer::top_k_sample(d_data.ptr, n, 40, 0.01f, seed);
        EXPECT_EQ(sampled, 42)
            << "Low temperature should produce argmax (seed=" << seed << ")";
    }
}

TEST_F(EmbeddingTest, TopK_K1IsArgmax) {
    // top-k with k=1 should always return the argmax.
    const int n = 200;
    auto h_data = random_floats(n, 99);

    int32_t cpu_max = cpu_argmax(h_data.data(), n);

    GpuFloatBuffer d_data(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_data.ptr, h_data.data(),
                                  n * sizeof(float), cudaMemcpyHostToDevice));

    int32_t sampled = cuinfer::top_k_sample(d_data.ptr, n, /*k=*/1, 1.0f, 42);
    EXPECT_EQ(sampled, cpu_max)
        << "top-k with k=1 should be equivalent to argmax";
}

// ════════════════════════════════════════════════════════════════════════
// End-to-End: Embed -> LM Head -> Argmax
// ════════════════════════════════════════════════════════════════════════

TEST_F(EmbeddingTest, EndToEnd_EmbedProjectArgmax) {
    // Full pipeline: embed tokens, project through LM head, pick argmax.
    // Uses tied weights (same matrix for embedding and LM head).
    const int vocab_size = 128;
    const int dim = 32;

    auto h_weight = random_floats(vocab_size * dim);
    cuinfer::Tensor weight = cuinfer::Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    // Embed a single token
    std::vector<int32_t> h_ids = {7};
    auto d_ids = upload_ints(h_ids);

    cuinfer::Embedding emb(vocab_size, dim);
    cuinfer::Tensor hidden = emb.forward(d_ids.ptr, 1, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Project to logits
    cuinfer::LMHead lm_head(vocab_size, dim);
    cuinfer::Tensor logits = lm_head.forward(hidden, weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Argmax over the logit vector
    int32_t next_token = cuinfer::argmax_sample(
        logits.data_as<float>(), vocab_size);

    // Compute CPU reference for the entire pipeline
    std::vector<float> cpu_hidden(dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_hidden.data(),
                         1, dim);

    std::vector<float> cpu_logits(vocab_size);
    cpu_matmul_transposed_b(cpu_hidden.data(), h_weight.data(),
                            cpu_logits.data(), 1, vocab_size, dim);

    int32_t cpu_next = cpu_argmax(cpu_logits.data(), vocab_size);

    EXPECT_EQ(next_token, cpu_next)
        << "End-to-end pipeline: GPU and CPU should agree on next token";
}
