// ── test_model.cu — Unit Tests for the TransformerModel and its Integration ──
//
// Since TransformerModel requires a GGUF file for construction, these tests
// exercise the individual building blocks that the model wires together,
// using synthetic data throughout. No model file is needed.
//
// Test categories:
//   1. ModelConfig struct — field initialization and consistency
//   2. Weight transposition — CPU reference for the transpose logic
//   3. Embedding -> RMSNorm -> LMHead forward flow
//   4. Residual connection (element-wise add) in-place correctness
//   5. Sampling (argmax and top-k) with known logit distributions
//   6. Weight tying — same Tensor used for Embedding and LMHead

#include <gtest/gtest.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "cuinfer/model.h"
#include "cuinfer/embedding.h"
#include "cuinfer/tensor.h"
#include "cuinfer/cuda_utils.h"
#include "cuinfer/kernels/elementwise.h"

using namespace cuinfer;

// ════════════════════════════════════════════════════════════════════════
// Test Helpers
// ════════════════════════════════════════════════════════════════════════

// RAII GPU buffer for float data.
struct GpuBuf {
    float* ptr = nullptr;
    GpuBuf() = default;
    explicit GpuBuf(size_t n) {
        CUINFER_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float)));
    }
    ~GpuBuf() { if (ptr) cudaFree(ptr); }
    GpuBuf(GpuBuf&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    GpuBuf& operator=(GpuBuf&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }
    GpuBuf(const GpuBuf&) = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;
};

// RAII GPU buffer for int32 data (token IDs).
struct GpuIntBuf {
    int32_t* ptr = nullptr;
    GpuIntBuf() = default;
    explicit GpuIntBuf(size_t n) {
        CUINFER_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(int32_t)));
    }
    ~GpuIntBuf() { if (ptr) cudaFree(ptr); }
    GpuIntBuf(GpuIntBuf&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    GpuIntBuf& operator=(GpuIntBuf&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }
    GpuIntBuf(const GpuIntBuf&) = delete;
    GpuIntBuf& operator=(const GpuIntBuf&) = delete;
};

static GpuBuf upload(const std::vector<float>& h) {
    GpuBuf buf(h.size());
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.ptr, h.data(),
                                   h.size() * sizeof(float),
                                   cudaMemcpyHostToDevice));
    return buf;
}

static GpuIntBuf upload_ints(const std::vector<int32_t>& h) {
    GpuIntBuf buf(h.size());
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.ptr, h.data(),
                                   h.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice));
    return buf;
}

static std::vector<float> download(const float* d_ptr, size_t n) {
    std::vector<float> h(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(h.data(), d_ptr,
                                   n * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<float> random_floats(int n, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) {
        v[i] = lo + (hi - lo) * (static_cast<float>(rand()) / RAND_MAX);
    }
    return v;
}

// CPU reference: matmul with transposed B. C[i,j] = sum_k A[i,k] * B[j,k].
// A is [M, K], B is [N, K], C is [M, N].
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

// CPU reference: RMSNorm per row.
// output[i] = input[i] * weight[i] * rsqrt(mean(input^2) + eps)
static void cpu_rmsnorm(const float* input, const float* weight, float* output,
                        int rows, int n, float eps = 1e-5f) {
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int j = 0; j < n; ++j) {
            float val = input[r * n + j];
            ss += val * val;
        }
        float rms = std::sqrt(ss / static_cast<float>(n) + eps);
        float inv_rms = 1.0f / rms;
        for (int j = 0; j < n; ++j) {
            output[r * n + j] = input[r * n + j] * weight[j] * inv_rms;
        }
    }
}

// CPU reference: argmax.
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

// CPU reference: embedding lookup.
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

// Tolerances.
constexpr float TOL = 1e-5f;
constexpr float MATMUL_TOL = 1e-3f;
constexpr float RMSNORM_TOL = 1e-4f;

// ════════════════════════════════════════════════════════════════════════
// Test Fixture
// ════════════════════════════════════════════════════════════════════════

class TransformerModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(42);
    }
};

// ════════════════════════════════════════════════════════════════════════
// 1. ModelConfig Struct Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, ModelConfig_DefaultInitialization) {
    // Verify that ModelConfig fields can be initialized and read back.
    ModelConfig config;
    config.architecture = "llama";
    config.n_layers = 30;
    config.dim = 576;
    config.n_heads = 9;
    config.n_kv_heads = 3;
    config.head_dim = 64;
    config.intermediate_size = 1536;
    config.vocab_size = 49152;
    config.max_seq_len = 2048;
    config.rms_norm_eps = 1e-5f;
    config.rope_theta = 100000.0f;
    config.tie_embeddings = true;

    EXPECT_EQ(config.architecture, "llama");
    EXPECT_EQ(config.n_layers, 30);
    EXPECT_EQ(config.dim, 576);
    EXPECT_EQ(config.n_heads, 9);
    EXPECT_EQ(config.n_kv_heads, 3);
    EXPECT_EQ(config.head_dim, 64);
    EXPECT_EQ(config.intermediate_size, 1536);
    EXPECT_EQ(config.vocab_size, 49152);
    EXPECT_EQ(config.max_seq_len, 2048);
    EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-5f);
    EXPECT_FLOAT_EQ(config.rope_theta, 100000.0f);
    EXPECT_TRUE(config.tie_embeddings);
}

TEST_F(TransformerModelTest, ModelConfig_SmolLM2_135M) {
    // Verify config matches the SmolLM2-135M architecture.
    ModelConfig config;
    config.architecture = "llama";
    config.n_layers = 30;
    config.dim = 576;
    config.n_heads = 9;
    config.n_kv_heads = 3;
    config.head_dim = 64;
    config.intermediate_size = 1536;
    config.vocab_size = 49152;
    config.max_seq_len = 2048;
    config.rms_norm_eps = 1e-5f;
    config.rope_theta = 100000.0f;
    config.tie_embeddings = true;

    // Verify derived relationships.
    EXPECT_EQ(config.dim, config.n_heads * config.head_dim)
        << "dim must equal n_heads * head_dim";
    EXPECT_EQ(config.n_heads % config.n_kv_heads, 0)
        << "n_heads must be divisible by n_kv_heads for GQA";
    EXPECT_EQ(config.n_heads / config.n_kv_heads, 3)
        << "GQA ratio should be 3:1 for SmolLM2-135M";
}

TEST_F(TransformerModelTest, ModelConfig_GQA_Ratio) {
    // Test various GQA ratios are representable.
    // MHA: n_heads == n_kv_heads
    ModelConfig mha;
    mha.n_heads = 8;
    mha.n_kv_heads = 8;
    EXPECT_EQ(mha.n_heads / mha.n_kv_heads, 1)
        << "MHA should have 1:1 ratio";

    // GQA 4:1
    ModelConfig gqa;
    gqa.n_heads = 32;
    gqa.n_kv_heads = 8;
    EXPECT_EQ(gqa.n_heads / gqa.n_kv_heads, 4)
        << "GQA should have 4:1 ratio";

    // MQA: n_kv_heads == 1
    ModelConfig mqa;
    mqa.n_heads = 16;
    mqa.n_kv_heads = 1;
    EXPECT_EQ(mqa.n_heads / mqa.n_kv_heads, 16)
        << "MQA should have n_heads:1 ratio";
}

// ════════════════════════════════════════════════════════════════════════
// 2. Weight Transposition Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, WeightTranspose_Small) {
    // Verify the transpose concept that load_weight_transposed uses.
    // GGUF stores weights as [out_features, in_features].
    // For matmul_tiled (C = A * B where B is [K, N]),
    // we need [in_features, out_features] = transposed.
    //
    // Test: create a known 3x2 matrix and transpose it to 2x3.
    const int rows = 3, cols = 2;

    // Original matrix [3, 2]:
    //   [ 1, 2 ]
    //   [ 3, 4 ]
    //   [ 5, 6 ]
    std::vector<float> original = {1, 2, 3, 4, 5, 6};

    // Expected transpose [2, 3]:
    //   [ 1, 3, 5 ]
    //   [ 2, 4, 6 ]
    std::vector<float> expected_transposed = {1, 3, 5, 2, 4, 6};

    // CPU transpose
    std::vector<float> transposed(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j * rows + i] = original[i * cols + j];
        }
    }

    ASSERT_EQ(transposed.size(), expected_transposed.size());
    for (size_t i = 0; i < transposed.size(); ++i) {
        EXPECT_FLOAT_EQ(transposed[i], expected_transposed[i])
            << "Transpose mismatch at index " << i;
    }
}

TEST_F(TransformerModelTest, WeightTranspose_Square) {
    // Square matrix transpose: A^T^T == A.
    const int n = 4;
    std::vector<float> original(n * n);
    for (int i = 0; i < n * n; ++i) {
        original[i] = static_cast<float>(i + 1);
    }

    // Transpose
    std::vector<float> transposed(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[j * n + i] = original[i * n + j];
        }
    }

    // Double transpose should recover the original.
    std::vector<float> double_transposed(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double_transposed[j * n + i] = transposed[i * n + j];
        }
    }

    for (int i = 0; i < n * n; ++i) {
        EXPECT_FLOAT_EQ(double_transposed[i], original[i])
            << "Double transpose should recover original at index " << i;
    }
}

TEST_F(TransformerModelTest, WeightTranspose_MatmulEquivalence) {
    // Verify that multiplying with a transposed weight gives the same
    // result as the standard matmul_transposed_b with the original weight.
    //
    // If W is [out, in] (GGUF format), then:
    //   hidden [1, in] * W^T [in, out] = logits [1, out]
    // is the same as:
    //   matmul_transposed_b(hidden, W) where W is [out, in]
    //   because matmul_transposed_b computes C[i,j] = sum_k A[i,k] * B[j,k]
    //
    // This test verifies the equivalence.

    const int in_features = 32;
    const int out_features = 16;
    const int seq_len = 1;

    auto h_hidden = random_floats(seq_len * in_features, -0.5f, 0.5f);
    auto h_weight = random_floats(out_features * in_features, -0.5f, 0.5f);

    // Method 1: CPU matmul_transposed_b (using GGUF-style [out, in] weight)
    std::vector<float> result_transb(seq_len * out_features);
    cpu_matmul_transposed_b(h_hidden.data(), h_weight.data(),
                            result_transb.data(),
                            seq_len, out_features, in_features);

    // Method 2: Standard matmul with an explicitly transposed weight
    // Transpose W from [out, in] to [in, out]
    std::vector<float> h_weight_T(in_features * out_features);
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            h_weight_T[j * out_features + i] = h_weight[i * in_features + j];
        }
    }

    // C = hidden [1, in] * W_T [in, out] = [1, out]
    std::vector<float> result_standard(seq_len * out_features);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += h_hidden[i * in_features + k] *
                       h_weight_T[k * out_features + j];
            }
            result_standard[i * out_features + j] = sum;
        }
    }

    for (int i = 0; i < seq_len * out_features; ++i) {
        EXPECT_NEAR(result_transb[i], result_standard[i], 1e-5f)
            << "Transposed-B and standard matmul should give identical results at "
            << i;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 3. Forward Pass Components — Integration-Style Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, EmbedToRMSNorm_Flow) {
    // Tests the first two operations in the transformer forward pass:
    //   1. Embedding lookup: token IDs -> dense vectors
    //   2. RMSNorm: normalize before attention
    //
    // Verifies that the pipeline produces correct results when composed.

    const int vocab_size = 128;
    const int dim = 64;
    const int seq_len = 4;
    const float eps = 1e-5f;

    auto h_weight = random_floats(vocab_size * dim, -0.5f, 0.5f);
    auto h_norm_weight = random_floats(dim, 0.5f, 1.5f); // RMSNorm weights are typically positive

    std::vector<int32_t> h_ids = {5, 20, 42, 100};

    // CPU reference: embed -> rmsnorm
    std::vector<float> cpu_embedded(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_embedded.data(),
                         seq_len, dim);

    std::vector<float> cpu_normed(seq_len * dim);
    cpu_rmsnorm(cpu_embedded.data(), h_norm_weight.data(), cpu_normed.data(),
                seq_len, dim, eps);

    // GPU pipeline
    Tensor t_weight = Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});
    auto d_ids = upload_ints(h_ids);

    Embedding emb(vocab_size, dim);
    Tensor t_embedded = emb.forward(d_ids.ptr, seq_len, t_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // RMSNorm
    auto d_norm_weight = upload(h_norm_weight);
    GpuBuf d_normed(seq_len * dim);
    rmsnorm(t_embedded.data_as<float>(), d_norm_weight.ptr, d_normed.ptr,
            seq_len, dim, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_normed = download(d_normed.ptr, seq_len * dim);

    for (int i = 0; i < seq_len * dim; ++i) {
        EXPECT_NEAR(gpu_normed[i], cpu_normed[i], RMSNORM_TOL)
            << "Embed->RMSNorm mismatch at index " << i
            << " (token=" << i / dim << ", dim=" << i % dim << ")";
    }
}

TEST_F(TransformerModelTest, EmbedNormLMHead_FullForwardFlow) {
    // Tests a simplified forward flow: embed -> rmsnorm -> lm_head -> argmax.
    // This mimics a single-layer model without attention or FFN.

    const int vocab_size = 64;
    const int dim = 32;
    const int seq_len = 1;
    const float eps = 1e-5f;

    auto h_emb_weight = random_floats(vocab_size * dim, -0.5f, 0.5f);
    auto h_norm_weight = random_floats(dim, 0.8f, 1.2f);
    // Use separate LM head weight (non-tied) for this test.
    auto h_lm_weight = random_floats(vocab_size * dim, -0.5f, 0.5f);

    std::vector<int32_t> h_ids = {7};

    // ── CPU Reference ──
    // Step 1: Embed
    std::vector<float> cpu_embedded(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_emb_weight.data(), cpu_embedded.data(),
                         seq_len, dim);

    // Step 2: RMSNorm
    std::vector<float> cpu_normed(seq_len * dim);
    cpu_rmsnorm(cpu_embedded.data(), h_norm_weight.data(), cpu_normed.data(),
                seq_len, dim, eps);

    // Step 3: LM Head (matmul_transposed_b)
    std::vector<float> cpu_logits(seq_len * vocab_size);
    cpu_matmul_transposed_b(cpu_normed.data(), h_lm_weight.data(),
                            cpu_logits.data(), seq_len, vocab_size, dim);

    // Step 4: Argmax
    int32_t cpu_next_token = cpu_argmax(cpu_logits.data(), vocab_size);

    // ── GPU Pipeline ──
    Tensor t_emb_weight = Tensor::from_host(
        h_emb_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});
    Tensor t_lm_weight = Tensor::from_host(
        h_lm_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    auto d_ids = upload_ints(h_ids);
    auto d_norm_weight = upload(h_norm_weight);

    // Step 1: Embed
    Embedding emb(vocab_size, dim);
    Tensor t_embedded = emb.forward(d_ids.ptr, seq_len, t_emb_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: RMSNorm
    GpuBuf d_normed(seq_len * dim);
    rmsnorm(t_embedded.data_as<float>(), d_norm_weight.ptr, d_normed.ptr,
            seq_len, dim, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Step 3: LM Head
    Tensor t_normed_input = Tensor::from_host(
        download(d_normed.ptr, seq_len * dim).data(),
        {static_cast<int64_t>(seq_len), static_cast<int64_t>(dim)});

    LMHead lm_head(vocab_size, dim);
    Tensor t_logits = lm_head.forward(t_normed_input, t_lm_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Step 4: Argmax
    int32_t gpu_next_token = argmax_sample(t_logits.data_as<float>(), vocab_size);

    EXPECT_EQ(gpu_next_token, cpu_next_token)
        << "Full forward flow should produce the same next token on GPU and CPU";

    // Also verify logit values match.
    auto gpu_logits = download(t_logits.data_as<float>(), seq_len * vocab_size);
    for (int i = 0; i < seq_len * vocab_size; ++i) {
        EXPECT_NEAR(gpu_logits[i], cpu_logits[i], MATMUL_TOL)
            << "Logit mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 4. Residual Connection Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, ResidualAdd_Basic) {
    // In the transformer forward pass, residual connections add the
    // block output back to the input: x = x + block(x).
    // Verify element-wise add works correctly.

    const int n = 256;
    auto h_a = random_floats(n, -1.0f, 1.0f);
    auto h_b = random_floats(n, -1.0f, 1.0f);

    // CPU reference
    std::vector<float> cpu_result(n);
    for (int i = 0; i < n; ++i) {
        cpu_result[i] = h_a[i] + h_b[i];
    }

    // GPU
    auto d_a = upload(h_a);
    auto d_b = upload(h_b);
    GpuBuf d_output(n);

    add(d_a.ptr, d_b.ptr, d_output.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_result = download(d_output.ptr, n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], TOL)
            << "Residual add mismatch at index " << i;
    }
}

TEST_F(TransformerModelTest, ResidualAdd_InPlace) {
    // In the model's forward pass, residual connections are computed as:
    //   add(residual, block_output, residual, n)
    // where the output overwrites the residual buffer. Verify this
    // in-place pattern works correctly.

    const int seq_len = 4;
    const int dim = 64;
    const int n = seq_len * dim;

    auto h_residual = random_floats(n, -1.0f, 1.0f);
    auto h_block_output = random_floats(n, -0.5f, 0.5f);

    // CPU reference: residual += block_output
    std::vector<float> cpu_result(n);
    for (int i = 0; i < n; ++i) {
        cpu_result[i] = h_residual[i] + h_block_output[i];
    }

    // GPU: add in-place (output pointer == first input pointer)
    auto d_residual = upload(h_residual);
    auto d_block_output = upload(h_block_output);

    add(d_residual.ptr, d_block_output.ptr, d_residual.ptr, n);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_result = download(d_residual.ptr, n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], TOL)
            << "In-place residual add mismatch at index " << i;
    }
}

TEST_F(TransformerModelTest, ResidualAdd_ChainedResiduals) {
    // Simulate two transformer blocks with residual connections:
    //   x = x + block1(x)
    //   x = x + block2(x)
    // The key property: residuals accumulate, they don't replace.

    const int n = 128;

    // Initial hidden state (e.g., from embedding).
    auto h_x = random_floats(n, -1.0f, 1.0f);
    auto h_block1_out = random_floats(n, -0.1f, 0.1f);
    auto h_block2_out = random_floats(n, -0.1f, 0.1f);

    // CPU reference
    std::vector<float> cpu_x(h_x);
    for (int i = 0; i < n; ++i) {
        cpu_x[i] += h_block1_out[i];  // After block 1
    }
    for (int i = 0; i < n; ++i) {
        cpu_x[i] += h_block2_out[i];  // After block 2
    }

    // GPU: chain two in-place adds
    auto d_x = upload(h_x);
    auto d_b1 = upload(h_block1_out);
    auto d_b2 = upload(h_block2_out);

    add(d_x.ptr, d_b1.ptr, d_x.ptr, n);  // x += block1
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    add(d_x.ptr, d_b2.ptr, d_x.ptr, n);  // x += block2
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_x = download(d_x.ptr, n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_x[i], cpu_x[i], TOL)
            << "Chained residual mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 5. Sampling Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, ArgmaxSample_ClearMaximum) {
    // A logit distribution with one value clearly dominating.
    const int vocab_size = 256;
    std::vector<float> h_logits(vocab_size, -10.0f);
    h_logits[42] = 100.0f;  // Clear winner

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    int32_t result = argmax_sample(d_logits.ptr, vocab_size);
    EXPECT_EQ(result, 42) << "Argmax should return index 42";
}

TEST_F(TransformerModelTest, ArgmaxSample_AllEqual) {
    // All logits equal: argmax should return the first index (index 0).
    const int vocab_size = 100;
    std::vector<float> h_logits(vocab_size, 5.0f);

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    int32_t result = argmax_sample(d_logits.ptr, vocab_size);
    EXPECT_EQ(result, 0) << "When all logits are equal, argmax should return 0";
}

TEST_F(TransformerModelTest, ArgmaxSample_LargeVocab) {
    // Simulates a realistic vocab size with a known maximum.
    const int vocab_size = 49152;  // SmolLM2-135M vocab
    std::vector<float> h_logits(vocab_size);
    // Create a gradual distribution with a peak at a specific position.
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = -5.0f + 0.0001f * static_cast<float>(i);
    }
    // Plant a clear maximum
    const int expected_max = 12345;
    h_logits[expected_max] = 50.0f;

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    int32_t result = argmax_sample(d_logits.ptr, vocab_size);
    EXPECT_EQ(result, expected_max);
}

TEST_F(TransformerModelTest, TopKSample_TemperatureZero_BehavesLikeArgmax) {
    // top_k_sample with near-zero temperature should pick the argmax.
    const int vocab_size = 500;
    auto h_logits = random_floats(vocab_size, -2.0f, 2.0f);

    // Plant a clear maximum
    h_logits[123] = 50.0f;

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    // Very low temperature makes softmax peaked at the max.
    for (unsigned int seed = 0; seed < 10; ++seed) {
        int32_t result = top_k_sample(d_logits.ptr, vocab_size,
                                       /*k=*/40, /*temperature=*/0.01f, seed);
        EXPECT_EQ(result, 123)
            << "top_k with near-zero temperature should behave like argmax "
            << "(seed=" << seed << ")";
    }
}

TEST_F(TransformerModelTest, TopKSample_K1_IsArgmax) {
    // top-k with k=1 should always return the argmax regardless of temp.
    const int vocab_size = 200;
    auto h_logits = random_floats(vocab_size, -1.0f, 1.0f);

    int32_t cpu_max = cpu_argmax(h_logits.data(), vocab_size);

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    int32_t result = top_k_sample(d_logits.ptr, vocab_size,
                                   /*k=*/1, /*temperature=*/1.0f, /*seed=*/42);
    EXPECT_EQ(result, cpu_max)
        << "top_k with k=1 must be equivalent to argmax";
}

TEST_F(TransformerModelTest, TopKSample_ResultInTopK) {
    // Verify the sampled token is actually among the top-k candidates.
    const int vocab_size = 1000;
    const int k = 10;
    auto h_logits = random_floats(vocab_size, -3.0f, 3.0f);

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    // Run several seeds to test the sampling distribution.
    for (unsigned int seed = 0; seed < 20; ++seed) {
        int32_t sampled = top_k_sample(d_logits.ptr, vocab_size,
                                        k, 1.0f, seed);

        // Find actual top-k indices on CPU.
        std::vector<int32_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&h_logits](int32_t a, int32_t b) {
                              return h_logits[a] > h_logits[b];
                          });

        bool found = false;
        for (int i = 0; i < k; ++i) {
            if (indices[i] == sampled) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found)
            << "Sampled token " << sampled << " is not in the top-" << k
            << " (seed=" << seed << ")";
    }
}

TEST_F(TransformerModelTest, TopKSample_HighTemperature_MoreVariety) {
    // With high temperature, sampling should produce varied results.
    // With low temperature, results should be more concentrated.
    const int vocab_size = 100;
    const int k = 10;

    // Create logits with a few strong candidates.
    std::vector<float> h_logits(vocab_size, 0.0f);
    h_logits[10] = 5.0f;
    h_logits[20] = 4.8f;
    h_logits[30] = 4.5f;
    h_logits[40] = 4.0f;
    h_logits[50] = 3.5f;

    GpuBuf d_logits(vocab_size);
    CUINFER_CHECK_CUDA(cudaMemcpy(d_logits.ptr, h_logits.data(),
                                   vocab_size * sizeof(float),
                                   cudaMemcpyHostToDevice));

    // Low temperature: should mostly return index 10 (the maximum).
    int low_temp_top_count = 0;
    for (unsigned int seed = 0; seed < 50; ++seed) {
        int32_t result = top_k_sample(d_logits.ptr, vocab_size,
                                       k, 0.1f, seed);
        if (result == 10) ++low_temp_top_count;
    }

    // High temperature: should produce more variety.
    std::set<int32_t> high_temp_unique;
    for (unsigned int seed = 0; seed < 50; ++seed) {
        int32_t result = top_k_sample(d_logits.ptr, vocab_size,
                                       k, 5.0f, seed);
        high_temp_unique.insert(result);
    }

    // Low temperature should strongly favor the top token.
    EXPECT_GT(low_temp_top_count, 40)
        << "Low temperature should produce the argmax most of the time";

    // High temperature should produce at least a few different tokens.
    EXPECT_GT(static_cast<int>(high_temp_unique.size()), 1)
        << "High temperature should produce some variety";
}

// ════════════════════════════════════════════════════════════════════════
// 6. Weight Tying Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, WeightTying_SameTensorForEmbedAndLMHead) {
    // In many LLMs (Llama, SmolLM2), the token embedding weight and the
    // LM head weight are the same tensor (tied weights). This test verifies
    // that passing the same Tensor to both Embedding::forward and
    // LMHead::forward works correctly.

    const int vocab_size = 64;
    const int dim = 32;

    auto h_weight = random_floats(vocab_size * dim, -0.5f, 0.5f);
    Tensor shared_weight = Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    // Step 1: Embed token 7 using the shared weight.
    std::vector<int32_t> h_ids = {7};
    auto d_ids = upload_ints(h_ids);

    Embedding emb(vocab_size, dim);
    Tensor hidden = emb.forward(d_ids.ptr, 1, shared_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: Project through LM head using the SAME shared weight.
    LMHead lm_head(vocab_size, dim);
    Tensor logits = lm_head.forward(hidden, shared_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Step 3: Verify correctness.
    // hidden = weight[7] (the 7th row of the embedding matrix).
    // logits[j] = dot(hidden, weight[j]) = dot(weight[7], weight[j]).
    // The self-dot product (j=7) should equal ||weight[7]||^2.
    auto gpu_logits = download(logits.data_as<float>(), vocab_size);

    float self_dot = 0.0f;
    for (int k = 0; k < dim; ++k) {
        float w = h_weight[7 * dim + k];
        self_dot += w * w;
    }

    EXPECT_NEAR(gpu_logits[7], self_dot, MATMUL_TOL)
        << "Tied weight self-dot product should equal ||weight[7]||^2";
}

TEST_F(TransformerModelTest, WeightTying_ArgmaxMatchesCPU) {
    // With tied weights, the full pipeline embed -> lm_head -> argmax
    // should agree between GPU and CPU.

    const int vocab_size = 128;
    const int dim = 64;

    auto h_weight = random_floats(vocab_size * dim, -0.3f, 0.3f);

    // CPU pipeline
    std::vector<int32_t> h_ids = {15};
    std::vector<float> cpu_hidden(dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_hidden.data(), 1, dim);

    std::vector<float> cpu_logits(vocab_size);
    cpu_matmul_transposed_b(cpu_hidden.data(), h_weight.data(),
                            cpu_logits.data(), 1, vocab_size, dim);
    int32_t cpu_next = cpu_argmax(cpu_logits.data(), vocab_size);

    // GPU pipeline with tied weights
    Tensor shared_weight = Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});
    auto d_ids = upload_ints(h_ids);

    Embedding emb(vocab_size, dim);
    Tensor hidden = emb.forward(d_ids.ptr, 1, shared_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    LMHead lm_head(vocab_size, dim);
    Tensor logits = lm_head.forward(hidden, shared_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    int32_t gpu_next = argmax_sample(logits.data_as<float>(), vocab_size);

    EXPECT_EQ(gpu_next, cpu_next)
        << "Tied-weight pipeline: GPU and CPU should agree on next token "
        << "(GPU=" << gpu_next << ", CPU=" << cpu_next << ")";
}

TEST_F(TransformerModelTest, WeightTying_SharedWeightNotCorrupted) {
    // Verify that using the same Tensor for embedding lookup followed by
    // LM head projection does not corrupt the weight data. The weight
    // tensor is read-only in both operations, so it must remain unchanged.

    const int vocab_size = 64;
    const int dim = 32;

    auto h_weight = random_floats(vocab_size * dim, -0.5f, 0.5f);
    Tensor shared_weight = Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});

    // Snapshot the weight data before operations.
    auto weight_before = download(shared_weight.data_as<float>(),
                                   vocab_size * dim);

    // Run embedding lookup.
    std::vector<int32_t> h_ids = {0, 10, 20, 30};
    auto d_ids = upload_ints(h_ids);
    Embedding emb(vocab_size, dim);
    Tensor hidden = emb.forward(d_ids.ptr, 4, shared_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Run LM head projection.
    LMHead lm_head(vocab_size, dim);
    Tensor logits = lm_head.forward(hidden, shared_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Snapshot the weight data after operations.
    auto weight_after = download(shared_weight.data_as<float>(),
                                  vocab_size * dim);

    // Weight data must be identical before and after.
    for (int i = 0; i < vocab_size * dim; ++i) {
        EXPECT_FLOAT_EQ(weight_before[i], weight_after[i])
            << "Shared weight corrupted at index " << i
            << " after embedding + LM head operations";
    }
}

// ════════════════════════════════════════════════════════════════════════
// Additional Integration Tests
// ════════════════════════════════════════════════════════════════════════

TEST_F(TransformerModelTest, RMSNorm_PreservesDirection) {
    // RMSNorm normalizes the magnitude of a vector but preserves its
    // direction (scaled by learned weights). With all norm weights = 1,
    // the output should have the same direction as input.

    const int dim = 64;
    const int rows = 1;
    const float eps = 1e-5f;

    auto h_input = random_floats(dim, -2.0f, 2.0f);
    std::vector<float> h_norm_weight(dim, 1.0f); // All ones

    // CPU reference
    std::vector<float> cpu_output(dim);
    cpu_rmsnorm(h_input.data(), h_norm_weight.data(), cpu_output.data(),
                rows, dim, eps);

    // Verify direction: normalized output should be proportional to input.
    // Find the ratio output/input for the first nonzero element,
    // then verify all other elements have the same ratio.
    float ratio = 0.0f;
    bool found_nonzero = false;
    for (int i = 0; i < dim; ++i) {
        if (std::abs(h_input[i]) > 1e-7f) {
            if (!found_nonzero) {
                ratio = cpu_output[i] / h_input[i];
                found_nonzero = true;
            } else {
                float this_ratio = cpu_output[i] / h_input[i];
                EXPECT_NEAR(this_ratio, ratio, 1e-5f)
                    << "RMSNorm with unit weights should preserve direction "
                    << "(index " << i << ")";
            }
        }
    }
    EXPECT_TRUE(found_nonzero) << "Need at least one nonzero input";
}

TEST_F(TransformerModelTest, MultiToken_EmbedNormProjectPipeline) {
    // Test the pipeline with multiple tokens (prefill scenario).
    // embed -> rmsnorm -> lm_head should produce [seq_len, vocab_size] logits.

    const int vocab_size = 32;
    const int dim = 16;
    const int seq_len = 3;
    const float eps = 1e-5f;

    auto h_weight = random_floats(vocab_size * dim, -0.5f, 0.5f);
    auto h_norm_weight = random_floats(dim, 0.8f, 1.2f);
    std::vector<int32_t> h_ids = {1, 5, 10};

    // CPU reference
    std::vector<float> cpu_embedded(seq_len * dim);
    cpu_embedding_lookup(h_ids.data(), h_weight.data(), cpu_embedded.data(),
                         seq_len, dim);

    std::vector<float> cpu_normed(seq_len * dim);
    cpu_rmsnorm(cpu_embedded.data(), h_norm_weight.data(), cpu_normed.data(),
                seq_len, dim, eps);

    std::vector<float> cpu_logits(seq_len * vocab_size);
    cpu_matmul_transposed_b(cpu_normed.data(), h_weight.data(),
                            cpu_logits.data(), seq_len, vocab_size, dim);

    // GPU pipeline
    Tensor t_weight = Tensor::from_host(
        h_weight.data(),
        {static_cast<int64_t>(vocab_size), static_cast<int64_t>(dim)});
    auto d_ids = upload_ints(h_ids);
    auto d_norm_weight = upload(h_norm_weight);

    Embedding emb(vocab_size, dim);
    Tensor t_embedded = emb.forward(d_ids.ptr, seq_len, t_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // RMSNorm
    GpuBuf d_normed(seq_len * dim);
    rmsnorm(t_embedded.data_as<float>(), d_norm_weight.ptr, d_normed.ptr,
            seq_len, dim, eps);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // LM Head
    // Build a Tensor from the normalized output for the LMHead.
    auto h_normed_gpu = download(d_normed.ptr, seq_len * dim);
    Tensor t_normed = Tensor::from_host(
        h_normed_gpu.data(),
        {static_cast<int64_t>(seq_len), static_cast<int64_t>(dim)});

    LMHead lm_head(vocab_size, dim);
    Tensor t_logits = lm_head.forward(t_normed, t_weight);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Verify shape: [seq_len, vocab_size]
    EXPECT_EQ(t_logits.shape(2), seq_len);
    EXPECT_EQ(t_logits.shape(3), vocab_size);

    // Verify values
    auto gpu_logits = download(t_logits.data_as<float>(), seq_len * vocab_size);
    for (int i = 0; i < seq_len * vocab_size; ++i) {
        EXPECT_NEAR(gpu_logits[i], cpu_logits[i], MATMUL_TOL)
            << "Multi-token logit mismatch at flat index " << i
            << " (token=" << i / vocab_size << ", vocab=" << i % vocab_size << ")";
    }

    // The argmax of the last token's logits should match CPU.
    int cpu_last_argmax = cpu_argmax(
        cpu_logits.data() + (seq_len - 1) * vocab_size, vocab_size);
    int gpu_last_argmax = argmax_sample(
        t_logits.data_as<float>() + (seq_len - 1) * vocab_size, vocab_size);
    EXPECT_EQ(gpu_last_argmax, cpu_last_argmax)
        << "Last-token argmax should match between GPU and CPU";
}
