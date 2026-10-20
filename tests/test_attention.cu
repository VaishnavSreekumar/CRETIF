// ── test_attention.cu — Unit Tests for the Attention Block ──────────────
//
// Tests the attention block and its constituent kernels against CPU
// reference implementations. All tests use synthetic data with fixed
// seeds for reproducibility.
//
// Test categories:
//   1. Individual kernel tests (causal_mask, scale, batched_matmul)
//   2. Single-head attention end-to-end
//   3. Multi-head attention (MHA) end-to-end
//   4. Grouped-query attention (GQA) end-to-end
//   5. Causal masking verification (future tokens cannot attend)
//   6. KV-cache correctness across multiple forward calls

#include <gtest/gtest.h>
#include "cuinfer/attention.h"
#include "cuinfer/cuda_utils.h"
#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/kernels/matmul.h"

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

using namespace cuinfer;

// ════════════════════════════════════════════════════════════════
// Test Helpers
// ════════════════════════════════════════════════════════════════

// RAII GPU buffer for test convenience.
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

static GpuBuf upload(const std::vector<float>& h) {
    GpuBuf buf(h.size());
    CUINFER_CHECK_CUDA(cudaMemcpy(buf.ptr, h.data(),
                                   h.size() * sizeof(float),
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

// CPU matmul: C = A * B, A [M,K], B [K,N], C [M,N].
static void cpu_matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// CPU RoPE (in-place), same as in test_elementwise.cu.
static void cpu_rope(float* qk, int pos, int dim, int n_heads,
                     float theta_base = 10000.0f) {
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

// CPU softmax (per-row, numerically stable).
static void cpu_softmax(const float* input, float* output, int n) {
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

// ── CPU reference for the full attention forward pass ───────────
// Computes single-position (decode) or multi-position (prefill)
// attention with causal mask and GQA support.
//
// All arrays are in row-major layout.
// Returns the output [seq_len, dim].
static std::vector<float> cpu_attention(
    const float* input,    // [seq_len, dim]
    const float* w_q,      // [dim, dim]
    const float* w_k,      // [dim, kv_dim]
    const float* w_v,      // [dim, kv_dim]
    const float* w_o,      // [dim, dim]
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    int pos,
    // The full K/V cache up to and including this call's tokens.
    // k_cache/v_cache: [kv_len, n_kv_heads, head_dim]
    // where kv_len = pos + seq_len.
    float* k_cache,
    float* v_cache) {

    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int kv_len = pos + seq_len;
    const int n_groups = n_heads / n_kv_heads;

    // Step 1: QKV projection
    std::vector<float> q_proj(seq_len * dim);
    std::vector<float> k_proj(seq_len * kv_dim);
    std::vector<float> v_proj(seq_len * kv_dim);

    cpu_matmul(input, w_q, q_proj.data(), seq_len, dim, dim);
    cpu_matmul(input, w_k, k_proj.data(), seq_len, kv_dim, dim);
    cpu_matmul(input, w_v, v_proj.data(), seq_len, kv_dim, dim);

    // Step 2: RoPE
    for (int t = 0; t < seq_len; t++) {
        cpu_rope(q_proj.data() + t * dim, pos + t, head_dim, n_heads);
        cpu_rope(k_proj.data() + t * kv_dim, pos + t, head_dim, n_kv_heads);
    }

    // Step 3: Update KV cache (copy k_proj/v_proj into cache at position pos).
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < kv_dim; i++) {
            k_cache[(pos + t) * kv_dim + i] = k_proj[t * kv_dim + i];
            v_cache[(pos + t) * kv_dim + i] = v_proj[t * kv_dim + i];
        }
    }

    // Step 4: Scaled dot-product attention with causal mask.
    // For each query head h (0..n_heads-1):
    //   kv_head = h / n_groups
    //   scores[q][k] = sum_d Q[h][q][d] * K[kv_head][k][d] / sqrt(head_dim)
    //   Apply causal mask: scores[q][k] = -inf if k > pos + q
    //   attn_weights = softmax(scores)  [per row]
    //   attn_out[h][q][d] = sum_k attn_weights[q][k] * V[kv_head][k][d]

    std::vector<float> attn_output(n_heads * seq_len * head_dim, 0.0f);
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < n_heads; h++) {
        const int kv_h = h / n_groups;

        for (int q = 0; q < seq_len; q++) {
            // Compute scores for this query position.
            std::vector<float> scores(kv_len);
            for (int k = 0; k < kv_len; k++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    // Q: q_proj[q * dim + h * head_dim + d]
                    // K from cache: k_cache[k * kv_dim + kv_h * head_dim + d]
                    float q_val = q_proj[q * dim + h * head_dim + d];
                    float k_val = k_cache[k * kv_dim + kv_h * head_dim + d];
                    dot += q_val * k_val;
                }
                scores[k] = dot * scale_factor;
            }

            // Causal mask: query at absolute position (pos + q) can attend
            // to keys at positions 0..pos+q.
            for (int k = 0; k < kv_len; k++) {
                if (k > pos + q) {
                    scores[k] = -INFINITY;
                }
            }

            // Softmax
            std::vector<float> attn_w(kv_len);
            cpu_softmax(scores.data(), attn_w.data(), kv_len);

            // Weighted sum of V
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int k = 0; k < kv_len; k++) {
                    val += attn_w[k] *
                           v_cache[k * kv_dim + kv_h * head_dim + d];
                }
                attn_output[(h * seq_len + q) * head_dim + d] = val;
            }
        }
    }

    // Step 5: Transpose from [n_heads, seq_len, head_dim] to [seq_len, dim]
    // and then multiply by W_o.
    std::vector<float> concat(seq_len * dim);
    for (int h = 0; h < n_heads; h++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                concat[s * dim + h * head_dim + d] =
                    attn_output[(h * seq_len + s) * head_dim + d];
            }
        }
    }

    std::vector<float> output(seq_len * dim);
    cpu_matmul(concat.data(), w_o, output.data(), seq_len, dim, dim);

    return output;
}

// Test fixture that seeds the RNG for reproducibility.
class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(42);
    }
};

// ════════════════════════════════════════════════════════════════
// Kernel Tests: Causal Mask
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, CausalMask_Basic) {
    // 1 head, seq_len=3, kv_len=3, start_pos=0.
    // Expected mask (1=keep, 0=mask):
    //   q=0: [1, 0, 0]  (pos 0 can attend to pos 0 only)
    //   q=1: [1, 1, 0]  (pos 1 can attend to pos 0,1)
    //   q=2: [1, 1, 1]  (pos 2 can attend to pos 0,1,2)
    const int n_heads = 1, seq_len = 3, kv_len = 3, start_pos = 0;
    std::vector<float> h_scores(n_heads * seq_len * kv_len, 1.0f);
    auto d_scores = upload(h_scores);

    causal_mask(d_scores.ptr, n_heads, seq_len, kv_len, start_pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = download(d_scores.ptr, h_scores.size());

    // Row 0 (q=0): positions 1,2 should be -inf
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_TRUE(std::isinf(result[1]) && result[1] < 0);
    EXPECT_TRUE(std::isinf(result[2]) && result[2] < 0);

    // Row 1 (q=1): position 2 should be -inf
    EXPECT_FLOAT_EQ(result[3], 1.0f);
    EXPECT_FLOAT_EQ(result[4], 1.0f);
    EXPECT_TRUE(std::isinf(result[5]) && result[5] < 0);

    // Row 2 (q=2): all should be 1.0 (no masking)
    EXPECT_FLOAT_EQ(result[6], 1.0f);
    EXPECT_FLOAT_EQ(result[7], 1.0f);
    EXPECT_FLOAT_EQ(result[8], 1.0f);
}

TEST_F(AttentionTest, CausalMask_WithStartPos) {
    // Simulates decode step: seq_len=1, kv_len=4, start_pos=3.
    // The single query token is at absolute position 3.
    // It can attend to keys 0,1,2,3 — so no masking should occur.
    const int n_heads = 2, seq_len = 1, kv_len = 4, start_pos = 3;
    const int total = n_heads * seq_len * kv_len;
    std::vector<float> h_scores(total, 1.0f);
    auto d_scores = upload(h_scores);

    causal_mask(d_scores.ptr, n_heads, seq_len, kv_len, start_pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = download(d_scores.ptr, total);

    // All positions should remain unmasked.
    for (int i = 0; i < total; i++) {
        EXPECT_FLOAT_EQ(result[i], 1.0f)
            << "Unexpected mask at index " << i;
    }
}

TEST_F(AttentionTest, CausalMask_PrefillWithOffset) {
    // Prefill: seq_len=2, kv_len=5, start_pos=3.
    // Query 0 is at absolute position 3, query 1 at position 4.
    // q=0 can attend to keys 0..3 (mask key 4)
    // q=1 can attend to keys 0..4 (no masking)
    const int n_heads = 1, seq_len = 2, kv_len = 5, start_pos = 3;
    const int total = n_heads * seq_len * kv_len;
    std::vector<float> h_scores(total, 1.0f);
    auto d_scores = upload(h_scores);

    causal_mask(d_scores.ptr, n_heads, seq_len, kv_len, start_pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = download(d_scores.ptr, total);

    // Row 0 (q=0, abs_pos=3): keys 0..3 OK, key 4 masked
    for (int k = 0; k < 4; k++) {
        EXPECT_FLOAT_EQ(result[k], 1.0f) << "k=" << k;
    }
    EXPECT_TRUE(std::isinf(result[4]) && result[4] < 0);

    // Row 1 (q=1, abs_pos=4): all keys 0..4 OK
    for (int k = 0; k < 5; k++) {
        EXPECT_FLOAT_EQ(result[5 + k], 1.0f) << "k=" << k;
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel Tests: Scale
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, Scale_Basic) {
    const int n = 256;
    std::vector<float> h_data(n);
    for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i + 1);

    auto d_data = upload(h_data);
    const float factor = 0.125f; // 1/sqrt(64)

    scale(d_data.ptr, n, factor);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = download(d_data.ptr, n);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(result[i], h_data[i] * factor, 1e-6f)
            << "Mismatch at index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel Tests: Batched MatMul
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, BatchedMatmul_Basic) {
    const int batch = 4, M = 2, N = 3, K = 4;
    auto h_A = random_floats(batch * M * K);
    auto h_B = random_floats(batch * K * N);

    // CPU reference
    std::vector<float> h_C_ref(batch * M * N, 0.0f);
    for (int b = 0; b < batch; b++) {
        cpu_matmul(h_A.data() + b * M * K,
                   h_B.data() + b * K * N,
                   h_C_ref.data() + b * M * N,
                   M, N, K);
    }

    auto d_A = upload(h_A);
    auto d_B = upload(h_B);
    GpuBuf d_C(batch * M * N);
    CUINFER_CHECK_CUDA(cudaMemset(d_C.ptr, 0, batch * M * N * sizeof(float)));

    batched_matmul(d_A.ptr, d_B.ptr, d_C.ptr, batch, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = download(d_C.ptr, batch * M * N);
    for (int i = 0; i < batch * M * N; i++) {
        EXPECT_NEAR(result[i], h_C_ref[i], 1e-4f)
            << "Mismatch at flat index " << i;
    }
}

TEST_F(AttentionTest, BatchedMatmulGQA_Basic) {
    // 4 query heads, 2 KV heads -> groups of 2.
    // Head 0,1 use KV head 0; Head 2,3 use KV head 1.
    const int n_heads = 4, n_kv_heads = 2;
    const int M = 2, N = 3, K = 4;

    auto h_A = random_floats(n_heads * M * K);
    auto h_B = random_floats(n_kv_heads * K * N);

    // CPU reference with GQA mapping.
    const int n_groups = n_heads / n_kv_heads;
    std::vector<float> h_C_ref(n_heads * M * N, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_groups;
        cpu_matmul(h_A.data() + h * M * K,
                   h_B.data() + kv_h * K * N,
                   h_C_ref.data() + h * M * N,
                   M, N, K);
    }

    auto d_A = upload(h_A);
    auto d_B = upload(h_B);
    GpuBuf d_C(n_heads * M * N);
    CUINFER_CHECK_CUDA(cudaMemset(d_C.ptr, 0, n_heads * M * N * sizeof(float)));

    batched_matmul_gqa(d_A.ptr, d_B.ptr, d_C.ptr,
                       n_heads, n_kv_heads, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = download(d_C.ptr, n_heads * M * N);
    for (int i = 0; i < n_heads * M * N; i++) {
        EXPECT_NEAR(result[i], h_C_ref[i], 1e-4f)
            << "Mismatch at flat index " << i;
    }
}

// ════════════════════════════════════════════════════════════════
// End-to-End: Single-Head Attention
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, SingleHead_SingleToken) {
    // Minimal config: 1 head, head_dim=8, seq_len=1, pos=0.
    const int n_heads = 1, n_kv_heads = 1, head_dim = 8;
    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int seq_len = 1;
    const int pos = 0;
    const int max_seq_len = 16;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    // Generate random weights and input.
    auto h_input = random_floats(seq_len * dim, -0.5f, 0.5f);
    auto h_w_q   = random_floats(dim * dim, -0.5f, 0.5f);
    auto h_w_k   = random_floats(dim * kv_dim, -0.5f, 0.5f);
    auto h_w_v   = random_floats(dim * kv_dim, -0.5f, 0.5f);
    auto h_w_o   = random_floats(dim * dim, -0.5f, 0.5f);

    // Create GPU tensors.
    Tensor t_input = Tensor::from_host(h_input.data(), {seq_len, (int64_t)dim});
    Tensor t_w_q   = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k   = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v   = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o   = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    // KV cache (1 layer).
    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    // GPU forward pass.
    Tensor output = attn.forward(t_input, t_w_q, t_w_k, t_w_v, t_w_o,
                                 kv_cache, /*layer_idx=*/0, pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // CPU reference.
    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);

    auto cpu_out = cpu_attention(
        h_input.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        seq_len, n_heads, n_kv_heads, head_dim, pos,
        cpu_k_cache.data(), cpu_v_cache.data());

    // Compare.
    std::vector<float> gpu_out(seq_len * dim);
    output.copy_to_host(gpu_out.data(), gpu_out.size() * sizeof(float));

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 5e-4f)
            << "Mismatch at index " << i
            << " gpu=" << gpu_out[i] << " cpu=" << cpu_out[i];
    }
}

// ════════════════════════════════════════════════════════════════
// End-to-End: Multi-Head Attention
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, MultiHead_SingleToken) {
    // 4 heads, head_dim=16 -> dim=64.
    const int n_heads = 4, n_kv_heads = 4, head_dim = 16;
    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int seq_len = 1;
    const int pos = 0;
    const int max_seq_len = 32;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    auto h_input = random_floats(seq_len * dim, -0.5f, 0.5f);
    auto h_w_q   = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o   = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_input = Tensor::from_host(h_input.data(), {seq_len, (int64_t)dim});
    Tensor t_w_q   = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k   = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v   = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o   = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    Tensor output = attn.forward(t_input, t_w_q, t_w_k, t_w_v, t_w_o,
                                 kv_cache, 0, pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);
    auto cpu_out = cpu_attention(
        h_input.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        seq_len, n_heads, n_kv_heads, head_dim, pos,
        cpu_k_cache.data(), cpu_v_cache.data());

    std::vector<float> gpu_out(seq_len * dim);
    output.copy_to_host(gpu_out.data(), gpu_out.size() * sizeof(float));

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "Mismatch at index " << i
            << " gpu=" << gpu_out[i] << " cpu=" << cpu_out[i];
    }
}

TEST_F(AttentionTest, MultiHead_Prefill) {
    // 4 heads, head_dim=16, prefill 4 tokens at once.
    const int n_heads = 4, n_kv_heads = 4, head_dim = 16;
    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int seq_len = 4;
    const int pos = 0;
    const int max_seq_len = 32;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    auto h_input = random_floats(seq_len * dim, -0.5f, 0.5f);
    auto h_w_q   = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o   = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_input = Tensor::from_host(h_input.data(), {seq_len, (int64_t)dim});
    Tensor t_w_q   = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k   = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v   = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o   = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    Tensor output = attn.forward(t_input, t_w_q, t_w_k, t_w_v, t_w_o,
                                 kv_cache, 0, pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);
    auto cpu_out = cpu_attention(
        h_input.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        seq_len, n_heads, n_kv_heads, head_dim, pos,
        cpu_k_cache.data(), cpu_v_cache.data());

    std::vector<float> gpu_out(seq_len * dim);
    output.copy_to_host(gpu_out.data(), gpu_out.size() * sizeof(float));

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 2e-3f)
            << "Mismatch at index " << i
            << " gpu=" << gpu_out[i] << " cpu=" << cpu_out[i];
    }
}

// ════════════════════════════════════════════════════════════════
// End-to-End: Grouped-Query Attention (GQA)
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, GQA_SingleToken) {
    // 8 query heads, 2 KV heads -> 4:1 ratio.
    const int n_heads = 8, n_kv_heads = 2, head_dim = 16;
    const int dim = n_heads * head_dim;    // 128
    const int kv_dim = n_kv_heads * head_dim; // 32
    const int seq_len = 1;
    const int pos = 0;
    const int max_seq_len = 32;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    auto h_input = random_floats(seq_len * dim, -0.5f, 0.5f);
    auto h_w_q   = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o   = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_input = Tensor::from_host(h_input.data(), {seq_len, (int64_t)dim});
    Tensor t_w_q   = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k   = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v   = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o   = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    Tensor output = attn.forward(t_input, t_w_q, t_w_k, t_w_v, t_w_o,
                                 kv_cache, 0, pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);
    auto cpu_out = cpu_attention(
        h_input.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        seq_len, n_heads, n_kv_heads, head_dim, pos,
        cpu_k_cache.data(), cpu_v_cache.data());

    std::vector<float> gpu_out(seq_len * dim);
    output.copy_to_host(gpu_out.data(), gpu_out.size() * sizeof(float));

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "GQA mismatch at index " << i
            << " gpu=" << gpu_out[i] << " cpu=" << cpu_out[i];
    }
}

TEST_F(AttentionTest, GQA_Prefill) {
    // 4 query heads, 2 KV heads, prefill 3 tokens.
    const int n_heads = 4, n_kv_heads = 2, head_dim = 8;
    const int dim = n_heads * head_dim;    // 32
    const int kv_dim = n_kv_heads * head_dim; // 16
    const int seq_len = 3;
    const int pos = 0;
    const int max_seq_len = 16;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    auto h_input = random_floats(seq_len * dim, -0.5f, 0.5f);
    auto h_w_q   = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v   = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o   = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_input = Tensor::from_host(h_input.data(), {seq_len, (int64_t)dim});
    Tensor t_w_q   = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k   = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v   = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o   = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    Tensor output = attn.forward(t_input, t_w_q, t_w_k, t_w_v, t_w_o,
                                 kv_cache, 0, pos);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);
    auto cpu_out = cpu_attention(
        h_input.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        seq_len, n_heads, n_kv_heads, head_dim, pos,
        cpu_k_cache.data(), cpu_v_cache.data());

    std::vector<float> gpu_out(seq_len * dim);
    output.copy_to_host(gpu_out.data(), gpu_out.size() * sizeof(float));

    for (int i = 0; i < seq_len * dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 2e-3f)
            << "GQA prefill mismatch at index " << i
            << " gpu=" << gpu_out[i] << " cpu=" << cpu_out[i];
    }
}

// ════════════════════════════════════════════════════════════════
// Causal Masking Verification
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, CausalMask_FutureTokensIgnored) {
    // Verify that during prefill, the first token's output is identical
    // regardless of what tokens come after it (proving future tokens
    // are truly masked out and don't influence earlier positions).
    //
    // Strategy: run attention twice with different second tokens but the
    // same first token. The output at position 0 should be identical.

    const int n_heads = 2, n_kv_heads = 2, head_dim = 8;
    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int max_seq_len = 16;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    // Fixed weights.
    auto h_w_q = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_w_q = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    // Input A: [token_0, token_1_A]
    auto h_token0 = random_floats(dim, -0.5f, 0.5f);
    auto h_token1_A = random_floats(dim, -0.5f, 0.5f);
    std::vector<float> h_input_A(2 * dim);
    std::copy(h_token0.begin(), h_token0.end(), h_input_A.begin());
    std::copy(h_token1_A.begin(), h_token1_A.end(), h_input_A.begin() + dim);

    // Input B: [token_0, token_1_B] (different second token)
    auto h_token1_B = random_floats(dim, -0.5f, 0.5f);
    std::vector<float> h_input_B(2 * dim);
    std::copy(h_token0.begin(), h_token0.end(), h_input_B.begin());
    std::copy(h_token1_B.begin(), h_token1_B.end(), h_input_B.begin() + dim);

    // Run A
    {
        AttentionBlock attn(attn_config);
        KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
        KVCache kv_cache(kv_cfg);

        Tensor t_input_A = Tensor::from_host(h_input_A.data(), {2, (int64_t)dim});
        Tensor out_A = attn.forward(t_input_A, t_w_q, t_w_k, t_w_v, t_w_o,
                                    kv_cache, 0, 0);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        // Run B
        KVCache kv_cache_B(kv_cfg);
        Tensor t_input_B = Tensor::from_host(h_input_B.data(), {2, (int64_t)dim});
        Tensor out_B = attn.forward(t_input_B, t_w_q, t_w_k, t_w_v, t_w_o,
                                    kv_cache_B, 0, 0);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        // Compare only the first token's output (first `dim` elements).
        std::vector<float> gpu_A(2 * dim), gpu_B(2 * dim);
        out_A.copy_to_host(gpu_A.data(), 2 * dim * sizeof(float));
        out_B.copy_to_host(gpu_B.data(), 2 * dim * sizeof(float));

        for (int i = 0; i < dim; i++) {
            EXPECT_NEAR(gpu_A[i], gpu_B[i], 1e-5f)
                << "Causal mask failure: first token output differs at index "
                << i << " (A=" << gpu_A[i] << ", B=" << gpu_B[i] << ")";
        }

        // The second token's output should differ (since they see different inputs).
        bool any_differ = false;
        for (int i = dim; i < 2 * dim; i++) {
            if (std::abs(gpu_A[i] - gpu_B[i]) > 1e-5f) {
                any_differ = true;
                break;
            }
        }
        EXPECT_TRUE(any_differ)
            << "Second token outputs should differ when inputs differ";
    }
}

// ════════════════════════════════════════════════════════════════
// KV-Cache Correctness Across Multiple Forward Calls
// ════════════════════════════════════════════════════════════════

TEST_F(AttentionTest, KVCache_PrefillThenDecode) {
    // Prefill 3 tokens, then decode 1 token.
    // Compare the decode output against CPU reference that processes
    // all 4 tokens with correct KV-cache state.
    const int n_heads = 2, n_kv_heads = 2, head_dim = 8;
    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int max_seq_len = 16;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    // Fixed weights (shared across calls).
    auto h_w_q = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_w_q = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    // Prefill input: 3 tokens.
    auto h_prefill = random_floats(3 * dim, -0.5f, 0.5f);
    Tensor t_prefill = Tensor::from_host(h_prefill.data(), {3, (int64_t)dim});

    Tensor prefill_out = attn.forward(t_prefill, t_w_q, t_w_k, t_w_v, t_w_o,
                                      kv_cache, 0, /*pos=*/0);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());
    EXPECT_EQ(kv_cache.current_length(), 3);

    // Decode input: 1 token at position 3.
    auto h_decode = random_floats(1 * dim, -0.5f, 0.5f);
    Tensor t_decode = Tensor::from_host(h_decode.data(), {1, (int64_t)dim});

    Tensor decode_out = attn.forward(t_decode, t_w_q, t_w_k, t_w_v, t_w_o,
                                     kv_cache, 0, /*pos=*/3);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());
    EXPECT_EQ(kv_cache.current_length(), 4);

    // CPU reference: replay both prefill and decode.
    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);

    // CPU prefill
    auto cpu_prefill_out = cpu_attention(
        h_prefill.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        3, n_heads, n_kv_heads, head_dim, 0,
        cpu_k_cache.data(), cpu_v_cache.data());

    // CPU decode
    auto cpu_decode_out = cpu_attention(
        h_decode.data(), h_w_q.data(), h_w_k.data(),
        h_w_v.data(), h_w_o.data(),
        1, n_heads, n_kv_heads, head_dim, 3,
        cpu_k_cache.data(), cpu_v_cache.data());

    // Verify prefill output.
    std::vector<float> gpu_prefill(3 * dim);
    prefill_out.copy_to_host(gpu_prefill.data(), 3 * dim * sizeof(float));
    for (int i = 0; i < 3 * dim; i++) {
        EXPECT_NEAR(gpu_prefill[i], cpu_prefill_out[i], 2e-3f)
            << "Prefill mismatch at index " << i;
    }

    // Verify decode output.
    std::vector<float> gpu_decode(1 * dim);
    decode_out.copy_to_host(gpu_decode.data(), 1 * dim * sizeof(float));
    for (int i = 0; i < 1 * dim; i++) {
        EXPECT_NEAR(gpu_decode[i], cpu_decode_out[i], 2e-3f)
            << "Decode mismatch at index " << i;
    }
}

TEST_F(AttentionTest, KVCache_MultipleDecodeSteps) {
    // Process tokens one at a time for 4 steps.
    // After each step, compare against CPU reference.
    const int n_heads = 2, n_kv_heads = 1, head_dim = 8;
    const int dim = n_heads * head_dim;    // 16
    const int kv_dim = n_kv_heads * head_dim; // 8
    const int max_seq_len = 16;
    const int n_steps = 4;

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    AttentionBlock attn(attn_config);

    auto h_w_q = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_w_q = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_cache(kv_cfg);

    std::vector<float> cpu_k_cache(max_seq_len * kv_dim, 0.0f);
    std::vector<float> cpu_v_cache(max_seq_len * kv_dim, 0.0f);

    // Pre-generate all inputs so that both GPU and CPU use the same data.
    std::vector<std::vector<float>> all_inputs(n_steps);
    for (int t = 0; t < n_steps; t++) {
        all_inputs[t] = random_floats(dim, -0.5f, 0.5f);
    }

    for (int t = 0; t < n_steps; t++) {
        Tensor t_input = Tensor::from_host(all_inputs[t].data(),
                                           {1, (int64_t)dim});

        Tensor gpu_output = attn.forward(t_input, t_w_q, t_w_k, t_w_v, t_w_o,
                                         kv_cache, 0, /*pos=*/t);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        EXPECT_EQ(kv_cache.current_length(), t + 1);

        auto cpu_output = cpu_attention(
            all_inputs[t].data(), h_w_q.data(), h_w_k.data(),
            h_w_v.data(), h_w_o.data(),
            1, n_heads, n_kv_heads, head_dim, t,
            cpu_k_cache.data(), cpu_v_cache.data());

        std::vector<float> gpu_out(dim);
        gpu_output.copy_to_host(gpu_out.data(), dim * sizeof(float));

        for (int i = 0; i < dim; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_output[i], 2e-3f)
                << "Step " << t << " mismatch at index " << i
                << " gpu=" << gpu_out[i] << " cpu=" << cpu_output[i];
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Config Validation
// ════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════
// Fused Decode Attention Tests
// ════════════════════════════════════════════════════════════════
//
// Tests fused_decode_attention() against CPU reference for various
// configurations. The CPU reference computes:
//   for each head h:
//     scores[k] = dot(Q[h,:], K[k, kv_h, :]) * scale
//     attn_weights = softmax(scores)
//     output[h,:] = sum_k attn_weights[k] * V[k, kv_h, :]

static void cpu_fused_decode_attention(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* output,
    int n_heads, int n_kv_heads, int head_dim, int kv_len,
    float scale_factor) {

    const int n_groups = n_heads / n_kv_heads;
    const int kv_dim = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_groups;
        const float* q_ptr = Q + h * head_dim;

        // Compute attention scores
        std::vector<float> scores(kv_len);
        for (int k = 0; k < kv_len; k++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_ptr[d] * K_cache[k * kv_dim + kv_h * head_dim + d];
            }
            scores[k] = dot * scale_factor;
        }

        // Softmax
        std::vector<float> attn_weights(kv_len);
        cpu_softmax(scores.data(), attn_weights.data(), kv_len);

        // Weighted sum of V
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                val += attn_weights[k] * V_cache[k * kv_dim + kv_h * head_dim + d];
            }
            output[h * head_dim + d] = val;
        }
    }
}

TEST_F(AttentionTest, FusedDecodeAttention_SingleHead) {
    const int n_heads = 1, n_kv_heads = 1, head_dim = 8, kv_len = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int kv_dim = n_kv_heads * head_dim;

    auto h_Q = random_floats(n_heads * head_dim, -0.5f, 0.5f);
    auto h_K = random_floats(kv_len * kv_dim, -0.5f, 0.5f);
    auto h_V = random_floats(kv_len * kv_dim, -0.5f, 0.5f);

    // CPU reference
    std::vector<float> cpu_out(n_heads * head_dim, 0.0f);
    cpu_fused_decode_attention(h_Q.data(), h_K.data(), h_V.data(),
                               cpu_out.data(), n_heads, n_kv_heads,
                               head_dim, kv_len, scale);

    // GPU
    auto d_Q = upload(h_Q);
    auto d_K = upload(h_K);
    auto d_V = upload(h_V);
    GpuBuf d_out(n_heads * head_dim);

    fused_decode_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_out.ptr,
                           n_heads, n_kv_heads, head_dim, kv_len, scale);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_out.ptr, n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f)
            << "Mismatch at index " << i
            << " gpu=" << gpu_out[i] << " cpu=" << cpu_out[i];
    }
}

TEST_F(AttentionTest, FusedDecodeAttention_MultiHead) {
    const int n_heads = 4, n_kv_heads = 4, head_dim = 16, kv_len = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int kv_dim = n_kv_heads * head_dim;

    auto h_Q = random_floats(n_heads * head_dim, -0.5f, 0.5f);
    auto h_K = random_floats(kv_len * kv_dim, -0.5f, 0.5f);
    auto h_V = random_floats(kv_len * kv_dim, -0.5f, 0.5f);

    std::vector<float> cpu_out(n_heads * head_dim, 0.0f);
    cpu_fused_decode_attention(h_Q.data(), h_K.data(), h_V.data(),
                               cpu_out.data(), n_heads, n_kv_heads,
                               head_dim, kv_len, scale);

    auto d_Q = upload(h_Q);
    auto d_K = upload(h_K);
    auto d_V = upload(h_V);
    GpuBuf d_out(n_heads * head_dim);

    fused_decode_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_out.ptr,
                           n_heads, n_kv_heads, head_dim, kv_len, scale);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_out.ptr, n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "Mismatch at index " << i;
    }
}

TEST_F(AttentionTest, FusedDecodeAttention_GQA) {
    const int n_heads = 8, n_kv_heads = 2, head_dim = 16, kv_len = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int kv_dim = n_kv_heads * head_dim;

    auto h_Q = random_floats(n_heads * head_dim, -0.5f, 0.5f);
    auto h_K = random_floats(kv_len * kv_dim, -0.5f, 0.5f);
    auto h_V = random_floats(kv_len * kv_dim, -0.5f, 0.5f);

    std::vector<float> cpu_out(n_heads * head_dim, 0.0f);
    cpu_fused_decode_attention(h_Q.data(), h_K.data(), h_V.data(),
                               cpu_out.data(), n_heads, n_kv_heads,
                               head_dim, kv_len, scale);

    auto d_Q = upload(h_Q);
    auto d_K = upload(h_K);
    auto d_V = upload(h_V);
    GpuBuf d_out(n_heads * head_dim);

    fused_decode_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_out.ptr,
                           n_heads, n_kv_heads, head_dim, kv_len, scale);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_out.ptr, n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-3f)
            << "GQA mismatch at index " << i;
    }
}

TEST_F(AttentionTest, FusedDecodeAttention_LongContext) {
    const int n_heads = 4, n_kv_heads = 4, head_dim = 64, kv_len = 512;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int kv_dim = n_kv_heads * head_dim;

    auto h_Q = random_floats(n_heads * head_dim, -0.3f, 0.3f);
    auto h_K = random_floats(kv_len * kv_dim, -0.3f, 0.3f);
    auto h_V = random_floats(kv_len * kv_dim, -0.3f, 0.3f);

    std::vector<float> cpu_out(n_heads * head_dim, 0.0f);
    cpu_fused_decode_attention(h_Q.data(), h_K.data(), h_V.data(),
                               cpu_out.data(), n_heads, n_kv_heads,
                               head_dim, kv_len, scale);

    auto d_Q = upload(h_Q);
    auto d_K = upload(h_K);
    auto d_V = upload(h_V);
    GpuBuf d_out(n_heads * head_dim);

    fused_decode_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_out.ptr,
                           n_heads, n_kv_heads, head_dim, kv_len, scale);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    auto gpu_out = download(d_out.ptr, n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++) {
        EXPECT_NEAR(gpu_out[i], cpu_out[i], 2e-3f)
            << "Long context mismatch at index " << i;
    }
}

TEST_F(AttentionTest, FusedDecodeAttention_MatchesUnfused) {
    // Compare fused decode against the full unfused attention pipeline
    // (via AttentionBlock::forward with seq_len=1).
    const int n_heads = 4, n_kv_heads = 2, head_dim = 16;
    const int dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int max_seq_len = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    AttentionConfig attn_config;
    attn_config.n_heads = n_heads;
    attn_config.n_kv_heads = n_kv_heads;
    attn_config.head_dim = head_dim;
    attn_config.max_seq_len = max_seq_len;

    auto h_w_q = random_floats(dim * dim, -0.3f, 0.3f);
    auto h_w_k = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_v = random_floats(dim * kv_dim, -0.3f, 0.3f);
    auto h_w_o = random_floats(dim * dim, -0.3f, 0.3f);

    Tensor t_w_q = Tensor::from_host(h_w_q.data(), {(int64_t)dim, (int64_t)dim});
    Tensor t_w_k = Tensor::from_host(h_w_k.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_v = Tensor::from_host(h_w_v.data(), {(int64_t)dim, (int64_t)kv_dim});
    Tensor t_w_o = Tensor::from_host(h_w_o.data(), {(int64_t)dim, (int64_t)dim});

    // Pre-fill 5 tokens to populate the KV cache, then test decode at pos=5.
    const int prefill_len = 5;
    auto h_prefill = random_floats(prefill_len * dim, -0.5f, 0.5f);
    auto h_decode = random_floats(1 * dim, -0.5f, 0.5f);

    // Run unfused path
    AttentionBlock attn_unfused(attn_config);
    KVCacheConfig kv_cfg = {1, max_seq_len, n_kv_heads, head_dim};
    KVCache kv_unfused(kv_cfg);

    Tensor t_prefill = Tensor::from_host(h_prefill.data(), {prefill_len, (int64_t)dim});
    attn_unfused.forward(t_prefill, t_w_q, t_w_k, t_w_v, t_w_o,
                         kv_unfused, 0, 0);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    Tensor t_decode = Tensor::from_host(h_decode.data(), {1, (int64_t)dim});
    Tensor unfused_out = attn_unfused.forward(t_decode, t_w_q, t_w_k, t_w_v, t_w_o,
                                              kv_unfused, 0, prefill_len);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Run fused path: need to replicate the QKV projections and RoPE
    // to get the same Q, then use fused_decode_attention on the same KV cache.
    AttentionBlock attn_fused(attn_config);
    KVCache kv_fused(kv_cfg);

    Tensor t_prefill2 = Tensor::from_host(h_prefill.data(), {prefill_len, (int64_t)dim});
    attn_fused.forward(t_prefill2, t_w_q, t_w_k, t_w_v, t_w_o,
                       kv_fused, 0, 0);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Manually do QKV projection for the decode token
    Tensor t_decode2 = Tensor::from_host(h_decode.data(), {1, (int64_t)dim});
    Tensor q_proj = Tensor::zeros({1, (int64_t)dim});
    Tensor k_proj = Tensor::zeros({1, (int64_t)kv_dim});
    Tensor v_proj = Tensor::zeros({1, (int64_t)kv_dim});

    matmul_tiled(t_decode2.data_as<float>(), t_w_q.data_as<float>(),
                 q_proj.data_as<float>(), 1, dim, dim);
    matmul_tiled(t_decode2.data_as<float>(), t_w_k.data_as<float>(),
                 k_proj.data_as<float>(), 1, kv_dim, dim);
    matmul_tiled(t_decode2.data_as<float>(), t_w_v.data_as<float>(),
                 v_proj.data_as<float>(), 1, kv_dim, dim);

    rope(q_proj.data_as<float>(), prefill_len, head_dim, n_heads);
    rope(k_proj.data_as<float>(), prefill_len, head_dim, n_kv_heads);

    // Append K/V to cache
    kv_fused.append(0, k_proj.data_as<float>(), v_proj.data_as<float>(), 1);
    int kv_len = kv_fused.current_length();

    // Run fused decode attention
    Tensor fused_attn_out = Tensor::zeros({1, (int64_t)dim});
    fused_decode_attention(q_proj.data_as<float>(),
                           kv_fused.k_cache(0), kv_fused.v_cache(0),
                           fused_attn_out.data_as<float>(),
                           n_heads, n_kv_heads, head_dim, kv_len, scale);

    // Apply Wo projection
    Tensor fused_out = Tensor::zeros({1, (int64_t)dim});
    matmul_tiled(fused_attn_out.data_as<float>(), t_w_o.data_as<float>(),
                 fused_out.data_as<float>(), 1, dim, dim);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    // Compare
    std::vector<float> unfused_host(dim), fused_host(dim);
    unfused_out.copy_to_host(unfused_host.data(), dim * sizeof(float));
    fused_out.copy_to_host(fused_host.data(), dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(fused_host[i], unfused_host[i], 2e-3f)
            << "Fused vs unfused mismatch at index " << i
            << " fused=" << fused_host[i] << " unfused=" << unfused_host[i];
    }
}

TEST_F(AttentionTest, ConfigValidation) {
    // n_heads not divisible by n_kv_heads should throw.
    AttentionConfig bad_config = {3, 2, 16, 64};
    EXPECT_THROW(AttentionBlock{bad_config}, std::invalid_argument);

    // Odd head_dim should throw (RoPE requires even).
    bad_config = {4, 4, 15, 64};
    EXPECT_THROW(AttentionBlock{bad_config}, std::invalid_argument);

    // Zero n_heads.
    bad_config = {0, 1, 16, 64};
    EXPECT_THROW(AttentionBlock{bad_config}, std::invalid_argument);

    // Valid config should not throw.
    AttentionConfig good_config = {4, 2, 16, 64};
    EXPECT_NO_THROW(AttentionBlock{good_config});
}
