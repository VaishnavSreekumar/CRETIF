#include <gtest/gtest.h>
#include "cuinfer/kv_cache.h"
#include "cuinfer/cuda_utils.h"

#include <cstring>
#include <numeric>
#include <vector>

using namespace cuinfer;

// ── Helper: allocate device memory and copy host data to GPU ─

/// Allocates device memory for `n` floats, copies `host_data` to it,
/// and returns the device pointer. Caller must cudaFree the result.
static float* to_device(const std::vector<float>& host_data) {
    float* d_ptr = nullptr;
    const size_t bytes = host_data.size() * sizeof(float);
    CUINFER_CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_ptr, host_data.data(), bytes,
                                  cudaMemcpyHostToDevice));
    return d_ptr;
}

/// Copies `n` floats from device to a host vector.
static std::vector<float> to_host(const float* d_ptr, size_t n) {
    std::vector<float> result(n);
    CUINFER_CHECK_CUDA(cudaMemcpy(result.data(), d_ptr,
                                  n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return result;
}

// ── Test: Construction and memory sizing ────────────────────

TEST(KVCache, ConstructionAndMemory) {
    // SmolLM2-135M-like config (scaled down max_seq_len for test speed).
    KVCacheConfig config;
    config.n_layers    = 4;
    config.max_seq_len = 64;
    config.n_kv_heads  = 3;
    config.head_dim    = 16;

    KVCache cache(config);

    // Verify total_bytes matches manual calculation:
    //   per-layer K bytes = 64 * 3 * 16 * 4 = 12288
    //   per-layer V bytes = 12288
    //   total = 4 layers * (12288 + 12288) = 98304
    const size_t expected_per_layer = 64 * 3 * 16 * sizeof(float);
    EXPECT_EQ(config.layer_k_bytes(), expected_per_layer);
    EXPECT_EQ(config.layer_v_bytes(), expected_per_layer);
    EXPECT_EQ(config.total_bytes(), 4 * 2 * expected_per_layer);
    EXPECT_EQ(cache.total_bytes(), config.total_bytes());

    // Cache starts empty.
    EXPECT_EQ(cache.current_length(), 0);
    EXPECT_EQ(cache.used_bytes(), 0u);
}

// ── Test: Config validation ─────────────────────────────────

TEST(KVCache, ConfigValidation) {
    // Zero or negative config values should throw.
    KVCacheConfig bad_config = {0, 64, 3, 16};
    EXPECT_THROW(KVCache{bad_config}, std::invalid_argument);

    bad_config = {4, 0, 3, 16};
    EXPECT_THROW(KVCache{bad_config}, std::invalid_argument);

    bad_config = {4, 64, 0, 16};
    EXPECT_THROW(KVCache{bad_config}, std::invalid_argument);

    bad_config = {4, 64, 3, 0};
    EXPECT_THROW(KVCache{bad_config}, std::invalid_argument);
}

// ── Test: Append single token and read back ─────────────────

TEST(KVCache, AppendSingleToken) {
    // Small cache: 2 layers, max_seq=8, 2 kv_heads, head_dim=4.
    // Per position: 2 heads * 4 dim = 8 floats.
    KVCacheConfig config = {2, 8, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim; // 8

    // Generate known K and V data for one token.
    std::vector<float> k_host(elems_per_token);
    std::vector<float> v_host(elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        k_host[i] = static_cast<float>(i + 1);          // 1, 2, ..., 8
        v_host[i] = static_cast<float>((i + 1) * 10);   // 10, 20, ..., 80
    }

    float* d_k = to_device(k_host);
    float* d_v = to_device(v_host);

    // Append to layer 0 (does not increment current_length yet).
    cache.append(/*layer=*/0, d_k, d_v, /*n_new_tokens=*/1);

    // Append to layer 1 (last layer -- this increments current_length).
    // Use different data for layer 1 to verify layer isolation.
    std::vector<float> k_host_l1(elems_per_token);
    std::vector<float> v_host_l1(elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        k_host_l1[i] = static_cast<float>((i + 1) * 100);
        v_host_l1[i] = static_cast<float>((i + 1) * 1000);
    }
    float* d_k_l1 = to_device(k_host_l1);
    float* d_v_l1 = to_device(v_host_l1);

    cache.append(/*layer=*/1, d_k_l1, d_v_l1, /*n_new_tokens=*/1);

    EXPECT_EQ(cache.current_length(), 1);

    // Read back layer 0 K cache and verify.
    auto k0_readback = to_host(cache.k_cache(0), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(k0_readback[i], k_host[i])
            << "K cache layer 0 mismatch at index " << i;
    }

    // Read back layer 0 V cache.
    auto v0_readback = to_host(cache.v_cache(0), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(v0_readback[i], v_host[i])
            << "V cache layer 0 mismatch at index " << i;
    }

    // Read back layer 1 K cache -- should have the layer-1-specific data.
    auto k1_readback = to_host(cache.k_cache(1), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(k1_readback[i], k_host_l1[i])
            << "K cache layer 1 mismatch at index " << i;
    }

    // Read back layer 1 V cache.
    auto v1_readback = to_host(cache.v_cache(1), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(v1_readback[i], v_host_l1[i])
            << "V cache layer 1 mismatch at index " << i;
    }

    // Cleanup temporary device allocations.
    CUINFER_CHECK_CUDA(cudaFree(d_k));
    CUINFER_CHECK_CUDA(cudaFree(d_v));
    CUINFER_CHECK_CUDA(cudaFree(d_k_l1));
    CUINFER_CHECK_CUDA(cudaFree(d_v_l1));
}

// ── Test: Multi-token append (prefill) ──────────────────────

TEST(KVCache, MultiTokenPrefill) {
    // 1 layer, max_seq=16, 2 kv_heads, head_dim=4.
    KVCacheConfig config = {1, 16, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim; // 8
    const int n_prefill = 4;
    const int total_elems = n_prefill * elems_per_token; // 32

    // Fill K with sequential values [1..32], V with [100..131].
    std::vector<float> k_host(total_elems);
    std::vector<float> v_host(total_elems);
    for (int i = 0; i < total_elems; ++i) {
        k_host[i] = static_cast<float>(i + 1);
        v_host[i] = static_cast<float>(i + 100);
    }

    float* d_k = to_device(k_host);
    float* d_v = to_device(v_host);

    // Single layer, so this append increments current_length.
    cache.append(0, d_k, d_v, n_prefill);
    EXPECT_EQ(cache.current_length(), n_prefill);

    // Read back all 4 positions from the K cache.
    auto k_readback = to_host(cache.k_cache(0), total_elems);
    for (int i = 0; i < total_elems; ++i) {
        EXPECT_FLOAT_EQ(k_readback[i], k_host[i])
            << "K prefill mismatch at index " << i;
    }

    // Same for V.
    auto v_readback = to_host(cache.v_cache(0), total_elems);
    for (int i = 0; i < total_elems; ++i) {
        EXPECT_FLOAT_EQ(v_readback[i], v_host[i])
            << "V prefill mismatch at index " << i;
    }

    CUINFER_CHECK_CUDA(cudaFree(d_k));
    CUINFER_CHECK_CUDA(cudaFree(d_v));
}

// ── Test: Sequential single-token appends (decode) ──────────

TEST(KVCache, SequentialDecode) {
    // 1 layer so current_length increments on every append call.
    KVCacheConfig config = {1, 16, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim; // 8
    const int n_tokens = 3;

    // Append three tokens one at a time, each with distinct data.
    std::vector<std::vector<float>> k_tokens(n_tokens);
    std::vector<std::vector<float>> v_tokens(n_tokens);

    for (int t = 0; t < n_tokens; ++t) {
        k_tokens[t].resize(elems_per_token);
        v_tokens[t].resize(elems_per_token);
        for (int i = 0; i < elems_per_token; ++i) {
            k_tokens[t][i] = static_cast<float>(t * 100 + i);
            v_tokens[t][i] = static_cast<float>(t * 1000 + i);
        }

        float* d_k = to_device(k_tokens[t]);
        float* d_v = to_device(v_tokens[t]);
        cache.append(0, d_k, d_v, 1);
        CUINFER_CHECK_CUDA(cudaFree(d_k));
        CUINFER_CHECK_CUDA(cudaFree(d_v));
    }

    EXPECT_EQ(cache.current_length(), n_tokens);

    // Read back all 3 positions and verify each token's data is correct.
    auto k_all = to_host(cache.k_cache(0), n_tokens * elems_per_token);
    auto v_all = to_host(cache.v_cache(0), n_tokens * elems_per_token);

    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < elems_per_token; ++i) {
            const int idx = t * elems_per_token + i;
            EXPECT_FLOAT_EQ(k_all[idx], k_tokens[t][i])
                << "K decode mismatch at token=" << t << " elem=" << i;
            EXPECT_FLOAT_EQ(v_all[idx], v_tokens[t][i])
                << "V decode mismatch at token=" << t << " elem=" << i;
        }
    }
}

// ── Test: Clear and reuse ───────────────────────────────────

TEST(KVCache, ClearAndReuse) {
    KVCacheConfig config = {1, 8, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim;

    // Fill 3 positions.
    std::vector<float> k_host(elems_per_token, 42.0f);
    std::vector<float> v_host(elems_per_token, 99.0f);
    float* d_k = to_device(k_host);
    float* d_v = to_device(v_host);

    for (int t = 0; t < 3; ++t) {
        cache.append(0, d_k, d_v, 1);
    }
    EXPECT_EQ(cache.current_length(), 3);
    EXPECT_GT(cache.used_bytes(), 0u);

    // Clear.
    cache.clear();
    EXPECT_EQ(cache.current_length(), 0);
    EXPECT_EQ(cache.used_bytes(), 0u);

    // Fill again with different data and verify the new data is correct.
    std::vector<float> k_new(elems_per_token, 7.0f);
    std::vector<float> v_new(elems_per_token, 13.0f);
    float* d_k_new = to_device(k_new);
    float* d_v_new = to_device(v_new);

    cache.append(0, d_k_new, d_v_new, 1);
    EXPECT_EQ(cache.current_length(), 1);

    auto k_readback = to_host(cache.k_cache(0), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(k_readback[i], 7.0f)
            << "K post-clear mismatch at index " << i;
    }

    auto v_readback = to_host(cache.v_cache(0), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(v_readback[i], 13.0f)
            << "V post-clear mismatch at index " << i;
    }

    CUINFER_CHECK_CUDA(cudaFree(d_k));
    CUINFER_CHECK_CUDA(cudaFree(d_v));
    CUINFER_CHECK_CUDA(cudaFree(d_k_new));
    CUINFER_CHECK_CUDA(cudaFree(d_v_new));
}

// ── Test: Overflow detection ────────────────────────────────

TEST(KVCache, OverflowDetection) {
    // Cache with max_seq=4.
    KVCacheConfig config = {1, 4, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim;

    std::vector<float> data(elems_per_token, 1.0f);
    float* d_ptr = to_device(data);

    // Fill exactly to capacity (4 tokens, one at a time).
    for (int t = 0; t < 4; ++t) {
        EXPECT_NO_THROW(cache.append(0, d_ptr, d_ptr, 1));
    }
    EXPECT_EQ(cache.current_length(), 4);

    // One more should throw overflow_error.
    EXPECT_THROW(cache.append(0, d_ptr, d_ptr, 1), std::overflow_error);

    // Also test: appending too many at once.
    cache.clear();
    EXPECT_THROW(cache.append(0, d_ptr, d_ptr, 5), std::overflow_error);

    // Appending exactly max_seq_len from empty should succeed.
    std::vector<float> big_data(4 * elems_per_token, 1.0f);
    float* d_big = to_device(big_data);
    EXPECT_NO_THROW(cache.append(0, d_big, d_big, 4));
    EXPECT_EQ(cache.current_length(), 4);

    CUINFER_CHECK_CUDA(cudaFree(d_ptr));
    CUINFER_CHECK_CUDA(cudaFree(d_big));
}

// ── Test: Memory reporting ──────────────────────────────────

TEST(KVCache, MemoryReporting) {
    KVCacheConfig config = {2, 16, 3, 8};
    KVCache cache(config);

    // Total bytes: 2 layers * 2 caches * 16 * 3 * 8 * 4 = 6144
    const size_t expected_total = 2 * 2 * 16 * 3 * 8 * sizeof(float);
    EXPECT_EQ(cache.total_bytes(), expected_total);

    // Empty cache -- zero used.
    EXPECT_EQ(cache.used_bytes(), 0u);

    const int elems_per_token = config.n_kv_heads * config.head_dim;
    std::vector<float> data(elems_per_token, 1.0f);
    float* d_ptr = to_device(data);

    // Append one token to both layers.
    cache.append(0, d_ptr, d_ptr, 1);
    cache.append(1, d_ptr, d_ptr, 1);
    EXPECT_EQ(cache.current_length(), 1);

    // Used bytes: 2 layers * 2 caches * 1 * 3 * 8 * 4 = 384
    const size_t expected_used_1 = 2 * 2 * 1 * 3 * 8 * sizeof(float);
    EXPECT_EQ(cache.used_bytes(), expected_used_1);

    // Append another token.
    cache.append(0, d_ptr, d_ptr, 1);
    cache.append(1, d_ptr, d_ptr, 1);
    EXPECT_EQ(cache.current_length(), 2);

    const size_t expected_used_2 = 2 * 2 * 2 * 3 * 8 * sizeof(float);
    EXPECT_EQ(cache.used_bytes(), expected_used_2);

    // After clear, used should be zero again.
    cache.clear();
    EXPECT_EQ(cache.used_bytes(), 0u);
    // Total allocated is unchanged.
    EXPECT_EQ(cache.total_bytes(), expected_total);

    CUINFER_CHECK_CUDA(cudaFree(d_ptr));
}

// ── Test: Multi-layer data isolation ────────────────────────

TEST(KVCache, MultiLayerIsolation) {
    // 3 layers, max_seq=4, 2 heads, dim=4.
    KVCacheConfig config = {3, 4, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim; // 8

    // Write a unique pattern to each layer.
    for (int layer = 0; layer < config.n_layers; ++layer) {
        std::vector<float> k_data(elems_per_token);
        std::vector<float> v_data(elems_per_token);
        for (int i = 0; i < elems_per_token; ++i) {
            // Pattern: layer * 1000 + element index for K,
            //          layer * 2000 + element index for V.
            k_data[i] = static_cast<float>(layer * 1000 + i);
            v_data[i] = static_cast<float>(layer * 2000 + i);
        }
        float* d_k = to_device(k_data);
        float* d_v = to_device(v_data);
        cache.append(layer, d_k, d_v, 1);
        CUINFER_CHECK_CUDA(cudaFree(d_k));
        CUINFER_CHECK_CUDA(cudaFree(d_v));
    }

    EXPECT_EQ(cache.current_length(), 1);

    // Verify each layer's data is correct and not corrupted by other layers.
    for (int layer = 0; layer < config.n_layers; ++layer) {
        auto k_readback = to_host(cache.k_cache(layer), elems_per_token);
        auto v_readback = to_host(cache.v_cache(layer), elems_per_token);

        for (int i = 0; i < elems_per_token; ++i) {
            const float expected_k = static_cast<float>(layer * 1000 + i);
            const float expected_v = static_cast<float>(layer * 2000 + i);

            EXPECT_FLOAT_EQ(k_readback[i], expected_k)
                << "K layer " << layer << " index " << i;
            EXPECT_FLOAT_EQ(v_readback[i], expected_v)
                << "V layer " << layer << " index " << i;
        }
    }
}

// ── Test: Move semantics ────────────────────────────────────

TEST(KVCache, MoveSemantics) {
    KVCacheConfig config = {2, 8, 2, 4};
    KVCache original(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim;
    std::vector<float> data(elems_per_token, 5.0f);
    float* d_ptr = to_device(data);

    original.append(0, d_ptr, d_ptr, 1);
    original.append(1, d_ptr, d_ptr, 1);
    EXPECT_EQ(original.current_length(), 1);

    // Move-construct.
    KVCache moved(std::move(original));

    EXPECT_EQ(moved.current_length(), 1);
    EXPECT_EQ(moved.config().n_layers, 2);
    EXPECT_EQ(moved.config().max_seq_len, 8);

    // Verify data survived the move.
    auto k_readback = to_host(moved.k_cache(0), elems_per_token);
    for (int i = 0; i < elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(k_readback[i], 5.0f)
            << "Move: K data mismatch at index " << i;
    }

    CUINFER_CHECK_CUDA(cudaFree(d_ptr));
}

// ── Test: Layer bounds checking ─────────────────────────────

TEST(KVCache, LayerBoundsCheck) {
    KVCacheConfig config = {3, 8, 2, 4};
    KVCache cache(config);

    // Valid layers: 0, 1, 2.
    EXPECT_NO_THROW(cache.k_cache(0));
    EXPECT_NO_THROW(cache.k_cache(2));

    // Out of bounds.
    EXPECT_THROW(cache.k_cache(-1), std::out_of_range);
    EXPECT_THROW(cache.k_cache(3), std::out_of_range);
    EXPECT_THROW(cache.v_cache(3), std::out_of_range);
}

// ── Test: Prefill then decode pattern ───────────────────────

TEST(KVCache, PrefillThenDecode) {
    // Simulates the real usage pattern: prefill N tokens, then decode 1 at a time.
    KVCacheConfig config = {1, 16, 2, 4};
    KVCache cache(config);

    const int elems_per_token = config.n_kv_heads * config.head_dim;
    const int prefill_len = 5;

    // Prefill: 5 tokens at once.
    std::vector<float> prefill_k(prefill_len * elems_per_token);
    std::vector<float> prefill_v(prefill_len * elems_per_token);
    for (int i = 0; i < prefill_len * elems_per_token; ++i) {
        prefill_k[i] = static_cast<float>(i);
        prefill_v[i] = static_cast<float>(i + 1000);
    }

    float* d_pk = to_device(prefill_k);
    float* d_pv = to_device(prefill_v);
    cache.append(0, d_pk, d_pv, prefill_len);
    EXPECT_EQ(cache.current_length(), prefill_len);

    // Decode: 3 more tokens, one at a time.
    for (int t = 0; t < 3; ++t) {
        std::vector<float> decode_k(elems_per_token);
        std::vector<float> decode_v(elems_per_token);
        for (int i = 0; i < elems_per_token; ++i) {
            decode_k[i] = static_cast<float>((prefill_len + t) * 100 + i);
            decode_v[i] = static_cast<float>((prefill_len + t) * 200 + i);
        }

        float* d_dk = to_device(decode_k);
        float* d_dv = to_device(decode_v);
        cache.append(0, d_dk, d_dv, 1);
        CUINFER_CHECK_CUDA(cudaFree(d_dk));
        CUINFER_CHECK_CUDA(cudaFree(d_dv));
    }

    EXPECT_EQ(cache.current_length(), prefill_len + 3);

    // Verify prefill data is still intact.
    auto k_all = to_host(cache.k_cache(0),
                         (prefill_len + 3) * elems_per_token);

    for (int i = 0; i < prefill_len * elems_per_token; ++i) {
        EXPECT_FLOAT_EQ(k_all[i], prefill_k[i])
            << "Prefill K corrupted at index " << i;
    }

    // Verify decode token 0 (position 5).
    const int decode_start = prefill_len * elems_per_token;
    for (int i = 0; i < elems_per_token; ++i) {
        const float expected = static_cast<float>(prefill_len * 100 + i);
        EXPECT_FLOAT_EQ(k_all[decode_start + i], expected)
            << "Decode token 0 K mismatch at elem " << i;
    }

    CUINFER_CHECK_CUDA(cudaFree(d_pk));
    CUINFER_CHECK_CUDA(cudaFree(d_pv));
}

// ── Test: Realistic model config (SmolLM2-135M-like) ────────

TEST(KVCache, RealisticSmolLM2Config) {
    // SmolLM2-135M: 30 layers, 3 KV heads, head_dim=64, max_seq=256 (reduced).
    KVCacheConfig config = {30, 256, 3, 64};
    KVCache cache(config);

    // Expected total: 30 * 2 * 256 * 3 * 64 * 4 = 11,796,480 bytes (~11.25 MB)
    const size_t expected = 30ull * 2 * 256 * 3 * 64 * sizeof(float);
    EXPECT_EQ(cache.total_bytes(), expected);

    // Should be constructible without exceeding VRAM budget.
    EXPECT_EQ(cache.current_length(), 0);

    // Smoke test: append a small prefill and read back.
    const int elems_per_token = config.n_kv_heads * config.head_dim;
    std::vector<float> data(elems_per_token, 1.0f);
    float* d_ptr = to_device(data);

    for (int layer = 0; layer < config.n_layers; ++layer) {
        cache.append(layer, d_ptr, d_ptr, 1);
    }
    EXPECT_EQ(cache.current_length(), 1);

    CUINFER_CHECK_CUDA(cudaFree(d_ptr));
}

// ── Test: GQA config (Llama 3.2 1B-like) ───────────────────

TEST(KVCache, GQAConfig) {
    // Llama 3.2 1B: 22 layers, 8 KV heads (32 query heads, ratio 4:1),
    // head_dim=64, max_seq=128 (reduced for testing).
    KVCacheConfig config = {22, 128, 8, 64};
    KVCache cache(config);

    // Verify the KV cache only stores n_kv_heads (8), not n_query_heads (32).
    // This is implicit in the config -- just verify the math is right.
    // Per layer K: 128 * 8 * 64 * 4 = 262144 bytes
    const size_t expected_layer_k = 128ull * 8 * 64 * sizeof(float);
    EXPECT_EQ(config.layer_k_bytes(), expected_layer_k);

    // Total: 22 * 2 * 262144 = 11,534,336 bytes
    const size_t expected_total = 22ull * 2 * expected_layer_k;
    EXPECT_EQ(cache.total_bytes(), expected_total);
}

// ── Test: Print memory report (smoke test) ──────────────────

TEST(KVCache, PrintMemoryReport) {
    KVCacheConfig config = {2, 16, 2, 4};
    KVCache cache(config);

    // Just verify it doesn't crash. Output goes to stdout.
    EXPECT_NO_THROW(cache.print_memory_report());

    const int elems_per_token = config.n_kv_heads * config.head_dim;
    std::vector<float> data(elems_per_token, 1.0f);
    float* d_ptr = to_device(data);

    cache.append(0, d_ptr, d_ptr, 1);
    cache.append(1, d_ptr, d_ptr, 1);

    EXPECT_NO_THROW(cache.print_memory_report());

    CUINFER_CHECK_CUDA(cudaFree(d_ptr));
}
