// ── test_dequantize.cu — Correctness tests for dequantization kernels ────
//
// Tests GPU dequantize kernels against CPU reference implementations.
// Uses synthetic data with known values for reproducible verification.
// Tolerance: 1e-5 for Q8_0/Q6_K (lossless-ish), 1e-5 for Q4_0/Q4_K
// (the dequant itself is exact — precision loss is from quantization).

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "cuinfer/quantization.h"
#include "cuinfer/cuda_utils.h"

namespace cuinfer {
namespace {

// ── CPU Reference: Q8_0 Dequantization ──────────────────────

void cpu_dequantize_q8_0(const BlockQ8_0* blocks, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        float scale = __half2float(blocks[b].scale);
        for (int i = 0; i < Q8_0_BLOCK_SIZE; ++i) {
            out[b * Q8_0_BLOCK_SIZE + i] = scale * static_cast<float>(blocks[b].qs[i]);
        }
    }
}

// ── CPU Reference: Q4_0 Dequantization ──────────────────────

void cpu_dequantize_q4_0(const BlockQ4_0* blocks, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        float scale = __half2float(blocks[b].scale);
        for (int i = 0; i < Q4_0_BLOCK_SIZE; ++i) {
            int byte_idx = i / 2;
            uint8_t packed = blocks[b].qs[byte_idx];
            int nibble;
            if (i % 2 == 0) {
                nibble = packed & 0x0F;
            } else {
                nibble = packed >> 4;
            }
            out[b * Q4_0_BLOCK_SIZE + i] = scale * static_cast<float>(nibble - 8);
        }
    }
}

// ── CPU Reference: Q6_K Dequantization ──────────────────────
// Matches GGML's dequantize_row_q6_K interleaved layout exactly.

void cpu_dequantize_q6_k(const BlockQ6_K* blocks, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        float d = __half2float(blocks[b].d);
        const uint8_t* ql = blocks[b].ql;
        const uint8_t* qh = blocks[b].qh;
        const int8_t*  sc = blocks[b].scales;

        // Process 2 chunks of 128 elements each
        for (int n = 0; n < Q6_K_BLOCK_SIZE; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = n / 16 + l / 16;
                int8_t q1 = static_cast<int8_t>(((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                int8_t q2 = static_cast<int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                int8_t q3 = static_cast<int8_t>(((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32);
                int8_t q4 = static_cast<int8_t>(((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32);

                int base = b * Q6_K_BLOCK_SIZE + n;
                out[base + l +  0] = d * sc[is + 0] * q1;
                out[base + l + 32] = d * sc[is + 2] * q2;
                out[base + l + 64] = d * sc[is + 4] * q3;
                out[base + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
        }
    }
}

// ── CPU Reference: Q4_K Dequantization ──────────────────────
// Matches GGML's dequantize_row_q4_K half-interleaved nibble layout.

void cpu_dequantize_q4_k(const BlockQ4_K* blocks, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        float d    = __half2float(blocks[b].d);
        float dmin = __half2float(blocks[b].dmin);
        const uint8_t* q = blocks[b].qs;

        int is = 0;
        int out_idx = b * Q4_K_BLOCK_SIZE;
        for (int j = 0; j < Q4_K_BLOCK_SIZE; j += 64) {
            // Decode scale and min for sub-blocks 'is' and 'is+1'
            uint8_t sc1, mn1, sc2, mn2;
            if (is < 4) {
                sc1 = blocks[b].scales[is] & 0x3F;
                mn1 = blocks[b].scales[is + 4] & 0x3F;
            } else {
                sc1 = (blocks[b].scales[is + 4] & 0x0F) |
                      ((blocks[b].scales[is - 4] >> 6) << 4);
                mn1 = (blocks[b].scales[is + 4] >> 4) |
                      ((blocks[b].scales[is] >> 6) << 4);
            }
            if (is + 1 < 4) {
                sc2 = blocks[b].scales[is + 1] & 0x3F;
                mn2 = blocks[b].scales[is + 1 + 4] & 0x3F;
            } else {
                sc2 = (blocks[b].scales[is + 1 + 4] & 0x0F) |
                      ((blocks[b].scales[is + 1 - 4] >> 6) << 4);
                mn2 = (blocks[b].scales[is + 1 + 4] >> 4) |
                      ((blocks[b].scales[is + 1] >> 6) << 4);
            }

            float d1 = d * static_cast<float>(sc1);
            float m1 = dmin * static_cast<float>(mn1);
            float d2 = d * static_cast<float>(sc2);
            float m2 = dmin * static_cast<float>(mn2);

            // 32 bytes of qs: lower nibbles → first 32 elements, upper → next 32
            for (int l = 0; l < 32; ++l) {
                out[out_idx + l +  0] = d1 * static_cast<float>(q[l] & 0xF) - m1;
                out[out_idx + l + 32] = d2 * static_cast<float>(q[l] >> 4)  - m2;
            }

            is += 2;
            q += 32;
            out_idx += 64;
        }
    }
}

// ── Helper: Create synthetic Q8_0 blocks ────────────────────

std::vector<BlockQ8_0> make_q8_0_blocks(int n_blocks) {
    std::vector<BlockQ8_0> blocks(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        // Varied scale factors
        float scale = 0.1f * (b + 1);
        blocks[b].scale = __float2half(scale);
        for (int i = 0; i < 32; ++i) {
            // Values in [-127, 127] range
            blocks[b].qs[i] = static_cast<int8_t>((i * 7 + b * 13) % 255 - 127);
        }
    }
    return blocks;
}

// ── Helper: Create synthetic Q4_0 blocks ────────────────────

std::vector<BlockQ4_0> make_q4_0_blocks(int n_blocks) {
    std::vector<BlockQ4_0> blocks(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        float scale = 0.05f * (b + 1);
        blocks[b].scale = __float2half(scale);
        for (int i = 0; i < 16; ++i) {
            // Pack two nibbles per byte: low = [0,15], high = [0,15]
            uint8_t lo = (i + b) % 16;
            uint8_t hi = (i * 3 + b * 5) % 16;
            blocks[b].qs[i] = lo | (hi << 4);
        }
    }
    return blocks;
}

// ── Helper: Create synthetic Q6_K blocks ────────────────────

std::vector<BlockQ6_K> make_q6_k_blocks(int n_blocks) {
    std::vector<BlockQ6_K> blocks(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        blocks[b].d = __float2half(0.02f * (b + 1));
        for (int i = 0; i < 16; ++i) {
            blocks[b].scales[i] = static_cast<int8_t>((i * 3 + b) % 127 - 63);
        }
        // ql: packed nibbles
        for (int i = 0; i < 128; ++i) {
            uint8_t lo = (i + b) % 16;
            uint8_t hi = (i * 3 + b) % 16;
            blocks[b].ql[i] = lo | (hi << 4);
        }
        // qh: packed 2-bit values
        for (int i = 0; i < 64; ++i) {
            blocks[b].qh[i] = static_cast<uint8_t>((i + b * 7) % 256);
        }
    }
    return blocks;
}

// ── Helper: Create synthetic Q4_K blocks ────────────────────

std::vector<BlockQ4_K> make_q4_k_blocks(int n_blocks) {
    std::vector<BlockQ4_K> blocks(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        blocks[b].d    = __float2half(0.03f * (b + 1));
        blocks[b].dmin = __float2half(0.01f * (b + 1));
        // Fill scales with values that exercise the 6-bit packing
        for (int i = 0; i < 12; ++i) {
            blocks[b].scales[i] = static_cast<uint8_t>((i * 17 + b * 11) % 256);
        }
        // Fill qs with packed nibbles
        for (int i = 0; i < 128; ++i) {
            uint8_t lo = (i + b * 3) % 16;
            uint8_t hi = (i * 5 + b) % 16;
            blocks[b].qs[i] = lo | (hi << 4);
        }
    }
    return blocks;
}

// ════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════

TEST(DequantizeQ8_0, MatchesCPUReference) {
    constexpr int N_BLOCKS = 64;
    constexpr int N_ELEMENTS = N_BLOCKS * Q8_0_BLOCK_SIZE;

    auto blocks = make_q8_0_blocks(N_BLOCKS);

    // CPU reference
    std::vector<float> cpu_out(N_ELEMENTS);
    cpu_dequantize_q8_0(blocks.data(), cpu_out.data(), N_BLOCKS);

    // GPU dequantize
    void* d_src = nullptr;
    float* d_dst = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_src, N_BLOCKS * sizeof(BlockQ8_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_dst, N_ELEMENTS * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_src, blocks.data(),
                                  N_BLOCKS * sizeof(BlockQ8_0),
                                  cudaMemcpyHostToDevice));

    dequantize_q8_0(d_src, d_dst, N_BLOCKS);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> gpu_out(N_ELEMENTS);
    CUINFER_CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_dst,
                                  N_ELEMENTS * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_ELEMENTS; ++i) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(DequantizeQ8_0, SingleBlock) {
    // Verify with a single block of known values
    BlockQ8_0 block;
    block.scale = __float2half(0.5f);
    for (int i = 0; i < 32; ++i) {
        block.qs[i] = static_cast<int8_t>(i - 16);  // [-16, 15]
    }

    void* d_src = nullptr;
    float* d_dst = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_src, sizeof(BlockQ8_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_dst, 32 * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_src, &block, sizeof(BlockQ8_0),
                                  cudaMemcpyHostToDevice));

    dequantize_q8_0(d_src, d_dst, 1);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out(32);
    CUINFER_CHECK_CUDA(cudaMemcpy(out.data(), d_dst, 32 * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; ++i) {
        float expected = 0.5f * static_cast<float>(i - 16);
        ASSERT_NEAR(out[i], expected, 1e-3f) << "At element " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(DequantizeQ4_0, MatchesCPUReference) {
    constexpr int N_BLOCKS = 64;
    constexpr int N_ELEMENTS = N_BLOCKS * Q4_0_BLOCK_SIZE;

    auto blocks = make_q4_0_blocks(N_BLOCKS);

    // CPU reference
    std::vector<float> cpu_out(N_ELEMENTS);
    cpu_dequantize_q4_0(blocks.data(), cpu_out.data(), N_BLOCKS);

    // GPU dequantize
    void* d_src = nullptr;
    float* d_dst = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_src, N_BLOCKS * sizeof(BlockQ4_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_dst, N_ELEMENTS * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_src, blocks.data(),
                                  N_BLOCKS * sizeof(BlockQ4_0),
                                  cudaMemcpyHostToDevice));

    dequantize_q4_0(d_src, d_dst, N_BLOCKS);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> gpu_out(N_ELEMENTS);
    CUINFER_CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_dst,
                                  N_ELEMENTS * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_ELEMENTS; ++i) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(DequantizeQ4_0, SingleBlock) {
    BlockQ4_0 block;
    block.scale = __float2half(1.0f);
    // All nibbles set to 8 (which minus 8 = 0)
    memset(block.qs, 0x88, 16);

    void* d_src = nullptr;
    float* d_dst = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_src, sizeof(BlockQ4_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_dst, 32 * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_src, &block, sizeof(BlockQ4_0),
                                  cudaMemcpyHostToDevice));

    dequantize_q4_0(d_src, d_dst, 1);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out(32);
    CUINFER_CHECK_CUDA(cudaMemcpy(out.data(), d_dst, 32 * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    // All values should be 1.0 * (8 - 8) = 0.0
    for (int i = 0; i < 32; ++i) {
        ASSERT_NEAR(out[i], 0.0f, 1e-5f) << "At element " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(DequantizeQ6_K, MatchesCPUReference) {
    constexpr int N_BLOCKS = 8;
    constexpr int N_ELEMENTS = N_BLOCKS * Q6_K_BLOCK_SIZE;

    auto blocks = make_q6_k_blocks(N_BLOCKS);

    // CPU reference
    std::vector<float> cpu_out(N_ELEMENTS);
    cpu_dequantize_q6_k(blocks.data(), cpu_out.data(), N_BLOCKS);

    // GPU dequantize
    void* d_src = nullptr;
    float* d_dst = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_src, N_BLOCKS * sizeof(BlockQ6_K)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_dst, N_ELEMENTS * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_src, blocks.data(),
                                  N_BLOCKS * sizeof(BlockQ6_K),
                                  cudaMemcpyHostToDevice));

    dequantize_q6_k(d_src, d_dst, N_BLOCKS);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> gpu_out(N_ELEMENTS);
    CUINFER_CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_dst,
                                  N_ELEMENTS * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_ELEMENTS; ++i) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(DequantizeQ4_K, MatchesCPUReference) {
    constexpr int N_BLOCKS = 8;
    constexpr int N_ELEMENTS = N_BLOCKS * Q4_K_BLOCK_SIZE;

    auto blocks = make_q4_k_blocks(N_BLOCKS);

    // CPU reference
    std::vector<float> cpu_out(N_ELEMENTS);
    cpu_dequantize_q4_k(blocks.data(), cpu_out.data(), N_BLOCKS);

    // GPU dequantize
    void* d_src = nullptr;
    float* d_dst = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_src, N_BLOCKS * sizeof(BlockQ4_K)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_dst, N_ELEMENTS * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_src, blocks.data(),
                                  N_BLOCKS * sizeof(BlockQ4_K),
                                  cudaMemcpyHostToDevice));

    dequantize_q4_k(d_src, d_dst, N_BLOCKS);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> gpu_out(N_ELEMENTS);
    CUINFER_CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_dst,
                                  N_ELEMENTS * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_ELEMENTS; ++i) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-5f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

} // namespace
} // namespace cuinfer
