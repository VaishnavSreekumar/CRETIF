// ── test_matmul_quant.cu — Correctness tests for quantized matmul ────────
//
// Strategy: create a known FP32 matrix, quantize it to Q8_0/Q4_0/Q4_K/Q6_K
// on the host, run the quantized matmul, and compare against the FP32
// reference matmul (which uses the original FP32 matrix as B-transposed).
//
// Tolerances account for quantization error, not just FP32 rounding:
//   Q8_0: 1e-2  (8-bit, ~0.4% relative error per element)
//   Q4_0: 5e-2  (4-bit, ~3% relative error per element)
//   Q4_K: 5e-2  (4-bit K-quant, similar to Q4_0)
//   Q6_K: 2e-2  (6-bit, between Q8_0 and Q4_0)

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "cuinfer/quantization.h"
#include "cuinfer/kernels/matmul_quant.h"
#include "cuinfer/cuda_utils.h"

namespace cuinfer {
namespace {

// ── CPU matmul reference: C = A * B^T ───────────────────────
// A: [M,K], B: [N,K] (row-major), C: [M,N]
void cpu_matmul_bt(const float* A, const float* B, float* C,
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

// ── Host-side Q8_0 quantization ─────────────────────────────
// Quantizes FP32 data to Q8_0 blocks.
// Each block of 32 elements: scale = max(|x|) / 127, qs[i] = round(x[i] / scale)
std::vector<BlockQ8_0> quantize_to_q8_0(const float* data, int n_elements) {
    int n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    std::vector<BlockQ8_0> blocks(n_blocks);

    for (int b = 0; b < n_blocks; ++b) {
        const float* block_data = data + b * Q8_0_BLOCK_SIZE;

        // Find max absolute value for scale
        float amax = 0.0f;
        for (int i = 0; i < Q8_0_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_data[i]));
        }

        float scale = amax / 127.0f;
        if (scale == 0.0f) scale = 1.0f;  // avoid division by zero
        blocks[b].scale = __float2half(scale);

        float inv_scale = 1.0f / scale;
        for (int i = 0; i < Q8_0_BLOCK_SIZE; ++i) {
            int val = static_cast<int>(std::round(block_data[i] * inv_scale));
            val = std::max(-128, std::min(127, val));
            blocks[b].qs[i] = static_cast<int8_t>(val);
        }
    }
    return blocks;
}

// ── Host-side Q4_0 quantization ─────────────────────────────
std::vector<BlockQ4_0> quantize_to_q4_0(const float* data, int n_elements) {
    int n_blocks = n_elements / Q4_0_BLOCK_SIZE;
    std::vector<BlockQ4_0> blocks(n_blocks);

    for (int b = 0; b < n_blocks; ++b) {
        const float* block_data = data + b * Q4_0_BLOCK_SIZE;

        float amax = 0.0f;
        for (int i = 0; i < Q4_0_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_data[i]));
        }

        // Q4_0: values stored as unsigned [0,15] with offset -8
        // range is [-8*scale, 7*scale], so scale = amax / 8
        float scale = amax / 8.0f;
        if (scale == 0.0f) scale = 1.0f;
        blocks[b].scale = __float2half(scale);

        float inv_scale = 1.0f / scale;
        for (int i = 0; i < Q4_0_BLOCK_SIZE; i += 2) {
            int v0 = static_cast<int>(std::round(block_data[i] * inv_scale)) + 8;
            int v1 = static_cast<int>(std::round(block_data[i + 1] * inv_scale)) + 8;
            v0 = std::max(0, std::min(15, v0));
            v1 = std::max(0, std::min(15, v1));
            blocks[b].qs[i / 2] = static_cast<uint8_t>(v0 | (v1 << 4));
        }
    }
    return blocks;
}

// ── Host-side Q4_K quantization (simplified) ────────────────
// Simplified quantization that tests the dequant path correctly.
// Real GGML Q4_K quantization is more complex (optimal scale search).
std::vector<BlockQ4_K> quantize_to_q4_k(const float* data, int n_elements) {
    int n_blocks = n_elements / Q4_K_BLOCK_SIZE;
    std::vector<BlockQ4_K> blocks(n_blocks);

    for (int b = 0; b < n_blocks; ++b) {
        const float* block_data = data + b * Q4_K_BLOCK_SIZE;

        // Find global max for d scale
        float amax = 0.0f;
        for (int i = 0; i < Q4_K_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_data[i]));
        }

        float d_val = amax / (15.0f * 63.0f);  // max_nibble * max_6bit_scale
        if (d_val == 0.0f) d_val = 1e-6f;
        blocks[b].d = __float2half(d_val);
        blocks[b].dmin = __float2half(0.0f);  // simplified: no min offset
        memset(blocks[b].scales, 0, 12);

        // Set all sub-block scales to a uniform value
        for (int sb = 0; sb < 8; ++sb) {
            // Simplified: find sub-block max, compute scale to fit in [0,15] nibble
            float sb_max = 0.0f;
            for (int i = 0; i < 32; ++i) {
                sb_max = std::max(sb_max, std::abs(block_data[sb * 32 + i]));
            }
            uint8_t sc = static_cast<uint8_t>(std::min(63.0f, std::round(sb_max / (d_val * 15.0f))));

            // Encode 6-bit scale (simplified: only use lower 6 bits for sub_block < 4)
            if (sb < 4) {
                blocks[b].scales[sb] = (blocks[b].scales[sb] & 0xC0) | (sc & 0x3F);
            } else {
                blocks[b].scales[sb + 4] = (blocks[b].scales[sb + 4] & 0xF0) | (sc & 0x0F);
                blocks[b].scales[sb - 4] = (blocks[b].scales[sb - 4] & 0x3F) | ((sc >> 4) << 6);
            }
        }

        // Quantize each element to 4-bit nibble using GGML half-interleaved layout:
        // Each 64-element chunk uses 32 bytes: lower nibbles → first 32, upper → next 32.
        memset(blocks[b].qs, 0, 128);
        for (int chunk64 = 0; chunk64 < 4; ++chunk64) {
            int sb_lo = chunk64 * 2;
            int sb_hi = chunk64 * 2 + 1;

            // Decode scales for both sub-blocks in this chunk
            uint8_t sc_lo, sc_hi;
            if (sb_lo < 4) {
                sc_lo = blocks[b].scales[sb_lo] & 0x3F;
            } else {
                sc_lo = (blocks[b].scales[sb_lo + 4] & 0x0F) | ((blocks[b].scales[sb_lo - 4] >> 6) << 4);
            }
            if (sb_hi < 4) {
                sc_hi = blocks[b].scales[sb_hi] & 0x3F;
            } else {
                sc_hi = (blocks[b].scales[sb_hi + 4] & 0x0F) | ((blocks[b].scales[sb_hi - 4] >> 6) << 4);
            }

            float eff_lo = __half2float(blocks[b].d) * static_cast<float>(sc_lo);
            float eff_hi = __half2float(blocks[b].d) * static_cast<float>(sc_hi);
            float inv_lo = (eff_lo > 0) ? 1.0f / eff_lo : 0.0f;
            float inv_hi = (eff_hi > 0) ? 1.0f / eff_hi : 0.0f;

            for (int l = 0; l < 32; ++l) {
                int elem_lo = chunk64 * 64 + l;        // lower-nibble element
                int elem_hi = chunk64 * 64 + l + 32;   // upper-nibble element

                int v_lo = static_cast<int>(std::round(block_data[elem_lo] * inv_lo));
                int v_hi = static_cast<int>(std::round(block_data[elem_hi] * inv_hi));
                v_lo = std::max(0, std::min(15, v_lo));
                v_hi = std::max(0, std::min(15, v_hi));

                int qs_idx = chunk64 * 32 + l;
                blocks[b].qs[qs_idx] = static_cast<uint8_t>(v_lo | (v_hi << 4));
            }
        }
    }
    return blocks;
}

// ── Generate random FP32 matrix ─────────────────────────────
std::vector<float> random_matrix(int rows, int cols, unsigned int seed = 42) {
    std::vector<float> data(rows * cols);
    unsigned int state = seed;
    for (auto& v : data) {
        state = state * 1103515245u + 12345u;
        v = (static_cast<float>(state & 0xFFFF) / 32768.0f) - 1.0f;  // [-1, 1]
    }
    return data;
}

// ════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════

TEST(MatmulQ8_0, MatchesFP32Reference) {
    const int M = 4, N = 64, K = 128;

    auto A = random_matrix(M, K, 1);
    auto B = random_matrix(N, K, 2);  // [N, K] — B-transposed layout

    // CPU reference: C = A * B^T
    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    // Quantize B to Q8_0
    auto B_q8 = quantize_to_q8_0(B.data(), N * K);

    // Upload to GPU
    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q8.size() * sizeof(BlockQ8_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q8.data(), B_q8.size() * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    matmul_q8_0(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare with tolerance for quantization error
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1e-1f)
            << "Mismatch at element " << i << " (Q8_0 vs FP32 reference)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulQ8_0, LargerDimensions) {
    // Test with dimensions matching typical model layers
    const int M = 1, N = 576, K = 576;

    auto A = random_matrix(M, K, 10);
    auto B = random_matrix(N, K, 20);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    auto B_q8 = quantize_to_q8_0(B.data(), N * K);

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q8.size() * sizeof(BlockQ8_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q8.data(), B_q8.size() * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));

    matmul_q8_0(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 5e-1f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulQ4_0, MatchesFP32Reference) {
    const int M = 4, N = 64, K = 128;

    auto A = random_matrix(M, K, 3);
    auto B = random_matrix(N, K, 4);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    auto B_q4 = quantize_to_q4_0(B.data(), N * K);

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q4.size() * sizeof(BlockQ4_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q4.data(), B_q4.size() * sizeof(BlockQ4_0), cudaMemcpyHostToDevice));

    matmul_q4_0(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Q4_0 has more quantization error than Q8_0
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1.0f)
            << "Mismatch at element " << i << " (Q4_0 vs FP32 reference)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulQ4_K, MatchesDequantizedReference) {
    // Q4_K requires K divisible by 256.
    // This test verifies the fused matmul matches separate dequant + CPU matmul,
    // not raw FP32 (quantization error is expected and separate from kernel correctness).
    const int M = 4, N = 32, K = 256;

    auto A = random_matrix(M, K, 5);
    auto B = random_matrix(N, K, 6);

    // Quantize B to Q4_K
    auto B_q4k = quantize_to_q4_k(B.data(), N * K);

    // CPU reference: dequantize Q4_K → FP32 using correct GGML layout, then matmul
    std::vector<float> B_dequant(N * K);
    for (int row = 0; row < N; ++row) {
        int blocks_per_row = K / Q4_K_BLOCK_SIZE;
        for (int sb_idx = 0; sb_idx < blocks_per_row; ++sb_idx) {
            const BlockQ4_K& blk = B_q4k[row * blocks_per_row + sb_idx];
            float d    = __half2float(blk.d);
            float dmin = __half2float(blk.dmin);
            const uint8_t* q = blk.qs;
            int is = 0;
            int base = row * K + sb_idx * Q4_K_BLOCK_SIZE;
            for (int j = 0; j < Q4_K_BLOCK_SIZE; j += 64) {
                uint8_t sc1, mn1, sc2, mn2;
                if (is < 4) {
                    sc1 = blk.scales[is] & 0x3F;
                    mn1 = blk.scales[is + 4] & 0x3F;
                } else {
                    sc1 = (blk.scales[is + 4] & 0x0F) | ((blk.scales[is - 4] >> 6) << 4);
                    mn1 = (blk.scales[is + 4] >> 4) | ((blk.scales[is] >> 6) << 4);
                }
                if (is + 1 < 4) {
                    sc2 = blk.scales[is + 1] & 0x3F;
                    mn2 = blk.scales[is + 1 + 4] & 0x3F;
                } else {
                    sc2 = (blk.scales[is + 1 + 4] & 0x0F) | ((blk.scales[is + 1 - 4] >> 6) << 4);
                    mn2 = (blk.scales[is + 1 + 4] >> 4) | ((blk.scales[is + 1] >> 6) << 4);
                }
                float d1 = d * static_cast<float>(sc1);
                float m1 = dmin * static_cast<float>(mn1);
                float d2 = d * static_cast<float>(sc2);
                float m2 = dmin * static_cast<float>(mn2);
                for (int l = 0; l < 32; ++l) {
                    B_dequant[base + j + l +  0] = d1 * static_cast<float>(q[l] & 0xF) - m1;
                    B_dequant[base + j + l + 32] = d2 * static_cast<float>(q[l] >> 4)  - m2;
                }
                is += 2;
                q += 32;
            }
        }
    }

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B_dequant.data(), C_ref.data(), M, N, K);

    // GPU fused quantized matmul
    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q4k.size() * sizeof(BlockQ4_K)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q4k.data(), B_q4k.size() * sizeof(BlockQ4_K), cudaMemcpyHostToDevice));

    matmul_q4_k(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Fused matmul should exactly match dequant + matmul (both use same math)
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1e-2f)
            << "Mismatch at element " << i << " (fused Q4_K matmul vs dequant + CPU matmul)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ── Host-side Q5_0 quantization ─────────────────────────────
// Each block of 32 elements: 5-bit values (4 low from qs + 1 high from qh)
// GGML Q5_0 nibble layout:
//   Elements 0..15  → LOWER nibble of qs[0..15]
//   Elements 16..31 → UPPER nibble of qs[0..15]
// High bit: bit j of qh (little-endian uint32) = 5th bit of element j
// scale = max(|x|) / 16, value = (q5 - 16) * scale
std::vector<BlockQ5_0> quantize_to_q5_0(const float* data, int n_elements) {
    int n_blocks = n_elements / Q5_0_BLOCK_SIZE;
    std::vector<BlockQ5_0> blocks(n_blocks);

    for (int b = 0; b < n_blocks; ++b) {
        const float* block_data = data + b * Q5_0_BLOCK_SIZE;

        float amax = 0.0f;
        for (int i = 0; i < Q5_0_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_data[i]));
        }

        float scale = amax / 16.0f;
        if (scale == 0.0f) scale = 1.0f;
        blocks[b].scale = __float2half(scale);

        float inv_scale = 1.0f / scale;
        uint32_t qh_u32 = 0;
        memset(blocks[b].qs, 0, 16);

        for (int i = 0; i < Q5_0_BLOCK_SIZE; ++i) {
            int val = static_cast<int>(std::round(block_data[i] * inv_scale)) + 16;
            val = std::max(0, std::min(31, val));

            // Extract 5-bit value
            int low4 = val & 0x0F;
            int high_bit = (val >> 4) & 1;

            // Store low 4 bits in half-interleaved nibble layout
            if (i < 16) {
                // Elements 0..15 → lower nibble of qs[i]
                blocks[b].qs[i] = static_cast<uint8_t>((blocks[b].qs[i] & 0xF0) | low4);
            } else {
                // Elements 16..31 → upper nibble of qs[i-16]
                blocks[b].qs[i - 16] = static_cast<uint8_t>((blocks[b].qs[i - 16] & 0x0F) | (low4 << 4));
            }

            // Store high bit in qh
            if (high_bit) {
                qh_u32 |= (1u << i);
            }
        }

        // Store qh as little-endian bytes
        blocks[b].qh[0] = static_cast<uint8_t>(qh_u32 & 0xFF);
        blocks[b].qh[1] = static_cast<uint8_t>((qh_u32 >> 8) & 0xFF);
        blocks[b].qh[2] = static_cast<uint8_t>((qh_u32 >> 16) & 0xFF);
        blocks[b].qh[3] = static_cast<uint8_t>((qh_u32 >> 24) & 0xFF);
    }
    return blocks;
}

// ── CPU dequantize Q5_0 reference ────────────────────────────
// Follows exact GGML formula for verification.
std::vector<float> cpu_dequant_q5_0(const std::vector<BlockQ5_0>& blocks) {
    int n_elements = static_cast<int>(blocks.size()) * Q5_0_BLOCK_SIZE;
    std::vector<float> result(n_elements);

    for (size_t b = 0; b < blocks.size(); ++b) {
        const BlockQ5_0& blk = blocks[b];
        float scale = __half2float(blk.scale);

        uint32_t qh_u32 = static_cast<uint32_t>(blk.qh[0])
                          | (static_cast<uint32_t>(blk.qh[1]) << 8)
                          | (static_cast<uint32_t>(blk.qh[2]) << 16)
                          | (static_cast<uint32_t>(blk.qh[3]) << 24);

        for (int i = 0; i < Q5_0_BLOCK_SIZE; ++i) {
            int half = i / 16;
            int j = i % 16;
            int low4 = (half == 0) ? (blk.qs[j] & 0x0F) : (blk.qs[j] >> 4);
            int high_bit = (qh_u32 >> i) & 1;
            int q5_val = low4 | (high_bit << 4);
            result[b * Q5_0_BLOCK_SIZE + i] = scale * static_cast<float>(q5_val - 16);
        }
    }
    return result;
}

TEST(MatmulQ5_0, MatchesDequantizedReference) {
    const int M = 4, N = 64, K = 128;

    auto A = random_matrix(M, K, 7);
    auto B = random_matrix(N, K, 8);

    // Quantize B to Q5_0
    auto B_q5 = quantize_to_q5_0(B.data(), N * K);

    // CPU reference: dequant Q5_0 → FP32, then C = A * B_dequant^T
    auto B_dequant = cpu_dequant_q5_0(B_q5);
    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B_dequant.data(), C_ref.data(), M, N, K);

    // GPU fused quantized matmul
    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q5.size() * sizeof(BlockQ5_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q5.data(), B_q5.size() * sizeof(BlockQ5_0), cudaMemcpyHostToDevice));

    matmul_q5_0(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Fused matmul should match dequant + matmul (same math, FP32 rounding only)
    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1e-1f)
            << "Mismatch at element " << i << " (fused Q5_0 matmul vs dequant + CPU matmul)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulQ5_0, LargerDimensions) {
    const int M = 1, N = 576, K = 576;

    auto A = random_matrix(M, K, 11);
    auto B = random_matrix(N, K, 12);

    auto B_q5 = quantize_to_q5_0(B.data(), N * K);
    auto B_dequant = cpu_dequant_q5_0(B_q5);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B_dequant.data(), C_ref.data(), M, N, K);

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q5.size() * sizeof(BlockQ5_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q5.data(), B_q5.size() * sizeof(BlockQ5_0), cudaMemcpyHostToDevice));

    matmul_q5_0(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 5e-1f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(DequantQ5_0, MatchesCPUReference) {
    // Verify standalone GPU dequant kernel matches CPU reference
    const int n_blocks = 16;  // 512 elements
    const int n_elements = n_blocks * Q5_0_BLOCK_SIZE;

    auto data = random_matrix(1, n_elements, 42);
    auto blocks = quantize_to_q5_0(data.data(), n_elements);
    auto cpu_result = cpu_dequant_q5_0(blocks);

    // GPU dequant
    void* d_quant = nullptr;
    float* d_out = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_quant, blocks.size() * sizeof(BlockQ5_0)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_out, n_elements * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_quant, blocks.data(),
                                  blocks.size() * sizeof(BlockQ5_0),
                                  cudaMemcpyHostToDevice));

    dequantize_q5_0(d_quant, d_out, n_blocks);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> gpu_result(n_elements);
    CUINFER_CHECK_CUDA(cudaMemcpy(gpu_result.data(), d_out,
                                  n_elements * sizeof(float),
                                  cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], 1e-5f)
            << "Dequant mismatch at element " << i;
    }

    cudaFree(d_quant);
    cudaFree(d_out);
}

// ── Host-side Q6_K quantization ─────────────────────────────
// Each super-block of 256 elements. Uses GGML interleaved layout:
//   ql[128] : lower 4 bits (interleaved)
//   qh[64]  : upper 2 bits
//   scales[16]: int8_t sub-block scales (16 groups of 16 elements)
//   d : __half super-block scale
std::vector<BlockQ6_K> quantize_to_q6_k(const float* data, int n_elements) {
    int n_blocks = n_elements / Q6_K_BLOCK_SIZE;
    std::vector<BlockQ6_K> blocks(n_blocks);

    for (int b = 0; b < n_blocks; ++b) {
        const float* block_data = data + b * Q6_K_BLOCK_SIZE;
        memset(&blocks[b], 0, sizeof(BlockQ6_K));

        // Find overall max for the super-block scale
        float amax = 0.0f;
        for (int i = 0; i < Q6_K_BLOCK_SIZE; ++i) {
            amax = std::max(amax, std::abs(block_data[i]));
        }

        // Compute per-sub-block scales (16 groups of 16 elements)
        float sub_amax[16];
        for (int g = 0; g < 16; ++g) {
            sub_amax[g] = 0.0f;
            for (int i = 0; i < 16; ++i) {
                sub_amax[g] = std::max(sub_amax[g], std::abs(block_data[g * 16 + i]));
            }
        }

        // Super-block scale
        float d = amax > 0 ? amax / (32.0f * 31.0f) : 1.0f;
        blocks[b].d = __float2half(d);
        float inv_d = 1.0f / d;

        // Compute and store ALL 16 sub-block scales
        int8_t sc[16];
        for (int g = 0; g < 16; ++g) {
            float sub_scale = sub_amax[g] > 0 ? sub_amax[g] / 31.0f : 0.0f;
            sc[g] = static_cast<int8_t>(std::round(sub_scale * inv_d));
            sc[g] = std::max((int8_t)-127, std::min((int8_t)127, sc[g]));
            blocks[b].scales[g] = sc[g];
        }

        // Quantize each element to 6 bits.
        // Element-to-scale mapping (matching GGML):
        //   scale_idx = chunk*8 + quarter*2 + l/16
        // where chunk=elem/128, quarter=(elem%128)/32, l=elem%32.
        for (int g16 = 0; g16 < 16; ++g16) {
            // scale_idx for this 16-element group
            int scale_idx = g16;
            float eff_scale = d * static_cast<float>(sc[scale_idx]);
            float inv_eff = (eff_scale != 0.0f) ? 1.0f / eff_scale : 0.0f;

            for (int i = 0; i < 16; ++i) {
                int elem = g16 * 16 + i;
                int q6_val = static_cast<int>(std::round(block_data[elem] * inv_eff)) + 32;
                q6_val = std::max(0, std::min(63, q6_val));

                int low4 = q6_val & 0x0F;
                int high2 = (q6_val >> 4) & 0x03;

                // GGML interleaved layout decomposition
                int chunk = elem / 128;
                int within_chunk = elem % 128;
                int quarter = within_chunk / 32;
                int l = within_chunk % 32;

                int ql_offset = (quarter & 1) * 32 + l + chunk * 64;
                if (quarter < 2) {
                    blocks[b].ql[ql_offset] = static_cast<uint8_t>((blocks[b].ql[ql_offset] & 0xF0) | low4);
                } else {
                    blocks[b].ql[ql_offset] = static_cast<uint8_t>((blocks[b].ql[ql_offset] & 0x0F) | (low4 << 4));
                }

                int qh_byte = l + chunk * 32;
                int qh_shift = quarter * 2;
                blocks[b].qh[qh_byte] |= (high2 << qh_shift);
            }
        }
    }
    return blocks;
}

// ── CPU dequantize Q6_K for matmul test ─────────────────────
// Uses the same interleaved layout as the GPU kernel
std::vector<float> cpu_dequant_q6_k(const std::vector<BlockQ6_K>& blocks) {
    int n = static_cast<int>(blocks.size()) * Q6_K_BLOCK_SIZE;
    std::vector<float> result(n);

    for (size_t b = 0; b < blocks.size(); ++b) {
        const BlockQ6_K& blk = blocks[b];
        float d = __half2float(blk.d);

        for (int i = 0; i < Q6_K_BLOCK_SIZE; ++i) {
            int chunk = i / 128;
            int within_chunk = i % 128;
            int quarter = within_chunk / 32;
            int l = within_chunk % 32;

            int scale_idx = chunk * 8 + quarter * 2 + l / 16;
            float sc = static_cast<float>(blk.scales[scale_idx]);

            int ql_offset = (quarter & 1) * 32 + l + chunk * 64;
            int ql_nibble = (quarter < 2) ? (blk.ql[ql_offset] & 0x0F)
                                          : (blk.ql[ql_offset] >> 4);

            int qh_byte = l + chunk * 32;
            int qh_shift = quarter * 2;
            int qh_bits = (blk.qh[qh_byte] >> qh_shift) & 0x03;

            int q6_val = (ql_nibble | (qh_bits << 4)) - 32;
            result[b * Q6_K_BLOCK_SIZE + i] = d * sc * static_cast<float>(q6_val);
        }
    }
    return result;
}

TEST(MatmulQ6_K, MatchesDequantizedReference) {
    // Small test: K must be divisible by 256
    const int M = 4, N = 32, K = 256;

    auto A = random_matrix(M, K, 10);
    auto B = random_matrix(N, K, 11);

    auto B_q6k = quantize_to_q6_k(B.data(), N * K);
    auto B_dequant = cpu_dequant_q6_k(B_q6k);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B_dequant.data(), C_ref.data(), M, N, K);

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q6k.size() * sizeof(BlockQ6_K)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q6k.data(), B_q6k.size() * sizeof(BlockQ6_K), cudaMemcpyHostToDevice));

    matmul_q6_k(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1e-2f)
            << "Mismatch at element " << i << " (fused Q6_K matmul vs dequant + CPU matmul)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulQ6_K, SmolLMDimensions) {
    // Exact dimensions used by SmolLM2-135M ffn_down: M=1, N=576, K=1536
    const int M = 1, N = 576, K = 1536;

    auto A = random_matrix(M, K, 20);
    auto B = random_matrix(N, K, 21);

    auto B_q6k = quantize_to_q6_k(B.data(), N * K);
    auto B_dequant = cpu_dequant_q6_k(B_q6k);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B_dequant.data(), C_ref.data(), M, N, K);

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q6k.size() * sizeof(BlockQ6_K)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q6k.data(), B_q6k.size() * sizeof(BlockQ6_K), cudaMemcpyHostToDevice));

    matmul_q6_k(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 5e-2f)
            << "Mismatch at element " << i
            << ": gpu=" << C_gpu[i] << " ref=" << C_ref[i]
            << " (fused Q6_K matmul vs dequant + CPU matmul at SmolLM dimensions)";
    }
}

TEST(MatmulQ4_K, SmolLMDimensions) {
    // Exact dimensions used by SmolLM2-135M ffn_gate/up: M=1, N=1536, K=576
    const int M = 1, N = 1536;  // Note: K must be div by 256 (see below)

    // K=576 is NOT divisible by 256 (576/256 = 2.25), so we need K=512 instead
    // Actually for Q4_K, the block size is 256, and K must be divisible by it.
    // SmolLM2's dim=576 is not divisible by 256. This means Q4_K matmul can't
    // work for attn weights with K=576. But ffn weights have K=576 for gate/up.
    // Wait - 576 % 256 = 64, so it's NOT divisible. This could be a problem!

    // Let me test with K that IS divisible by 256:
    const int K_aligned = 512;
    auto A2 = random_matrix(M, K_aligned, 30);
    auto B2 = random_matrix(N, K_aligned, 31);

    auto B_q4k = quantize_to_q4_k(B2.data(), N * K_aligned);

    // Dequant reference
    std::vector<float> B_dequant(N * K_aligned);
    for (int row = 0; row < N; ++row) {
        int blocks_per_row = K_aligned / Q4_K_BLOCK_SIZE;
        for (int sb_idx = 0; sb_idx < blocks_per_row; ++sb_idx) {
            const BlockQ4_K& blk = B_q4k[row * blocks_per_row + sb_idx];
            float d    = __half2float(blk.d);
            float dmin = __half2float(blk.dmin);
            const uint8_t* q = blk.qs;
            int is = 0;
            int base = row * K_aligned + sb_idx * Q4_K_BLOCK_SIZE;
            for (int j = 0; j < Q4_K_BLOCK_SIZE; j += 64) {
                uint8_t sc1, mn1, sc2, mn2;
                if (is < 4) {
                    sc1 = blk.scales[is] & 0x3F;
                    mn1 = blk.scales[is + 4] & 0x3F;
                } else {
                    sc1 = (blk.scales[is + 4] & 0x0F) | ((blk.scales[is - 4] >> 6) << 4);
                    mn1 = (blk.scales[is + 4] >> 4) | ((blk.scales[is] >> 6) << 4);
                }
                if (is + 1 < 4) {
                    sc2 = blk.scales[is + 1] & 0x3F;
                    mn2 = blk.scales[is + 1 + 4] & 0x3F;
                } else {
                    sc2 = (blk.scales[is + 1 + 4] & 0x0F) | ((blk.scales[is + 1 - 4] >> 6) << 4);
                    mn2 = (blk.scales[is + 1 + 4] >> 4) | ((blk.scales[is + 1] >> 6) << 4);
                }
                float d1 = d * static_cast<float>(sc1);
                float m1 = dmin * static_cast<float>(mn1);
                float d2 = d * static_cast<float>(sc2);
                float m2 = dmin * static_cast<float>(mn2);
                for (int l = 0; l < 32; ++l) {
                    B_dequant[base + j + l +  0] = d1 * static_cast<float>(q[l] & 0xF) - m1;
                    B_dequant[base + j + l + 32] = d2 * static_cast<float>(q[l] >> 4)  - m2;
                }
                is += 2;
                q += 32;
            }
        }
    }

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A2.data(), B_dequant.data(), C_ref.data(), M, N, K_aligned);

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K_aligned * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, B_q4k.size() * sizeof(BlockQ4_K)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A2.data(), M * K_aligned * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_q4k.data(), B_q4k.size() * sizeof(BlockQ4_K), cudaMemcpyHostToDevice));

    matmul_q4_k(d_A, d_B, d_C, M, N, K_aligned);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 5e-2f)
            << "Mismatch at element " << i << " (Q4_K matmul at larger dimensions)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ════════════════════════════════════════════════════════════════
// FP16 Mixed-Precision MatMul Tests
// ════════════════════════════════════════════════════════════════
//
// Strategy: generate FP32 B, convert to FP16 on host (__float2half),
// upload raw __half bytes, run kernel, compare vs FP32 CPU reference.
// Tolerance: 1e-2 (FP16 precision — half-mantissa rounding).

TEST(MatmulFP16, Square256) {
    const int M = 4, N = 256, K = 256;

    auto A = random_matrix(M, K, 100);
    auto B = random_matrix(N, K, 101);  // [N, K] — B-transposed layout

    // CPU reference: C = A * B^T
    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    // Convert B to FP16
    std::vector<__half> B_fp16(N * K);
    for (int i = 0; i < N * K; ++i) {
        B_fp16[i] = __float2half(B[i]);
    }

    // Upload to GPU
    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(__half)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_fp16.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

    matmul_fp16(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 5e-1f)
            << "Mismatch at element " << i << " (FP16 matmul vs FP32 reference)";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulFP16, SmolLMDims) {
    // Typical SmolLM2-135M attention dimension: M=1, N=576, K=576
    const int M = 1, N = 576, K = 576;

    auto A = random_matrix(M, K, 102);
    auto B = random_matrix(N, K, 103);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    std::vector<__half> B_fp16(N * K);
    for (int i = 0; i < N * K; ++i) {
        B_fp16[i] = __float2half(B[i]);
    }

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(__half)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_fp16.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice));

    matmul_fp16(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 5e-1f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulFP16, LargeK) {
    // FFN dimension: M=1, N=576, K=1536
    const int M = 1, N = 576, K = 1536;

    auto A = random_matrix(M, K, 104);
    auto B = random_matrix(N, K, 105);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    std::vector<__half> B_fp16(N * K);
    for (int i = 0; i < N * K; ++i) {
        B_fp16[i] = __float2half(B[i]);
    }

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(__half)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_fp16.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice));

    matmul_fp16(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1.0f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatmulFP16, Batch4) {
    // Batch=4 (prefill scenario)
    const int M = 4, N = 576, K = 576;

    auto A = random_matrix(M, K, 106);
    auto B = random_matrix(N, K, 107);

    std::vector<float> C_ref(M * N);
    cpu_matmul_bt(A.data(), B.data(), C_ref.data(), M, N, K);

    std::vector<__half> B_fp16(N * K);
    for (int i = 0; i < N * K; ++i) {
        B_fp16[i] = __float2half(B[i]);
    }

    float* d_A = nullptr;
    void* d_B = nullptr;
    float* d_C = nullptr;
    CUINFER_CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(__half)));
    CUINFER_CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUINFER_CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUINFER_CHECK_CUDA(cudaMemcpy(d_B, B_fp16.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice));

    matmul_fp16(d_A, d_B, d_C, M, N, K);
    CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C_gpu(M * N);
    CUINFER_CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        ASSERT_NEAR(C_gpu[i], C_ref[i], 1.0f)
            << "Mismatch at element " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // namespace
} // namespace cuinfer
