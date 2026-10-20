#pragma once

// ── Quantization Primitives ──────────────────────────────────────────
//
// Defines GGML-compatible quantization block structures and GPU
// dequantization kernels. These match the GGUF on-disk format exactly,
// allowing quantized weights to be uploaded to GPU as raw bytes and
// dequantized on the fly during matmul.
//
// Block formats (from GGML specification):
//
// Q8_0:  34 bytes = 1 FP16 scale + 32 int8 values   (32 elements/block)
//        dequant: x[i] = scale * qs[i]
//
// Q4_0:  18 bytes = 1 FP16 scale + 16 packed-nibble bytes (32 elements/block)
//        Each byte holds two 4-bit values (lower nibble first).
//        Values are stored as unsigned [0, 15] with implicit offset of -8.
//        dequant: x[i] = scale * (nibble[i] - 8)
//
// Q4_K: 144 bytes = super-block of 256 elements with nested sub-block scales
//       Contains FP16 d/dmin + 12 bytes sub-block scales + 4 bytes sub-block
//       mins + 128 bytes packed nibbles. 8 sub-blocks of 32 elements each.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cuinfer {

// ── Q8_0 Block ──────────────────────────────────────────────
// 32 elements quantized to 8 bits with one shared scale factor.
// Total: 2 (scale) + 32 (values) = 34 bytes per block.
//
// Memory layout (packed, no padding):
//   [0..1]  : __half scale (FP16)
//   [2..33] : int8_t qs[32] (quantized values)

struct BlockQ8_0 {
    __half scale;       // FP16 scale factor
    int8_t qs[32];      // quantized values
};

static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

// ── Q4_0 Block ──────────────────────────────────────────────
// 32 elements quantized to 4 bits with one shared scale factor.
// Total: 2 (scale) + 16 (nibble pairs) = 18 bytes per block.
//
// Memory layout:
//   [0..1]  : __half scale (FP16)
//   [2..17] : uint8_t qs[16] (packed nibble pairs)
//
// Each byte qs[j] holds two 4-bit values:
//   element 2*j+0 = (qs[j] & 0x0F) - 8
//   element 2*j+1 = (qs[j] >> 4)   - 8

struct BlockQ4_0 {
    __half scale;       // FP16 scale factor
    uint8_t qs[16];     // packed 4-bit values (2 per byte)
};

static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

// ── Q5_0 Block ──────────────────────────────────────────────
// 32 elements quantized to 5 bits with one shared scale factor.
// Total: 2 (scale) + 4 (high-bit mask) + 16 (low nibbles) = 22 bytes per block.
//
// Memory layout:
//   [0..1]  : __half scale (FP16)
//   [2..5]  : uint8_t qh[4] — bit j of qh packs the 5th bit for element j
//   [6..21] : uint8_t qs[16] — packed low 4-bit nibbles (2 per byte)
//
// Reconstruction: q5 = (low_nibble | (high_bit << 4)), range [0, 31]
//   x[i] = scale * (q5 - 16)

struct BlockQ5_0 {
    __half  scale;      // FP16 scale factor
    uint8_t qh[4];      // high bits: bit j = 5th bit of element j
    uint8_t qs[16];     // packed low 4-bit values (2 per byte)
};

static_assert(sizeof(BlockQ5_0) == 22, "BlockQ5_0 must be 22 bytes");

// ── Q6_K Block ──────────────────────────────────────────────
// 256 elements quantized to 6 bits with sub-block scales.
// Total: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210 bytes.
//
// Memory layout:
//   ql[128]: lower 4 bits of each quantized value (packed nibble pairs)
//   qh[64]:  upper 2 bits of each quantized value (packed 4 per byte)
//   scales[16]: int8_t sub-block scales (16 sub-blocks of 16 elements)
//   d:       __half super-block scale

struct BlockQ6_K {
    uint8_t ql[128];    // lower 4 bits (packed, 2 per byte)
    uint8_t qh[64];     // upper 2 bits (packed, 4 per byte)
    int8_t  scales[16]; // sub-block scales
    __half  d;          // super-block scale
};

static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

// ── Q4_K Block ──────────────────────────────────────────────
// 256 elements in a super-block with sub-block scales and mins.
// Total: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144 bytes.
//
// Memory layout:
//   d:          __half — super-block scale factor
//   dmin:       __half — super-block minimum factor
//   scales[12]: packed 6-bit sub-block scales and mins
//               (8 sub-blocks × 2 values × 6 bits = 96 bits = 12 bytes)
//   qs[128]:    packed 4-bit quantized values (256 elements, 2 per byte)
//
// Dequantization for sub-block j (32 elements each, 8 sub-blocks):
//   sc = (decoded scale for sub-block j)
//   mn = (decoded min for sub-block j)
//   x[i] = d * sc * nibble[i] - dmin * mn

struct BlockQ4_K {
    __half  d;          // super-block scale
    __half  dmin;       // super-block min
    uint8_t scales[12]; // packed sub-block scales and mins
    uint8_t qs[128];    // packed 4-bit values (2 per byte)
};

static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

// ── Constants ───────────────────────────────────────────────

constexpr int Q8_0_BLOCK_SIZE = 32;   // elements per Q8_0 block
constexpr int Q4_0_BLOCK_SIZE = 32;   // elements per Q4_0 block
constexpr int Q5_0_BLOCK_SIZE = 32;   // elements per Q5_0 block
constexpr int Q4_K_BLOCK_SIZE = 256;  // elements per Q4_K super-block
constexpr int Q6_K_BLOCK_SIZE = 256;  // elements per Q6_K super-block

// ── Standalone Dequantization Kernels ───────────────────────
// These dequantize an entire quantized tensor to FP32.
// Used for norm weights and embedding tables that need FP32
// for elementwise operations or row lookups.

/// Dequantize Q8_0 blocks to FP32.
/// @param src      Device pointer to BlockQ8_0 array.
/// @param dst      Device pointer to float output (n_elements floats).
/// @param n_blocks Number of Q8_0 blocks (= n_elements / 32).
void dequantize_q8_0(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream = 0);

/// Dequantize Q4_0 blocks to FP32.
/// @param src      Device pointer to BlockQ4_0 array.
/// @param dst      Device pointer to float output (n_elements floats).
/// @param n_blocks Number of Q4_0 blocks (= n_elements / 32).
void dequantize_q4_0(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream = 0);

/// Dequantize Q5_0 blocks to FP32.
/// @param src      Device pointer to BlockQ5_0 array.
/// @param dst      Device pointer to float output (n_elements floats).
/// @param n_blocks Number of Q5_0 blocks (= n_elements / 32).
void dequantize_q5_0(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream = 0);

/// Dequantize Q6_K blocks to FP32.
/// @param src      Device pointer to BlockQ6_K array.
/// @param dst      Device pointer to float output (n_elements floats).
/// @param n_blocks Number of Q6_K blocks (= n_elements / 256).
void dequantize_q6_k(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream = 0);

/// Dequantize Q4_K blocks to FP32.
/// @param src      Device pointer to BlockQ4_K array.
/// @param dst      Device pointer to float output (n_elements floats).
/// @param n_blocks Number of Q4_K blocks (= n_elements / 256).
void dequantize_q4_k(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream = 0);

} // namespace cuinfer
