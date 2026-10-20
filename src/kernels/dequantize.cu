// ── dequantize.cu — Standalone GPU Dequantization Kernels ──────────────
//
// Converts quantized block data (Q8_0, Q4_0, Q6_K, Q4_K) to FP32.
// These are used for:
//   - Norm weights (small 1D tensors, always used in FP32 elementwise ops)
//   - Embedding tables (row-lookup pattern doesn't suit block quantization)
//   - Verification/testing (dequantize entire tensor for comparison)
//
// Each kernel processes one block per thread. The block formats match the
// GGML specification exactly, so raw GGUF data can be uploaded to the GPU
// and dequantized in-place without any host-side preprocessing.
//
// Thread launch strategy:
//   Grid: n_blocks (one block per quantization block)
//   Block: 32 threads (one thread per element within a Q8_0/Q4_0 block,
//          or multiple threads for K-quant super-blocks)

#include "cuinfer/quantization.h"
#include "cuinfer/cuda_utils.h"

namespace cuinfer {

// ════════════════════════════════════════════════════════════════
// Q8_0 Dequantization
// ════════════════════════════════════════════════════════════════
//
// Each Q8_0 block: 1 FP16 scale + 32 int8 values.
// Dequantize: x[i] = scale * qs[i]
//
// Launch: one CUDA thread per element.
// Each warp (32 threads) processes exactly one Q8_0 block.
// Grid: ceil(n_blocks * 32 / 256) blocks of 256 threads.

__global__ void kernel_dequantize_q8_0(
    const BlockQ8_0* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_id = idx / Q8_0_BLOCK_SIZE;
    const int elem_id  = idx % Q8_0_BLOCK_SIZE;

    if (block_id >= n_blocks) return;

    const BlockQ8_0& blk = src[block_id];
    const float scale = __half2float(blk.scale);
    dst[block_id * Q8_0_BLOCK_SIZE + elem_id] = scale * static_cast<float>(blk.qs[elem_id]);
}

void dequantize_q8_0(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream) {
    const int total_elements = n_blocks * Q8_0_BLOCK_SIZE;
    constexpr int BLOCK = 256;
    const int grid = (total_elements + BLOCK - 1) / BLOCK;

    kernel_dequantize_q8_0<<<grid, BLOCK, 0, stream>>>(
        static_cast<const BlockQ8_0*>(src), dst, n_blocks);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q4_0 Dequantization
// ════════════════════════════════════════════════════════════════
//
// Each Q4_0 block: 1 FP16 scale + 16 packed-nibble bytes = 32 elements.
// Byte qs[j] holds two 4-bit unsigned values:
//   element 2*j+0 = (qs[j] & 0x0F) - 8   (lower nibble)
//   element 2*j+1 = (qs[j] >> 4)   - 8   (upper nibble)
// Dequantize: x[i] = scale * (nibble_value - 8)
//
// Launch: one CUDA thread per element (32 per block).

__global__ void kernel_dequantize_q4_0(
    const BlockQ4_0* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_id = idx / Q4_0_BLOCK_SIZE;
    const int elem_id  = idx % Q4_0_BLOCK_SIZE;

    if (block_id >= n_blocks) return;

    const BlockQ4_0& blk = src[block_id];
    const float scale = __half2float(blk.scale);

    // Each byte holds two 4-bit values. elem_id [0..31] maps to byte [0..15].
    const int byte_idx = elem_id / 2;
    const uint8_t packed = blk.qs[byte_idx];

    // Lower nibble for even elements, upper nibble for odd elements.
    int nibble;
    if (elem_id % 2 == 0) {
        nibble = (packed & 0x0F);
    } else {
        nibble = (packed >> 4);
    }

    // GGML Q4_0 stores unsigned [0, 15] with implicit offset of -8.
    dst[block_id * Q4_0_BLOCK_SIZE + elem_id] = scale * static_cast<float>(nibble - 8);
}

void dequantize_q4_0(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream) {
    const int total_elements = n_blocks * Q4_0_BLOCK_SIZE;
    constexpr int BLOCK = 256;
    const int grid = (total_elements + BLOCK - 1) / BLOCK;

    kernel_dequantize_q4_0<<<grid, BLOCK, 0, stream>>>(
        static_cast<const BlockQ4_0*>(src), dst, n_blocks);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q5_0 Dequantization
// ════════════════════════════════════════════════════════════════
//
// Each Q5_0 block: 1 FP16 scale + 4 bytes high-bits + 16 packed-nibble bytes = 32 elements.
// Byte qs[j] holds two 4-bit unsigned values (lower nibble first).
// Byte qh packs the 5th (high) bit for all 32 elements across 4 bytes (bit-addressed).
// Reconstruction: q5 = low_nibble | (high_bit << 4), range [0, 31]
//   x[i] = scale * (q5 - 16)
//
// Launch: one CUDA thread per element (32 per block).

__global__ void kernel_dequantize_q5_0(
    const BlockQ5_0* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_id = idx / Q5_0_BLOCK_SIZE;
    const int elem_id  = idx % Q5_0_BLOCK_SIZE;

    if (block_id >= n_blocks) return;

    const BlockQ5_0& blk = src[block_id];
    const float scale = __half2float(blk.scale);

    // Q5_0 nibble layout (matches GGML convention):
    //   Elements 0..15  use the LOWER nibble of qs[0..15]
    //   Elements 16..31 use the UPPER nibble of qs[0..15]
    // This is different from Q4_0 which uses consecutive pairs!
    const int half = elem_id / 16;  // 0 = first half, 1 = second half
    const int j = elem_id % 16;     // index within half
    int low4;
    if (half == 0) {
        low4 = blk.qs[j] & 0x0F;
    } else {
        low4 = blk.qs[j] >> 4;
    }

    // High bit from qh[4] interpreted as a little-endian uint32_t.
    // Bit elem_id of qh stores the 5th bit for element elem_id.
    // Construct uint32 byte-by-byte to avoid misaligned access (qh is at
    // offset 2 in the struct, which is not 4-byte aligned).
    const uint32_t qh_u32 = static_cast<uint32_t>(blk.qh[0])
                           | (static_cast<uint32_t>(blk.qh[1]) << 8)
                           | (static_cast<uint32_t>(blk.qh[2]) << 16)
                           | (static_cast<uint32_t>(blk.qh[3]) << 24);
    const int high_bit = (qh_u32 >> elem_id) & 1;

    const int q5_val = low4 | (high_bit << 4);  // [0, 31]
    dst[block_id * Q5_0_BLOCK_SIZE + elem_id] = scale * static_cast<float>(q5_val - 16);
}

void dequantize_q5_0(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream) {
    const int total_elements = n_blocks * Q5_0_BLOCK_SIZE;
    constexpr int BLOCK = 256;
    const int grid = (total_elements + BLOCK - 1) / BLOCK;

    kernel_dequantize_q5_0<<<grid, BLOCK, 0, stream>>>(
        static_cast<const BlockQ5_0*>(src), dst, n_blocks);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q6_K Dequantization
// ════════════════════════════════════════════════════════════════
//
// Q6_K super-block: 256 elements, 210 bytes.
// Structure:
//   ql[128] : lower 4 bits (interleaved layout, NOT simple consecutive pairs)
//   qh[64]  : upper 2 bits (interleaved, 4 elements per byte)
//   scales[16]: int8_t sub-block scales (ALL 16 used, 16 elements each)
//   d       : __half super-block scale
//
// GGML interleaved layout (2 chunks of 128 elements):
//   For each 128-element chunk (chunk = 0 or 1):
//     Elements 0..31:   ql[l + chunk*64]      LOWER nibble,  qh[l + chunk*32] bits 0-1
//     Elements 32..63:  ql[l+32 + chunk*64]   LOWER nibble,  qh[l + chunk*32] bits 2-3
//     Elements 64..95:  ql[l + chunk*64]      UPPER nibble,  qh[l + chunk*32] bits 4-5
//     Elements 96..127: ql[l+32 + chunk*64]   UPPER nibble,  qh[l + chunk*32] bits 6-7
//   where l = position within quarter (0..31).
//
// Scale mapping: 16 groups of 16 elements. GGML uses is=l/16 to split each
// 32-element group into two 16-element subgroups with different scales.
//   scale_idx = chunk*8 + quarter*2 + l/16
//   Elements 0..15 → scales[0], 16..31 → scales[1], ..., 240..255 → scales[15]
//
// Launch: one CUDA thread per element (256 per super-block).

__global__ void kernel_dequantize_q6_k(
    const BlockQ6_K* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_id = idx / Q6_K_BLOCK_SIZE;
    const int elem_id  = idx % Q6_K_BLOCK_SIZE;

    if (block_id >= n_blocks) return;

    const BlockQ6_K& blk = src[block_id];
    const float d = __half2float(blk.d);

    // Decompose element position into the interleaved structure
    const int chunk = elem_id / 128;           // 0 or 1
    const int within_chunk = elem_id % 128;
    const int quarter = within_chunk / 32;     // 0..3
    const int l = within_chunk % 32;           // position within quarter

    // Scale: 16 groups of 16 elements, ALL scale indices used.
    // GGML reference: is = l/16 splits the 32-element group into two
    // 16-element subgroups, each with its own scale.
    const int scale_idx = chunk * 8 + quarter * 2 + l / 16;
    const float sc = static_cast<float>(blk.scales[scale_idx]);

    // ql: interleaved layout — quarters 0,2 use same bytes (lower/upper nibble),
    //     quarters 1,3 use bytes offset by 32
    const int ql_offset = (quarter & 1) * 32 + l + chunk * 64;
    const int ql_nibble = (quarter < 2) ? (blk.ql[ql_offset] & 0x0F)
                                        : (blk.ql[ql_offset] >> 4);

    // qh: each byte holds 4 elements (2 bits each), indexed by quarter
    const int qh_byte = l + chunk * 32;
    const int qh_shift = quarter * 2;
    const int qh_bits = (blk.qh[qh_byte] >> qh_shift) & 0x03;

    const int q6_val = (ql_nibble | (qh_bits << 4)) - 32;

    dst[block_id * Q6_K_BLOCK_SIZE + elem_id] = d * sc * static_cast<float>(q6_val);
}

void dequantize_q6_k(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream) {
    const int total_elements = n_blocks * Q6_K_BLOCK_SIZE;
    constexpr int BLOCK = 256;
    const int grid = (total_elements + BLOCK - 1) / BLOCK;

    kernel_dequantize_q6_k<<<grid, BLOCK, 0, stream>>>(
        static_cast<const BlockQ6_K*>(src), dst, n_blocks);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Q4_K Dequantization
// ════════════════════════════════════════════════════════════════
//
// Q4_K super-block: 256 elements, 144 bytes.
// Structure:
//   d       : __half — super-block scale factor
//   dmin    : __half — super-block minimum factor
//   scales[12]: packed 6-bit sub-block scales and mins
//   qs[128] : packed 4-bit values (half-interleaved, NOT consecutive pairs)
//
// GGML half-interleaved nibble layout (4 chunks of 64 elements):
//   For each 64-element chunk (chunk64 = 0..3):
//     Elements 0..31:  LOWER nibble of qs[chunk64*32 + l]   (scale = chunk64*2)
//     Elements 32..63: UPPER nibble of qs[chunk64*32 + l]   (scale = chunk64*2 + 1)
//   where l = position within half (0..31).
//
// Scale/min decoding uses GGML's get_scale_min_k4 packing (unchanged).
//
// Launch: one CUDA thread per element (256 per super-block).

__global__ void kernel_dequantize_q4_k(
    const BlockQ4_K* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_id = idx / Q4_K_BLOCK_SIZE;
    const int elem_id  = idx % Q4_K_BLOCK_SIZE;

    if (block_id >= n_blocks) return;

    const BlockQ4_K& blk = src[block_id];
    const float d    = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);

    // Determine which sub-block (0..7) this element belongs to
    const int sub_block = elem_id / 32;

    // Decode 6-bit scale and min for this sub-block from the packed scales array.
    // The packing is the same as GGML's get_scale_min_k4 function.
    uint8_t sc, mn;
    if (sub_block < 4) {
        sc = blk.scales[sub_block] & 0x3F;
        mn = blk.scales[sub_block + 4] & 0x3F;
    } else {
        sc = (blk.scales[sub_block + 4] & 0x0F) | ((blk.scales[sub_block - 4] >> 6) << 4);
        mn = (blk.scales[sub_block + 4] >> 4)    | ((blk.scales[sub_block]     >> 6) << 4);
    }

    // Extract 4-bit nibble using GGML half-interleaved layout:
    // Each 64-element chunk uses 32 bytes: lower nibbles → first 32 elements,
    // upper nibbles → next 32 elements.
    const int chunk64 = elem_id / 64;
    const int within_64 = elem_id % 64;
    const int l = within_64 % 32;
    const int qs_byte = chunk64 * 32 + l;
    int nibble;
    if (within_64 < 32) {
        nibble = blk.qs[qs_byte] & 0x0F;
    } else {
        nibble = blk.qs[qs_byte] >> 4;
    }

    // Dequantize: x = d * sc * nibble - dmin * mn
    dst[block_id * Q4_K_BLOCK_SIZE + elem_id] =
        d * static_cast<float>(sc) * static_cast<float>(nibble) -
        dmin * static_cast<float>(mn);
}

void dequantize_q4_k(const void* src, float* dst, int n_blocks,
                     cudaStream_t stream) {
    const int total_elements = n_blocks * Q4_K_BLOCK_SIZE;
    constexpr int BLOCK = 256;
    const int grid = (total_elements + BLOCK - 1) / BLOCK;

    kernel_dequantize_q4_k<<<grid, BLOCK, 0, stream>>>(
        static_cast<const BlockQ4_K*>(src), dst, n_blocks);
    CUINFER_CHECK_CUDA(cudaGetLastError());
}

} // namespace cuinfer
