#pragma once

// ── Quantized Matrix Multiplication ──────────────────────────────────
//
// Fused dequantize + matmul kernels:  C_fp32 = A_fp32 * dequant(B_quant)^T
//
// B is stored in GGUF-native [N, K] layout (= B-transposed with respect to
// the standard C = A * B matmul convention). This avoids any host-side
// transposition that would break quantization block alignment.
//
// Each tile iteration loads a strip of quantized B from global memory,
// dequantizes it to FP32 in shared memory, then performs standard tiled
// FP32 accumulation. TILE_K=32 matches Q8_0/Q4_0 block size (32 elements),
// so each tile-K strip = exactly one quantization block per B row.
//
// API convention:
//   A: [M, K] row-major FP32
//   B: [N, K] row-major quantized (each row is a sequence of quant blocks)
//   C: [M, N] row-major FP32 output
//
// This computes C[i,j] = sum_k A[i,k] * dequant(B[j,k])

#include <cuda_runtime.h>

namespace cuinfer {

/// Quantized matmul with Q8_0 weights (B-transposed layout).
/// C[M,N] = A[M,K] * dequant(B_q8_0[N,K])^T
///
/// @param A      Device pointer to FP32 activation matrix [M, K].
/// @param B_quant Device pointer to Q8_0 weight data [N, K] in block format.
///               Each row of B contains K/32 BlockQ8_0 structures.
/// @param C      Device pointer to FP32 output matrix [M, N].
/// @param M      Number of rows in A / rows in output.
/// @param N      Number of rows in B / columns in output (out_features).
/// @param K      Shared dimension (in_features, must be divisible by 32).
void matmul_q8_0(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

/// Quantized matmul with Q4_0 weights (B-transposed layout).
/// C[M,N] = A[M,K] * dequant(B_q4_0[N,K])^T
///
/// @param A       Device pointer to FP32 activation matrix [M, K].
/// @param B_quant Device pointer to Q4_0 weight data [N, K] in block format.
///               Each row of B contains K/32 BlockQ4_0 structures.
/// @param C      Device pointer to FP32 output matrix [M, N].
/// @param M      Number of rows in A / rows in output.
/// @param N      Number of rows in B / columns in output.
/// @param K      Shared dimension (must be divisible by 32).
void matmul_q4_0(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

/// Quantized matmul with Q5_0 weights (B-transposed layout).
/// C[M,N] = A[M,K] * dequant(B_q5_0[N,K])^T
///
/// @param A       Device pointer to FP32 activation matrix [M, K].
/// @param B_quant Device pointer to Q5_0 weight data [N, K] in block format.
///               Each row of B contains K/32 BlockQ5_0 structures.
/// @param C      Device pointer to FP32 output matrix [M, N].
/// @param M      Number of rows in A / rows in output.
/// @param N      Number of rows in B / columns in output.
/// @param K      Shared dimension (must be divisible by 32).
void matmul_q5_0(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

/// Quantized matmul with Q4_K weights (B-transposed layout).
/// C[M,N] = A[M,K] * dequant(B_q4_k[N,K])^T
///
/// @param A       Device pointer to FP32 activation matrix [M, K].
/// @param B_quant Device pointer to Q4_K weight data [N, K] in super-block format.
///               Each row of B contains K/256 BlockQ4_K structures.
/// @param C      Device pointer to FP32 output matrix [M, N].
/// @param M      Number of rows in A / rows in output.
/// @param N      Number of rows in B / columns in output.
/// @param K      Shared dimension (must be divisible by 256).
void matmul_q4_k(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

/// Quantized matmul with Q6_K weights (B-transposed layout).
/// C[M,N] = A[M,K] * dequant(B_q6_k[N,K])^T
///
/// @param A       Device pointer to FP32 activation matrix [M, K].
/// @param B_quant Device pointer to Q6_K weight data [N, K] in super-block format.
/// @param C      Device pointer to FP32 output matrix [M, N].
/// @param M      Number of rows in A / rows in output.
/// @param N      Number of rows in B / columns in output.
/// @param K      Shared dimension (must be divisible by 256).
void matmul_q6_k(const float* A, const void* B_quant, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

/// Mixed-precision matmul with FP16 weights (B-transposed layout).
/// C[M,N] = A[M,K] * dequant(B_fp16[N,K])^T
///
/// Weights are stored as native __half in GGUF [N, K] layout.
/// Dequantization is trivial: __half2float() per element.
/// Accumulation is always in FP32 for precision.
///
/// @param A       Device pointer to FP32 activation matrix [M, K].
/// @param B_fp16  Device pointer to FP16 weight data [N, K] (__half elements).
/// @param C       Device pointer to FP32 output matrix [M, N].
/// @param M       Number of rows in A / rows in output.
/// @param N       Number of rows in B / columns in output.
/// @param K       Shared dimension (must be divisible by 32).
void matmul_fp16(const float* A, const void* B_fp16, float* C,
                 int M, int N, int K, cudaStream_t stream = 0);

} // namespace cuinfer
