# CuInfer — Phase 3 Improvement Results & Verification Report

**Project:** CuInfer (C++17 / CUDA Transformer Inference Engine)
**Target Optimization:** Elimination of 32-Way Shared Memory Bank Conflicts in Quantized GEMM
**Date:** October 20, 2026
**Status:** IMPLEMENTED AND ANALYTICALLY VERIFIED

---

## 1. Problem Identified & Original Behavior

In `src/kernels/matmul_quant.cu`, all six quantized matrix multiplication kernels (`kernel_matmul_q8_0`, `kernel_matmul_q4_0`, `kernel_matmul_q5_0`, `kernel_matmul_q4_k`, `kernel_matmul_q6_k`, and `kernel_matmul_fp16`) compute matrix multiplication with on-the-fly dequantization in shared memory:

$$C[M, N] = A[M, K] \times \text{dequant}(B)^T[K, N]$$

Each thread block launches with dimensions $32 \times 32 = 1024$ threads. Each thread $(\text{threadIdx.y}, \text{threadIdx.x})$ dequantizes one element of the $B$ sub-matrix tile into shared memory:

```cpp
__shared__ float tile_a[TILE_M][TILE_K];
__shared__ float tile_b[TILE_K][TILE_N]; // 32 rows, 32 columns

// Thread mapping:
// tx in [0, 31], ty in [0, 31]
tile_b[threadIdx.x][threadIdx.y] = dequantized_value;
```

### The Architectural Defect: 32-Way Bank Conflict
In CUDA architecture:
- Shared memory is split into 32 independent memory banks, each 4 bytes wide.
- Successive 32-bit words are assigned to successive banks: $\text{Bank} = (\text{Word Index}) \pmod{32}$.
- When writing to `tile_b[threadIdx.x][threadIdx.y]`, the word index is:
  $$\text{Index} = \text{threadIdx.x} \times 32 + \text{threadIdx.y}$$
- Applying the bank mapping modulo:
  $$\text{Bank} = (\text{threadIdx.x} \times 32 + \text{threadIdx.y}) \pmod{32} \equiv \text{threadIdx.y} \pmod{32}$$
- Because threads in a warp share the same row index `threadIdx.y` (for Warp 0, `threadIdx.y = 0`), **all 32 threads in the warp write to Bank 0 simultaneously**.
- Because every thread writes a distinct value (individual quantized weights), the hardware cannot broadcast the write.
- **Consequence:** The shared memory controller serializes the memory transaction across all 32 threads, resulting in a 32-way bank conflict for every single warp on every tile iteration.

---

## 2. Implementation Details

We introduced a padded column stride `TILE_N_PAD = TILE_N + 1` ($33$ floats, $132$ bytes) for `tile_b` across all six quantized GEMM kernels:

```diff
 namespace cuinfer {

 // ── Tile dimensions ─────────────────────────────────────────
 constexpr int TILE_M = 32;
 constexpr int TILE_N = 32;
 constexpr int TILE_K = 32;
+// Padded column dimension for tile_b shared memory tile to eliminate 32-way bank conflicts.
+// Stride of 33 floats (132 bytes) ensures that (threadIdx.x * 33 + threadIdx.y) % 32
+// maps every thread in a warp to a distinct shared memory bank.
+constexpr int TILE_N_PAD = TILE_N + 1;
```

And in `kernel_matmul_q8_0`, `kernel_matmul_q4_0`, `kernel_matmul_q5_0`, `kernel_matmul_q4_k`, `kernel_matmul_q6_k`, and `kernel_matmul_fp16`:
```diff
     __shared__ float tile_a[TILE_M][TILE_K];
-    __shared__ float tile_b[TILE_K][TILE_N];
+    __shared__ float tile_b[TILE_K][TILE_N_PAD];
```

### Mathematical Proof of Conflict-Free Access
1. **Write Phase (`tile_b[threadIdx.x][threadIdx.y] = ...`):**
   $$\text{Index} = \text{threadIdx.x} \times 33 + \text{threadIdx.y} = \text{threadIdx.x} \times 32 + (\text{threadIdx.x} + \text{threadIdx.y})$$
   $$\text{Bank} = (\text{Index}) \pmod{32} = (\text{threadIdx.x} + \text{threadIdx.y}) \pmod{32}$$
   For any warp with fixed `threadIdx.y`, as `threadIdx.x` ranges from $0$ to $31$, the bank index $(\text{threadIdx.x} + \text{threadIdx.y}) \pmod{32}$ maps 1-to-1 onto all 32 banks $\{0, 1, \dots, 31\}$.
   **Bank Conflict Count: 0.**

2. **Read Phase (`sum += tile_a[ty][k] * tile_b[k][tx]`):**
   $$\text{Index} = k \times 33 + \text{threadIdx.x} = k \times 32 + (k + \text{threadIdx.x})$$
   $$\text{Bank} = (k + \text{threadIdx.x}) \pmod{32}$$
   For any iteration $k$, as `threadIdx.x` ranges from $0$ to $31$ across the warp, the bank index traverses $\{0, 1, \dots, 31\}$.
   **Bank Conflict Count: 0.**

3. **Memory Overhead:**
   Shared memory per block increases by $32 \times 1 \times 4 = 128$ bytes (from 8,192 bytes to 8,320 bytes). This represents an increase of only 1.56% and remains well below the per-SM limit (100 KB on RTX 3060 / RTX 4060).

---

## 3. Files Changed

- `src/kernels/matmul_quant.cu`: Applied `TILE_N_PAD = TILE_N + 1` to `tile_b` across all 6 kernels.
- `tests/test_bank_conflict_simulation.cpp`: Standalone C++ verification suite proving bank index distribution and numerical equivalence.
- `docs/research/CUINFER-TECHNICAL-DUE-DILIGENCE.md`: Complete Phase 1 audit report.
- `docs/research/CUINFER-IMPROVEMENT-PLAN.md`: Phase 2 selection and implementation plan.
- `docs/research/CUINFER-IMPROVEMENT-RESULTS.md`: This verification document.

---

## 4. Tests Executed & Actual Results

### Verification Suite: `tests/test_bank_conflict_simulation.cpp`
Built and executed with GCC 15.2.0 (`g++ -std=c++17 -O3`):

```text
========================================================
  CuInfer — Bank Conflict Elimination Verification Suite
========================================================

[Verification 1] Testing unpadded tile_b[32][32]...
  -> CONFIRMED: Unpadded tile produces 32-way bank conflict (only 1 bank active per warp write)!
[Verification 2] Testing padded tile_b[32][33]...
  -> CONFIRMED: Padded tile completely eliminates bank conflicts (all 32 banks active, 0 conflicts)!
[Verification 3] Testing read phase under padded tile...
  -> CONFIRMED: Padded tile read phase is also 100% conflict-free across all k!
[Verification 4] Testing numerical equivalence of padded simulation vs reference...
  -> CONFIRMED: Numerical output with padded tile matches unpadded bit-for-bit (diff < 1e-6)!

All verification assertions PASSED successfully!
```

> [!IMPORTANT]
> **Verification Scope Clarification:**
> The standalone C++ verification suite proves the mathematical mapping of thread indices to hardware banks and verifies arithmetic equivalence under host simulation. It does **not** establish CUDA kernel compilation, GPU runtime correctness, device memory safety via sanitizers, or physical performance improvements.

---

## 5. Architectural & Performance Comparison

| Metric / Parameter | Baseline (Original Code) | Optimized (Our Modification) | Status / Verification Level |
|---|---|---|---|
| **`tile_b` Dimension** | `[32][32]` (1024 floats) | `[32][33]` (1056 floats) | Verified in Source (+128 bytes/block) |
| **Dequantize Write Conflicts** | 32-way bank conflict | 0 conflicts (conflict-free) | Verified mathematically & in C++ test |
| **Compute Read Conflicts** | 0 conflicts | 0 conflicts | Verified mathematically & in C++ test |
| **Warp Access Pattern** | Serialized across single bank | Bijective spread across 32 banks | Source-level analysis prediction |
| **Numerical Error ($\Delta C$)** | 0.0f | 0.0f | Verified bit-for-bit in simulation |
| **Physical Performance Impact** | Baseline | Pending GPU compilation | **Unverified on physical GPU** |

Source-level analysis predicts a reduction in the identified bank-conflict pattern under the documented warp mapping. However, physical execution and wall-clock performance impact remain unverified until the modified CUDA kernels are compiled with `nvcc`, tested on device, and benchmarked on compatible hardware with Nsight Compute.

---

## 6. Reproduction Instructions

To reproduce the bank conflict verification test:
```bash
g++ -std=c++17 -O3 tests/test_bank_conflict_simulation.cpp -o test_bank_conflict.exe
./test_bank_conflict.exe
```

When building with CUDA Toolkit (`nvcc`):
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release
ctest --output-on-failure
```

---

## 7. Next Highest-Value Opportunity

The next most impactful optimization is in `src/attention.cu` (`kernel_fused_decode_attention`):
- Promote `s_acc[head_dim]` from shared memory to thread-private registers (`float reg_acc`).
- Replace shared memory broadcast of online softmax parameters (`s_ctl`) with intra-warp shuffle intrinsics (`__shfl_sync`).
- Eliminate the redundant `__syncthreads()` barrier inside the per-token KV decoding loop.
