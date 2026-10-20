# CuInfer — Phase 2 Implementation Decision & Improvement Plan

**Target Optimization:** Elimination of 32-Way Shared Memory Bank Conflicts in Quantized GEMM
**Target File:** `src/kernels/matmul_quant.cu`
**Test File:** `tests/test_matmul_quant.cu`
**Date:** October 20, 2026

---

## 1. Candidate Evaluation & Selection

### Shortlist Comparison
1. **Candidate 1 (Selected Winner): Elimination of 32-Way Shared Memory Bank Conflicts via Tile Padding in `src/kernels/matmul_quant.cu`**
   - **Target Kernels:** `kernel_matmul_q8_0`, `kernel_matmul_q4_0`, `kernel_matmul_q5_0`, `kernel_matmul_q4_k`, `kernel_matmul_q6_k`, `kernel_matmul_fp16`.
   - **Root Cause:** Inverted indexing into unpadded shared memory tile `tile_b[32][32]` causes all 32 threads in each warp to write to Bank 0 simultaneously ($(\text{tx} \cdot 32 + \text{ty}) \pmod{32} \equiv \text{ty}$).
   - **Defensibility:** Source-level analysis predicts a reduction in the identified bank-conflict pattern under the documented warp mapping.
   - **Interview Value:** Direct demonstration of CUDA memory subsystem analysis (banks, strides, conflict-free access, instruction replays).

2. **Candidate 2 (Alternative): Vectorized 128-bit Global Memory Coalescing in `src/kernels/matmul.cu`**
   - **Reason for Lower Priority:** Only impacts the unquantized FP32 matmul path. Since LLM inference in CuInfer almost exclusively runs on quantized weights, optimizing the unquantized FP32 kernel does not accelerate real-world model execution.

3. **Candidate 3 (Alternative): Register-Resident Accumulation in Fused Attention (`src/attention.cu`)**
   - **Reason for Lower Priority:** More invasive code change with higher regression risk across sequence lengths; limited to attention decode phase, whereas quantized GEMM accounts for >80% of total inference compute.

---

## 2. Technical Specification of Selected Improvement

### Affected Files & Functions
- **File:** `src/kernels/matmul_quant.cu`
  - Function 1: `kernel_matmul_q8_0`
  - Function 2: `kernel_matmul_q4_0`
  - Function 3: `kernel_matmul_q5_0`
  - Function 4: `kernel_matmul_q4_k`
  - Function 5: `kernel_matmul_q6_k`
  - Function 6: `kernel_matmul_fp16`

### Specific Code Change
Currently:
```cpp
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

// In each kernel:
__shared__ float tile_a[TILE_M][TILE_K];
__shared__ float tile_b[TILE_K][TILE_N];  // Stride = 32 floats (128 bytes)
```

Modification:
Introduce a padded column stride `TILE_N_PAD = TILE_N + 1` (33 floats, 132 bytes) for `tile_b`:
```cpp
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int TILE_N_PAD = TILE_N + 1;  // +1 float eliminates shared memory bank conflicts

// In each kernel:
__shared__ float tile_a[TILE_M][TILE_K];
__shared__ float tile_b[TILE_K][TILE_N_PAD]; // Stride = 33 floats (132 bytes)
```
Write and read indexing:
- Write phase: `tile_b[threadIdx.x][threadIdx.y] = ...;` writes to address $(\text{threadIdx.x} \cdot 33 + \text{threadIdx.y})$.
  $$\text{Bank} = (\text{threadIdx.x} \cdot 33 + \text{threadIdx.y}) \pmod{32} = (\text{threadIdx.x} + \text{threadIdx.y}) \pmod{32}$$
  Conflict count: **0** (each thread in warp accesses a unique bank).
- Read phase: `sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];` reads from address $(k \cdot 33 + \text{threadIdx.x})$.
  $$\text{Bank} = (k \cdot 33 + \text{threadIdx.x}) \pmod{32} = (k + \text{threadIdx.x}) \pmod{32}$$
  Conflict count: **0** (each thread in warp accesses a unique bank).

---

## 3. Milestones & Verification Strategy

### Milestone 1 — Establish Baseline and Correctness
- Inspect reference tests in `tests/test_matmul_quant.cu`.
- Verify the CPU reference calculation for Q8_0, Q4_0, Q5_0, Q4_K, Q6_K, and FP16.
- Confirm exact tolerance criteria ($\epsilon \le 1\times 10^{-4}$ for FP16/Q8_0; format-specific tolerances for lower-bit quants).

### Milestone 2 — Implement Conflict-Free Tile Padding
- Update all 6 quantized GEMM kernels in `src/kernels/matmul_quant.cu` with `TILE_N_PAD = TILE_N + 1`.
- Verify that shared memory allocation per block only increases by $32 \times 1 \times 4 = 128$ bytes (from 8,192 bytes to 8,320 bytes), well within SM capacity (100 KB on RTX 3060/4060).

### Milestone 3 — Validation, Documentation & Finalization
- Validate mathematical correctness via standalone simulation and ensure no out-of-bounds memory accesses or indexing offsets are introduced.
- Document the change in `docs/research/CUINFER-IMPROVEMENT-RESULTS.md`.
- Explicitly note that while mathematical and simulated access patterns are validated, physical GPU execution, runtime latency, and throughput impact remain unverified on hardware pending a full CUDA build toolchain.
