# CuInfer â€” PTX Compiler Investigation & Micro-Architectural Analysis

**Target Architecture:** NVIDIA Ada Lovelace (`sm_89`, GeForce RTX 4060 Laptop GPU)
**Compiler Backend:** LLVM NVPTX (`clang++` 18.1.3, PTX ISA 7.8, `.target sm_89`)
**Target Kernel:** `kernel_matmul_q8_0` (CuInfer GGUF Q8_0 On-the-Fly Dequantizing GEMM)
**Date:** October 20, 2026
**Audience:** NVIDIA PTX Compiler & System Software Engineering

---

## 1. Objective & Relevance to NVIDIA PTX Compiler Team

This investigation examines the intermediate PTX representation emitted for the quantized matrix multiplication kernel (`kernel_matmul_q8_0`) from CuInfer. Specifically, we investigate how high-level CUDA shared-memory tiling, on-the-fly FP16/INT8 dequantization, and tile padding (`TILE_N_PAD = 33` vs unpadded `32`) translate into PTX instructions, address arithmetic, register allocation, and memory space qualifiers (`.shared`, `.global`).

For an NVIDIA compiler engineer, understanding the structural mapping between CUDA C++ source, PTX intermediate code, and hardware bank address decoding is foundational:
- PTX address generation dictates whether the downstream SASS generator (`ptxas`) can maintain conflict-free shared memory instructions.
- Stride multipliers directly determine hardware shared memory bank conflict behavior.

---

## 2. Machine & Toolchain Details

* **Host GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89, 8,188 MiB VRAM), Driver 592.82.
* **Compiler Used:** Clang 18.1.3 targeting `nvptx64-nvidia-cuda` with `-march=sm_89`.
* **Emitted PTX Version:** PTX ISA Version 7.8, 64-bit address size (`.address_size 64`).
* **Source Harness:** `docs/research/artifacts/harness_q8_0_ptx.cpp` (reproducing the exact indexing and memory layout of `src/kernels/matmul_quant.cu`).

---

## 3. Compilation Commands & Artifacts Generated

All artifacts were generated directly by the compiler backend during this sprint and are committed under `docs/research/artifacts/`:

1. **Unpadded Baseline (O3):**
   ```bash
   clang++ --target=nvptx64-nvidia-cuda -march=sm_89 -S -O3 docs/research/artifacts/harness_q8_0_ptx.cpp -o docs/research/artifacts/kernel_q8_0_unpadded_O3_sm89.ptx
   ```
2. **Padded Conflict-Free (O3):**
   ```bash
   clang++ --target=nvptx64-nvidia-cuda -march=sm_89 -S -O3 -DPAD_TILE docs/research/artifacts/harness_q8_0_ptx.cpp -o docs/research/artifacts/kernel_q8_0_padded_O3_sm89.ptx
   ```
3. **Padded Unoptimized (O0):**
   ```bash
   clang++ --target=nvptx64-nvidia-cuda -march=sm_89 -S -O0 -DPAD_TILE docs/research/artifacts/harness_q8_0_ptx.cpp -o docs/research/artifacts/kernel_q8_0_padded_O0_sm89.ptx
   ```

---

## 4. PTX Instruction Mapping & Micro-Architectural Analysis

### A. Shared Memory Allocation Declarations
In PTX, shared memory is declared in the `.shared` state space:

```ptx
// Unpadded Baseline: 32 x 32 floats = 4,096 bytes
.visible .shared .align 4 .b8 tile_a[4096];
.visible .shared .align 4 .b8 tile_b[4096];

// Padded Optimization: 32 x 33 floats = 4,224 bytes
.visible .shared .align 4 .b8 tile_a[4096];
.visible .shared .align 4 .b8 tile_b[4224];
```
The allocation expands by exactly 128 bytes ($32 \times 1 \times 4\text{ B}$), increasing shared memory consumption per block by only 1.56%.

---

### B. Address Calculation for Dequantization Write
During the B-tile loading phase, threads compute `tile_b[threadIdx.x][threadIdx.y]`:

#### In Unpadded PTX (`kernel_q8_0_unpadded_O3_sm89.ptx`):
```ptx
	mul.wide.s32 	%rd17, %r27, 128;   // %r27 is threadIdx.x; offset = tx * 128 bytes
	mov.u64 	%rd18, tile_b;
	add.s64 	%rd19, %rd18, %rd17;        // tile_b + tx * 128
	mul.wide.s32 	%rd20, %r2, 4;      // %r2 is threadIdx.y; ty * 4 bytes
	add.s64 	%rd6, %rd19, %rd20;         // Effective address: tile_b + tx*128 + ty*4
...
	st.shared.f32 	[%rd6], %f112;      // Store dequantized float into shared memory
```
* **Hardware Bank Conflict Analysis:**
  Shared memory has 32 banks, each 4 bytes wide. The bank index of a 4-byte float at byte address $A$ is:
  $$\text{Bank} = \left(\frac{A}{4}\right) \pmod{32}$$
  Here, $A = \text{tile\_b} + \text{tx} \times 128 + \text{ty} \times 4$.
  Because $128$ is an exact multiple of $128$ (the warp bank stride, $32 \times 4\text{ B}$):
  $$\left(\frac{\text{tx} \times 128}{4}\right) \pmod{32} = (32 \cdot \text{tx}) \pmod{32} \equiv 0$$
  $$\text{Bank} = (0 + \text{ty}) \pmod{32} \equiv \text{ty} \pmod{32}$$
  For all 32 threads in a warp (which share identical $\text{ty}$), the bank evaluated is **identical**. All 32 threads write to Bank $\text{ty}$, causing a hardware 32-way bank conflict.

#### In Padded PTX (`kernel_q8_0_padded_O3_sm89.ptx`):
```ptx
	mul.wide.s32 	%rd17, %r27, 132;   // Offset = tx * 132 bytes (33 floats)
	mov.u64 	%rd18, tile_b;
	add.s64 	%rd19, %rd18, %rd17;        // tile_b + tx * 132
	mul.wide.s32 	%rd20, %r2, 4;      // ty * 4 bytes
	add.s64 	%rd6, %rd19, %rd20;         // Effective address: tile_b + tx*132 + ty*4
...
	st.shared.f32 	[%rd6], %f112;      // Store dequantized float
```
* **Hardware Bank Conflict Analysis:**
  $A = \text{tile\_b} + \text{tx} \times 132 + \text{ty} \times 4$.
  $$\frac{A}{4} = \frac{\text{tile\_b}}{4} + 33 \cdot \text{tx} + \text{ty} = \frac{\text{tile\_b}}{4} + 32 \cdot \text{tx} + (\text{tx} + \text{ty})$$
  $$\text{Bank} = (\text{tx} + \text{ty}) \pmod{32}$$
  For fixed $\text{ty}$, as $\text{tx}$ ranges $0..31$, $\text{Bank}$ spans all 32 banks bijectively. The 32-way collision is eliminated at the instruction level.

---

### C. Compute Loop Unrolling & Stride Differences
In the unrolled inner loop (`#pragma unroll`), the compiler emits 32 pairs of `ld.shared.f32` and `fma.rn.f32` instructions:

```ptx
// Unpadded compute offsets (128-byte increments):
ld.shared.f32 	%f14, [%rd7];
ld.shared.f32 	%f15, [%rd8];
fma.rn.f32 	%f16, %f14, %f15, %f110;
ld.shared.f32 	%f17, [%rd7+4];
ld.shared.f32 	%f18, [%rd8+128];       // Offset +128
fma.rn.f32 	%f19, %f17, %f18, %f16;
ld.shared.f32 	%f20, [%rd7+8];
ld.shared.f32 	%f21, [%rd8+256];       // Offset +256
...
ld.shared.f32 	%f107, [%rd7+124];
ld.shared.f32 	%f108, [%rd8+3968];      // Offset +3968 (31 * 128)
fma.rn.f32 	%f110, %f107, %f108, %f106;

// Padded compute offsets (132-byte increments):
ld.shared.f32 	%f14, [%rd7];
ld.shared.f32 	%f15, [%rd8];
fma.rn.f32 	%f16, %f14, %f15, %f110;
ld.shared.f32 	%f17, [%rd7+4];
ld.shared.f32 	%f18, [%rd8+132];       // Offset +132
fma.rn.f32 	%f19, %f17, %f18, %f16;
ld.shared.f32 	%f20, [%rd7+8];
ld.shared.f32 	%f21, [%rd8+264];       // Offset +264
...
ld.shared.f32 	%f107, [%rd7+124];
ld.shared.f32 	%f108, [%rd8+4092];      // Offset +4092 (31 * 132)
fma.rn.f32 	%f110, %f107, %f108, %f106;
```
Each iteration of $k$ reads from row $k$ with base offset $k \times 132\text{ B}$. Reading $\text{tile\_b}[k][\text{threadIdx.x}]$ remains 100% conflict-free across all 32 threads.

---

### D. FP16 Dequantization Instruction
The on-the-fly conversion from FP16 scale to FP32 is mapped to:
```ptx
ld.global.nc.u16 	%rs1, [%rd23];  // Load 16-bit half scale
cvt.f32.f16 	%f12, %rs1;             // Convert FP16 -> FP32 in hardware
ld.global.nc.u8 	%rs2, [%rd24+2]; // Load 8-bit quantized integer
cvt.s16.s8 	%rs3, %rs2;
cvt.rn.f32.s16 	%f13, %rs3;             // Convert int8 -> float
mul.rn.f32 	%f112, %f12, %f13;      // scale * float(qs[tx])
```
This confirms that the on-the-fly dequantization executes via a single hardware `cvt.f32.f16` conversion rather than expensive software emulation.

---

### E. Optimization Comparison (-O0 vs -O3)

| Metric | Padded -O0 (`kernel_q8_0_padded_O0_sm89.ptx`) | Padded -O3 (`kernel_q8_0_padded_O3_sm89.ptx`) |
|---|---|---|
| **File Size** | 8,995 bytes | 7,207 bytes |
| **Inner Loop Unrolling** | Rolled loop with branches (`@%p bra`), loop counter increments | Fully unrolled into 32 straight-line `fma.rn.f32` instructions |
| **Local Frame (`.local`)** | Declares `__local_depot0[32]` for stack spill | Zero `.local` stack allocation; pure register residency |
| **FMA Instructions** | Rolled scalar `fma` | 32 pipelined `fma.rn.f32` operations |

---

## 5. Register & Resource Declarations

In the emitted PTX (`.entry kernel_matmul_q8_0`):
* Predicate registers: `.reg .pred %p<11>;` (11 predicates)
* 16-bit registers: `.reg .b16 %rs<4>;` (4 registers)
* 32-bit registers: `.reg .b32 %r<30>;` (30 registers)
* Single-precision float registers: `.reg .f32 %f<113>;` (113 virtual float registers)
* 64-bit address registers: `.reg .b64 %rd<26>;` (26 registers)

> [!NOTE]
> **PTX vs Physical SASS Registers:**
> PTX registers are virtual SSA registers. During final machine compilation (`ptxas`), the register allocator maps these virtual registers onto physical GPU register files (up to 255 per thread on Ada Lovelace sm_89).

---

## 6. Verification & Test Outcomes

1. **Compiler Execution:**
   - Command: `clang++ --target=nvptx64-nvidia-cuda -march=sm_89 -S -O3 ...`
   - Exit Code: `0`
   - Status: **PASSED** (Real PTX generated and verified).
2. **Mathematical Bank Conflict Verification Suite (`tests/test_bank_conflict_simulation.cpp`):**
   - Command: `g++ -std=c++17 -O3 tests/test_bank_conflict_simulation.cpp -o test_bank_conflict.exe && ./test_bank_conflict.exe`
   - Exit Code: `0`
   - Status: **PASSED** (Confirmed 32-way collision on unpadded, 0 conflicts on padded stride 33, bit-for-bit numerical equivalence).

---

## 7. Limitations & Boundaries

1. **Hardware Execution:** The PTX was compiled through the LLVM NVPTX target. Execution of the generated kernel on the physical RTX 4060 GPU and hardware profiler traces via `ncu` remain unverified pending installation of the NVIDIA CUDA Toolkit.
2. **SASS Machine Code:** PTX is virtual bytecode; final machine instructions and instruction scheduling are determined by `ptxas`.

---

## 8. Reproduction Instructions

To reproduce the exact PTX files generated in this report:

```bash
# In WSL or Linux with clang++ installed:
# 1. Generate unpadded baseline PTX
clang++ --target=nvptx64-nvidia-cuda -march=sm_89 -S -O3 docs/research/artifacts/harness_q8_0_ptx.cpp -o docs/research/artifacts/kernel_q8_0_unpadded_O3_sm89.ptx

# 2. Generate padded conflict-free PTX
clang++ --target=nvptx64-nvidia-cuda -march=sm_89 -S -O3 -DPAD_TILE docs/research/artifacts/harness_q8_0_ptx.cpp -o docs/research/artifacts/kernel_q8_0_padded_O3_sm89.ptx

# 3. Compare stride multipliers (128 vs 132)
diff docs/research/artifacts/kernel_q8_0_unpadded_O3_sm89.ptx docs/research/artifacts/kernel_q8_0_padded_O3_sm89.ptx
```
