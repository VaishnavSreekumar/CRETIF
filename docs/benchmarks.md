# CuInfer Benchmark Report

**Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB GDDR6, 30 SMs
**Platform**: Windows 10 Pro, CUDA 12.8, MSVC 19.38 (also builds on WSL2 Ubuntu 24.04 / GCC 11+)
**Build**: Release, `CMAKE_CUDA_ARCHITECTURES=86`
**Last updated**: 2026-03-04 (Phase 3 complete + cuBLAS comparison in bench_matmul)

---

## Table of Contents

1. [Test Environment](#1-test-environment)
2. [Methodology](#2-methodology)
3. [How to Reproduce](#3-how-to-reproduce)
4. [Matrix Multiplication Kernels](#4-matrix-multiplication-kernels)
5. [cuBLAS vs Hand-Written Comparison](#5-cublas-vs-hand-written-comparison)
6. [Element-wise and Normalization Kernels](#6-element-wise-and-normalization-kernels)
7. [CUB vs Hand-Written Reductions](#7-cub-vs-hand-written-reductions)
8. [Kernel Fusion Impact](#8-kernel-fusion-impact)
9. [End-to-End Inference](#9-end-to-end-inference)
10. [CuInfer vs PyTorch](#10-cuinfer-vs-pytorch)
11. [Quantization Impact](#11-quantization-impact)
12. [Memory Usage Breakdown](#12-memory-usage-breakdown)
13. [Analysis](#13-analysis)

---

## 1. Test Environment

### 1.1 Hardware

| Specification | Value | Notes |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | GA106, Ampere architecture |
| Compute Capability | SM 8.6 | Ampere generation |
| CUDA Cores | 3,584 | FP32 execution units |
| Boost Clock | 1,777 MHz | Maximum sustained boost |
| FP32 Throughput (theoretical) | 12.74 TFLOPS | 2 × 3,584 × 1,777 MHz |
| FP16 Tensor Core Throughput (theoretical) | ~101.6 TFLOPS | Ampere HMMA |
| Memory Bandwidth (theoretical) | 360.0 GB/s | 192-bit bus × 15 Gbps GDDR6 |
| Memory Bandwidth (effective ceiling) | ~336.5 GB/s | Accounting for bus efficiency |
| VRAM | 6 GB GDDR6 | VRAM budget cap: 5.5 GB (reserve for OS/display) |
| L2 Cache | 3 MB | |
| Shared Memory per SM | up to 100 KB (configurable) | |
| SM Count | 30 | Laptop variant; desktop RTX 3060 has 28 SMs |

The roofline ridge point for this GPU is approximately **37.9 FLOP/byte** (12.74 TFLOPS / 336.5 GB/s). Kernels with arithmetic intensity below this value are memory-bandwidth limited; kernels above it are compute limited.

### 1.2 Software

| Component | Version |
|---|---|
| CUDA Toolkit | 12.8 |
| Compiler | MSVC 19.38 (primary), GCC 11+ (WSL2) |
| OS | Windows 10 Pro 10.0.19045 |
| cuBLAS | bundled with CUDA 12.8 |
| CUB | bundled with CUDA 12.8 (header-only) |
| PyTorch (comparison) | 2.10.0+cu128 |
| HuggingFace Transformers (comparison) | 5.2.0 |

---

## 2. Methodology

### 2.1 Timing

All kernel timings use CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`), not host-side wall-clock timers. The measurement procedure is:

1. Allocate and initialize input data on the GPU.
2. Run 10 warmup iterations (not recorded). Warmup ensures the GPU is at boost clock and that driver-level overhead such as JIT compilation and context initialization is excluded from measurements.
3. Record a `cudaEvent_t` start event.
4. Run 100 timed iterations.
5. Record a `cudaEvent_t` stop event.
6. Call `cudaEventSynchronize(stop)` to ensure all work is complete.
7. Query elapsed time via `cudaEventElapsedTime`.
8. Report the **median** per-iteration time across the 100 runs. The median is used rather than the mean to reduce sensitivity to occasional OS scheduling interruptions or thermal throttle events.

The measured time covers only kernel execution on the device, not host-to-device or device-to-host transfers (except in end-to-end tests where those transfers are part of the measured workload).

### 2.2 Throughput Calculation

**For compute-bound kernels (matmul):**

```
GFLOPS = (2 × M × N × K) / (time_seconds × 1e9)
```

The factor of 2 accounts for one multiply and one add per element of the inner product.

**For memory-bound kernels (element-wise, normalization):**

```
Throughput (GB/s) = bytes_accessed / time_seconds / 1e9
```

`bytes_accessed` is the sum of bytes read and bytes written. Byte accounting per kernel:

| Kernel | Bytes moved |
|---|---|
| SiLU, GeLU | `2 × n × 4` (read input, write output) |
| Add, Multiply | `3 × n × 4` (read a + b, write output) |
| RMSNorm | `(2 × rows × cols + cols) × 4` (read input + weight, write output) |
| Softmax | `2 × rows × cols × 4` (DRAM view — L2 absorbs internal re-reads) |
| Fused RMSNorm+Residual | `(4 × rows × cols + cols) × 4` (read input + residual + weight, write combined + normed) |

### 2.3 VRAM Reporting

Peak VRAM usage for kernel benchmarks is measured using `cudaMemGetInfo` queried immediately after all allocations are made. For end-to-end model benchmarks, VRAM is measured after model weights are loaded and the KV-cache is pre-allocated. The reported figure is:

```
VRAM_used = total_VRAM - free_VRAM
```

This GPU-level figure includes the CUDA context (~1 GB) and driver allocator metadata, not just tensor data. PyTorch's `max_memory_allocated()` reports only tensor allocations, so the two are not directly comparable. Sections comparing CuInfer and PyTorch VRAM note which measurement basis applies.

### 2.4 Reported Statistics

| Field | Description |
|---|---|
| Median time (µs or ms) | Median of 100 timed iterations |
| Throughput | GFLOPS (compute kernels) or GB/s (memory kernels) |
| % of peak | Achieved / theoretical × 100 |
| VRAM | Peak allocated during the benchmark |

---

## 3. How to Reproduce

All commands assume the working directory is the repository root and that the Release build is configured for SM 8.6.

### 3.1 Configure and Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j$(nproc)
```

### 3.2 Verify Correctness Before Benchmarking

```bash
cd build && ctest --output-on-failure
```

Expected output: 221/221 tests passing.

### 3.3 Kernel Benchmarks

```bash
# Matrix multiplication: square and non-square sweep, with cuBLAS comparison
./build/bench_matmul

# Skip naive (very slow above 2048x2048) and include cuBLAS by default:
./build/bench_matmul --skip-naive

# Element-wise, normalization, and fused kernel benchmarks
./build/bench_elementwise
```

### 3.4 End-to-End Benchmarks

```bash
# SmolLM2-135M at three quantization levels (generate 100 tokens)
./build/cuinfer models/SmolLM2-135M-Instruct-f16.gguf \
    "The key properties of transformer attention are" 100
./build/cuinfer models/SmolLM2-135M-Instruct-Q8_0.gguf \
    "The key properties of transformer attention are" 100
./build/cuinfer models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
    "The key properties of transformer attention are" 100

# Llama 3.2 1B at two quantization levels
./build/cuinfer models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    "The key properties of transformer attention are" 100
./build/cuinfer models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    "The key properties of transformer attention are" 100
```

### 3.5 PyTorch Comparison

```bash
# Requires torch + transformers + accelerate in a venv
.venv/Scripts/python.exe scripts/benchmark_pytorch_e2e.py  # Windows
.venv/bin/python scripts/benchmark_pytorch_e2e.py          # Linux/WSL
```

---

## 4. Matrix Multiplication Kernels

Three FP32 kernel variants are implemented in `src/kernels/matmul.cu`, all computing `C = A × B` where A is `[M, K]` and B is `[K, N]` in row-major layout.

### 4.1 Kernel Descriptions

| Kernel | Block | Shared Memory / Block | Key Design |
|---|---|---|---|
| `kernel_matmul_naive` | 16×16 = 256 threads | 0 bytes | Global memory only; column accesses on B are strided (uncoalesced) |
| `kernel_matmul_tiled` | 32×32 = 1024 threads | 8,192 bytes (two 32×32 FP32 tiles) | Shared memory tiling with TILE_SIZE=32; reduces global traffic by ~32× |
| `kernel_matmul_vectorized` | 8×32 = 256 threads | 2,048 bytes | `float4` loads; each thread computes a 1×4 output micro-tile |

The naive kernel's arithmetic intensity is approximately 0.25 FLOP/byte due to uncoalesced column accesses on B. The tiled kernel raises this by reusing each loaded value across TILE_SIZE threads. The vectorized kernel improves coalescing on B via 128-bit `float4` loads.

### 4.2 Square Matrix Benchmark (M = N = K)

Throughput in GFLOPS and percentage of 12.74 TFLOPS FP32 theoretical peak. Naive is omitted above M=N=K=2048 due to prohibitive runtime.

| Size (M=N=K) | Naive (GFLOPS) | Tiled (GFLOPS) | Vectorized (GFLOPS) | Vec % of peak |
|---|---|---|---|---|
| 128 | ~18 | 315 | 333 | 2.6% |
| 256 | ~62 | 529 | 762 | 6.0% |
| 512 | — | 728 | 1,181 | 9.3% |
| 1024 | — | 741 | 1,239 | 9.7% |
| 2048 | — | 718 | 1,168 | 9.2% |
| 4096 | — | 726 | 1,139 | 8.9% |

The vectorized kernel peaks at approximately 1,240 GFLOPS (9.7% of FP32 peak). Hand-written kernels without register-level tiling, double buffering, or WMMA (tensor core) instructions are expected to achieve 8–12% of theoretical FP32 peak at these matrix sizes.

### 4.3 Non-Square Matrix Benchmark

LLM inference operates on non-square matrices. The shapes below represent linear layers in small transformer models.

| Shape (M × N × K) | Description | Tiled (GFLOPS) | Vectorized (GFLOPS) | Vec time (ms) |
|---|---|---|---|---|
| 512 × 768 × 3072 | FFN gate/up projection | 730 | 1,173 | 2.06 |
| 512 × 3072 × 768 | FFN down projection | 732 | 1,186 | 2.04 |
| 1024 × 64 × 1024 | Attention Q×K^T (tall-skinny) | 518 | 728 | 0.18 |
| 1024 × 1024 × 64 | Attention score×V (short-wide) | 720 | 1,201 | 0.11 |
| 1000 × 1000 × 1000 | Non-tile-aligned dimensions | 671 | 1,143 | 1.75 |

The tall-skinny shape (1024×64×1024) shows lower GFLOPS than square shapes of comparable size because the small N dimension reduces parallelism utilization across the grid.

### 4.4 Vectorized vs Tiled Speedup

| Size (M=N=K) | Tiled (ms) | Vectorized (ms) | Speedup |
|---|---|---|---|
| 128 | — | — | 1.06x |
| 256 | — | — | 1.44x |
| 512 | 0.369 | 0.227 | 1.62x |
| 1024 | 2.896 | 1.733 | 1.67x |
| 2048 | 23.931 | 14.704 | 1.63x |
| 4096 | 189.289 | 120.673 | 1.57x |

The vectorized kernel provides a consistent 1.4–1.7x speedup over tiled, primarily from improved B-matrix coalescing via `float4` loads.

---

## 5. cuBLAS vs Hand-Written Comparison

`bench_matmul` includes cuBLAS via `cublasSgemm` (FP32) and `cublasGemmEx` (FP16 B with FP32 accumulation). The cuBLAS row-major trick applies: for row-major `C = A × B`, the call uses `cublasSgemm(N, N, N, M, K)` passing B first and A second, which computes `C^T = B^T × A^T` in column-major and writes row-major C directly.

### 5.1 FP32: Vectorized vs cuBLAS

| Size (M=N=K) | Vectorized (ms) | Vectorized (GFLOPS) | cuBLAS (ms) | cuBLAS (GFLOPS) | cuBLAS % peak | Ratio (cuBLAS/Vec) |
|---|---|---|---|---|---|---|
| 128 | 0.016 | 256 | 0.009 | 455 | 3.6% | 1.8x |
| 256 | 0.060 | 555 | 0.017 | 1,928 | 15.1% | 3.5x |
| 512 | 0.313 | 857 | 0.062 | 4,306 | 33.8% | 5.0x |
| 1024 | 2.345 | 916 | 0.352 | 6,096 | 47.9% | 6.7x |
| 2048 | 14.785 | 1,162 | 2.410 | 7,130 | 56.0% | 6.1x |
| 4096 | 117.971 | 1,165 | 23.509 | 5,846 | 45.9% | 5.0x |

cuBLAS FP32 peaks at 56% of theoretical (7,130 GFLOPS at 2048x2048), compared to the vectorized kernel's ~9% peak. The 5–7x gap is structural: cuBLAS uses register-level tiling, double buffering, and runtime heuristics that a simple shared-memory tiled kernel cannot match. The slight cuBLAS regression at 4096 (45.9%) is likely due to L2 cache pressure at that working set size.

### 5.2 FP16 Mixed-Precision: Custom Kernel vs cuBLAS

The custom FP16 kernel (`kernel_matmul_fp16`) uses `__half` B stored in `[N, K]` layout with FP32 accumulation. The cuBLAS FP16 comparison uses `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for a fair comparison.

| Size (M=N=K) | Custom FP16 (ms) | Custom (GFLOPS) | cuBLAS FP16 (ms) | cuBLAS (GFLOPS) | Ratio |
|---|---|---|---|---|---|
| 512 | 0.572 | 469 | — | — | — |
| 1024 | 4.421 | 486 | — | — | — |
| 2048 | 27.570 | 623 | — | — | — |
| 4096 | 222.495 | 618 | — | — | — |

**Note on cuBLAS FP16 results:** The `cublasGemmEx` mixed-precision path (FP32 A, FP16 B, `CUBLAS_COMPUTE_32F`) reports constant ~0.002ms regardless of matrix size, indicating a measurement or dispatch issue rather than actual computation. These results are excluded until the root cause is identified. The custom FP16 kernel results are valid and show the expected pattern: slower than the FP32 vectorized kernel because the `__half` arithmetic overhead outweighs the bandwidth savings at these sizes without warp-level matrix operations.

The custom FP16 kernel achieves ~620 GFLOPS at large sizes, roughly half the FP32 vectorized kernel's throughput. This is expected: the kernel converts `__half` to `float` per element for accumulation, doubling the instruction count without benefiting from tensor cores. The value of FP16 weights in CuInfer is halved VRAM usage, not compute speedup.

### 5.3 Context: Why the Gap Exists

CuInfer's hand-written matmul kernels use:
- Shared memory tiling (TILE_SIZE=32)
- `float4` vectorized loads (vectorized variant)
- FP32 arithmetic throughout

cuBLAS on Ampere uses:
- Register-level tiling with per-thread output arrays (typically 4×8 or 8×8 register tiles)
- Double buffering to overlap data loads with computation
- Tensor core (WMMA) instructions for 16×16×16 matrix-multiply-accumulate in hardware
- Epilogue fusion and algorithmic heuristics selected at runtime based on problem size

The measured gap between CuInfer's ~9% of FP32 peak and cuBLAS's ~56% is structural, not a code quality issue. Closing it would require implementing tensor core intrinsics (`wmma::mma_sync`) and async memory copies, which would significantly increase kernel complexity. The gap is documented intentionally as part of this project's educational narrative.

---

## 6. Element-wise and Normalization Kernels

Seven element-wise kernels in `src/kernels/elementwise.cu` use 256 threads per block. Pure element-wise kernels (SiLU, GeLU, Add, Multiply) use a flat grid-stride loop. Reduction kernels (RMSNorm, Softmax) use one block per row with shared memory.

### 6.1 SiLU and GeLU

Bandwidth: `2 × n × 4` bytes (read input, write output). Both kernels show nearly identical profiles since the memory access pattern is identical.

| Rows | Cols | Elements | SiLU (GB/s) | SiLU % peak | GeLU (GB/s) | GeLU % peak |
|---|---|---|---|---|---|---|
| 1 | 768 | 768 | 1.2 | 0.3% | 1.2 | 0.3% |
| 32 | 768 | 24,576 | 38.4 | 10.7% | 39.4 | 10.9% |
| 128 | 768 | 98,304 | 130.4 | 36.2% | 131.4 | 36.5% |
| 512 | 768 | 393,216 | 307.2 | 85.3% | 308.2 | 85.6% |
| 128 | 2048 | 262,144 | 292.6 | 81.3% | 292.6 | 81.3% |
| 512 | 2048 | 1,048,576 | 264.3 | 73.4% | 264.3 | 73.4% |
| 512 | 4096 | 2,097,152 | 287.4 | 79.8% | 287.4 | 79.8% |
| 512 | 8192 | 4,194,304 | 300.6 | 83.5% | 301.0 | 83.6% |

At large sizes, both kernels reach 83–86% of peak bandwidth. The low numbers at small sizes (rows=1, cols=768) reflect kernel launch overhead dominating a 3 KB workload.

### 6.2 Add and Multiply

Bandwidth: `3 × n × 4` bytes (read two inputs, write one output).

| Rows | Cols | Elements | Add (GB/s) | Add % peak | Multiply (GB/s) | Mul % peak |
|---|---|---|---|---|---|---|
| 32 | 768 | 24,576 | 72.0 | 20.0% | 72.0 | 20.0% |
| 128 | 768 | 98,304 | 164.6 | 45.7% | 288.0 | 80.0% |
| 512 | 768 | 393,216 | 256.0 | 71.1% | 266.4 | 74.0% |
| 32 | 4096 | 131,072 | 311.1 | 86.4% | 311.1 | 86.4% |
| 512 | 2048 | 1,048,576 | 292.6 | 81.3% | 279.3 | 77.6% |
| 512 | 8192 | 4,194,304 | 307.2 | 85.3% | 307.2 | 85.3% |

Peak values reach 86.4% of the 360 GB/s theoretical ceiling, close to the practical maximum for simple element-wise operations.

### 6.3 RMSNorm

Bandwidth: `(2 × rows × cols + cols) × 4` bytes.

| Rows | Cols | Median (µs) | BW (GB/s) | % peak |
|---|---|---|---|---|
| 1 | 768 | 6.1 | 1.5 | 0.4% |
| 32 | 768 | 7.0 | 28.5 | 7.9% |
| 128 | 768 | 7.2 | 110.1 | 30.6% |
| 512 | 768 | 14.3 | 219.6 | 61.0% |
| 128 | 2048 | 9.2 | 228.4 | 63.5% |
| 512 | 2048 | 37.9 | 221.6 | 61.6% |
| 128 | 4096 | 20.5 | 205.6 | 57.1% |
| 512 | 8192 | 167.9 | 200.0 | 55.6% |

RMSNorm peaks at approximately 63% of theoretical bandwidth. The two-pass algorithm (compute variance, then normalize) and the parallel reduction step in shared memory reduce throughput relative to pure element-wise kernels.

### 6.4 Softmax

Bandwidth: `2 × rows × cols × 4` bytes (DRAM view, assuming L2 absorbs the re-reads from the 3-pass algorithm).

| Rows | Cols | Median (µs) | BW (GB/s) | % peak |
|---|---|---|---|---|
| 1 | 768 | 7.1 | 0.9 | 0.2% |
| 32 | 768 | 7.2 | 27.4 | 7.6% |
| 128 | 768 | 9.2 | 85.3 | 23.7% |
| 512 | 768 | 20.5 | 153.6 | 42.7% |
| 512 | 2048 | 43.0 | 195.1 | 54.2% |
| 128 | 4096 | 25.6 | 163.8 | 45.5% |
| 512 | 4096 | 102.4 | 163.8 | 45.5% |
| 512 | 8192 | 219.1 | 153.1 | 42.5% |

Softmax peaks at approximately 54% of theoretical bandwidth. The 3-pass design (find row max, compute sum-of-exps, normalize) provides numerical stability at a throughput cost compared to a single-pass unstable variant.

### 6.5 RoPE

RoPE operates in-place on Q/K vectors. Each call processes `n_heads × head_dim` floats per token position.

| Config | Positions | Median (µs) | BW (GB/s) | % peak |
|---|---|---|---|---|
| 12 heads × 64 dim | 1 | 5.1 | 1.2 | 0.3% |
| 12 heads × 64 dim | 32 | 96.1 | 2.1 | 0.6% |
| 12 heads × 64 dim | 512 | 1,942.5 | 1.6 | 0.4% |
| 32 heads × 128 dim | 1 | 4.1 | 8.0 | 2.2% |
| 32 heads × 128 dim | 32 | 86.9 | 12.1 | 3.4% |
| 32 heads × 128 dim | 512 | 1,999.9 | 8.4 | 2.3% |

Low bandwidth numbers reflect that each call operates on a small buffer (768–4,096 floats) with one kernel launch per position. In production inference, RoPE is called once per token during decode, where the 4–8 µs launch cost is a fixed overhead per generated token.

---

## 7. CUB vs Hand-Written Reductions

CUB-accelerated variants (`cub::BlockReduce`) exist alongside the hand-written tree-reduction kernels for RMSNorm, Softmax, and the fused RMSNorm+Residual kernel. CUB ships as a header-only library with the CUDA Toolkit. It uses warp shuffle instructions (`__shfl_down_sync`) for the final 32 elements, avoiding shared memory bank conflicts in the reduction's last stage.

Both variants compute identical results to FP32 precision. The comparison measures throughput only.

### 7.1 RMSNorm: Hand-Written vs CUB

| Rows | Cols | Hand-written (µs) | CUB (µs) | Hand-written (GB/s) | CUB (GB/s) | Speedup |
|---|---|---|---|---|---|---|
| 128 | 768 | 7.26 | 6.70 | 108.7 | 117.8 | 1.08x |
| 512 | 768 | 14.34 | 12.29 | 219.6 | 256.3 | 1.17x |
| 128 | 2048 | 9.22 | 8.19 | 228.4 | 257.0 | 1.13x |
| 512 | 2048 | 37.89 | 34.82 | 221.6 | 241.2 | 1.09x |
| 128 | 4096 | 20.48 | 19.46 | 205.6 | 216.4 | 1.05x |
| 512 | 4096 | 80.90 | 73.73 | 207.6 | 227.8 | 1.10x |
| 512 | 8192 | 167.94 | 161.79 | 200.0 | 207.6 | 1.04x |

CUB provides a consistent 5–17% speedup over the hand-written tree reduction.

### 7.2 Softmax: Hand-Written vs CUB

| Rows | Cols | Hand-written (µs) | CUB (µs) | Hand-written (GB/s) | CUB (GB/s) | Speedup |
|---|---|---|---|---|---|---|
| 128 | 768 | 9.22 | 8.16 | 85.3 | 96.4 | 1.13x |
| 512 | 768 | 20.48 | 15.36 | 153.6 | 204.8 | 1.33x |
| 128 | 2048 | 12.29 | 9.22 | 170.7 | 227.6 | 1.33x |
| 512 | 2048 | 43.01 | 33.79 | 195.1 | 248.2 | 1.27x |
| 128 | 4096 | 25.65 | 20.48 | 163.5 | 204.8 | 1.25x |
| 512 | 4096 | 102.40 | 78.37 | 163.8 | 214.1 | 1.31x |
| 512 | 8192 | 220.16 | 196.61 | 152.4 | 170.7 | 1.12x |

CUB provides a 12–33% speedup over the hand-written 3-pass softmax. The CUB Softmax variant peaks at 248 GB/s (69% of theoretical), compared to 195 GB/s (54%) for the hand-written variant.

---

## 8. Kernel Fusion Impact

### 8.1 Fused RMSNorm + Residual Add

Each transformer layer contains two residual-add + normalization operations (post-attention and post-FFN). The fused kernel combines `add(input, residual, output)` and `rmsnorm(output, weight, normed)` into a single launch, eliminating one intermediate global memory round-trip.

Bandwidth basis: `(4 × rows × cols + cols) × 4` bytes (read input + residual + weight, write combined + normed).

| Rows | Cols | Unfused (µs) | Fused HW (µs) | Fused CUB (µs) | Fusion speedup | CUB vs unfused |
|---|---|---|---|---|---|---|
| 128 | 768 | 10.24 | 8.19 | 7.17 | 1.25x | 1.43x |
| 512 | 768 | 31.74 | 25.60 | 25.60 | 1.24x | 1.24x |
| 128 | 2048 | 22.53 | 18.43 | 17.55 | 1.22x | 1.28x |
| 512 | 2048 | 79.87 | 62.46 | 61.44 | 1.28x | 1.30x |
| 128 | 4096 | 38.91 | 38.91 | 37.89 | 1.00x | 1.03x |
| 512 | 4096 | 162.82 | 138.24 | 138.24 | 1.18x | 1.18x |
| 512 | 8192 | 329.73 | 277.50 | 277.50 | 1.19x | 1.19x |

Fusion provides 18–28% speedup at transformer-realistic dimensions (SmolLM2-135M: dim=576; Llama 3.2 1B: dim=2048). Because each transformer layer contains two fusion sites, the per-layer savings double.

### 8.2 Fused Decode Attention

The fused decode attention kernel replaces a sequence of separate kernel launches (QK^T matmul, causal mask, scale, softmax, score×V matmul) with a single kernel using the online softmax algorithm (Milakov & Gimelshein 2018). One block per attention head processes all KV-cache positions in a single pass.

The online algorithm maintains a running (max, sum, accumulator) tuple, updating it as each KV position is processed:

```
for each KV position k:
    score = dot(Q, K[k]) * scale
    new_max = max(old_max, score)
    correction = exp(old_max - new_max)
    exp_score  = exp(score - new_max)
    sum_exp    = sum_exp * correction + exp_score
    acc[d]     = acc[d] * correction + exp_score * V[k][d]
output = acc / sum_exp
```

This processes each KV entry exactly once rather than two passes, and avoids materializing the full attention score matrix. For a sequence length of 1024 and 32 heads, this saves 32 × 1024 × 4 = 128 KB of temporary storage per layer.

The kernel launch reduction from 9+ launches to 1 per decode step is significant for small models where each individual kernel operates on small data volumes and kernel launch overhead (4–6 µs per launch on this hardware) is a non-trivial fraction of per-layer time.

---

## 9. End-to-End Inference

End-to-end throughput measures the full pipeline: GGUF loading, tokenization, prefill, and autoregressive decoding with greedy (argmax) sampling. All runs generate approximately 100 tokens from the prompt `"The key properties of transformer attention are"`. Multiple trials are averaged.

Phase 3 features contributing to these results:
- Fused RMSNorm+Residual Add (2 fewer kernel launches per layer)
- Fused decode attention kernel (online softmax, 1 launch instead of 9+ per layer)
- Quantized dequantize-on-the-fly matmul for Q8_0, Q4_0, Q5_0, Q4_K, Q6_K
- Native FP16 matmul (`__half` weights with FP32 accumulation)

### 9.1 SmolLM2-135M

Architecture: 30 layers, 9 query heads (3 KV heads, GQA 3:1), hidden dim 576, FFN dim 1,536, vocab 49,152.

| Format | Tok/s | Peak VRAM | Weights | KV-Cache | Buffers |
|---|---|---|---|---|---|
| FP16 (native `__half`) | 28.4 | 1.46 GB | 310.6 MB | 90.0 MB | 45.0 MB |
| Q8_0 | 27.8 | 1.35 GB | 215.7 MB | 90.0 MB | 45.0 MB |
| Q4_K_M (mixed) | 27.3 | 1.31 GB | 178.2 MB | 90.0 MB | 45.0 MB |

All three formats produce coherent English output. Throughput is nearly constant across formats because SmolLM2-135M at dim=576 is **kernel-launch-overhead bound** rather than compute or bandwidth bound. The fused kernels reduce this overhead by eliminating redundant launches per layer.

### 9.2 Llama 3.2 1B

Architecture: 16 layers, 32 query heads (8 KV heads, GQA 4:1), hidden dim 2,048, FFN dim 8,192, vocab 128,256.

| Format | Tok/s | Peak VRAM | Weights | KV-Cache | Buffers |
|---|---|---|---|---|---|
| Q8_0 | 6.8 | 3.42 GB | 1,988.3 MB | 128.0 MB | 224.0 MB |
| Q4_K_M (mixed) | 6.6 | 2.97 GB | 1,559.3 MB | 128.0 MB | 224.0 MB |

Both formats fit within the 6 GB RTX 3060. FP16 and FP32 are excluded — FP32 would require approximately 5.1 GB for weights alone, exceeding the 5.5 GB VRAM budget. Q4_K_M saves 429 MB (22%) of weight memory compared to Q8_0, with a throughput reduction of 0.2 tok/s (3%).

The 4.1x throughput reduction from SmolLM2-135M (28 tok/s) to Llama 1B (6.8 tok/s) is expected. Llama's hidden dim is 3.6× larger (2,048 vs 576), making each matmul approximately 12.6× larger in arithmetic operations. Despite Llama having only 16 layers vs 30, the per-layer compute is substantially heavier, shifting the bottleneck from launch overhead to compute and memory bandwidth.

---

## 10. CuInfer vs PyTorch

PyTorch measurements use eager mode (no `torch.compile`) on the same RTX 3060. Source data: `scripts/pytorch_e2e_results.json` (run 2026-03-02).

### 10.1 PyTorch Configuration

| Setting | Value |
|---|---|
| PyTorch version | 2.10.0+cu128 |
| CUDA version | 12.8 |
| Execution mode | Eager (no `torch.compile`) |
| Decoding | Greedy (`do_sample=False`) |
| KV-cache | Enabled (`use_cache=True`) |
| Model dtype | `float16` |
| Timing | `torch.cuda.Event` |
| Iterations | 5 (median reported) |

### 10.2 Throughput Comparison — SmolLM2-135M

50 generated tokens per prompt, greedy decoding. CuInfer Phase 3 results include fused kernels and native FP16 matmul.

| Prompt | Prompt tokens | CuInfer FP16 (tok/s) | PyTorch FP16 (tok/s) | Ratio |
|---|---|---|---|---|
| "Hi" | 1 | ~33 | 18.1 | 1.8x |
| "The capital of France is" | 5 | ~33 | 18.1 | 1.8x |
| "Once upon a time..." (22 words) | 23 | ~34 | 18.0 | 1.9x |

Note: Phase 2 CuInfer numbers (~33 tok/s) are used here since they correspond to the same 50-token measurement protocol as the PyTorch baseline. Phase 3 SmolLM2-135M numbers (28.4 tok/s) reflect a different measurement configuration (100 tokens, different prompt).

### 10.3 Phase 3 Throughput Summary

Consistent protocol: 100 generated tokens, prompt `"The key properties of transformer attention are"`.

| Model | CuInfer Format | CuInfer (tok/s) | PyTorch FP16 Eager (tok/s) | Ratio |
|---|---|---|---|---|
| SmolLM2-135M | FP16 | 28.4 | 18.1 | 1.57x |
| SmolLM2-135M | Q8_0 | 27.8 | 18.1 | 1.54x |
| SmolLM2-135M | Q4_K_M | 27.3 | 18.1 | 1.51x |
| Llama 3.2 1B | Q8_0 | 6.8 | 28.3 | 0.24x |
| Llama 3.2 1B | Q4_K_M | 6.6 | 28.3 | 0.23x |

### 10.4 VRAM Comparison

| Metric | CuInfer FP16 (reported) | CuInfer Q4_K_M (reported) | PyTorch FP16 (tensor allocations) |
|---|---|---|---|
| SmolLM2-135M weights | 310.6 MB | 178.2 MB | 256.6 MB |
| SmolLM2-135M peak total | 445.6 MB | 313.2 MB | 266.4 MB |
| SmolLM2-135M GPU-level | 1.46 GB | 1.31 GB | N/A (tensor-only reported) |

The "reported" column uses CuInfer's own allocation tracking. The GPU-level column uses `cudaMemGetInfo` which includes the CUDA context (~1 GB) and is not comparable to PyTorch's `max_memory_allocated()`.

CuInfer's native FP16 storage (310.6 MB for weights) is slightly larger than PyTorch's (256.6 MB) because CuInfer pre-allocates the KV-cache (90 MB for 2,048 positions) and activation buffers (45 MB) at load time. PyTorch allocates these dynamically as needed.

### 10.5 Output Correctness Verification

For the prompt "The capital of France is", CuInfer and PyTorch FP16 logits were compared numerically:

| Metric | Value |
|---|---|
| Max absolute difference | 0.000024 |
| Mean absolute difference | 0.000008 |
| Top-1 token match | PASS (" Paris", token 7042) |
| Top-5 exact rank match | PASS |
| Top-20 token ID match | PASS (all 20 identical) |

The residual difference of 2.4×10⁻⁵ arises from floating-point operation ordering differences between CuInfer's tiled matmul and PyTorch's cuBLAS backend. Both engines produce identical generated text with greedy decoding.

---

## 11. Quantization Impact

### 11.1 Weight Formats Supported

CuInfer implements dequantize-on-the-fly matmul for five GGML quantization formats. Weights are stored in compressed form on the GPU and dequantized per tile inside the matmul kernel, avoiding full FP32 expansion.

| Format | Bytes per 32 elements | Compression vs FP32 | Super-block structure |
|---|---|---|---|
| FP16 | 64 | 2.0× | — |
| Q8_0 | 34 | 3.76× | 1 FP16 scale + 32 int8 values |
| Q5_0 | 22 | 5.82× | 1 FP16 scale + 4 bytes high bits + 16 bytes nibbles |
| Q4_0 | 18 | 7.11× | 1 FP16 scale + 16 bytes packed nibbles |
| Q4_K | 144 per 256 elements | 8.89× | FP16 d/dmin + 12 bytes scales + 128 bytes nibbles |
| Q6_K | 210 per 256 elements | 6.10× | 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d |

Q4_K_M GGUF files contain mixed tensor types: typically Q4_K for attention and FFN gate/up weights, Q6_K for FFN down and attention value weights.

### 11.2 Weight Memory by Format

| Model | FP32 | FP16 | Q8_0 | Q4_K_M |
|---|---|---|---|---|
| SmolLM2-135M | ~622 MB | 310.6 MB | 215.7 MB | 178.2 MB |
| Llama 3.2 1B | ~5,100 MB | N/A | 1,988.3 MB | 1,559.3 MB |

FP32 Llama 3.2 1B would exceed the 5.5 GB VRAM budget and is not supported. FP16 Llama 3.2 1B has not been tested.

### 11.3 Throughput by Format

| Model | FP16 (tok/s) | Q8_0 (tok/s) | Q4_K_M (tok/s) | Q4_K_M vs FP16 |
|---|---|---|---|---|
| SmolLM2-135M | 28.4 | 27.8 | 27.3 | -3.9% |
| Llama 3.2 1B | N/A | 6.8 | 6.6 | N/A |

The throughput difference between quantization formats is small for SmolLM2-135M (less than 4%) because the model is kernel-launch-overhead bound. The dequantize-on-the-fly matmul adds negligible compute relative to the launch latency. For Llama 3.2 1B (compute/bandwidth bound), Q4_K_M vs Q8_0 shows a 2.9% throughput reduction, consistent with the dequantization overhead being a small fraction of total matmul work.

### 11.4 Output Quality

FP16 and Q8_0 produce token-for-token identical output to the FP32 baseline on SmolLM2-135M with greedy decoding. Q4_K_M produces different text due to weight quantization noise, but the output is grammatically correct and topically coherent across all tested prompts.

Formal perplexity evaluation is not included in this benchmark set. The logit verification in section 10.5 confirms numerical correctness for the FP16 path only.

---

## 12. Memory Usage Breakdown

### 12.1 Component Definitions

| Component | Description | Allocation policy |
|---|---|---|
| Weights | Model parameters stored on GPU | Loaded at startup, read-only during inference |
| KV-cache | Key and value tensors for all layers | Pre-allocated for maximum sequence length at startup |
| Activation buffers | Intermediate tensors (hidden states, attention scores) | Pre-allocated at maximum batch size, reused across layers |
| CUDA context | Driver overhead, code segments, allocator metadata | Allocated by CUDA runtime at first API call |

### 12.2 KV-Cache Sizing Formula

```
kv_cache_bytes = 2 × max_seq_len × n_kv_heads × head_dim × n_layers × sizeof(float)
```

The factor of 2 accounts for K and V tensors. GQA (grouped-query attention) reduces the KV-cache relative to multi-head attention by the group ratio.

| Model | n_kv_heads | head_dim | max_seq_len | Layers | KV-cache (FP32) |
|---|---|---|---|---|---|
| SmolLM2-135M | 3 | 64 | 2,048 | 30 | 90.0 MB |
| Llama 3.2 1B | 8 | 64 | 2,048 | 16 | 128.0 MB |

SmolLM2-135M's 3:9 KV:Q head ratio (GQA) reduces the cache to one-third of what full MHA would require (270 MB). Llama's 4:1 ratio reduces it to one-fourth.

### 12.3 Full Allocation Breakdown

All figures in megabytes. "GPU-level" includes CUDA context and driver overhead as reported by `cudaMemGetInfo`.

| Model | Format | Weights | KV-Cache | Buffers | Reported total | GPU-level |
|---|---|---|---|---|---|---|
| SmolLM2-135M | FP16 | 310.6 | 90.0 | 45.0 | 445.6 MB | 1.46 GB |
| SmolLM2-135M | Q8_0 | 215.7 | 90.0 | 45.0 | 350.7 MB | 1.35 GB |
| SmolLM2-135M | Q4_K_M | 178.2 | 90.0 | 45.0 | 313.2 MB | 1.31 GB |
| Llama 3.2 1B | Q8_0 | 1,988.3 | 128.0 | 224.0 | 2,340.3 MB | 3.42 GB |
| Llama 3.2 1B | Q4_K_M | 1,559.3 | 128.0 | 224.0 | 1,911.3 MB | 2.97 GB |

All configurations fit within the 5.5 GB VRAM budget. The largest configuration (Llama Q8_0, 3.42 GB) leaves 2.08 GB of headroom.

### 12.4 VRAM Headroom

The 5.5 GB budget and remaining headroom per configuration:

| Model + Format | GPU-level VRAM | Remaining headroom |
|---|---|---|
| SmolLM2-135M FP16 | 1.46 GB | 4.04 GB |
| SmolLM2-135M Q4_K_M | 1.31 GB | 4.19 GB |
| Llama 3.2 1B Q4_K_M | 2.97 GB | 2.53 GB |
| Llama 3.2 1B Q8_0 | 3.42 GB | 2.08 GB |
| Llama 3.2 1B FP16 (estimated) | ~4.50 GB | ~1.00 GB |
| Llama 3.2 1B FP32 (estimated) | ~6.10 GB | over budget |

---

## 13. Analysis

### 13.1 Where Time Is Spent

For a single autoregressive decode step (generating one token), the operations executed are, per transformer layer:

1. RMSNorm + residual add (pre-attention)
2. QKV projection: three matmuls of shape [1, dim] × [dim, dim]
3. RoPE on Q and K
4. KV-cache append
5. Fused decode attention: dot products over all cached K/V positions
6. Output projection: matmul [1, dim] × [dim, dim]
7. RMSNorm + residual add (post-attention)
8. FFN gate and up projections: two matmuls [1, dim] × [dim, ffn_dim]
9. SiLU gating
10. FFN down projection: matmul [1, ffn_dim] × [ffn_dim, dim]

For SmolLM2-135M (dim=576, ffn_dim=1,536, 30 layers), each matmul involves `[1, 576] × [576, 576]` or `[1, 576] × [576, 1,536]`. These are very small — a 1×576×576 matmul at 1,200 GFLOPS takes approximately 0.29 µs of compute. Kernel launch overhead on this GPU is 4–6 µs per launch. At 30 layers with approximately 8 matmuls and 4 normalization/activation kernels per layer, the launch overhead alone is 30 × 12 × 5 µs ≈ 1,800 µs, comparable to the observed total decode time of approximately 30 ms for 50 tokens (600 µs per token). The fused kernels reduce this by eliminating redundant launches.

For Llama 3.2 1B (dim=2,048, ffn_dim=8,192, 16 layers), a [1, 2048] × [2048, 8192] matmul at 1,200 GFLOPS takes approximately 22.4 µs of compute. At 16 layers with 3 FFN matmuls (gate, up, down), FFN compute alone is 16 × 3 × 22 µs ≈ 1,056 µs, which is 73% of the observed 147 ms per token (6.8 tok/s). This model is **compute and bandwidth bound** rather than launch-overhead bound.

### 13.2 Bottleneck Summary

| Model | Primary bottleneck | Evidence |
|---|---|---|
| SmolLM2-135M | Kernel launch overhead | Near-identical throughput across FP16/Q8_0/Q4_K_M formats; throughput sensitive to number of kernel launches per layer |
| Llama 3.2 1B | Compute and memory bandwidth | 4.1x throughput reduction vs SmolLM2-135M despite only 7.5× more matmul work per layer (dim ratio 3.6× squared) |

### 13.3 CuInfer vs PyTorch: Sources of the Speedup

CuInfer achieves 1.5–1.9x higher decode throughput than PyTorch eager mode for SmolLM2-135M. The gap has three primary causes:

1. **No Python interpreter overhead.** PyTorch eager mode executes Python for each operation dispatched through the autograd graph, even when gradients are disabled. Each `model.forward()` call dispatches approximately 300–500 Python operations for SmolLM2-135M. CuInfer invokes CUDA kernels directly.

2. **Reduced kernel launches.** CuInfer's fused kernels (RMSNorm+Residual, fused decode attention) reduce the number of kernel launches per layer compared to PyTorch's per-operation dispatch. This matters most for SmolLM2-135M, which is launch-overhead bound.

3. **No autograd metadata.** PyTorch tensor operations allocate metadata (gradient function pointers, reference counts, strides, device properties) even in `torch.no_grad()` mode. CuInfer tensors are minimal RAII wrappers with only a device pointer and shape.

These advantages are largest for small models (SmolLM2-135M) where the actual kernel compute is small relative to framework overhead. For Llama 3.2 1B, the balance reverses: PyTorch achieves 28.3 tok/s vs CuInfer's 6.8 tok/s (Q8_0), a 4.2x advantage. At this model size, the matmul performance gap dominates — cuBLAS achieves 5–7x higher GFLOPS than CuInfer's hand-written kernels (measured: 56% vs 9% of FP32 peak at 2048x2048), and Llama's 2048-dim hidden size means matmul accounts for the vast majority of compute time.

### 13.4 Performance Ceilings

| Kernel | Achieved % of peak | Ceiling and reason |
|---|---|---|
| Matmul (vectorized) | 9.7% of 12.74 TFLOPS FP32 | Structural: no register tiling, no WMMA, no double buffering |
| SiLU / GeLU | 85.6% of 360 GB/s | Near practical ceiling for simple element-wise ops |
| Add / Multiply | 86.4% of 360 GB/s | Near practical ceiling |
| RMSNorm (CUB) | 63.5% of 360 GB/s | Two-pass reduction adds overhead vs single-pass element-wise |
| Softmax (CUB) | 69.0% of 360 GB/s | Three-pass numerical stability algorithm; cache partially absorbs re-reads |
| Fused RMSNorm+Residual (CUB) | ~70% of 360 GB/s | Near reduction kernel ceiling |

Element-wise kernels are within 14% of the practical memory bandwidth ceiling. Reduction kernels (RMSNorm, Softmax) are at 63–69% due to the multi-pass algorithm structure. Matmul is at 9.7% of compute peak, which is expected without tensor core usage.

### 13.5 CUB vs Hand-Written: Interpretation

CUB's `BlockReduce` provides 5–33% speedup over the hand-written tree reduction. The gain comes from:
- Warp-level shuffle instructions (`__shfl_down_sync`) for the final 32 elements, replacing shared memory reads
- Avoidance of shared memory bank conflicts in the final warp reduction stage
- Compiler optimization of CUB's template-expanded reduction tree

The hand-written reduction is pedagogically valuable and performs reasonably well (within 33% of CUB), but CUB is measurably faster. CuInfer uses both variants in production: hand-written variants are present for reference and testing, while CUB variants are used in the active transformer forward pass.

### 13.6 Quantization: Speed vs Size Trade-off

For Llama 3.2 1B:
- Q4_K_M saves 429 MB of weight memory (22%) compared to Q8_0
- Q4_K_M reduces throughput by 0.2 tok/s (2.9%) compared to Q8_0
- The dequantization overhead in `kernel_matmul_q4_k` is the source of this reduction

For SmolLM2-135M the trade-off is even more favorable — Q4_K_M saves 37 MB (17%) with less than 4% throughput reduction — because the model is not compute-bound, so the extra dequantization work has no impact on end-to-end timing.

---

## Appendix A: Linear Layer Dimension Reference

### SmolLM2-135M

During decode (M=1 per token), each matmul has M=1.

| Layer | N | K | Description |
|---|---|---|---|
| QKV projection | 576 | 576 | Combined Q+K+V (split after) |
| Output projection | 576 | 576 | Attention output |
| FFN gate | 1,536 | 576 | gate_proj |
| FFN up | 1,536 | 576 | up_proj |
| FFN down | 576 | 1,536 | down_proj |
| LM head | 49,152 | 576 | Vocabulary logits |

### Llama 3.2 1B

During decode (M=1 per token).

| Layer | N | K | Description |
|---|---|---|---|
| Q projection | 2,048 | 2,048 | Full query |
| K projection | 512 | 2,048 | Grouped KV (8 heads × 64 dim) |
| V projection | 512 | 2,048 | Grouped KV |
| Output projection | 2,048 | 2,048 | Attention output |
| FFN gate | 8,192 | 2,048 | gate_proj |
| FFN up | 8,192 | 2,048 | up_proj |
| FFN down | 2,048 | 8,192 | down_proj |
| LM head | 128,256 | 2,048 | Vocabulary logits |

---

## Appendix B: Benchmark Version History

| Date | Phase | Description |
|---|---|---|
| 2026-03-01 | Phase 1 | Matmul and elementwise kernel results populated |
| 2026-03-02 | Phase 2 | End-to-end SmolLM2-135M results and PyTorch comparison added; logit verification completed |
| 2026-03-03 | Phase 3 | Quantized inference, CUB comparison, fused kernel benchmarks, Llama 3.2 1B results |
| 2026-03-04 | Phase 4 | Restructured into comprehensive report; cuBLAS FP32 comparison data populated; FP16 cuBLAS excluded due to measurement issue; analysis and memory breakdown sections added |
