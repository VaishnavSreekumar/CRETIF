# CuInfer Kernel Design

All CUDA kernels in CuInfer are written from scratch — no cuBLAS, no cuDNN. This document covers the design patterns, algorithms, and implementation details of every kernel.

**36 kernels total** across 6 categories, targeting NVIDIA RTX 3060 (SM 8.6, 6 GB VRAM).

---

## Design Patterns

### Tiled Shared Memory (Matrix Multiplication)

The fundamental optimization across all matmul kernels. Instead of each thread independently reading global memory (400+ cycle latency), a block of threads cooperatively loads a tile into shared memory (~5 cycle latency), then all threads read from the shared copy.

**Data reuse argument**: For a TILE_SIZE=32 tile, each global memory load is reused by 32 threads. This reduces global traffic by ~32x compared to the naive kernel.

**Synchronization**: Two `__syncthreads()` per tile iteration — one after loading (all threads must finish before any thread reads) and one after computing (prevent next load from overwriting data still in use).

### Dequantize-on-the-fly (Quantized Matmul)

GGUF weights are stored in compressed formats (4-8 bits per element). Rather than dequantizing the entire weight matrix to FP32 (doubling VRAM), we dequantize each tile into shared memory during the matmul:

1. Load A tile (FP32) from global to shared
2. Load quantized B block from global, dequantize to FP32 in shared
3. Standard FP32 dot product from shared memory

**TILE_K=32 alignment**: Q8_0, Q4_0, and Q5_0 all use 32-element blocks. TILE_K=32 means each tile iteration processes exactly one quantization block per B row — no cross-block indexing needed.

**B-transposed layout**: GGUF stores weight matrices as [N, K] (output features x input features). Our quantized kernels read B in this native layout, avoiding a host-side transpose.

### One-Block-Per-Row Reductions (RMSNorm / Softmax)

For row-wise operations (normalization, softmax), each CUDA block processes one row:

1. **Accumulate**: Each thread processes `ceil(row_len / 256)` elements via stride loop
2. **Reduce**: Tree-based parallel reduction in shared memory (log2(256) = 8 steps)
3. **Broadcast**: Thread 0 computes the final scalar (norm factor / softmax denominator), writes to shared memory
4. **Apply**: All threads read the broadcast scalar and write normalized output

This pattern avoids inter-block synchronization entirely — each block is fully independent.

### Online Softmax in Fused Decode Attention

Standard softmax requires two passes (find max, compute exp/sum) before normalization. The fused decode attention kernel uses the Milakov & Gimelshein (2018) online algorithm to combine these into a single pass over KV-cache entries:

```
for each key k:
    score = dot(Q, K[k]) * scale
    new_max = max(old_max, score)
    correction = exp(old_max - new_max)    // rescale previous accumulator
    exp_score  = exp(score - new_max)      // current contribution
    sum_exp = sum_exp * correction + exp_score
    accumulator = accumulator * correction + exp_score * V[k]
output = accumulator / sum_exp
```

This processes each KV entry exactly once (O(kv_len) instead of O(2 * kv_len)), and critically avoids materializing the full attention score matrix — saving O(seq_len * kv_len) memory.

### GQA Broadcast (No K/V Replication)

Grouped-Query Attention shares K/V heads across multiple Q heads. Rather than replicating K/V tensors (wasting VRAM), the kernel computes which KV head to use:

```cuda
const int kv_h = h / n_groups;  // e.g., for 32 Q heads and 8 KV heads, n_groups=4
```

Each Q head reads from the correct KV head via this index — zero extra memory, zero extra copies.

### CUB vs Hand-Written Reductions

Both variants exist side-by-side for RMSNorm, Softmax, and fused RMSNorm+Residual:

- **Hand-written**: Tree reduction with `__syncthreads()` at each level, shared memory array
- **CUB**: `cub::BlockReduce<float, 256>::Sum()` using warp shuffles for the final 32 elements

CUB avoids shared memory bank conflicts in the final warp and uses register-level communication (`__shfl_down_sync`). Our benchmarks show CUB variants are 4–33% faster: RMSNorm sees 4–17% improvement, Softmax sees 12–33%.

---

## 1. Matrix Multiplication (3 kernels)

### kernel_matmul_naive

**Purpose**: Baseline — demonstrates the performance cost of unoptimized global memory access.

| Property | Value |
|---|---|
| Block | 16x16 = 256 threads |
| Grid | ceil(N/16) x ceil(M/16) |
| Shared memory | 0 bytes |
| Registers/thread | ~16 |
| Arithmetic intensity | ~0.25 FLOP/byte |

**Algorithm**: Each thread computes one C[row][col] by walking the full K-dimension dot product in global memory. Column accesses on B are strided (stride=N), causing poor coalescing.

**Why include it**: The performance gap vs tiled (10-50x) makes a compelling visual argument for shared memory optimization.

### kernel_matmul_tiled

**Purpose**: Shared memory tiling — the core optimization taught in every CUDA course, implemented cleanly.

| Property | Value |
|---|---|
| Block | 32x32 = 1024 threads |
| Grid | ceil(N/32) x ceil(M/32) |
| Shared memory | 8,192 bytes (2 x 32x32 float tiles) |
| Tile size | TILE_SIZE = 32 |

**Algorithm**: The K dimension is swept in 32-wide strips. Each iteration, all 1024 threads cooperatively load a 32x32 tile of A and B into shared memory, then compute partial dot products. The inner loop is `#pragma unroll`ed since TILE_SIZE is a compile-time constant.

**Boundary handling**: Threads outside the matrix bounds load 0.0f (neutral for addition).

### kernel_matmul_vectorized

**Purpose**: Demonstrates vectorized memory access (float4) and micro-tiling for higher arithmetic intensity.

| Property | Value |
|---|---|
| Block | 8x32 = 256 threads |
| Grid | ceil(N/32) x ceil(M/32) |
| Shared memory | 2,048 bytes (32x8 + 8x32 float tiles) |
| Tile dimensions | TILE_M=32, TILE_N=32, TILE_K=8 |
| Outputs/thread | 4 (1x4 micro-tile) |

**Key optimizations**:
- **float4 stores**: When 4 consecutive output columns are within bounds and 16-byte aligned, a single 128-bit write replaces 4 scalar writes.
- **Cooperative tile loading**: 256 threads load 256-element tiles (one element each), with thread linear ID remapped to 2D tile coordinates.
- **Higher arithmetic intensity**: Each thread computes 4 FMA operations per shared memory read of tile_a, compared to 1 in the tiled kernel.

**Tile_b loading**: The thread ID is linearized (`tid = ty * 8 + tx`) then decomposed into tile_b coordinates (`row = tid / 32, col = tid % 32`), ensuring coalesced global reads.

---

## 2. Quantized Matrix Multiplication (6 kernels)

All quantized matmul kernels share the same structure: 32x32 thread blocks, 8 KB shared memory (tile_a[32][32] + tile_b[32][32] in FP32), and TILE_K=32. They differ only in the dequantization logic applied during the B tile load.

**Common launch config**: Block 32x32 = 1024 threads, Grid ceil(N/32) x ceil(M/32).

### kernel_matmul_q8_0

**Format**: 34 bytes per 32 elements — 1 FP16 scale + 32 int8 values.

**Dequant**: `value = scale * (float)qs[i]`

One Q8_0 block maps exactly to one tile-K iteration (32 elements). The thread at position (ty, tx) dequantizes element tx of B row `blockIdx.x * 32 + ty`.

### kernel_matmul_q4_0

**Format**: 18 bytes per 32 elements — 1 FP16 scale + 16 packed nibble bytes.

**Nibble extraction**:
```cuda
byte_idx = tx / 2;
if (tx < 16) val = qs[byte_idx] & 0x0F;       // lower nibble
else         val = (qs[byte_idx] >> 4) & 0x0F; // upper nibble
value = scale * (val - 8);  // zero-point = 8
```

### kernel_matmul_q5_0

**Format**: 22 bytes per 32 elements — 1 FP16 scale + 4 bytes qh (high bits) + 16 bytes qs (low nibbles).

**Bit assembly**: Each element is 5 bits — 4 from qs nibble + 1 from the qh bitmask. The qh bytes are loaded byte-by-byte into a uint32_t to avoid misaligned access, then the high bit is extracted per element: `high_bit = (qh_u32 >> tx) & 1`.

### kernel_matmul_q4_k

**Format**: 144 bytes per 256-element super-block — FP16 d/dmin + 12 bytes packed scales + 128 bytes packed nibbles.

**Tile strategy**: 8 tile iterations per super-block (8 x 32-element sub-blocks). Each iteration computes which sub-block `j` (0-7) it's processing and decodes the corresponding scale/min pair.

**Scale decoding** (complex — follows GGML exactly):
- Sub-blocks 0-3: `sc = scales[j] & 0x3F`, `mn = scales[j+4] & 0x3F`
- Sub-blocks 4-7: upper bits of scales packed differently, requiring 6-bit reconstruction from two separate byte positions

**Nibble layout**: GGML half-interleaved — the first 64 bytes encode nibbles 0-31 and 128-159, the second 64 bytes encode nibbles 32-63 and 160-191.

### kernel_matmul_q6_k

**Format**: 210 bytes per 256-element super-block — 128 bytes ql (lower 4 bits) + 64 bytes qh (upper 2 bits) + 16 bytes scales + 2 bytes d.

**Layout**: GGML interleaved — 2 chunks of 128 elements, each split into 4 quarters of 32 elements. The ql offset, qh offset, and scale index all depend on chunk, quarter, and element position.

**Critical detail**: All 16 scale entries are used. Elements within a quarter are split into two 16-element halves, each using a different scale: `scale_idx = chunk*8 + quarter*2 + l/16`. Missing the `+ l/16` term (as we discovered debugging) causes half the values to use the wrong scale.

### kernel_matmul_fp16

**Format**: Native `__half` in [N, K] layout (GGUF FP16).

**Dequant**: Trivial — `__half2float()` per element. Accumulation in FP32 for precision.

Included for completeness and as a baseline for FP16 performance comparison.

---

## 3. Standalone Dequantization (5 kernels)

Used for norm weights, embedding tables, and verification. Each kernel converts one quantization format to FP32.

**Common launch config**: 256 threads/block, grid = ceil(n_elements / 256). One thread per output element.

| Kernel | Format | Bytes/block | Elements/block |
|---|---|---|---|
| kernel_dequantize_q8_0 | Q8_0 | 34 | 32 |
| kernel_dequantize_q4_0 | Q4_0 | 18 | 32 |
| kernel_dequantize_q5_0 | Q5_0 | 22 | 32 |
| kernel_dequantize_q6_k | Q6_K | 210 | 256 |
| kernel_dequantize_q4_k | Q4_K | 144 | 256 |

Each kernel uses the same dequantization math as its matmul counterpart, but writes to a contiguous FP32 output array instead of shared memory tiles.

---

## 4. Normalization & Activation (11 kernels)

### kernel_rmsnorm (hand-written)

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | rows (one block per row) |
| Shared memory | 1,024 bytes (256 floats, dynamic) |
| Syncs | 2 + log2(256) = 10 total |

Two-pass algorithm: (1) parallel sum-of-squares reduction, (2) normalize with `rsqrt(mean_sq + eps)`.

### kernel_rmsnorm_cub (CUB-accelerated)

Same algorithm, but the tree reduction is replaced by `cub::BlockReduce<float, 256>::Sum()`. CUB uses warp shuffles for the final 32 elements, avoiding shared memory bank conflicts.

### kernel_softmax (hand-written)

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | rows |
| Shared memory | 1,024 bytes (dynamic) |

Three-pass algorithm: (1) find row max, (2) compute `sum(exp(x - max))`, (3) normalize. The max subtraction prevents FP32 overflow (log-sum-exp trick).

### kernel_softmax_cub (CUB-accelerated)

Same three-pass algorithm with CUB for both max and sum reductions. A `__syncthreads()` separates the two CUB reductions to reuse the TempStorage union.

### kernel_silu

**Formula**: `silu(x) = x / (1 + exp(-x))` (Swish activation, used in LLaMA FFN gate).

Element-wise, 256 threads/block, grid-stride loop. Zero shared memory.

### kernel_gelu

**Formula**: `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

Element-wise, same launch config as SiLU. Used by some model architectures.

### kernel_rope

**Purpose**: Rotary Position Embeddings — applies 2D rotation to Q/K pairs.

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | ceil(n_heads * dim/2 / 256) |

Each thread processes one dimension pair (2i, 2i+1):
```
freq = 1 / (theta_base ^ (2*i / dim))
angle = pos * freq
q[2i]   = q[2i]*cos(angle) - q[2i+1]*sin(angle)
q[2i+1] = q[2i]*sin(angle) + q[2i+1]*cos(angle)
```

Modifies Q and K vectors in-place.

### kernel_add / kernel_multiply

Element-wise vector operations. 256 threads/block, flat indexing. Used for residual connections and gating.

### kernel_rmsnorm_residual (hand-written)

**Fused kernel**: Combines residual addition + RMSNorm into a single kernel launch.

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | rows |
| Shared memory | 1,024 bytes (dynamic) |

**Algorithm**:
1. Add residual: `combined[i] = input[i] + residual[i]`, write to output, accumulate sum-of-squares
2. Reduce sum-of-squares (tree reduction)
3. Normalize: `normed[i] = combined[i] * weight[i] * rsqrt(mean_sq + eps)`

**Benefit**: One global memory round-trip instead of two (separate add + rmsnorm would read/write the intermediate buffer twice). Output may alias input for in-place residual update.

### kernel_rmsnorm_residual_cub

Same fused algorithm with CUB BlockReduce for the sum-of-squares reduction.

---

## 5. Fused Kernels (2 main kernels, 2 CUB variants)

### kernel_fused_decode_attention

The most complex kernel in CuInfer. Computes the entire attention mechanism for decode (single-token generation) in one kernel launch.

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | n_heads (one block per attention head) |
| Shared memory | Dynamic: Q[hd] + accumulator[hd] + 4 control floats + CUB TempStorage |

**Algorithm** (online softmax):
1. Load Q vector for this head into shared memory
2. For each KV-cache position k = 0..kv_len-1:
   a. Compute dot(Q, K[k]) via parallel reduction (CUB BlockReduce)
   b. Thread 0: online softmax update (new_max, correction, exp_score)
   c. All threads: `acc[d] = acc[d] * correction + exp_score * V[k][d]`
3. Normalize: `output[d] = acc[d] / sum_exp`

**GQA support**: `kv_h = h / n_groups` maps each Q head to its corresponding KV head. K and V are read from the KV-cache using this index.

**Memory savings**: No materialized attention score matrix. For kv_len=1024 and 32 heads, this saves 32 * 1024 * 4 = 128 KB of temporary storage per layer.

### kernel_rmsnorm_residual / kernel_rmsnorm_residual_cub

Described in Section 4 above. Listed here as fused kernels since they combine two operations.

---

## 6. Attention Infrastructure (7 kernels)

These kernels support the attention mechanism with utility operations.

### kernel_causal_mask

Applies autoregressive mask to attention scores in-place. Sets `scores[q][k] = -INFINITY` where k > start_pos + q.

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | ceil(total_elements / 256) |

### kernel_scale

Scales attention scores by `1/sqrt(head_dim)`. Element-wise, 256 threads/block.

### kernel_batched_matmul_naive

Batched matmul for per-head attention: one block per (head, output_row, output_col_tile). Used for scores = Q * K^T and output = attn_weights * V.

| Property | Value |
|---|---|
| Block | 256 threads (1D) |
| Grid | n_heads x M x ceil(N/256) |

Each thread computes one output element via a K-length dot product. Suitable for small per-head matrices where launching tiled kernels would waste resources.

### kernel_batched_matmul_gqa

Same as batched_matmul_naive but with GQA head broadcast: `kv_h = h / n_groups`. Q heads share K/V heads without data replication.

### kernel_transpose_seq_heads

Transposes [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim]. Used to rearrange Q/K/V after projection for per-head attention computation.

### kernel_transpose_heads_seq

Inverse of transpose_seq_heads: [n_heads, seq_len, head_dim] to [seq_len, n_heads, head_dim]. Used after attention to merge heads back.

### kernel_transpose_k

Transposes K for attention score computation: rearranges K from [seq, heads, dim] layout to [heads, dim, seq] layout so that Q * K^T can be computed as a standard matmul.

---

## 7. Embedding & Sampling (3 kernels)

### kernel_embedding_lookup

**Purpose**: Maps discrete token IDs to dense embedding vectors.

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | ceil(seq_len * dim / 256) |

Each thread copies one element: `output[pos * dim + d] = weight[token_id * dim + d]`.

Memory pattern: consecutive threads access consecutive embedding dimensions (coalesced reads from the weight table).

### kernel_matmul_transposed_b

**Purpose**: LM head projection — `logits[i,j] = sum_k hidden[i,k] * weight[j,k]`.

| Property | Value |
|---|---|
| Block | 32x32 = 1024 threads |
| Grid | ceil(N/32) x ceil(M/32) |
| Shared memory | 8,192 bytes |

Tiled matmul with B accessed as B[j,k] instead of B[k,j]. The weight matrix is [vocab_size, dim] and the dot product walks along the K dimension within each row of B.

### kernel_argmax

**Purpose**: Greedy decoding — finds the index of the maximum value in logits.

| Property | Value |
|---|---|
| Block | 256 threads |
| Grid | 1 (single row) |
| Shared memory | 256 * (sizeof(float) + sizeof(int)) = 2,048 bytes |

Parallel reduction tracking both max value and its index. Each thread finds its local maximum via stride loop, then tree reduction finds the global maximum.

---

## Appendix A: Lessons Learned

### Q6_K Scale Indexing Bug

GGML splits each 32-element group into two 16-element subgroups, each with a **different** scale. The scale index is `chunk*8 + quarter*2 + l/16`. We initially omitted the `+ l/16` term, causing half of every Q6_K block to use the wrong scale. The bug was invisible in synthetic round-trip tests because both the test's quantize function and the kernel's dequant had the same error.

**Lesson**: Always verify quantized kernels against reference GGUF data from real models, not just synthetic round-trips.

### Synthetic Tests Masking Real Bugs

If a test's `quantize()` function and the kernel's `dequant()` share the same bug, the round-trip test `dequant(quantize(x)) == x` will pass perfectly. We discovered this when Q6_K tests passed but real model inference produced garbage.

**Lesson**: Use a second, independent reference implementation (GGML's `ggml-quants.c`) to validate at least one direction of the pipeline.

### GGUF Dimension Convention

GGUF stores tensor dimensions as `[inner, outer]` = `[K, N]`, which is the **opposite** of typical math notation `[N, K]`. Our quantized matmul kernels use B in its native GGUF [N, K] layout (B-transposed), avoiding a host-side transpose.

### 4D Shape Padding Pitfall

CuInfer pads shapes to 4D: a 2D tensor `{K, N}` becomes `{1, 1, K, N}`. Using `shape()[0] * shape()[1]` to compute total size gives `1 * 1 = 1` instead of the actual element count. Always use the dedicated `numel()` method.

---

## Appendix B: Kernel Summary Table

| # | Kernel | File | Block | Shmem | Category |
|---|---|---|---|---|---|
| 1 | kernel_matmul_naive | matmul.cu | 16x16 | 0 | MatMul |
| 2 | kernel_matmul_tiled | matmul.cu | 32x32 | 8 KB | MatMul |
| 3 | kernel_matmul_vectorized | matmul.cu | 8x32 | 2 KB | MatMul |
| 4 | kernel_matmul_q8_0 | matmul_quant.cu | 32x32 | 8 KB | Quant MatMul |
| 5 | kernel_matmul_q4_0 | matmul_quant.cu | 32x32 | 8 KB | Quant MatMul |
| 6 | kernel_matmul_q5_0 | matmul_quant.cu | 32x32 | 8 KB | Quant MatMul |
| 7 | kernel_matmul_q4_k | matmul_quant.cu | 32x32 | 8 KB | Quant MatMul |
| 8 | kernel_matmul_q6_k | matmul_quant.cu | 32x32 | 8 KB | Quant MatMul |
| 9 | kernel_matmul_fp16 | matmul_quant.cu | 32x32 | 8 KB | Quant MatMul |
| 10 | kernel_dequantize_q8_0 | dequantize.cu | 256 | 0 | Dequant |
| 11 | kernel_dequantize_q4_0 | dequantize.cu | 256 | 0 | Dequant |
| 12 | kernel_dequantize_q5_0 | dequantize.cu | 256 | 0 | Dequant |
| 13 | kernel_dequantize_q6_k | dequantize.cu | 256 | 0 | Dequant |
| 14 | kernel_dequantize_q4_k | dequantize.cu | 256 | 0 | Dequant |
| 15 | kernel_rmsnorm | elementwise.cu | 256 | 1 KB | Norm |
| 16 | kernel_rmsnorm_cub | elementwise.cu | 256 | 1 KB + CUB | Norm |
| 17 | kernel_softmax | elementwise.cu | 256 | 1 KB | Norm |
| 18 | kernel_softmax_cub | elementwise.cu | 256 | 1 KB + CUB | Norm |
| 19 | kernel_silu | elementwise.cu | 256 | 0 | Activation |
| 20 | kernel_gelu | elementwise.cu | 256 | 0 | Activation |
| 21 | kernel_rope | elementwise.cu | 256 | 0 | Position |
| 22 | kernel_add | elementwise.cu | 256 | 0 | Elementwise |
| 23 | kernel_multiply | elementwise.cu | 256 | 0 | Elementwise |
| 24 | kernel_rmsnorm_residual | elementwise.cu | 256 | 1 KB | Fused |
| 25 | kernel_rmsnorm_residual_cub | elementwise.cu | 256 | 1 KB + CUB | Fused |
| 26 | kernel_fused_decode_attention | attention.cu | 256 | Dynamic | Fused |
| 27 | kernel_causal_mask | attention.cu | 256 | 0 | Attention |
| 28 | kernel_scale | attention.cu | 256 | 0 | Attention |
| 29 | kernel_batched_matmul_naive | attention.cu | 256 | 0 | Attention |
| 30 | kernel_batched_matmul_gqa | attention.cu | 256 | 0 | Attention |
| 31 | kernel_transpose_seq_heads | attention.cu | 256 | 0 | Attention |
| 32 | kernel_transpose_heads_seq | attention.cu | 256 | 0 | Attention |
| 33 | kernel_transpose_k | attention.cu | 256 | 0 | Attention |
| 34 | kernel_embedding_lookup | embedding.cu | 256 | 0 | Embed |
| 35 | kernel_matmul_transposed_b | embedding.cu | 32x32 | 8 KB | Embed |
| 36 | kernel_argmax | embedding.cu | 256 | 2 KB | Sampling |
