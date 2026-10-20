# CuInfer Architecture

This document describes the system design, component interactions, memory layout, and kernel
internals of the CuInfer inference engine. It is intended for readers with a working knowledge
of transformer architectures and CUDA programming.

**Hardware target**: NVIDIA RTX 3060 (6 GB VRAM, SM 8.6, Ampere)
**Software stack**: C++17, CUDA 12.8, no external ML frameworks
**Build system**: CMake 3.24+, targeting SM 8.6 primary / SM 7.5 secondary

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow Diagram](#2-data-flow-diagram)
3. [Component Architecture](#3-component-architecture)
4. [Memory Management](#4-memory-management)
5. [Kernel Architecture](#5-kernel-architecture)
6. [Quantization Pipeline](#6-quantization-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Design Decisions](#8-design-decisions)

---

## 1. System Overview

CuInfer loads a transformer model from a GGUF file, runs autoregressive text generation on the
GPU, and returns decoded text to the caller. The entire inference path from token IDs to logits
executes on GPU with no host involvement between forward-pass steps.

The system has three layers:

- **Model I/O layer**: parses the GGUF file and tokenizer, loads weights to GPU
- **Forward pass layer**: orchestrates the transformer computation per token
- **Kernel layer**: executes individual CUDA kernels for matmul, normalization, attention, and
  activation functions

All components reside in the `cuinfer` namespace. The single public entry point for text
generation is `TransformerModel::generate()`, declared in
`include/cuinfer/model.h`.

### Verified Models

| Model | Format | VRAM | Decode throughput |
|---|---|---|---|
| SmolLM2-135M | FP16 | 1.46 GB | 28.4 tok/s |
| SmolLM2-135M | Q8_0 | 1.35 GB | 27.8 tok/s |
| SmolLM2-135M | Q4_K_M | 1.31 GB | 27.3 tok/s |
| Llama 3.2 1B | Q8_0 | 3.42 GB | 6.8 tok/s |
| Llama 3.2 1B | Q4_K_M | 2.97 GB | 6.6 tok/s |

Measurements on RTX 3060 (SM 8.6, 6 GB GDDR6). Timing uses CUDA events over 100 decode steps.

---

## 2. Data Flow Diagram

```
GGUF file (disk)
      |
      | mmap (zero-copy)
      v
+---------------------+
| GGUFParser          |  Parses header, metadata KV pairs, tensor
| (gguf_parser.cpp)   |  descriptors. Returns raw pointers into the
|                     |  memory-mapped region.
+---------------------+
      |           |
      | metadata  | tensor data pointers
      v           v
+----------+  +------------------+
|Tokenizer |  | Weight Loading   |  FP32/FP16: host transpose -> GPU
|(tokenizer|  | (model.cu)       |  Quantized: raw bytes -> GPU (no transpose)
| .cpp)    |  |                  |  1D norms: GPU (always FP32)
+----------+  +------------------+
      |                |
      | token IDs      | GPU weight tensors
      |                v
      |      +-----------------------+
      |      |  GPU Memory           |
      |      |  - weight tensors     | Permanent, read-only
      |      |  - KV-cache buffers   | Permanent, written per token
      |      |  - activation buffers | Reused per layer per call
      |      +-----------+-----------+
      |                  |
      v                  v
+----------------------------------------+
|          TransformerModel::forward()   |
|          (model.cu)                    |
|                                        |
|  token_ids -> Embedding lookup         |
|  for layer in 0..n_layers:             |
|    RMSNorm -> QKV matmul -> RoPE       |
|    KV-cache append                     |
|    Attention (fused or multi-kernel)   |
|    matmul_dispatch(W_o)                |
|    fused rmsnorm_residual              |
|    SwiGLU FFN (3 matmuls + SiLU)       |
|    fused rmsnorm_residual              |
|  LM head matmul                        |
+------------------+---------------------+
                   |
                   | logits [seq_len, vocab_size]  (GPU tensor)
                   v
+----------------------------------------+
|  Sampling (embedding.h)                |
|  temperature == 0: argmax_sample()     |
|  temperature > 0:  top_k_sample()      |
+------------------+---------------------+
                   |
                   | next token ID (int32_t)
                   v
+----------------------------------------+
|  Tokenizer::decode_token()             |
|  String output (streamed or buffered)  |
+----------------------------------------+
```

---

## 3. Component Architecture

### 3.1 GGUFParser (`include/cuinfer/gguf_parser.h`, `src/gguf_parser.cpp`)

Parses the GGUF binary format and provides zero-copy access to model weights.

**Responsibilities**:
- Memory-maps the GGUF file (Windows: `CreateFileMappingA`/`MapViewOfFile`; POSIX:
  `open`/`mmap`). Both paths are wrapped in a RAII struct inside `GGUFParser::Impl`.
- Parses the header (24 bytes: magic, version, tensor count, metadata count) and the
  variable-length metadata KV section into an `std::unordered_map<std::string, GGUFValue>`.
- Reads each `GGUFTensorInfo` entry (name, shape, type, byte offset within the data section).
- Provides `tensor_data()`, which returns a raw pointer directly into the mapped file at the
  tensor's computed absolute offset. No copies are made.
- Provides convenience accessors (`architecture()`, `head_count()`, `block_count()`, etc.) that
  read well-known GGUF metadata keys using the architecture prefix (e.g., `"llama"`).

**Key detail — GGUF dimension convention**: `GGUFTensorInfo::dimensions[0]` is the
innermost (K) dimension and `dimensions[1]` is the outermost (N) dimension. This is the
opposite of the conventional row-major `[rows, cols]` interpretation and must be accounted
for during weight loading.

**GGUF file layout** (parsed sequentially):
```
[Header]          4B magic + 4B version + 8B tensor_count + 8B metadata_count
[Metadata KVs]    variable-length typed key-value entries
[Tensor Info]     name + shape + type + data-section offset, one entry per tensor
[Padding]         aligned to 32-byte boundary
[Tensor Data]     raw weight bytes, each at its declared offset
```

### 3.2 Tokenizer (`include/cuinfer/tokenizer.h`, `src/tokenizer.cpp`)

CPU-only BPE tokenizer. Tokenization runs once per prompt and is not on the inference
critical path.

**Supported tokenizer styles**:

- **SentencePiece** (Llama, SmolLM2): Merges are determined by per-token scores stored in the
  `tokenizer.ggml.scores` GGUF array. Higher score = higher merge priority. Unknown characters
  fall back to `<0xNN>` byte tokens.
- **GPT2-style** (some models): Merges are an explicit ordered list from
  `tokenizer.ggml.merges`. Lower list index = higher merge priority. Byte-level BPE maps
  each byte to a Unicode codepoint and reverses the mapping at decode time.

Both styles use the same greedy merge loop: find the adjacent pair with the highest priority,
merge it, repeat until no mergeable pairs remain.

**Storage**:
- `vocab_`: `id -> string` (direct lookup by token ID)
- `token_to_id_`: `string -> id` (hash map for merge lookups)
- `scores_`: merge priority per token (SentencePiece mode)
- `merge_ranks_`: merged-string to rank (GPT2 mode)
- `byte_tokens_[256]`: precomputed byte-to-token-ID table

### 3.3 Tensor (`include/cuinfer/tensor.h`, `src/tensor.cu`)

RAII wrapper for a GPU allocation. All heap-allocated GPU memory in the system flows through
this class.

**Design**:
- Stores a `void* data_` pointer, a 4D shape (`std::array<int64_t, 4>`, right-aligned with
  leading 1s for shorter shapes), and row-major strides.
- The `owning_` flag distinguishes owned allocations (freed in destructor) from non-owning
  views created by `view()` and `reshape()`. Views share the underlying pointer without copying.
- Copy constructor is deleted to prevent accidental GPU allocations. Explicit deep copies
  require `clone()`.
- Quantized types (`Q8_0`, `Q4_0`, etc.) are tracked via the `DType` enum. `size_bytes()`
  uses `dtype_block_bytes()` and `dtype_block_elements()` to compute the correct storage
  size for block-quantized formats.

**Factory methods**:
- `Tensor(shape, dtype)` — allocates uninitialized GPU memory via `cudaMalloc`
- `Tensor::from_host(ptr, shape, dtype)` — allocates and copies from host
- `Tensor::zeros(shape, dtype)` — allocates and zeroes via `cudaMemset`
- `view(dim, start, end)` — returns a non-owning slice along one dimension
- `reshape(new_shape)` — returns a non-owning view with a different shape (no data movement)

**Error handling**: every CUDA call inside `Tensor` is wrapped in `CUINFER_CHECK_CUDA`, which
throws `std::runtime_error` on failure.

### 3.4 KVCache (`include/cuinfer/kv_cache.h`, `src/kv_cache.cu`)

Pre-allocated GPU buffer for key and value projections across all transformer layers and
sequence positions.

**Memory layout**:
```
k_data_: float[n_layers][max_seq_len][n_kv_heads][head_dim]   (single contiguous allocation)
v_data_: float[n_layers][max_seq_len][n_kv_heads][head_dim]   (single contiguous allocation)
```

Total bytes: `2 * n_layers * max_seq_len * n_kv_heads * head_dim * sizeof(float)`.

For Llama 3.2 1B at max_seq_len=2048: 2 × 16 × 2048 × 8 × 64 × 4 = 128 MB.

**Append protocol**: `append(layer, k_new, v_new, n_new_tokens)` copies new K/V data into
the cache at the current write position. The position counter `current_length_` advances by
`n_new_tokens` only after the last layer (`layer == n_layers - 1`) is appended. This matches
the layer-by-layer forward pass where all layers process the same position set before moving
to the next token.

**GQA support**: The cache stores only `n_kv_heads` entries per position (not `n_heads`).
During attention, the mapping `kv_h = h / (n_heads / n_kv_heads)` selects the correct KV
head without physical replication. The fused decode attention kernel implements this mapping
directly.

**Reset**: `clear()` resets `current_length_` to zero without zeroing or freeing GPU memory.
Subsequent appends overwrite old data.

### 3.5 AttentionBlock (`include/cuinfer/attention.h`, `src/attention.cu`)

Implements multi-head and grouped-query attention for both prefill (seq_len > 1) and decode
(seq_len = 1).

**Two forward paths**:

- `forward(input, w_q, w_k, w_v, w_o, kv_cache, layer_idx, pos)`: Used for FP32 weights.
  Projects input via `matmul_tiled`, applies RoPE, appends to KV-cache, computes attention
  using the multi-kernel path, and projects output via `matmul_tiled`.

- `forward_from_qkv(q_proj, k_proj, v_proj, w_o, w_o_format, kv_cache, layer_idx, pos)`:
  Used for prefill in the quantized path, after QKV projections have already been computed
  by `matmul_dispatch`. Handles KV-cache update and the attention computation, then applies
  the output projection.

**Multi-kernel attention path** (used for prefill):
1. `batched_matmul_gqa(Q, K^T, scores)` — `[n_heads, seq_len, kv_len]` scores
2. `scale(scores, factor)` — multiply by `1/sqrt(head_dim)`
3. `causal_mask(scores)` — set future positions to -infinity
4. `softmax(scores)` — row-wise softmax per head
5. `batched_matmul_gqa(scores, V, output)` — weighted sum of values

**Fused decode path** (seq_len = 1, quantized forward): replaced by a single call to
`fused_decode_attention()`, which eliminates all five steps above. See Section 5.4.

### 3.6 GatedFFN (`include/cuinfer/ffn.h`, `src/ffn.cu`)

Implements the SwiGLU feed-forward network used by Llama-family models:

```
hidden = SiLU(input * gate_proj) * (input * up_proj)
output = hidden * down_proj
```

Pre-allocates three intermediate buffers at construction time:
- `gate_buf_` [max_seq_len, intermediate_size]
- `up_buf_` [max_seq_len, intermediate_size]
- `hidden_buf_` [max_seq_len, intermediate_size]

The quantized forward path accesses these buffers directly via `gate_buf()`, `up_buf()`, and
`hidden_buf()` accessor methods, avoiding re-allocation. The model's forward loop calls
`matmul_dispatch` for each projection, then applies `silu` and `multiply` element-wise before
dispatching the down-projection.

A `StandardFFN` class (GPT2-style GeLU FFN) is also provided but is not used by current
models.

### 3.7 Embedding and LMHead (`include/cuinfer/embedding.h`, `src/embedding.cu`)

**Embedding**: `embedding_lookup` launches `kernel_embedding_lookup`, where each thread copies
one float from the weight table row `weight[token_id]` to the output. Grid: `ceil(seq_len *
dim / 256)` blocks of 256 threads. The embedding weight (`token_embd_weight_`) is always
stored in FP32 regardless of the model's quantization format, because the row-lookup access
pattern is not compatible with block-quantized storage.

**LMHead**: Projects hidden states to vocabulary logits via `matmul_transposed_b`, which
computes `C = A * B^T` where B is the `[vocab_size, dim]` embedding table. When
`tie_embeddings` is true (common for Llama-family models), the same `token_embd_weight_`
tensor serves as both the embedding table and the LM head weight.

**Sampling**:
- `argmax_sample`: copies logits to host, finds the maximum index via `std::max_element`.
  Used when `temperature == 0`.
- `top_k_sample`: copies logits to host, performs a partial sort to find the top-k
  candidates, applies temperature scaling and softmax over those candidates, then samples
  using a discrete distribution seeded by the `seed` parameter.

### 3.8 TransformerModel (`include/cuinfer/model.h`, `src/model.cu`)

The top-level class. Owns all GPU memory and orchestrates the forward pass.

**Construction sequence**:
1. Open and parse the GGUF file via `GGUFParser`.
2. Read `ModelConfig` from GGUF metadata (architecture, dimensions, head counts, RoPE theta).
3. Detect `WeightFormat` by inspecting the GGUF type of the first layer's attention Q weight.
4. Build the `Tokenizer` from `tokenizer.ggml.*` metadata arrays.
5. Construct sub-components (`AttentionBlock`, `GatedFFN`, `KVCache`, `Embedding`, `LMHead`).
6. Load weights: embedding and norm weights are always loaded as FP32; projection weights are
   loaded in their native quantized format (raw bytes) or transposed FP32/FP16.
7. Pre-allocate activation buffers (`normed_buf_`, `ffn_output_buf_`, `token_ids_gpu_`).

**Weight layout decision**: GGUF stores 2D weights as `[out_features, in_features]`. The FP32
matmul kernel (`matmul_tiled`) expects B as `[K, N] = [in_features, out_features]`. Therefore,
FP32 and FP16 projection weights are transposed on the host during loading via `transpose_host`
before upload. Quantized projection weights are uploaded as-is because the quantized matmul
kernels read B in the native `[N, K]` layout.

**Per-tensor format tracking**: GGUF files with the `Q4_K_M` quant descriptor contain tensors
in mixed formats (some Q4_K, some Q6_K, some Q8_0). `TransformerModel` stores a
`WeightFormat` per tensor per layer (e.g., `attn_q_format_[i]`, `ffn_gate_format_[i]`). The
`matmul_dispatch` function selects the correct kernel based on this per-tensor format.

---

## 4. Memory Management

### 4.1 Allocation Strategy

All GPU memory is allocated during `TransformerModel` construction. The generation loop makes
zero `cudaMalloc` calls. This is a deliberate design constraint: dynamic allocation during the
hot loop would add unpredictable latency and fragment the allocator.

| Buffer category | Allocation site | Lifetime | Notes |
|---|---|---|---|
| Model weights | `TransformerModel` ctor | Permanent (freed in dtor) | Read-only after load |
| KV-cache | `KVCache` ctor | Permanent | Written per token, reset via `clear()` |
| Activation buffers | `TransformerModel` ctor | Reused per layer per call | See below |
| FFN intermediates | `GatedFFN` ctor | Reused per layer per call | gate/up/hidden bufs |
| Token IDs on GPU | `TransformerModel` ctor | Reused per call | `int32_t[max_seq_len]` |

**Exception**: `Tensor` allocations for `q_proj`, `k_proj`, `v_proj`, and `attn_concat` in
the quantized prefill and decode paths are allocated per forward call. These are small compared
to weights and KV-cache.

### 4.2 Activation Buffer Reuse

Two globally pre-allocated buffers are reused across all layers:

- `normed_buf_` [max_seq_len, dim]: holds RMSNorm output, overwritten at every layer
- `ffn_output_buf_` [max_seq_len, dim]: holds FFN output, overwritten at every layer

The fused `rmsnorm_residual()` kernel writes both `hidden` (the updated residual stream) and
`normed` (the RMSNorm output for the next sub-block) in one pass. This means there is no
separate "after residual add" buffer — the hidden tensor updates in-place while the normed
output goes directly to `normed_buf_`.

### 4.3 VRAM Budget (SmolLM2-135M vs Llama 3.2 1B)

At max_seq_len = 2048, FP32 activations:

| Component | SmolLM2-135M | Llama 3.2 1B |
|---|---|---|
| Weights (Q4_K_M) | ~178 MB | ~1,559 MB |
| Weights (Q8_0) | ~311 MB | ~1,988 MB |
| KV-Cache (FP32) | 90 MB | 128 MB |
| Activation buffers | ~4 MB | ~16 MB |
| FFN intermediates | ~12 MB | ~48 MB |

The Llama 3.2 1B Q8_0 configuration stays within the 5.5 GB budget on a 6 GB GPU.

### 4.4 Tensor Ownership and Views

`Tensor` uses an `owning_` flag to distinguish ownership:

- **Owning tensors**: allocated by `Tensor(shape, dtype)`, `from_host()`, or `zeros()`.
  Destructor calls `cudaFree(data_)`.
- **Views**: created by `view()` or `reshape()`. Destructor is a no-op. Views share the
  same `data_` pointer as the source tensor, offset by the appropriate byte count.

Callers must ensure that any owning tensor outlives all views into it. Views are used in the
forward pass to slice `normed_buf_` to the current sequence length (`normed_buf_.view(2, 0,
seq_len)`) without copying.

---

## 5. Kernel Architecture

36 CUDA kernels across 6 categories. All targets SM 8.6 (RTX 3060 Ampere). All kernel
launches are followed by `CUINFER_CHECK_CUDA(cudaGetLastError())`. Source files are in
`src/kernels/`.

### 5.1 Matrix Multiplication (`src/kernels/matmul.cu`)

Three FP32 kernels with increasing optimization levels:

**Naive** (`kernel_matmul_naive`):
- Block: 16×16 = 256 threads. Grid: `ceil(N/16) × ceil(M/16)`.
- One thread per output element. Reads A and B entirely from global memory.
- Arithmetic intensity: ~0.25 FLOP/byte. Baseline for benchmarking only.

**Tiled** (`kernel_matmul_tiled`):
- Block: 32×32 = 1024 threads. Grid: `ceil(N/32) × ceil(M/32)`.
- Shared memory: two 32×32 tiles of float = 8 KB per block.
- Each tile iteration loads one 32-column strip of A and one 32-row strip of B cooperatively,
  then accumulates 32 dot-product steps. Two `__syncthreads()` barriers per iteration.
- Used for FP32 projection matmuls in the FP32 forward path and for the LM head.

**Vectorized** (`kernel_matmul_vectorized`):
- Block: 8×32 = 256 threads. Each thread computes four consecutive C columns.
- Tile: TILE_M=32, TILE_N=32, TILE_K=8. Shared memory: 2 KB per block.
- When N is a multiple of 4 and the output address is 16-byte aligned, uses `float4` stores
  (one 128-bit write per thread). Benchmark reference only; not used in the forward pass.

### 5.2 Quantized Matmul (`src/kernels/matmul_quant.cu`)

Five quantized kernels and one FP16 kernel, all computing C = A × dequant(B)^T where B is in
GGUF-native [N, K] layout.

All use a 32×32 block (1024 threads) and 8 KB shared memory (two 32×32 FP32 tiles). Each
kernel differs only in how it loads and dequantizes the B tile.

**Tile K alignment**: TILE_K = 32 matches the block size of Q8_0, Q4_0, and Q5_0 (32
elements per block). Each tile iteration processes exactly one quantization block per B row,
avoiding cross-block index arithmetic for these formats.

**FP16** (`kernel_matmul_fp16`): reads `__half` values from B, converts to float via
`__half2float()` in shared memory. Accumulates in FP32.

**Q8_0** (`kernel_matmul_q8_0`): each thread reads `blk.qs[threadIdx.x]` (int8) and
`blk.scale` (FP16), dequantizes as `scale * (float)qs[i]`. Block index:
`b_row * blocks_per_row + t`.

**Q4_0** (`kernel_matmul_q4_0`): unpacks one nibble per thread from the 16-byte `qs` array.
Even-indexed threads take the lower nibble, odd-indexed take the upper nibble. Dequantizes as
`scale * (nibble - 8)`.

**Q5_0** (`kernel_matmul_q5_0`): reconstructs the 5th bit from the 4-byte `qh` bit-field.
Builds a `uint32_t` from the four `qh` bytes to avoid misaligned access, then extracts bit
`threadIdx.x`. Dequantizes as `scale * (q5_val - 16)`.

**Q4_K** (`kernel_matmul_q4_k`): 256-element super-blocks require 8 tile iterations per
super-block. For tile iteration `t`: `sb_idx = t / 8`, `sub_block = t % 8`. Decodes 6-bit
sub-block scale and min from the 12-byte packed `scales` field. Dequantizes as:
`d * sc * nibble - dmin * mn`. Uses GGML's half-interleaved nibble layout for the `qs` array.

**Q6_K** (`kernel_matmul_q6_k`): 256-element super-blocks, 210 bytes each. 8 tile iterations
per super-block. Each 6-bit value is reconstructed from 4-bit `ql` (lower nibble) and 2-bit
`qh` (packed 4 per byte). Uses GGML's interleaved chunk layout. Scale index:
`chunk * 8 + quarter * 2 + l / 16`, where all 16 scale entries are used.

### 5.3 Elementwise Kernels (`src/kernels/elementwise.cu`)

All elementwise kernels use 256 threads per block.

**RMSNorm** (`kernel_rmsnorm`): One block per row. Two passes:
1. Each thread accumulates `x[i]^2` via stride loop. Stores partial sum in shared memory.
   Tree reduction (log2(256) = 8 steps) computes `sum_sq`. Thread 0 broadcasts
   `rsqrt(sum_sq / n + eps)` via `sdata[0]`.
2. Each thread writes `input[i] * weight[i] * norm_factor`.

A CUB variant (`kernel_rmsnorm_cub`) replaces the hand-written tree reduction with
`cub::BlockReduce<float, 256>::Sum()`, which uses warp shuffles for the final 32 elements.
Both variants are available for benchmarking comparison.

**Softmax** (`kernel_softmax`): One block per row. Three passes: find max (reduction), compute
`sum(exp(x - max))` (reduction), normalize. Uses the log-sum-exp trick for numerical
stability. CUB variant available.

**Fused RMSNorm + Residual** (`kernel_rmsnorm_residual`): Combines residual addition and
RMSNorm into one kernel pass, writing two outputs simultaneously. During decode (seq_len=1),
this eliminates one kernel launch and one full read/write of the hidden state per layer:
```
Pass 1: val = input[i] + residual[i]; output[i] = val; accumulate val^2
Reduce: sum_sq -> norm_factor = rsqrt(sum_sq/n + eps)
Pass 2: normed[i] = output[i] * weight[i] * norm_factor
```
Output may alias input (in-place residual update) because each element is read before being
written at the same index.

**RoPE** (`kernel_rope`): Each thread rotates one pair of dimensions across all heads.
For dimension pair index `i` in head `h`: `freq = 1 / (theta^(2i/dim))`,
`theta_val = pos * freq`. Applies the 2D rotation in-place. Grid: `ceil(n_heads * dim/2 / 256)`.

**SiLU** (`kernel_silu`): `x / (1 + exp(-x))`, element-wise, no shared memory.

**GeLU** (`kernel_gelu`): tanh approximation `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

**Add / Multiply**: element-wise with grid-stride loop.

### 5.4 Fused Decode Attention (`src/attention.cu`, `fused_decode_attention`)

Single-kernel implementation of the entire attention mechanism for seq_len = 1. Replaces the
multi-kernel path (batched Q×K^T, scale, causal mask, softmax, attn×V, output projection) with
one kernel using online softmax.

**Grid**: `n_heads` blocks (one block per query head).
**Block**: 256 threads.
**Shared memory per block**:
- `Q[head_dim]`: query vector for this head
- `acc[head_dim]`: running weighted-sum accumulator
- 4 control floats: `running_max`, `sum_exp`, `correction`, `exp_score`
- `cub::BlockReduce<float, 256>::TempStorage` for the dot-product reduction

**Algorithm** (online softmax, Milakov & Gimelshein 2018):
```
Load Q[h, :] into shared memory
running_max = -inf, sum_exp = 0, acc[:] = 0

For k = 0 to kv_len - 1:
  kv_h = h / (n_heads / n_kv_heads)          # GQA mapping
  score = dot(Q, K_cache[kv_h, k, :]) * scale  # CUB BlockReduce
  new_max = max(running_max, score)
  correction = exp(running_max - new_max)
  exp_score  = exp(score - new_max)
  sum_exp = sum_exp * correction + exp_score
  acc[d] = acc[d] * correction + exp_score * V_cache[kv_h, k, d]
  running_max = new_max

output[d] = acc[d] / sum_exp
```

Each KV position is read exactly once. The full attention score matrix is never materialized,
saving O(n_heads × kv_len) memory and avoiding the overhead of a separate softmax kernel.

### 5.5 Supporting Attention Kernels (`src/attention.cu`)

Used in the multi-kernel prefill path:

**Causal mask** (`kernel_causal_mask`): For query position `q` at absolute position
`start_pos + q`, sets `scores[h, q, k] = -inf` when `k > start_pos + q`. Flat index
decomposition: `q = (idx / kv_len) % seq_len`, `k = idx % kv_len`.

**Scale** (`kernel_scale`): Multiplies all elements by a constant (typically `1/sqrt(head_dim)`).

**Batched matmul** (`kernel_batched_matmul_naive`): Computes C[b] = A[b] × B[b] for each
batch element. Block: 16×16. Grid: `ceil(N/16) × ceil(M/16) × batch`. Used for attention
score computation and value-weighted sum.

**Batched matmul GQA** (`kernel_batched_matmul_gqa`): Same as batched matmul, but B is
indexed as `B[h / n_groups]` instead of `B[h]`, implementing GQA broadcast without physical
K/V replication.

---

## 6. Quantization Pipeline

### 6.1 Supported Formats

| Format | Bytes/block | Elements/block | Structure |
|---|---|---|---|
| FP32 | 4 | 1 | Native float |
| FP16 | 2 | 1 | Native `__half` |
| Q8_0 | 34 | 32 | 1 FP16 scale + 32 int8 |
| Q4_0 | 18 | 32 | 1 FP16 scale + 16 packed nibbles |
| Q5_0 | 22 | 32 | 1 FP16 scale + 4 high-bit bytes + 16 packed nibbles |
| Q4_K | 144 | 256 | 2 FP16 super-scales + 12 packed 6-bit sub-scales + 128 packed nibbles |
| Q6_K | 210 | 256 | 128 lower-nibble bytes + 64 upper-2-bit bytes + 16 int8 sub-scales + 1 FP16 scale |

Block structures are defined in `include/cuinfer/quantization.h` as packed C structs
(`BlockQ8_0`, `BlockQ4_0`, `BlockQ5_0`, `BlockQ4_K`, `BlockQ6_K`). Static assertions verify
the sizes match the GGML specification.

### 6.2 Weight Loading by Format

During `TransformerModel` construction, the appropriate loading function is called per tensor:

**Embedding and norm weights** (always FP32 on GPU):
- `load_weight_1d()`: reads 1D tensors (norm weights). If the GGUF type is quantized, GPU
  dequant kernels convert to FP32 (dequant on device, copy back to host, upload as FP32).
- `load_weight_2d()`: reads the embedding table. Same dequant-then-upload path.

**FP32/FP16 projection weights**:
- `load_weight_transposed()`: reads the weight, dequantizes FP16 to FP32 on host if needed,
  then calls `transpose_host()` to convert from GGUF [out, in] layout to matmul-ready [in, out].

**Quantized projection weights**:
- `load_weight_quantized()`: reads the raw quantized bytes from the memory-mapped file and
  uploads them to GPU without transposition. Also outputs the per-tensor `WeightFormat`.

**Format detection**: `detect_weight_format()` inspects the GGML type of the first layer's
attention Q weight (`"blk.0.attn_q.weight"`) and maps it to `WeightFormat`.

### 6.3 Dequantize-on-the-Fly Matmul

Quantized weight tensors stay in compressed format on the GPU throughout inference. Each
`matmul_dispatch()` call selects the kernel matching the weight's format. Within the kernel,
dequantization happens per tile per thread:

1. Thread loads its B element from the quantized block in global memory.
2. Thread dequantizes to float in shared memory.
3. All threads accumulate the FP32 dot product from shared memory.

This avoids materializing the full dequantized weight matrix. Peak additional memory for
dequantization is 8 KB per block (one 32×32 FP32 tile in shared memory).

### 6.4 Mixed-Format Models (Q4_K_M)

GGUF files with the `Q4_K_M` descriptor use different quantization types for different
tensors within the same model. A typical assignment:

- Attention Q, K, O weights: Q4_K
- Attention V weights: Q6_K
- FFN gate, up weights: Q4_K
- FFN down weights: Q6_K

`TransformerModel` stores `std::vector<WeightFormat>` for each weight category
(e.g., `attn_q_format_`, `ffn_down_format_`). The `matmul_dispatch` static function reads
the per-tensor format and dispatches accordingly.

### 6.5 Standalone Dequantization Kernels (`src/kernels/dequantize.cu`)

Five standalone kernels (`dequantize_q8_0`, `dequantize_q4_0`, `dequantize_q5_0`,
`dequantize_q4_k`, `dequantize_q6_k`) convert full tensors to FP32. Used only during model
loading for 1D norm weights and the embedding table, not during the inference hot path.

---

## 7. Inference Pipeline

### 7.1 Prefill (seq_len > 1)

Prefill processes the entire prompt in one forward call.

```
TransformerModel::generate(prompt):
  1. reset()                     -- kv_cache_->clear()
  2. tokenizer_->encode(prompt)  -- CPU BPE encoding
  3. forward(token_ids)          -- seq_len = prompt_len
```

Inside `forward(token_ids)` during prefill:

```
Copy token_ids to device (cudaMemcpy, seq_len * 4 bytes)
hidden = embedding_.forward(token_ids_gpu_, seq_len, token_embd_weight_)
  -- kernel_embedding_lookup: each thread copies one float from embedding table

Pre-attention RMSNorm for layer 0:
  rmsnorm(hidden, attn_norm_weight_[0], normed_buf_, seq_len, dim, eps)

For each layer i in [0, n_layers):
  -- normed_buf_ already contains RMSNorm(hidden, attn_norm[i])

  [Quantized path, prefill:]
  q_proj = matmul_dispatch(normed, W_q, M=seq_len, N=dim,     K=dim)
  k_proj = matmul_dispatch(normed, W_k, M=seq_len, N=kv_dim,  K=dim)
  v_proj = matmul_dispatch(normed, W_v, M=seq_len, N=kv_dim,  K=dim)

  For each token t in [0, seq_len):
    rope(q_proj[t], pos=pos+t, dim=head_dim, n_heads)
    rope(k_proj[t], pos=pos+t, dim=head_dim, n_kv_heads)

  -- forward_from_qkv (kv_cache append + multi-kernel attention):
    kv_cache_->append(i, k_proj, v_proj, seq_len)
    scores = batched_matmul_gqa(Q, K_cache^T)    -- [n_heads, seq_len, kv_len]
    scale(scores, 1/sqrt(head_dim))
    causal_mask(scores, start_pos=pos)
    softmax(scores)                               -- row-wise per head
    attn_out = batched_matmul_gqa(scores, V_cache) -- [n_heads, seq_len, head_dim]
    output = matmul_dispatch(attn_out, W_o)

  FUSED: hidden += attn_out; normed_buf_ = RMSNorm(hidden, ffn_norm[i])
    rmsnorm_residual(hidden, attn_out, ffn_norm_weight_[i], hidden, normed_buf_)

  gate_buf = matmul_dispatch(normed, W_gate, M=seq_len, N=inter, K=dim)
  up_buf   = matmul_dispatch(normed, W_up,   M=seq_len, N=inter, K=dim)
  silu(gate_buf, hidden_buf, seq_len*inter)
  multiply(hidden_buf, up_buf, gate_buf, seq_len*inter)
  ffn_out  = matmul_dispatch(gate_buf, W_down, M=seq_len, N=dim,  K=inter)

  FUSED: hidden += ffn_out; normed_buf_ = RMSNorm(hidden, next_norm_weight)
    next_norm = (i < n_layers-1) ? attn_norm_weight_[i+1] : output_norm_weight_
    rmsnorm_residual(hidden, ffn_out, next_norm, hidden, normed_buf_)

-- normed_buf_ holds RMSNorm(hidden, output_norm) after the last layer

logits = lm_head_.forward(normed_buf_[0:seq_len], token_embd_weight_)
  -- matmul_transposed_b: [seq_len, dim] * [vocab_size, dim]^T = [seq_len, vocab_size]

Return logits
```

### 7.2 Decode (seq_len = 1)

Each decode step processes one token. Differs from prefill in the attention step only.

```
TransformerModel::generate continuation:
  next_token = sample(logits[prompt_len-1, :])

  For step in [1, max_tokens):
    If next_token == eos_token_id: break
    step_logits = forward({next_token})      -- seq_len = 1
    next_token = sample(step_logits[0, :])
    tokens.push_back(next_token)
```

Inside `forward({next_token})` during decode, the attention path uses the fused kernel:

```
[Fused decode path for seq_len=1:]
q_proj = matmul_dispatch(normed, W_q, M=1, N=dim,    K=dim)
k_proj = matmul_dispatch(normed, W_k, M=1, N=kv_dim, K=dim)
v_proj = matmul_dispatch(normed, W_v, M=1, N=kv_dim, K=dim)
rope(q_proj, pos, head_dim, n_heads)
rope(k_proj, pos, head_dim, n_kv_heads)
kv_cache_->append(i, k_proj, v_proj, 1)       -- kv_len becomes pos+1
fused_decode_attention(q_proj, K_cache[i], V_cache[i], attn_concat, n_heads, ...)
  -- Online softmax, one block per head, reads kv_len positions
attn_out = matmul_dispatch(attn_concat, W_o, M=1, N=dim, K=dim)
```

### 7.3 KV-Cache Position Tracking

The KV-cache `current_length_` tracks how many complete token positions have been processed.
Before the first forward call, `current_length_ = 0`. After prefill with prompt_len tokens,
`current_length_ = prompt_len`. Each subsequent decode step increments it by 1.

The `pos` variable in `forward()` is read as `kv_cache_->current_length()` before appending,
giving the absolute sequence position of the first new token. RoPE encodes each token at
its correct absolute position via `pos + t` for token `t` in the sequence.

---

## 8. Design Decisions

### Custom kernels vs cuBLAS

All matrix multiplications are implemented from scratch. This is a deliberate choice for a
portfolio project: the tiled shared memory kernel, dequantize-on-the-fly technique, and
fused attention kernel demonstrate understanding of GPU memory hierarchy, occupancy, and
quantized arithmetic. cuBLAS is included only in `benchmarks/bench_matmul.cu` for comparison,
linked via `CUDA::cublas` in `CMakeLists.txt`. The benchmarks document the gap.

CUB (`cub::BlockReduce`) is used alongside hand-written tree reductions for RMSNorm and
softmax. CUB ships with the CUDA toolkit and is not an external dependency. Both variants
are kept to show the comparison explicitly.

### GGUF format

GGUF is the distribution format for quantized GGML-compatible models (llama.cpp, Ollama,
and similar). Using it directly avoids requiring Python, HuggingFace Transformers, or any
ML framework at load time. The format is well-documented and the quantization block structures
are byte-compatible with GGML, allowing weights to be uploaded to GPU as raw bytes.

### FP32 activations throughout

All activations, KV-cache, and intermediate buffers are FP32 regardless of weight quantization.
Weights are dequantized to FP32 within the matmul kernels. Using FP16 activations would
halve the KV-cache memory but require careful handling of accumulation precision. The current
approach prioritizes correctness and debuggability over maximum memory efficiency.

### Pre-allocation for max sequence length

The KV-cache is allocated for `max_seq_len` positions at construction time. `clear()` resets
the position counter without freeing memory. This eliminates all allocator calls from the
generation hot loop and guarantees that `append()` cannot fail due to out-of-memory conditions
mid-generation (it will throw `std::overflow_error` only if the counter exceeds the
pre-allocated length).

### Fused kernel boundaries

The forward pass uses two fusion points per layer:
1. **Post-attention**: `rmsnorm_residual(hidden, attn_out, ffn_norm[i], hidden, normed_buf_)`
   fuses the residual add with the pre-FFN RMSNorm.
2. **Post-FFN**: `rmsnorm_residual(hidden, ffn_out, next_norm, hidden, normed_buf_)` fuses
   the residual add with the next layer's pre-attention RMSNorm (or the final output norm).

This means each layer's forward pass makes exactly two calls to `rmsnorm_residual` and zero
calls to standalone `add` or `rmsnorm`. During decode (seq_len=1), each call saves one full
read/write of a [1, dim] buffer.

For decode attention (seq_len=1), `fused_decode_attention` replaces five kernel launches
(scale, causal mask, softmax, batched Q×K^T, batched attn×V) plus associated temporary buffer
allocations with one kernel that reads the KV-cache in a single pass per head.

---

*Last updated: 2026-03-04. Reflects Phase 3 implementation (quantized inference, fused kernels,
Llama 3.2 1B support). Source at `E:/Repos/CuInfer`.*
