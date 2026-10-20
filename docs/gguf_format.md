# GGUF Format: Parser Implementation Reference

This document describes the GGUF binary format as it is parsed by CuInfer's
`GGUFParser` class, covering the on-disk layout, parsing strategy, metadata
key conventions, tensor loading, supported quantization block formats, and
weight name mapping used by `TransformerModel`.

Relevant source files:
- `include/cuinfer/gguf_parser.h` — public API and type definitions
- `src/gguf_parser.cpp` — parser implementation
- `include/cuinfer/quantization.h` — quantized block structures
- `src/model.cu` — weight loading and name mapping

---

## 1. GGUF Format Overview

GGUF (GGML Universal File) is a self-describing binary format for LLM weight
storage. It embeds model hyperparameters (layer counts, dimensions, RoPE
parameters, tokenizer vocabulary) together with raw tensor data in a single
file. The format is documented at
`https://github.com/ggerganov/ggml/blob/master/docs/gguf.md`.

### 1.1 File Layout

A GGUF file consists of four sections laid out sequentially:

```
Offset 0
+-------------------------------------------------------+
| Header (24 bytes)                                     |
|   magic          : uint32  — 0x46554747 ("GGUF")     |
|   version        : uint32  — format version (2 or 3) |
|   tensor_count   : uint64  — number of tensors        |
|   metadata_count : uint64  — number of KV pairs       |
+-------------------------------------------------------+
| Metadata KV Pairs (variable length)                   |
|   Repeated metadata_count times:                      |
|     key_len : uint64                                  |
|     key      : char[key_len]  (no null terminator)    |
|     val_type : uint32  (GGUFValueType enum)           |
|     value    : variable (depends on val_type)         |
+-------------------------------------------------------+
| Tensor Info (variable length)                         |
|   Repeated tensor_count times:                        |
|     name_len   : uint64                               |
|     name        : char[name_len]                      |
|     n_dims      : uint32                              |
|     dims        : uint64[n_dims]                      |
|     type        : uint32  (GGMLType enum)             |
|     offset      : uint64  (relative to data section)  |
+-------------------------------------------------------+
| Alignment Padding                                     |
|   Zero bytes to reach the next alignment boundary    |
|   Default alignment: 32 bytes                        |
|   Override: "general.alignment" metadata key (uint32)|
+-------------------------------------------------------+
| Tensor Data                                           |
|   Each tensor's raw bytes at its declared offset      |
+-------------------------------------------------------+
```

All integer fields are stored in little-endian byte order. Strings are stored
as a `uint64` byte count followed by the raw UTF-8 bytes, with no null
terminator.

### 1.2 Magic Number

The four-byte magic at offset 0 encodes the ASCII string "GGUF" in
little-endian order: `0x46 0x55 0x47 0x47`. In CuInfer this is defined as:

```cpp
constexpr uint32_t GGUF_MAGIC = 0x46554747;
```

A file that does not begin with this value is rejected immediately with a
`std::runtime_error`.

### 1.3 Versioning

CuInfer accepts GGUF versions 2 and 3. The version field controls no
behavioral differences in the parser — both versions use the same binary layout
for all sections parsed by this implementation. Version 1 files are rejected.

---

## 2. CuInfer's Parser Implementation

### 2.1 Class and Pimpl Structure

The public API is `cuinfer::GGUFParser`, declared in
`include/cuinfer/gguf_parser.h`. Its private state is held in
`GGUFParser::Impl`, defined only in `src/gguf_parser.cpp`. This keeps
platform-specific memory-mapping types (Windows `HANDLE`, POSIX `int fd`)
out of the public header.

```cpp
struct GGUFParser::Impl {
    MemoryMappedFile mmap;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    std::unordered_map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;
    size_t data_section_offset;   // absolute byte offset to tensor data start
};
```

`GGUFParser` is non-copyable because it owns the memory-mapped region, and is
movable for use in containers.

### 2.2 Memory Mapping Strategy

The entire file is memory-mapped read-only at construction time via the
`MemoryMappedFile` RAII class. On POSIX systems (Linux, WSL2) this uses
`open` + `mmap(PROT_READ, MAP_PRIVATE)`. On Windows it uses
`CreateFile` + `CreateFileMapping` + `MapViewOfFile`.

Memory mapping provides two advantages for model files:

1. The tensor data section (typically 0.5–3 GB) is never copied into process
   heap memory. `tensor_data()` returns a pointer directly into the mapped
   region.
2. The OS page cache handles the actual I/O, avoiding redundant buffering.

The mapped region remains valid for the lifetime of the `GGUFParser` instance.
All pointers returned by `tensor_data()` become invalid after the parser is
destroyed or moved from.

### 2.3 Sequential Parse Strategy

Parsing is performed by `StreamReader`, a bounds-checked cursor over the
memory-mapped region. The cursor advances forward-only through each section:

```
StreamReader: [data_ptr, size, pos]
  read<T>()       — memcpy T from pos, advance pos by sizeof(T)
  read_string()   — read uint64 length, then read that many bytes
  seek(pos)       — set pos directly (used only for alignment)
```

The parse sequence in the constructor is:

1. Validate header (magic, version, counts).
2. Iterate `metadata_kv_count` times: read key string, value type tag, then
   value payload. Results go into `impl_->metadata`.
3. Iterate `tensor_count` times: read name, dimension count, dimension array,
   type tag, and relative offset. Results go into `impl_->tensors`.
4. Round the cursor position up to the alignment boundary to compute
   `data_section_offset`.
5. For each tensor, set `data_offset = data_section_offset + tensor.offset`.

### 2.4 Endianness Handling

The parser uses `std::memcpy` to read multi-byte values from the mapped
region. GGUF specifies little-endian storage. CuInfer targets x86-64 hardware
(Windows 10 and WSL2 Ubuntu on RTX 3060), which is natively little-endian.
The implementation does not perform any byte-swapping. Running the parser on
a big-endian host would produce incorrect results.

### 2.5 Alignment Computation

After all tensor info entries are read, the cursor position is rounded up to
the alignment boundary:

```cpp
size_t data_start = ((cursor + alignment - 1) / alignment) * alignment;
```

The default alignment is 32 bytes (`GGUF_DEFAULT_ALIGNMENT`). If the metadata
contains `"general.alignment"` as a `UINT32` value, that value is used
instead; a value of zero falls back to 32.

---

## 3. Metadata Extraction

### 3.1 Value Types

GGUF metadata values are tagged with a `GGUFValueType` enum (`uint32` on
disk):

| Tag | C++ Type | Size on disk |
|-----|----------|--------------|
| 0 UINT8 | `uint8_t` | 1 byte |
| 1 INT8 | `int8_t` | 1 byte |
| 2 UINT16 | `uint16_t` | 2 bytes |
| 3 INT16 | `int16_t` | 2 bytes |
| 4 UINT32 | `uint32_t` | 4 bytes |
| 5 INT32 | `int32_t` | 4 bytes |
| 6 FLOAT32 | `float` | 4 bytes |
| 7 BOOL | `uint8_t` on disk, `bool` in memory | 1 byte |
| 8 STRING | `uint64` length + chars | variable |
| 9 ARRAY | `uint32` element_type + `uint64` count + elements | variable |
| 10 UINT64 | `uint64_t` | 8 bytes |
| 11 INT64 | `int64_t` | 8 bytes |
| 12 FLOAT64 | `double` | 8 bytes |

`BOOL` values are stored as a single byte; any nonzero byte is interpreted as
`true`.

`ARRAY` values contain a nested element type tag and count, followed by that
many values of the declared element type. Arrays can contain any scalar or
string type. Nested arrays are not used in practice by LLM GGUF files.

CuInfer stores all metadata values in `GGUFValue`, a struct holding a
`GGUFValueType` tag and a `std::variant<...>` payload. Typed accessors (e.g.,
`as_uint32()`, `as_string()`) check the tag at runtime and throw
`std::runtime_error` on mismatch.

### 3.2 Architecture Key Convention

All architecture-specific metadata keys follow the pattern
`{arch}.{property}`, where `{arch}` is read from `"general.architecture"`.
For Llama-family models (SmolLM2, Llama 3.2) this value is `"llama"`, making
keys such as `"llama.context_length"`.

`GGUFParser::architecture()` reads `"general.architecture"` and the remaining
helper methods on `GGUFParser` construct their key strings by prepending this
value. Callers that need architecture-specific keys not exposed by the helpers
can use `get_uint32(parser.architecture() + ".some_key")` directly.

### 3.3 Model Hyperparameter Keys

The following table lists the metadata keys read by `TransformerModel` and
the `GGUFParser` helper that reads each one. Values shown are for
SmolLM2-135M (`"llama"` architecture) and Llama 3.2 1B (same architecture
name).

| GGUF Key | Type | GGUFParser Method | SmolLM2-135M | Llama 3.2 1B |
|---|---|---|---|---|
| `general.architecture` | STRING | `architecture()` | `"llama"` | `"llama"` |
| `llama.context_length` | UINT32 | `context_length()` | 8192 | 131072 |
| `llama.embedding_length` | UINT32 | `embedding_length()` | 576 | 2048 |
| `llama.block_count` | UINT32 | `block_count()` | 30 | 16 |
| `llama.attention.head_count` | UINT32 | `head_count()` | 9 | 32 |
| `llama.attention.head_count_kv` | UINT32 | `head_count_kv()` | 3 | 8 |
| `llama.feed_forward_length` | UINT32 | `feed_forward_length()` | 1536 | 8192 |
| `llama.rope.freq_base` | FLOAT32 | `rope_freq_base()` | 100000.0 | 500000.0 |
| `llama.rope.dimension_count` | UINT32 | `rope_dimension_count()` | 64 | 64 |
| `llama.attention.layer_norm_rms_epsilon` | FLOAT32 | `rms_norm_eps()` | 1e-5 | 1e-5 |
| `llama.vocab_size` | UINT32 | `vocab_size()` | 49152 | 128256 |

Keys with defaults:
- `rope_freq_base()` returns `10000.0f` if the key is absent.
- `rms_norm_eps()` returns `1e-5f` if the key is absent.
- `vocab_size()` falls back to the length of the `"tokenizer.ggml.tokens"`
  array if `"{arch}.vocab_size"` is absent.

### 3.4 Tokenizer Metadata Keys

The tokenizer is built entirely from GGUF metadata arrays by
`TransformerModel::build_tokenizer()`. The following keys are read:

| GGUF Key | Type | Contents |
|---|---|---|
| `tokenizer.ggml.model` | STRING | Tokenizer type: `"gpt2"` for BPE, `"llama"` for SentencePiece |
| `tokenizer.ggml.tokens` | ARRAY[STRING] | Token strings, one per vocabulary entry |
| `tokenizer.ggml.scores` | ARRAY[FLOAT32] | SentencePiece merge scores |
| `tokenizer.ggml.token_type` | ARRAY[INT32] | Token classification (normal, BOS, EOS, etc.) |
| `tokenizer.ggml.merges` | ARRAY[STRING] | BPE merge rules (one per line, format `"a b"`) |
| `tokenizer.ggml.bos_token_id` | UINT32 | Beginning-of-sequence token ID |
| `tokenizer.ggml.eos_token_id` | UINT32 | End-of-sequence token ID |
| `tokenizer.ggml.padding_token_id` | UINT32 | Padding token ID |
| `tokenizer.ggml.add_bos_token` | BOOL | Whether to prepend BOS on encode |
| `tokenizer.ggml.add_eos_token` | BOOL | Whether to append EOS on encode |

When `tokenizer.ggml.model` is `"gpt2"` and `tokenizer.ggml.merges` is
present, the BPE path is used. Otherwise the SentencePiece score-based path
is used with `tokenizer.ggml.scores`.

---

## 4. Tensor Loading

### 4.1 Tensor Info Structure

Each tensor is described by a `GGUFTensorInfo` struct:

```cpp
struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;  // e.g., {4096, 4096} for a 2D weight
    GGMLType type;
    uint64_t offset;       // relative offset within the data section
    uint64_t data_offset;  // absolute byte offset from start of file
};
```

`byte_size()` computes the tensor's byte footprint:
```
byte_size = (element_count / block_size) * block_byte_size
```
For unquantized types `block_size = 1`, so `byte_size = element_count * sizeof(element)`.
For quantized types the block formula accounts for the packed block structure.
`byte_size()` throws if `element_count` is not divisible by `block_size`.

### 4.2 GGUF Dimension Convention

GGUF stores dimensions in GGML internal order: `dims[0]` is the innermost
(fastest-varying) dimension and `dims[1]` is the outermost. For a 2D weight
matrix that PyTorch stores as `[out_features, in_features]`:

```
dims[0] = in_features   (columns / K dimension)
dims[1] = out_features  (rows / N dimension)
```

This is the opposite of the C row-major convention. The parser stores
dimensions verbatim as read from the file; callers are responsible for
interpreting them correctly.

### 4.3 Tensor Data Access

`tensor_data(const GGUFTensorInfo& info)` returns a pointer directly into the
memory-mapped region:

```cpp
const void* GGUFParser::tensor_data(const GGUFTensorInfo& info) const {
    // bounds check: data_offset + byte_size <= file_size
    return impl_->mmap.data() + info.data_offset;
}
```

The pointer is valid until the `GGUFParser` is destroyed. No data is copied.
The overload `tensor_data(const std::string& name)` calls `find_tensor()`
first; it throws `std::runtime_error` if the tensor is not found.

`find_tensor()` performs a linear scan of `impl_->tensors`. For typical model
files with hundreds of tensors this is not a bottleneck during loading.

### 4.4 GPU Upload Path

`GGUFParser` itself does not interact with CUDA. GPU upload is performed by
`TransformerModel` weight-loading helpers in `src/model.cu`. The three
loading paths are:

**FP32 / FP16 with transposition (`load_weight_transposed`):**
Used for non-quantized 2D weights in the FP32 inference path. The raw data
pointer from `tensor_data()` is used as the CPU source. F16 weights are first
converted to F32 on the CPU via `__half2float`. The resulting F32 buffer is
transposed from GGUF's `[N, K]` layout to `[K, N]` for `matmul_tiled`, then
uploaded via `cudaMemcpy(HostToDevice)`.

**Quantized raw upload (`load_weight_quantized`):**
Used for Q8_0, Q4_0, Q5_0, Q4_K, Q6_K weights. The quantized bytes are
uploaded directly to the GPU without any dequantization or transposition. The
quantized matmul kernels (`src/kernels/matmul_quant.cu`) consume the GGUF
native `[N, K]` layout directly. For F16 tensors in the quantized path, the
raw half-precision bytes are uploaded to a `DType::FP16` Tensor.

**Dequantize-then-use (`load_weight_1d`, `load_weight_2d`):**
Used for 1D norm weights and the 2D embedding table, which require FP32 for
elementwise operations and row-lookup access respectively. The bytes are
uploaded to a temporary GPU buffer, the appropriate
`dequantize_q*` kernel is called to produce FP32 output in the final Tensor,
and the temporary buffer is freed.

---

## 5. Supported Quantization Formats

CuInfer recognizes the `GGMLType` enum values below. Types marked as
"dequant kernels" have custom GPU dequantization kernels in
`src/kernels/matmul_quant.cu`. Types marked "no kernel" are parsed and
stored but cannot be used as quantized weights.

### 5.1 Format Table

| GGMLType | Enum Value | Block Size | Block Bytes | Bits/Weight | Status |
|---|---|---|---|---|---|
| F32 | 0 | 1 | 4 | 32 | Supported |
| F16 | 1 | 1 | 2 | 16 | Supported |
| Q4_0 | 2 | 32 | 18 | 4.5 | Supported, dequant kernel |
| Q4_1 | 3 | 32 | 20 | 5.0 | Parsed only, no matmul kernel |
| Q5_0 | 6 | 32 | 22 | 5.5 | Supported, dequant kernel |
| Q5_1 | 7 | 32 | 24 | 6.0 | Parsed only, no matmul kernel |
| Q8_0 | 8 | 32 | 34 | 8.5 | Supported, dequant kernel |
| Q8_1 | 9 | 32 | 36 | 9.0 | Parsed only, no matmul kernel |
| Q2_K | 10 | 256 | 256 | 2.0 approx | Parsed only, no matmul kernel |
| Q3_K | 11 | 256 | 256 | 3.0 approx | Parsed only, no matmul kernel |
| Q4_K | 12 | 256 | 144 | 4.5 | Supported, dequant kernel |
| Q5_K | 13 | 256 | 176 | 5.5 | Parsed only, no matmul kernel |
| Q6_K | 14 | 256 | 210 | 6.5 | Supported, dequant kernel |

"Bits/weight" = `(block_bytes * 8) / block_size`. For unquantized types this
equals the element bit width. For quantized types the overhead of scale fields
raises the effective rate above the nominal quantization bit depth.

The `Q3_K` block size is listed as 256 to match the super-block convention,
but the actual byte size varies by implementation; the value 256 in
`ggml_type_size` is documented as approximate.

### 5.2 Block Structures

The C structs in `include/cuinfer/quantization.h` reproduce the GGML
on-disk format exactly. Each `static_assert` verifies the compiled size.

**Q8_0 — 34 bytes, 32 elements:**
```
Bytes  0- 1 : __half  scale         (FP16)
Bytes  2-33  : int8_t  qs[32]       (quantized values)
Dequant: x[i] = float(scale) * float(qs[i])
```

**Q4_0 — 18 bytes, 32 elements:**
```
Bytes  0- 1 : __half  scale         (FP16)
Bytes  2-17  : uint8_t qs[16]       (packed nibble pairs, 2 elements per byte)
Nibble layout: qs[j] & 0x0F = element 2j+0 (unsigned [0,15], implicit -8)
               qs[j]  >> 4  = element 2j+1
Dequant: x[i] = float(scale) * (nibble[i] - 8)
```

**Q5_0 — 22 bytes, 32 elements:**
```
Bytes  0- 1 : __half  scale         (FP16)
Bytes  2- 5  : uint8_t qh[4]        (high bits: bit j = 5th bit of element j)
Bytes  6-21  : uint8_t qs[16]       (low 4-bit nibbles, 2 per byte)
Reconstruction: q5 = (low_nibble | (high_bit << 4)), range [0, 31]
Dequant: x[i] = float(scale) * (q5[i] - 16)
```

**Q4_K — 144 bytes, 256 elements (super-block):**
```
Bytes  0- 1 : __half  d             (super-block scale)
Bytes  2- 3 : __half  dmin          (super-block min factor)
Bytes  4-15  : uint8_t scales[12]   (8 sub-block scales + 8 sub-block mins,
                                     packed as 6-bit fields in 12 bytes)
Bytes 16-143 : uint8_t qs[128]      (packed 4-bit values, 2 per byte)
8 sub-blocks of 32 elements each.
Dequant for sub-block j:
  sc  = decoded_scale(scales, j)
  mn  = decoded_min(scales, j)
  x[i] = float(d) * sc * nibble[i] - float(dmin) * mn
```

**Q6_K — 210 bytes, 256 elements (super-block):**
```
Bytes   0-127 : uint8_t ql[128]    (lower 4 bits, packed 2 per byte)
Bytes 128-191 : uint8_t qh[64]     (upper 2 bits, packed 4 per byte)
Bytes 192-207 : int8_t  scales[16] (sub-block scales, 16 sub-blocks of 16 el.)
Bytes 208-209 : __half  d          (super-block scale)
6-bit value reconstruction per element i:
  low4 = ql nibble for element i
  high2 = qh 2-bit field for element i
  q6 = low4 | (high2 << 4), range [0, 63]
Dequant: x[i] = float(d) * scales[i/16] * (q6[i] - 32)
```

The Q6_K scale index is `i / 16`, which means 16 distinct scale entries are
used within a 256-element super-block (two entries per 32-element group). Using
the wrong index (e.g., `i / 32`) causes half of the values to use an incorrect
scale.

---

## 6. Weight Name Mapping

### 6.1 GGUF Tensor Naming Convention

For Llama-family models, GGUF uses the following naming patterns:

| Category | GGUF Name Pattern | Notes |
|---|---|---|
| Token embedding | `token_embd.weight` | [vocab_size, dim] |
| Final norm | `output_norm.weight` | [dim] |
| LM head | `output.weight` | [vocab_size, dim]; may be absent if weights are tied |
| Per-layer attention norm | `blk.{i}.attn_norm.weight` | [dim] |
| Per-layer Q projection | `blk.{i}.attn_q.weight` | [dim, dim] |
| Per-layer K projection | `blk.{i}.attn_k.weight` | [n_kv_heads * head_dim, dim] |
| Per-layer V projection | `blk.{i}.attn_v.weight` | [n_kv_heads * head_dim, dim] |
| Per-layer output projection | `blk.{i}.attn_output.weight` | [dim, dim] |
| Per-layer FFN norm | `blk.{i}.ffn_norm.weight` | [dim] |
| Per-layer FFN gate | `blk.{i}.ffn_gate.weight` | [intermediate_size, dim] |
| Per-layer FFN up | `blk.{i}.ffn_up.weight` | [intermediate_size, dim] |
| Per-layer FFN down | `blk.{i}.ffn_down.weight` | [dim, intermediate_size] |

`{i}` is the zero-indexed layer number. Dimensions shown are in HuggingFace
convention (`[out_features, in_features]`), which corresponds to GGUF
`[dims[1], dims[0]]`.

### 6.2 TransformerModel Internal Names

`TransformerModel` (declared in `include/cuinfer/model.h`) stores weights in
member arrays indexed by layer:

```cpp
Tensor token_embd_weight_;        // [vocab_size, dim]     — always FP32
Tensor output_norm_weight_;       // [dim]                  — always FP32
Tensor lm_head_weight_;           // [vocab_size, dim]      — quantized or empty if tied

std::vector<Tensor> attn_norm_weight_;   // [dim] each     — always FP32
std::vector<Tensor> attn_q_weight_;      // [dim, dim] each
std::vector<Tensor> attn_k_weight_;      // [kv_dim, dim] each
std::vector<Tensor> attn_v_weight_;      // [kv_dim, dim] each
std::vector<Tensor> attn_o_weight_;      // [dim, dim] each
std::vector<Tensor> ffn_norm_weight_;    // [dim] each     — always FP32
std::vector<Tensor> ffn_gate_weight_;    // [dim, intermediate] each
std::vector<Tensor> ffn_up_weight_;      // [dim, intermediate] each
std::vector<Tensor> ffn_down_weight_;    // [intermediate, dim] each
```

The constructor builds the GGUF tensor name as:
```cpp
std::string prefix = "blk." + std::to_string(i) + ".";
// e.g., "blk.0.attn_q.weight"
```

### 6.3 Layout Transformation During Load

The memory layout of a tensor differs depending on the load path:

**FP32/F16 2D weights — transposed:**
GGUF stores `attn_q.weight` as `[dim, dim]` in `[N, K]` order (outer first).
`load_weight_transposed()` reads `dims[0]` as `in_feat` (K) and `dims[1]` as
`out_feat` (N), and produces a host-transposed `[in_feat, out_feat]` = `[K, N]`
tensor. This is the layout expected by `matmul_tiled(A, B)` where `B` is `[K, N]`.

**Quantized 2D weights — not transposed:**
`load_weight_quantized()` uploads raw quantized bytes directly without
transposition. The tensor is stored on GPU as `[N, K]` — the GGUF native
layout. The quantized matmul kernels read it in this order natively (they
dequantize B row by row, where each row is a set of output-dimension blocks).

**Embedding and norm weights:**
`load_weight_2d()` (for `token_embd.weight`) and `load_weight_1d()` (for norm
weights) always dequantize to FP32. The embedding table is stored as
`[vocab_size, dim]` without transposition, because it is accessed via row
lookup rather than matrix multiplication.

### 6.4 Weight Tying Detection

`GGUFParser::tie_word_embeddings()` returns `true` if the `"output.weight"`
tensor is absent from the file:

```cpp
bool GGUFParser::tie_word_embeddings() const {
    return find_tensor("output.weight") == nullptr;
}
```

When `config_.tie_embeddings` is `true`, `TransformerModel` does not load
`output.weight` and instead reuses `token_embd_weight_` for the LM head
computation.

### 6.5 Format Detection

The overall model weight format is detected by examining the first layer's Q
projection:

```cpp
WeightFormat TransformerModel::detect_weight_format(const GGUFParser& parser) {
    const auto* info = parser.find_tensor("blk.0.attn_q.weight");
    ...
    return ggml_type_to_weight_format(info->type);
}
```

For Q4_K_M quantized files, individual tensors may have different types within
the same file (typically a mix of Q4_K and Q6_K). Per-tensor format is
recorded in `attn_q_format_`, `attn_k_format_`, etc., and the correct matmul
kernel variant is dispatched at runtime via `matmul_dispatch()`.

---

## 7. Limitations and Known Issues

### Unsupported GGUF Versions

Versions other than 2 and 3 are rejected at parse time. Version 1 is an older
format from before the GGUF specification stabilized and is not handled.

### Unsupported Quantization Types

Q4_1, Q5_1, Q8_1, Q2_K, Q3_K, and Q5_K are recognized in the `GGMLType` enum
and can be parsed (names and dimensions are read), but no dequantization
kernels or matmul kernels exist for them. Attempting to load a model file that
uses these types for weight tensors will produce a `std::runtime_error` from
the weight loading helpers.

### No Big-Endian Support

The parser uses `memcpy`-based reads without byte-swapping. Running on a
big-endian host would silently produce wrong values throughout all scalar
fields.

### No GGUF Version 3 Sparse Tensor Support

GGUF version 3 added provisions for sparse tensor metadata. CuInfer does not
implement sparse tensors; all tensors are assumed to be dense.

### Quantization Block Alignment Requirement

`GGUFTensorInfo::byte_size()` requires that the total element count be evenly
divisible by the block size. If a quantized tensor in a GGUF file has a size
that is not a multiple of its block size, the parser throws at the time
`byte_size()` is first called (during model loading). Unquantized tensors
(F32, F16) have a block size of 1 and are not subject to this constraint.

### Linear Tensor Search

`find_tensor()` performs a linear scan over the tensor list. For files with
hundreds of tensors this is acceptable during the one-time model loading phase.
It is not appropriate for repeated lookups at inference time, but no inference
path calls `find_tensor()`.

### Single-Architecture Metadata Helpers

The architecture helper methods on `GGUFParser` (`context_length()`,
`embedding_length()`, etc.) all read keys under the architecture name returned
by `architecture()`. If a GGUF file contains tensors from two architectures or
uses non-standard metadata key names, these helpers will not find the expected
keys and will throw `std::runtime_error`. Only Llama-family architectures have
been tested.

### No Validation of Tensor Data Integrity

The parser verifies that tensor data offsets fall within the file boundary but
does not validate the content of tensor data against any checksum. Truncated
or corrupted model files may result in silently incorrect values after upload
to the GPU.
