# SmolLM2-135M GGUF Tensor & Architecture Mapping

Reference document for implementing the `TransformerModel` class (Task 2.5).
Maps SmolLM2-135M architecture parameters to GGUF metadata keys, and maps
GGUF tensor names to their roles in the transformer forward pass.

**Source model**: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
**GGUF files**: [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF)

---

## 1. Architecture Parameters

SmolLM2-135M uses the **Llama architecture** (`LlamaForCausalLM`) with Grouped Query
Attention (GQA). The config reports `general.architecture = "llama"` in GGUF metadata.

| Parameter              | Value   | Notes                                          |
|------------------------|---------|-------------------------------------------------|
| `num_hidden_layers`    | 30      | Number of transformer blocks                    |
| `hidden_size`          | 576     | Model / embedding dimension (d_model)           |
| `num_attention_heads`  | 9       | Query heads                                     |
| `num_key_value_heads`  | 3       | KV heads (GQA, ratio 3:1 per group)             |
| `head_dim`             | 64      | = hidden_size / num_attention_heads = 576 / 9   |
| `intermediate_size`    | 1536    | FFN hidden dimension (gate/up projection width)  |
| `vocab_size`           | 49152   | BPE vocabulary size                             |
| `max_position_embeddings` | 8192 | Maximum context / sequence length               |
| `rms_norm_eps`         | 1e-5    | RMSNorm epsilon                                 |
| `rope_theta`           | 100000  | RoPE frequency base                             |
| `rope_scaling`         | null    | No RoPE scaling applied                         |
| `hidden_act`           | silu    | Activation in FFN (SiLU / Swish)                |
| `tie_word_embeddings`  | true    | LM head shares weights with token embedding     |
| `attention_bias`       | false   | No bias in Q/K/V/O projections                  |
| `torch_dtype`          | bfloat16| Training precision                              |

### GQA Layout

With 9 query heads and 3 KV heads, each KV head serves a group of 3 query heads.
During attention, the K and V tensors are either:
- Repeated/broadcast 3x to match the 9 query heads, or
- Computed with an efficient grouped-attention kernel.

```
Query heads:  [h0 h1 h2] [h3 h4 h5] [h6 h7 h8]
KV heads:     [kv0      ] [kv1      ] [kv2      ]
```

### Tensor Dimensions Quick Reference

| Tensor                | Shape                                              |
|-----------------------|----------------------------------------------------|
| Token embedding       | [vocab_size, hidden_size] = [49152, 576]           |
| Attention Q weight    | [hidden_size, hidden_size] = [576, 576]            |
| Attention K weight    | [n_kv_heads * head_dim, hidden_size] = [192, 576]  |
| Attention V weight    | [n_kv_heads * head_dim, hidden_size] = [192, 576]  |
| Attention O weight    | [hidden_size, hidden_size] = [576, 576]            |
| FFN gate_proj weight  | [intermediate_size, hidden_size] = [1536, 576]     |
| FFN up_proj weight    | [intermediate_size, hidden_size] = [1536, 576]     |
| FFN down_proj weight  | [hidden_size, intermediate_size] = [576, 1536]     |
| RMSNorm weight        | [hidden_size] = [576]                              |
| LM head (output)      | Tied to token embedding = [49152, 576]             |

**Note on GGUF dimension ordering**: GGUF stores dimensions in the order they appear
in the binary layout. For 2D weight matrices, GGUF dimensions are typically
`[n_rows, n_cols]` which corresponds to `[out_features, in_features]` in PyTorch.

---

## 2. GGUF Metadata Keys

The GGUF metadata keys follow the pattern `{arch}.{property}` where `{arch}` is
the architecture name read from `general.architecture` (in this case, `"llama"`).

### General Keys

| GGUF Metadata Key               | Type     | SmolLM2-135M Value | Description                      |
|----------------------------------|----------|---------------------|----------------------------------|
| `general.architecture`           | string   | `"llama"`           | Architecture identifier          |
| `general.name`                   | string   | (varies by GGUF)    | Human-readable model name        |
| `general.file_type`             | uint32   | (varies)            | Quantization type of the file    |
| `general.quantization_version`  | uint32   | (varies)            | Quantization format version      |

### Architecture-Specific Keys

| GGUF Metadata Key                                | Type     | SmolLM2-135M Value | Maps to HF Config Field       |
|--------------------------------------------------|----------|---------------------|-------------------------------|
| `llama.context_length`                           | uint32   | 8192                | `max_position_embeddings`     |
| `llama.embedding_length`                         | uint32   | 576                 | `hidden_size`                 |
| `llama.block_count`                              | uint32   | 30                  | `num_hidden_layers`           |
| `llama.feed_forward_length`                      | uint32   | 1536                | `intermediate_size`           |
| `llama.attention.head_count`                     | uint32   | 9                   | `num_attention_heads`         |
| `llama.attention.head_count_kv`                  | uint32   | 3                   | `num_key_value_heads`         |
| `llama.attention.layer_norm_rms_epsilon`         | float32  | 1e-5                | `rms_norm_eps`                |
| `llama.rope.freq_base`                           | float32  | 100000.0            | `rope_theta`                  |
| `llama.rope.dimension_count`                     | uint32   | 64                  | `head_dim` (= hidden_size / num_attention_heads) |
| `llama.vocab_size`                               | uint32   | 49152               | `vocab_size`                  |

### Tokenizer Keys

| GGUF Metadata Key                   | Type           | Description                              |
|--------------------------------------|----------------|------------------------------------------|
| `tokenizer.ggml.model`             | string         | Tokenizer type (e.g., `"gpt2"` for BPE) |
| `tokenizer.ggml.tokens`            | array[string]  | Full vocabulary (token strings)          |
| `tokenizer.ggml.scores`            | array[float32] | Token scores / probabilities             |
| `tokenizer.ggml.token_type`        | array[int32]   | Token type classification                |
| `tokenizer.ggml.bos_token_id`      | uint32         | Beginning-of-sequence token ID           |
| `tokenizer.ggml.eos_token_id`      | uint32         | End-of-sequence token ID                 |
| `tokenizer.ggml.merges`            | array[string]  | BPE merge rules (if applicable)          |

**Note**: SmolLM2 uses `bos_token_id = 0` and `eos_token_id = 0` per its HF config.
Some GGUF files may also include `tokenizer.huggingface.json` as a string containing
the full HuggingFace tokenizer JSON.

### Reading Metadata with GGUFParser

The existing `GGUFParser` class already provides convenience methods for most of these:

```cpp
GGUFParser parser("smollm2-135m.gguf");

std::string arch   = parser.architecture();          // "llama"
uint32_t n_layers  = parser.block_count();           // 30
uint32_t dim       = parser.embedding_length();      // 576
uint32_t n_heads   = parser.head_count();            // 9
uint32_t n_kv_heads = parser.head_count_kv();        // 3
uint32_t ffn_dim   = parser.feed_forward_length();   // 1536
uint32_t ctx_len   = parser.context_length();        // 8192
float rope_base    = parser.rope_freq_base();        // 100000.0

// Keys not yet wrapped — access directly from metadata:
float norm_eps = parser.get_float32("llama.attention.layer_norm_rms_epsilon"); // 1e-5
uint32_t rope_dim = parser.get_uint32("llama.rope.dimension_count");          // 64
```

These methods are now implemented as `GGUFParser::rms_norm_eps()` and `GGUFParser::rope_dimension_count()`.

---

## 3. GGUF Tensor Name to Component Mapping

GGUF uses a standardized tensor naming convention. For Llama-family models, tensors
follow the pattern `blk.{layer_idx}.{component}.weight` for per-layer tensors, and
simple names for global tensors.

### Global Tensors (Not Per-Layer)

| GGUF Tensor Name         | Role                     | Shape               | HuggingFace Name         |
|--------------------------|--------------------------|----------------------|--------------------------|
| `token_embd.weight`     | Token embedding matrix   | [49152, 576]         | `model.embed_tokens.weight` |
| `output_norm.weight`    | Final RMSNorm            | [576]                | `model.norm.weight`      |
| `output.weight`         | LM head projection       | [49152, 576]         | `lm_head.weight`         |

### Per-Layer Attention Tensors

Pattern: `blk.{i}.{component}.weight` where `{i}` is the 0-indexed layer number (0-29).

| GGUF Tensor Name                | Role                     | Shape       | HuggingFace Name                                |
|---------------------------------|--------------------------|-------------|--------------------------------------------------|
| `blk.{i}.attn_norm.weight`    | Pre-attention RMSNorm    | [576]       | `model.layers.{i}.input_layernorm.weight`       |
| `blk.{i}.attn_q.weight`       | Query projection         | [576, 576]  | `model.layers.{i}.self_attn.q_proj.weight`      |
| `blk.{i}.attn_k.weight`       | Key projection           | [192, 576]  | `model.layers.{i}.self_attn.k_proj.weight`      |
| `blk.{i}.attn_v.weight`       | Value projection         | [192, 576]  | `model.layers.{i}.self_attn.v_proj.weight`      |
| `blk.{i}.attn_output.weight`  | Output projection        | [576, 576]  | `model.layers.{i}.self_attn.o_proj.weight`      |

### Per-Layer Feed-Forward Tensors

| GGUF Tensor Name                | Role                     | Shape         | HuggingFace Name                                |
|---------------------------------|--------------------------|---------------|--------------------------------------------------|
| `blk.{i}.ffn_norm.weight`     | Pre-FFN RMSNorm          | [576]         | `model.layers.{i}.post_attention_layernorm.weight` |
| `blk.{i}.ffn_gate.weight`     | Gate projection (SiLU)   | [1536, 576]   | `model.layers.{i}.mlp.gate_proj.weight`         |
| `blk.{i}.ffn_up.weight`       | Up projection            | [1536, 576]   | `model.layers.{i}.mlp.up_proj.weight`           |
| `blk.{i}.ffn_down.weight`     | Down projection          | [576, 1536]   | `model.layers.{i}.mlp.down_proj.weight`         |

### Complete Tensor Count

For SmolLM2-135M with 30 layers:

| Category              | Tensors per Layer | Total Tensors  |
|-----------------------|-------------------|----------------|
| Global tensors        | -                 | 3 (token_embd, output_norm, output) |
| Attention (per layer) | 5                 | 150 (5 x 30)   |
| FFN (per layer)       | 4                 | 120 (4 x 30)   |
| **Total**             |                   | **273**         |

**Note**: If `output.weight` is omitted due to weight tying (see Section 4), the
total is **272** tensors.

---

## 4. Weight Tying (Embedding / LM Head)

SmolLM2-135M has `tie_word_embeddings = true` in its HuggingFace config. This means
the LM head (`lm_head.weight`) shares the same weight matrix as the token embedding
(`model.embed_tokens.weight`).

### Behavior in GGUF Files

The handling of weight tying depends on how the GGUF file was produced:

1. **`output.weight` is present**: The converter may explicitly duplicate the
   embedding weights into the `output.weight` tensor. In this case, load both
   tensors normally. llama.cpp sometimes duplicates for offloading purposes
   (embedding on CPU, output on GPU).

2. **`output.weight` is absent**: The converter omitted the duplicate because
   the weights are tied. In this case, reuse the `token_embd.weight` data pointer
   for the LM head computation.

### Implementation Strategy

```cpp
// In Model class weight loading:
const auto* output_tensor = parser.find_tensor("output.weight");
if (output_tensor) {
    // Explicit output weight — load normally
    load_tensor(lm_head_weight, *output_tensor);
} else {
    // Weight tying — reuse embedding data
    lm_head_weight = token_embedding_weight;  // share the same Tensor (non-owning view)
}
```

---

## 5. Forward Pass Data Flow

Reference for implementing the forward pass with the above tensor mapping.

```
Input token IDs: [batch, seq_len]
        |
        v
[token_embd.weight]  -- Embedding lookup --> hidden_states: [batch, seq_len, 576]
        |
        v
  +----- For each layer i = 0..29 -----+
  |                                      |
  |  [blk.{i}.attn_norm.weight]         |
  |      RMSNorm(hidden_states)          |
  |             |                        |
  |  [blk.{i}.attn_q.weight]  --> Q: [batch, seq, 9, 64]   |
  |  [blk.{i}.attn_k.weight]  --> K: [batch, seq, 3, 64]   |
  |  [blk.{i}.attn_v.weight]  --> V: [batch, seq, 3, 64]   |
  |             |                        |
  |      RoPE(Q, K, position)            |
  |             |                        |
  |      GQA Attention:                  |
  |        Q * K^T / sqrt(64) --> scores |
  |        causal_mask(scores)           |
  |        softmax(scores)               |
  |        scores * V --> attn_out       |
  |             |                        |
  |  [blk.{i}.attn_output.weight]       |
  |      Linear projection              |
  |             |                        |
  |      + hidden_states  (residual)     |
  |             |                        |
  |  [blk.{i}.ffn_norm.weight]          |
  |      RMSNorm(hidden_states)          |
  |             |                        |
  |  [blk.{i}.ffn_gate.weight] --> gate: [batch, seq, 1536] |
  |  [blk.{i}.ffn_up.weight]   --> up:   [batch, seq, 1536] |
  |      SiLU(gate) * up                 |
  |             |                        |
  |  [blk.{i}.ffn_down.weight] --> out:  [batch, seq, 576]  |
  |             |                        |
  |      + hidden_states  (residual)     |
  |                                      |
  +--------------------------------------+
        |
        v
[output_norm.weight]  -- RMSNorm --> normalized: [batch, seq, 576]
        |
        v
[output.weight]       -- Linear  --> logits: [batch, seq, 49152]
  (or token_embd.weight if tied)
```

---

## 6. Memory Budget Estimate (FP32)

Quick estimate for the RTX 3060 6 GB VRAM budget.

### Model Weights (FP32)

| Component                   | Parameters           | Bytes (FP32)  |
|-----------------------------|----------------------|---------------|
| Token embedding             | 49152 x 576 = 28.3M | 113.2 MB      |
| Per-layer attn (Q+K+V+O)   | (576x576 + 192x576 + 192x576 + 576x576) x 30 = 26.5M | 106.2 MB |
| Per-layer norms (attn+ffn)  | (576 + 576) x 30 = 34.6K | 0.1 MB   |
| Per-layer FFN (gate+up+down)| (1536x576 + 1536x576 + 576x1536) x 30 = 79.6M | 318.5 MB |
| Output norm                 | 576 = 576            | 0.002 MB      |
| LM head (if not tied)       | 49152 x 576 = 28.3M | 113.2 MB      |
| **Total (with tying)**      | **~134.5M params**   | **~538 MB**   |
| **Total (without tying)**   | **~162.8M params**   | **~651 MB**   |

### KV-Cache (FP32, max seq_len = 2048)

| Component                   | Size                                        | Bytes       |
|-----------------------------|---------------------------------------------|-------------|
| Per-layer K cache           | 2048 x 3 x 64 = 393K elements              | 1.5 MB      |
| Per-layer V cache           | 2048 x 3 x 64 = 393K elements              | 1.5 MB      |
| All layers                  | 30 x 2 x 1.5 MB                            | **90 MB**   |

### Total VRAM Estimate (FP32, seq_len=2048)

| Component        | Bytes    |
|------------------|----------|
| Weights (tied)   | ~538 MB  |
| KV-cache         | ~90 MB   |
| Activations      | ~20 MB   |
| **Total**        | **~648 MB** |

This fits comfortably within the 5.5 GB VRAM budget. Even at FP32 with full 8192
context, the KV-cache would only grow to ~360 MB, keeping total under ~920 MB.

With FP16 weights, all sizes halve, bringing the total to ~340 MB.

---

## 7. Implementation Notes

### No Bias Terms

SmolLM2-135M has `attention_bias = false`. None of the Q/K/V/O projections or FFN
projections have bias terms. The GGUF file will not contain any `.bias` tensors for
these layers. Only `.weight` tensors need to be loaded.

### RoPE Configuration

- `rope_theta = 100000.0` (higher than the original Llama's 10000.0)
- `rope_dimension_count = 64` (equals head_dim, so RoPE is applied to all dimensions)
- `rope_interleaved = false` (standard non-interleaved RoPE)
- `rope_scaling = null` (no NTK-aware or YaRN scaling)

The RoPE frequencies are precomputed as:
```
freq[j] = 1.0 / (theta ^ (2j / head_dim))   for j = 0, 1, ..., head_dim/2 - 1
```

### Normalization

- Pre-norm architecture: RMSNorm is applied **before** both attention and FFN
- RMSNorm epsilon: 1e-5
- Final output norm applied before the LM head projection

### FFN Structure (SwiGLU Variant)

The FFN uses a gated architecture (sometimes called SwiGLU):
```
FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
This requires three weight matrices per layer instead of the standard two.

### Quantization Considerations

When loading quantized GGUF files (Q8_0, Q4_K_M, etc.):
- Token embedding and output weights are typically quantized to Q8_0 even in
  lower-precision files
- RMSNorm weights are always stored as F32 (1D, small, quantization hurts quality)
- Attention and FFN weight matrices may be quantized to the file's target precision
- Dequantization must happen before or during matmul
