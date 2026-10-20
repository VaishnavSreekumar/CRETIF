// ── model.cu — TransformerModel Implementation ──────────────────────────
//
// Orchestrates the full transformer forward pass: embedding lookup,
// N transformer layers (RMSNorm → Attention → Residual → RMSNorm → FFN
// → Residual), final RMSNorm, and LM head projection.
//
// Weight loading strategy:
//   - GGUF stores 2D weights as [out_features, in_features] (row-major).
//   - matmul_tiled expects B as [K, N] = [in_features, out_features].
//   - Therefore, all 2D weight matrices (except embedding/LM head) are
//     transposed during loading via a host-side CPU transpose.
//   - Embedding weights are NOT transposed (used for row lookup).
//   - LM head uses matmul_transposed_b which handles [vocab_size, dim] directly.
//
// FP16 support: If the GGUF file contains F16 weights, they are
// dequantized to FP32 on the host during loading. This is acceptable
// for Phase 2 (FP32 inference path); native FP16 inference is Phase 3.

#include "cuinfer/model.h"
#include "cuinfer/attention.h"
#include "cuinfer/cuda_utils.h"
#include "cuinfer/embedding.h"
#include "cuinfer/kernels/elementwise.h"
#include "cuinfer/kernels/matmul.h"
#include "cuinfer/kernels/matmul_quant.h"
#include "cuinfer/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// FP16 type for dequantization (CUDA's __half is defined in cuda_fp16.h)
#include <cuda_fp16.h>

namespace cuinfer {

// ════════════════════════════════════════════════════════════════
// Helper: Host-side FP16 → FP32 dequantization
// ════════════════════════════════════════════════════════════════

static std::vector<float> dequantize_f16_to_f32(const void* f16_data,
                                                  size_t n_elements) {
    const __half* src = static_cast<const __half*>(f16_data);
    std::vector<float> dst(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
        dst[i] = __half2float(src[i]);
    }
    return dst;
}

// ════════════════════════════════════════════════════════════════
// Helper: Host-side matrix transpose [rows, cols] → [cols, rows]
// ════════════════════════════════════════════════════════════════

static std::vector<float> transpose_host(const float* src,
                                          int rows, int cols) {
    std::vector<float> dst(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[static_cast<size_t>(c) * rows + r] =
                src[static_cast<size_t>(r) * cols + c];
        }
    }
    return dst;
}

// ════════════════════════════════════════════════════════════════
// Weight Loading
// ════════════════════════════════════════════════════════════════

Tensor TransformerModel::load_weight_transposed(const GGUFParser& parser,
                                                 const std::string& name) {
    const auto* info = parser.find_tensor(name);
    if (!info) {
        throw std::runtime_error("TransformerModel: tensor '" + name +
                                 "' not found in GGUF file");
    }

    if (info->dimensions.size() != 2) {
        throw std::runtime_error("TransformerModel: expected 2D tensor for '" +
                                 name + "', got " +
                                 std::to_string(info->dimensions.size()) + "D");
    }

    // GGUF/ggml dimension convention: dims[0] = inner/columns, dims[1] = outer/rows.
    // So for a weight [out_features, in_features] in memory:
    //   dims[0] = in_features (columns), dims[1] = out_features (rows)
    const int in_feat  = static_cast<int>(info->dimensions[0]);
    const int out_feat = static_cast<int>(info->dimensions[1]);
    const size_t n_elements = static_cast<size_t>(out_feat) * in_feat;

    const void* raw_data = parser.tensor_data(*info);

    // Get FP32 host data (dequantize if F16)
    std::vector<float> f32_data;
    const float* host_ptr = nullptr;

    if (info->type == GGMLType::F32) {
        host_ptr = static_cast<const float*>(raw_data);
    } else if (info->type == GGMLType::F16) {
        f32_data = dequantize_f16_to_f32(raw_data, n_elements);
        host_ptr = f32_data.data();
    } else if (info->type == GGMLType::Q8_0 || info->type == GGMLType::Q4_0 ||
               info->type == GGMLType::Q5_0 || info->type == GGMLType::Q4_K ||
               info->type == GGMLType::Q6_K) {
        // Dequantize on GPU, then copy back to host for transpose
        const size_t block_elems = ggml_block_size(info->type);
        const size_t block_bytes = ggml_type_size(info->type);
        const int n_blocks = static_cast<int>(n_elements / block_elems);
        const size_t total_bytes = static_cast<size_t>(n_blocks) * block_bytes;

        void* d_quant = nullptr;
        float* d_fp32 = nullptr;
        CUINFER_CHECK_CUDA(cudaMalloc(&d_quant, total_bytes));
        CUINFER_CHECK_CUDA(cudaMalloc(&d_fp32, n_elements * sizeof(float)));
        CUINFER_CHECK_CUDA(cudaMemcpy(d_quant, raw_data, total_bytes,
                                      cudaMemcpyHostToDevice));

        if (info->type == GGMLType::Q8_0) dequantize_q8_0(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q4_0) dequantize_q4_0(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q5_0) dequantize_q5_0(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q4_K) dequantize_q4_k(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q6_K) dequantize_q6_k(d_quant, d_fp32, n_blocks);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        f32_data.resize(n_elements);
        CUINFER_CHECK_CUDA(cudaMemcpy(f32_data.data(), d_fp32,
                                      n_elements * sizeof(float),
                                      cudaMemcpyDeviceToHost));
        cudaFree(d_quant);
        cudaFree(d_fp32);
        host_ptr = f32_data.data();
    } else {
        throw std::runtime_error(
            "TransformerModel: unsupported tensor type for '" + name +
            "': " + ggml_type_name(info->type));
    }

    // Transpose [out_feat, in_feat] → [in_feat, out_feat]
    std::vector<float> transposed = transpose_host(host_ptr, out_feat, in_feat);

    return Tensor::from_host(transposed.data(), {in_feat, out_feat});
}

Tensor TransformerModel::load_weight_1d(const GGUFParser& parser,
                                         const std::string& name) {
    const auto* info = parser.find_tensor(name);
    if (!info) {
        throw std::runtime_error("TransformerModel: tensor '" + name +
                                 "' not found in GGUF file");
    }

    const size_t n_elements = info->element_count();
    const void* raw_data = parser.tensor_data(*info);

    if (info->type == GGMLType::F32) {
        return Tensor::from_host(raw_data,
                                 {static_cast<int64_t>(n_elements)});
    } else if (info->type == GGMLType::F16) {
        auto f32_data = dequantize_f16_to_f32(raw_data, n_elements);
        return Tensor::from_host(f32_data.data(),
                                 {static_cast<int64_t>(n_elements)});
    }

    // Quantized 1D weights: upload to GPU, then dequantize to FP32.
    // Norm weights are small (dim-sized), so dequant cost is negligible.
    if (info->type == GGMLType::Q8_0 || info->type == GGMLType::Q4_0 ||
        info->type == GGMLType::Q5_0 || info->type == GGMLType::Q4_K ||
        info->type == GGMLType::Q6_K) {

        const size_t block_elems = ggml_block_size(info->type);
        const size_t block_bytes = ggml_type_size(info->type);
        const int n_blocks = static_cast<int>(n_elements / block_elems);
        const size_t total_bytes = static_cast<size_t>(n_blocks) * block_bytes;

        // Upload quantized data to GPU temp buffer
        void* d_quant = nullptr;
        CUINFER_CHECK_CUDA(cudaMalloc(&d_quant, total_bytes));
        CUINFER_CHECK_CUDA(cudaMemcpy(d_quant, raw_data, total_bytes,
                                      cudaMemcpyHostToDevice));

        // Allocate FP32 output
        Tensor result({static_cast<int64_t>(n_elements)}, DType::FP32);

        // Dequantize on GPU
        if (info->type == GGMLType::Q8_0) {
            dequantize_q8_0(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q4_0) {
            dequantize_q4_0(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q5_0) {
            dequantize_q5_0(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q4_K) {
            dequantize_q4_k(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q6_K) {
            dequantize_q6_k(d_quant, result.data_as<float>(), n_blocks);
        }
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_quant);
        return result;
    }

    throw std::runtime_error(
        "TransformerModel: unsupported type for 1D tensor '" + name +
        "': " + ggml_type_name(info->type));
}

Tensor TransformerModel::load_weight_2d(const GGUFParser& parser,
                                         const std::string& name) {
    const auto* info = parser.find_tensor(name);
    if (!info) {
        throw std::runtime_error("TransformerModel: tensor '" + name +
                                 "' not found in GGUF file");
    }

    if (info->dimensions.size() != 2) {
        throw std::runtime_error("TransformerModel: expected 2D tensor for '" +
                                 name + "', got " +
                                 std::to_string(info->dimensions.size()) + "D");
    }

    // GGUF/ggml convention: dims[0] = inner/columns, dims[1] = outer/rows.
    const int cols = static_cast<int>(info->dimensions[0]);
    const int rows = static_cast<int>(info->dimensions[1]);
    const size_t n_elements = static_cast<size_t>(rows) * cols;
    const void* raw_data = parser.tensor_data(*info);

    if (info->type == GGMLType::F32) {
        return Tensor::from_host(raw_data, {rows, cols});
    } else if (info->type == GGMLType::F16) {
        auto f32_data = dequantize_f16_to_f32(raw_data, n_elements);
        return Tensor::from_host(f32_data.data(), {rows, cols});
    }

    // Quantized 2D weights (embedding table): dequantize to FP32 on GPU.
    // Embedding tables need FP32 for row-lookup access.
    if (info->type == GGMLType::Q8_0 || info->type == GGMLType::Q4_0 ||
        info->type == GGMLType::Q5_0 || info->type == GGMLType::Q4_K ||
        info->type == GGMLType::Q6_K) {

        const size_t block_elems = ggml_block_size(info->type);
        const size_t block_bytes_sz = ggml_type_size(info->type);
        const int n_blocks = static_cast<int>(n_elements / block_elems);
        const size_t total_bytes = static_cast<size_t>(n_blocks) * block_bytes_sz;

        void* d_quant = nullptr;
        CUINFER_CHECK_CUDA(cudaMalloc(&d_quant, total_bytes));
        CUINFER_CHECK_CUDA(cudaMemcpy(d_quant, raw_data, total_bytes,
                                      cudaMemcpyHostToDevice));

        Tensor result({rows, cols}, DType::FP32);
        if (info->type == GGMLType::Q8_0) {
            dequantize_q8_0(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q4_0) {
            dequantize_q4_0(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q5_0) {
            dequantize_q5_0(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q4_K) {
            dequantize_q4_k(d_quant, result.data_as<float>(), n_blocks);
        } else if (info->type == GGMLType::Q6_K) {
            dequantize_q6_k(d_quant, result.data_as<float>(), n_blocks);
        }
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_quant);
        return result;
    }

    throw std::runtime_error(
        "TransformerModel: unsupported type for 2D tensor '" + name +
        "': " + ggml_type_name(info->type));
}

// ════════════════════════════════════════════════════════════════
// Quantized Weight Loading
// ════════════════════════════════════════════════════════════════

/// Converts a GGML type to our WeightFormat enum.
static WeightFormat ggml_type_to_weight_format(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return WeightFormat::FP32;
        case GGMLType::F16:  return WeightFormat::FP16;
        case GGMLType::Q8_0: return WeightFormat::Q8_0;
        case GGMLType::Q4_0: return WeightFormat::Q4_0;
        case GGMLType::Q5_0: return WeightFormat::Q5_0;
        case GGMLType::Q4_K: return WeightFormat::Q4_K;
        case GGMLType::Q6_K: return WeightFormat::Q6_K;
        default: return WeightFormat::FP32;
    }
}

/// Converts our DType to WeightFormat.
static DType weight_format_to_dtype(WeightFormat fmt) {
    switch (fmt) {
        case WeightFormat::FP32: return DType::FP32;
        case WeightFormat::FP16: return DType::FP16;
        case WeightFormat::Q8_0: return DType::Q8_0;
        case WeightFormat::Q4_0: return DType::Q4_0;
        case WeightFormat::Q5_0: return DType::Q5_0;
        case WeightFormat::Q4_K: return DType::Q4_K;
        case WeightFormat::Q6_K: return DType::Q6_K;
    }
    return DType::FP32;
}

WeightFormat TransformerModel::detect_weight_format(const GGUFParser& parser) {
    // Look at the first layer's attention Q weight to determine the format.
    const auto* info = parser.find_tensor("blk.0.attn_q.weight");
    if (!info) {
        // Fallback: look for any tensor to determine format
        const auto& tensors = parser.tensors();
        for (const auto& t : tensors) {
            if (t.dimensions.size() == 2) {
                return ggml_type_to_weight_format(t.type);
            }
        }
        return WeightFormat::FP32;
    }
    return ggml_type_to_weight_format(info->type);
}

Tensor TransformerModel::load_weight_quantized(const GGUFParser& parser,
                                                const std::string& name,
                                                WeightFormat& out_format) {
    const auto* info = parser.find_tensor(name);
    if (!info) {
        throw std::runtime_error("TransformerModel: tensor '" + name +
                                 "' not found in GGUF file");
    }

    if (info->dimensions.size() != 2) {
        throw std::runtime_error("TransformerModel: expected 2D tensor for '" +
                                 name + "', got " +
                                 std::to_string(info->dimensions.size()) + "D");
    }

    // Detect per-tensor format (Q4_K_M files mix Q4_K and Q6_K)
    out_format = ggml_type_to_weight_format(info->type);

    // FP32: fall back to transposed loading (Phase 2 path)
    if (info->type == GGMLType::F32) {
        out_format = WeightFormat::FP32;
        return load_weight_transposed(parser, name);
    }

    // GGUF dims: dims[0] = inner (K), dims[1] = outer (N)
    const int K = static_cast<int>(info->dimensions[0]);
    const int N = static_cast<int>(info->dimensions[1]);
    const size_t n_elements = static_cast<size_t>(N) * K;

    // FP16: upload raw __half bytes to GPU (2 bytes per element).
    // No dequant or transpose — the FP16 matmul kernel handles [N,K] layout.
    if (info->type == GGMLType::F16) {
        out_format = WeightFormat::FP16;
        const size_t fp16_bytes = n_elements * sizeof(__half);
        const void* raw_data = parser.tensor_data(*info);

        Tensor t({static_cast<int64_t>(N), static_cast<int64_t>(K)}, DType::FP16);
        CUINFER_CHECK_CUDA(cudaMemcpy(t.data(), raw_data, fp16_bytes,
                                      cudaMemcpyHostToDevice));
        return t;
    }

    // Compute byte size from GGML type info
    const size_t block_elems = ggml_block_size(info->type);
    const size_t block_bytes = ggml_type_size(info->type);
    const size_t total_bytes = (n_elements / block_elems) * block_bytes;

    const void* raw_data = parser.tensor_data(*info);

#ifdef CUINFER_DEBUG_FORCE_FP32
    // Debug: dequantize to FP32 + transpose (bypass quantized matmul).
    {
        const int n_blocks = static_cast<int>(n_elements / block_elems);

        void* d_quant = nullptr;
        float* d_fp32 = nullptr;
        CUINFER_CHECK_CUDA(cudaMalloc(&d_quant, total_bytes));
        CUINFER_CHECK_CUDA(cudaMalloc(&d_fp32, n_elements * sizeof(float)));
        CUINFER_CHECK_CUDA(cudaMemcpy(d_quant, raw_data, total_bytes,
                                      cudaMemcpyHostToDevice));

        if (info->type == GGMLType::Q8_0) dequantize_q8_0(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q4_0) dequantize_q4_0(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q5_0) dequantize_q5_0(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q4_K) dequantize_q4_k(d_quant, d_fp32, n_blocks);
        else if (info->type == GGMLType::Q6_K) dequantize_q6_k(d_quant, d_fp32, n_blocks);
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> fp32_data(n_elements);
        CUINFER_CHECK_CUDA(cudaMemcpy(fp32_data.data(), d_fp32,
                                      n_elements * sizeof(float),
                                      cudaMemcpyDeviceToHost));
        cudaFree(d_quant);
        cudaFree(d_fp32);

        auto transposed = transpose_host(fp32_data.data(), N, K);
        out_format = WeightFormat::FP32;
        return Tensor::from_host(transposed.data(),
                                 {static_cast<int64_t>(K),
                                  static_cast<int64_t>(N)});
    }
#endif

    // Upload raw quantized bytes to GPU.
    // Shape is set to [N, K] for logical tracking, but the tensor stores
    // raw bytes — size_bytes() uses the block-quantized formula.
    DType dtype = weight_format_to_dtype(out_format);
    Tensor t({static_cast<int64_t>(N), static_cast<int64_t>(K)}, dtype);
    CUINFER_CHECK_CUDA(cudaMemcpy(t.data(), raw_data, total_bytes,
                                  cudaMemcpyHostToDevice));

    return t;
}

// ════════════════════════════════════════════════════════════════
// Tokenizer Construction from GGUF Metadata
// ════════════════════════════════════════════════════════════════

std::unique_ptr<Tokenizer> TransformerModel::build_tokenizer(
    const GGUFParser& parser) {

    const auto& meta = parser.metadata();

    // Extract vocabulary: tokenizer.ggml.tokens (array of strings)
    auto it_tokens = meta.find("tokenizer.ggml.tokens");
    if (it_tokens == meta.end()) {
        throw std::runtime_error(
            "TransformerModel: GGUF missing 'tokenizer.ggml.tokens'");
    }
    const auto& token_array = it_tokens->second.as_array();

    std::vector<std::string> vocab;
    vocab.reserve(token_array.size());
    for (const auto& val : token_array) {
        vocab.push_back(val.as_string());
    }

    // Extract scores: tokenizer.ggml.scores (array of float32)
    std::vector<float> scores;
    auto it_scores = meta.find("tokenizer.ggml.scores");
    if (it_scores != meta.end()) {
        const auto& score_array = it_scores->second.as_array();
        scores.reserve(score_array.size());
        for (const auto& val : score_array) {
            scores.push_back(val.as_float32());
        }
    } else {
        // Default scores: all zeros (merge priority not defined)
        scores.resize(vocab.size(), 0.0f);
    }

    // Extract token types: tokenizer.ggml.token_type (array of int32)
    std::vector<TokenType> token_types;
    auto it_types = meta.find("tokenizer.ggml.token_type");
    if (it_types != meta.end()) {
        const auto& type_array = it_types->second.as_array();
        token_types.reserve(type_array.size());
        for (const auto& val : type_array) {
            token_types.push_back(
                static_cast<TokenType>(val.as_int32()));
        }
    } else {
        token_types.resize(vocab.size(), TokenType::NORMAL);
    }

    // Build tokenizer config
    TokenizerConfig config;
    auto it_model = meta.find("tokenizer.ggml.model");
    if (it_model != meta.end()) {
        config.model_type = it_model->second.as_string();
    }
    if (parser.has_metadata("tokenizer.ggml.bos_token_id")) {
        config.bos_token_id = parser.get_uint32("tokenizer.ggml.bos_token_id");
    }
    if (parser.has_metadata("tokenizer.ggml.eos_token_id")) {
        config.eos_token_id = parser.get_uint32("tokenizer.ggml.eos_token_id");
    }
    if (parser.has_metadata("tokenizer.ggml.padding_token_id")) {
        config.pad_token_id = parser.get_uint32("tokenizer.ggml.padding_token_id");
    }
    if (parser.has_metadata("tokenizer.ggml.add_bos_token")) {
        // GGUF stores this as a bool
        config.add_bos_token =
            meta.at("tokenizer.ggml.add_bos_token").as_bool();
    }
    if (parser.has_metadata("tokenizer.ggml.add_eos_token")) {
        config.add_eos_token =
            meta.at("tokenizer.ggml.add_eos_token").as_bool();
    }

    // Check if GPT2-style merges are available
    auto it_merges = meta.find("tokenizer.ggml.merges");
    if (it_merges != meta.end() &&
        (config.model_type == "gpt2" || config.model_type.empty())) {
        // GPT2-style: use explicit merge list
        const auto& merge_array = it_merges->second.as_array();
        std::vector<std::pair<std::string, std::string>> merges;
        merges.reserve(merge_array.size());
        for (const auto& val : merge_array) {
            const auto& s = val.as_string();
            auto space_pos = s.find(' ');
            if (space_pos != std::string::npos) {
                merges.emplace_back(s.substr(0, space_pos),
                                    s.substr(space_pos + 1));
            }
        }
        return std::make_unique<Tokenizer>(
            std::move(vocab), std::move(merges),
            std::move(token_types), config);
    }

    // SentencePiece-style: use scores
    return std::make_unique<Tokenizer>(
        std::move(vocab), std::move(scores),
        std::move(token_types), config);
}

// ════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════

TransformerModel::TransformerModel(const std::string& gguf_path,
                                   int max_seq_len)
    : config_{}
    , embedding_(1, 1)   // placeholder, will be reassigned
    , lm_head_(1, 1)     // placeholder, will be reassigned
    , attention_(AttentionConfig{1, 1, 2, 1})  // placeholder (head_dim=2 to pass even check)
    , ffn_(FFNConfig{1, 1}, 1)                 // placeholder
{
    printf("Loading model from: %s\n", gguf_path.c_str());

    GGUFParser parser(gguf_path);

    // ── Extract model configuration from GGUF metadata ──────
    config_.architecture    = parser.architecture();
    config_.n_layers        = static_cast<int>(parser.block_count());
    config_.dim             = static_cast<int>(parser.embedding_length());
    config_.n_heads         = static_cast<int>(parser.head_count());
    config_.n_kv_heads      = static_cast<int>(parser.head_count_kv());
    config_.head_dim        = config_.dim / config_.n_heads;
    config_.intermediate_size = static_cast<int>(parser.feed_forward_length());
    config_.vocab_size      = static_cast<int>(parser.vocab_size());
    config_.max_seq_len     = std::min(max_seq_len,
                                        static_cast<int>(parser.context_length()));
    config_.rms_norm_eps    = parser.rms_norm_eps();
    config_.rope_theta      = parser.rope_freq_base();
    config_.tie_embeddings  = parser.tie_word_embeddings();

    printf("  Architecture:    %s\n", config_.architecture.c_str());
    printf("  Layers:          %d\n", config_.n_layers);
    printf("  Dim:             %d\n", config_.dim);
    printf("  Heads:           %d (KV: %d)\n", config_.n_heads, config_.n_kv_heads);
    printf("  Head dim:        %d\n", config_.head_dim);
    printf("  FFN hidden:      %d\n", config_.intermediate_size);
    printf("  Vocab:           %d\n", config_.vocab_size);
    printf("  Max seq len:     %d\n", config_.max_seq_len);
    printf("  RoPE theta:      %.1f\n", config_.rope_theta);
    printf("  Tied embeddings: %s\n", config_.tie_embeddings ? "yes" : "no");

    // ── Detect weight format ────────────────────────────────
    config_.weight_format = detect_weight_format(parser);
    printf("  Weight format:   %s\n", weight_format_name(config_.weight_format));

#ifdef CUINFER_DEBUG_FORCE_FP32
    const bool use_quantized = false;
#else
    const bool use_quantized = (config_.weight_format != WeightFormat::FP32);
#endif

    // ── Build tokenizer ─────────────────────────────────────
    printf("  Building tokenizer...\n");
    tokenizer_ = build_tokenizer(parser);

    // ── Initialize components ───────────────────────────────
    AttentionConfig attn_config;
    attn_config.n_heads     = config_.n_heads;
    attn_config.n_kv_heads  = config_.n_kv_heads;
    attn_config.head_dim    = config_.head_dim;
    attn_config.max_seq_len = config_.max_seq_len;
    attn_config.rope_theta  = config_.rope_theta;
    attention_ = AttentionBlock(attn_config);

    embedding_ = Embedding(config_.vocab_size, config_.dim);
    lm_head_   = LMHead(config_.vocab_size, config_.dim);

    FFNConfig ffn_config;
    ffn_config.dim              = config_.dim;
    ffn_config.intermediate_size = config_.intermediate_size;
    ffn_ = GatedFFN(ffn_config, config_.max_seq_len);

    KVCacheConfig kv_config;
    kv_config.n_layers    = config_.n_layers;
    kv_config.max_seq_len = config_.max_seq_len;
    kv_config.n_kv_heads  = config_.n_kv_heads;
    kv_config.head_dim    = config_.head_dim;
    kv_cache_ = std::make_unique<KVCache>(kv_config);

    // ── Load weights ────────────────────────────────────────
    printf("  Loading weights...\n");

    // Global tensors — embedding always dequantized to FP32 (row lookup pattern)
    token_embd_weight_ = load_weight_2d(parser, "token_embd.weight");
    output_norm_weight_ = load_weight_1d(parser, "output_norm.weight");

    // LM head: check for weight tying
    if (!config_.tie_embeddings) {
        if (use_quantized) {
            lm_head_weight_ = load_weight_quantized(parser, "output.weight", lm_head_format_);
        } else {
            lm_head_weight_ = load_weight_2d(parser, "output.weight");
        }
    }

    // Per-layer weights
    attn_norm_weight_.reserve(config_.n_layers);
    attn_q_weight_.reserve(config_.n_layers);
    attn_k_weight_.reserve(config_.n_layers);
    attn_v_weight_.reserve(config_.n_layers);
    attn_o_weight_.reserve(config_.n_layers);
    ffn_norm_weight_.reserve(config_.n_layers);
    ffn_gate_weight_.reserve(config_.n_layers);
    ffn_up_weight_.reserve(config_.n_layers);
    ffn_down_weight_.reserve(config_.n_layers);

    attn_q_format_.reserve(config_.n_layers);
    attn_k_format_.reserve(config_.n_layers);
    attn_v_format_.reserve(config_.n_layers);
    attn_o_format_.reserve(config_.n_layers);
    ffn_gate_format_.reserve(config_.n_layers);
    ffn_up_format_.reserve(config_.n_layers);
    ffn_down_format_.reserve(config_.n_layers);

    for (int i = 0; i < config_.n_layers; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";

        // Norm weights — always FP32
        attn_norm_weight_.push_back(
            load_weight_1d(parser, prefix + "attn_norm.weight"));
        ffn_norm_weight_.push_back(
            load_weight_1d(parser, prefix + "ffn_norm.weight"));

        if (use_quantized) {
            // Quantized path: upload raw bytes, no transposition
            WeightFormat fmt;

            attn_q_weight_.push_back(load_weight_quantized(parser, prefix + "attn_q.weight", fmt));
            attn_q_format_.push_back(fmt);

            attn_k_weight_.push_back(load_weight_quantized(parser, prefix + "attn_k.weight", fmt));
            attn_k_format_.push_back(fmt);

            attn_v_weight_.push_back(load_weight_quantized(parser, prefix + "attn_v.weight", fmt));
            attn_v_format_.push_back(fmt);

            attn_o_weight_.push_back(load_weight_quantized(parser, prefix + "attn_output.weight", fmt));
            attn_o_format_.push_back(fmt);

            ffn_gate_weight_.push_back(load_weight_quantized(parser, prefix + "ffn_gate.weight", fmt));
            ffn_gate_format_.push_back(fmt);

            ffn_up_weight_.push_back(load_weight_quantized(parser, prefix + "ffn_up.weight", fmt));
            ffn_up_format_.push_back(fmt);

            ffn_down_weight_.push_back(load_weight_quantized(parser, prefix + "ffn_down.weight", fmt));
            ffn_down_format_.push_back(fmt);
        } else {
            // FP32 path: host transpose as before
            attn_q_weight_.push_back(
                load_weight_transposed(parser, prefix + "attn_q.weight"));
            attn_q_format_.push_back(WeightFormat::FP32);

            attn_k_weight_.push_back(
                load_weight_transposed(parser, prefix + "attn_k.weight"));
            attn_k_format_.push_back(WeightFormat::FP32);

            attn_v_weight_.push_back(
                load_weight_transposed(parser, prefix + "attn_v.weight"));
            attn_v_format_.push_back(WeightFormat::FP32);

            attn_o_weight_.push_back(
                load_weight_transposed(parser, prefix + "attn_output.weight"));
            attn_o_format_.push_back(WeightFormat::FP32);

            ffn_gate_weight_.push_back(
                load_weight_transposed(parser, prefix + "ffn_gate.weight"));
            ffn_gate_format_.push_back(WeightFormat::FP32);

            ffn_up_weight_.push_back(
                load_weight_transposed(parser, prefix + "ffn_up.weight"));
            ffn_up_format_.push_back(WeightFormat::FP32);

            ffn_down_weight_.push_back(
                load_weight_transposed(parser, prefix + "ffn_down.weight"));
            ffn_down_format_.push_back(WeightFormat::FP32);
        }

        if (i == 0 && use_quantized) {
            printf("    Layer 0 formats: Q=%s K=%s V=%s O=%s gate=%s up=%s down=%s\n",
                   weight_format_name(attn_q_format_[0]),
                   weight_format_name(attn_k_format_[0]),
                   weight_format_name(attn_v_format_[0]),
                   weight_format_name(attn_o_format_[0]),
                   weight_format_name(ffn_gate_format_[0]),
                   weight_format_name(ffn_up_format_[0]),
                   weight_format_name(ffn_down_format_[0]));
        }
        if ((i + 1) % 10 == 0 || i == config_.n_layers - 1) {
            printf("    Loaded layer %d/%d\n", i + 1, config_.n_layers);
        }
    }

    // ── Pre-allocate activation buffers ─────────────────────
    normed_buf_     = Tensor::empty({config_.max_seq_len, config_.dim});
    ffn_output_buf_ = Tensor::empty({config_.max_seq_len, config_.dim});

    CUINFER_CHECK_CUDA(cudaMalloc(&token_ids_gpu_,
                                   config_.max_seq_len * sizeof(int32_t)));

    printf("  Model loaded successfully.\n");
    print_memory_report();
}

TransformerModel::~TransformerModel() {
    if (token_ids_gpu_) {
        cudaFree(token_ids_gpu_);
        token_ids_gpu_ = nullptr;
    }
}

// Move constructor
TransformerModel::TransformerModel(TransformerModel&& other) noexcept
    : config_(std::move(other.config_))
    , tokenizer_(std::move(other.tokenizer_))
    , kv_cache_(std::move(other.kv_cache_))
    , embedding_(std::move(other.embedding_))
    , lm_head_(std::move(other.lm_head_))
    , attention_(std::move(other.attention_))
    , ffn_(std::move(other.ffn_))
    , token_embd_weight_(std::move(other.token_embd_weight_))
    , output_norm_weight_(std::move(other.output_norm_weight_))
    , lm_head_weight_(std::move(other.lm_head_weight_))
    , attn_norm_weight_(std::move(other.attn_norm_weight_))
    , attn_q_weight_(std::move(other.attn_q_weight_))
    , attn_k_weight_(std::move(other.attn_k_weight_))
    , attn_v_weight_(std::move(other.attn_v_weight_))
    , attn_o_weight_(std::move(other.attn_o_weight_))
    , ffn_norm_weight_(std::move(other.ffn_norm_weight_))
    , ffn_gate_weight_(std::move(other.ffn_gate_weight_))
    , ffn_up_weight_(std::move(other.ffn_up_weight_))
    , ffn_down_weight_(std::move(other.ffn_down_weight_))
    , normed_buf_(std::move(other.normed_buf_))
    , ffn_output_buf_(std::move(other.ffn_output_buf_))
    , token_ids_gpu_(other.token_ids_gpu_)
{
    other.token_ids_gpu_ = nullptr;
}

TransformerModel& TransformerModel::operator=(TransformerModel&& other) noexcept {
    if (this != &other) {
        if (token_ids_gpu_) cudaFree(token_ids_gpu_);

        config_             = std::move(other.config_);
        tokenizer_          = std::move(other.tokenizer_);
        kv_cache_           = std::move(other.kv_cache_);
        embedding_          = std::move(other.embedding_);
        lm_head_            = std::move(other.lm_head_);
        attention_          = std::move(other.attention_);
        ffn_                = std::move(other.ffn_);
        token_embd_weight_  = std::move(other.token_embd_weight_);
        output_norm_weight_ = std::move(other.output_norm_weight_);
        lm_head_weight_     = std::move(other.lm_head_weight_);
        attn_norm_weight_   = std::move(other.attn_norm_weight_);
        attn_q_weight_      = std::move(other.attn_q_weight_);
        attn_k_weight_      = std::move(other.attn_k_weight_);
        attn_v_weight_      = std::move(other.attn_v_weight_);
        attn_o_weight_      = std::move(other.attn_o_weight_);
        ffn_norm_weight_    = std::move(other.ffn_norm_weight_);
        ffn_gate_weight_    = std::move(other.ffn_gate_weight_);
        ffn_up_weight_      = std::move(other.ffn_up_weight_);
        ffn_down_weight_    = std::move(other.ffn_down_weight_);
        normed_buf_         = std::move(other.normed_buf_);
        ffn_output_buf_     = std::move(other.ffn_output_buf_);
        token_ids_gpu_      = other.token_ids_gpu_;
        other.token_ids_gpu_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════
// Matmul Dispatch — routes to FP32 or quantized matmul kernel
// ════════════════════════════════════════════════════════════════

/// Dispatches a matmul based on the weight format.
/// For FP32: C = A * B where B is [K, N] (already transposed on host).
/// For quantized: C = A * dequant(B)^T where B is [N, K] (GGUF native).
static void matmul_dispatch(const float* A, const Tensor& weight,
                            float* C, int M, int N, int K,
                            WeightFormat format,
                            cudaStream_t stream = 0) {
    switch (format) {
        case WeightFormat::FP32:
            // FP32 weights are pre-transposed to [K, N] during loading
            matmul_tiled(A, weight.data_as<float>(), C, M, N, K, stream);
            break;
        case WeightFormat::FP16:
            matmul_fp16(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q8_0:
            matmul_q8_0(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q4_0:
            matmul_q4_0(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q5_0:
            matmul_q5_0(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q4_K:
            matmul_q4_k(A, weight.data(), C, M, N, K, stream);
            break;
        case WeightFormat::Q6_K:
            matmul_q6_k(A, weight.data(), C, M, N, K, stream);
            break;
    }
}

// ════════════════════════════════════════════════════════════════
// Forward Pass
// ════════════════════════════════════════════════════════════════

Tensor TransformerModel::forward(const std::vector<int32_t>& token_ids) {
    const int seq_len = static_cast<int>(token_ids.size());
    if (seq_len <= 0) {
        throw std::invalid_argument("TransformerModel::forward: empty token_ids");
    }
    if (seq_len > config_.max_seq_len) {
        throw std::invalid_argument(
            "TransformerModel::forward: seq_len (" +
            std::to_string(seq_len) + ") exceeds max_seq_len (" +
            std::to_string(config_.max_seq_len) + ")");
    }

    const int pos = kv_cache_->current_length();
    if (pos + seq_len > config_.max_seq_len) {
        throw std::overflow_error(
            "TransformerModel::forward: would exceed max sequence length ("
            + std::to_string(pos + seq_len) + " > "
            + std::to_string(config_.max_seq_len) + ")");
    }

    const int dim = config_.dim;

    // Copy token IDs to device
    CUINFER_CHECK_CUDA(cudaMemcpy(
        token_ids_gpu_, token_ids.data(),
        seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    // ── Embedding lookup ────────────────────────────────────
    Tensor hidden = embedding_.forward(
        token_ids_gpu_, seq_len, token_embd_weight_);

    // ── Transformer layers ──────────────────────────────────
#ifdef CUINFER_DEBUG_FORCE_FP32
    const bool use_quantized = false;
#else
    const bool use_quantized = (config_.weight_format != WeightFormat::FP32);
#endif

    // ── Fused forward loop ─────────────────────────────────────
    //
    // Uses rmsnorm_residual() to fuse each residual add with the
    // following RMSNorm into a single kernel launch. This saves
    // 2 kernel launches and 2 global memory round-trips per layer.
    //
    // Structure:
    //   - First layer's pre-attention RMSNorm is standalone.
    //   - Post-attention: fused add + pre-FFN RMSNorm.
    //   - Post-FFN: fused add + next layer's pre-attention RMSNorm
    //     (or the final output norm for the last layer).

    // Standalone pre-attention RMSNorm for the first layer
    rmsnorm(hidden.data_as<float>(),
            attn_norm_weight_[0].data_as<float>(),
            normed_buf_.data_as<float>(),
            seq_len, dim, config_.rms_norm_eps);

    for (int i = 0; i < config_.n_layers; ++i) {

        // normed_buf_ already contains RMSNorm(hidden, attn_norm[i])
        Tensor normed_view = normed_buf_.view(2, 0, seq_len);

        Tensor attn_out;
        if (!use_quantized) {
            // FP32 path: use existing AttentionBlock (weights are pre-transposed)
            attn_out = attention_.forward(
                normed_view,
                attn_q_weight_[i], attn_k_weight_[i],
                attn_v_weight_[i], attn_o_weight_[i],
                *kv_cache_, i, pos);
        } else {
            // Quantized path: do projections with matmul_dispatch,
            // then handle RoPE, KV-cache, and attention in FP32.
            const int kv_dim = config_.n_kv_heads * config_.head_dim;

            // QKV projections using quantized matmul
            Tensor q_proj = Tensor::zeros({seq_len, (int64_t)dim});
            Tensor k_proj = Tensor::zeros({seq_len, (int64_t)kv_dim});
            Tensor v_proj = Tensor::zeros({seq_len, (int64_t)kv_dim});

            matmul_dispatch(normed_view.data_as<float>(), attn_q_weight_[i],
                           q_proj.data_as<float>(), seq_len, dim, dim,
                           attn_q_format_[i]);
            matmul_dispatch(normed_view.data_as<float>(), attn_k_weight_[i],
                           k_proj.data_as<float>(), seq_len, kv_dim, dim,
                           attn_k_format_[i]);
            matmul_dispatch(normed_view.data_as<float>(), attn_v_weight_[i],
                           v_proj.data_as<float>(), seq_len, kv_dim, dim,
                           attn_v_format_[i]);

            // Apply RoPE to Q and K
            for (int t = 0; t < seq_len; ++t) {
                rope(q_proj.data_as<float>() + t * dim, pos + t,
                     config_.head_dim, config_.n_heads, config_.rope_theta);
                rope(k_proj.data_as<float>() + t * kv_dim, pos + t,
                     config_.head_dim, config_.n_kv_heads, config_.rope_theta);
            }

            if (seq_len == 1) {
                // Fused decode attention: single kernel replaces 9+ kernels.
                // Append K/V to cache manually (forward_from_qkv not called).
                kv_cache_->append(i, k_proj.data_as<float>(),
                                  v_proj.data_as<float>(), 1);
                const int kv_len = pos + 1;
                const float attn_scale = 1.0f / std::sqrt(
                    static_cast<float>(config_.head_dim));

                Tensor attn_concat = Tensor::zeros({1, (int64_t)dim});
                fused_decode_attention(
                    q_proj.data_as<float>(),
                    kv_cache_->k_cache(i), kv_cache_->v_cache(i),
                    attn_concat.data_as<float>(),
                    config_.n_heads, config_.n_kv_heads,
                    config_.head_dim, kv_len, attn_scale);

                // Output projection (Wo)
                attn_out = Tensor::zeros({1, (int64_t)dim});
                matmul_dispatch(attn_concat.data_as<float>(),
                               attn_o_weight_[i],
                               attn_out.data_as<float>(),
                               1, dim, dim, attn_o_format_[i]);
            } else {
                // Prefill: use multi-kernel path (forward_from_qkv handles
                // KV-cache update internally — do NOT append here).
                attn_out = attention_.forward_from_qkv(
                    q_proj, k_proj, v_proj,
                    attn_o_weight_[i], attn_o_format_[i],
                    *kv_cache_, i, pos);
            }
        }

        // FUSED: hidden += attn_out; normed_buf = RMSNorm(hidden, ffn_norm[i])
        // Replaces separate add() + rmsnorm() — saves 1 launch + 1 memory pass.
        rmsnorm_residual(hidden.data_as<float>(),
                         attn_out.data_as<float>(),
                         ffn_norm_weight_[i].data_as<float>(),
                         hidden.data_as<float>(),
                         normed_buf_.data_as<float>(),
                         seq_len, dim, config_.rms_norm_eps);

        // Feed-forward network
        const int inter = config_.intermediate_size;
        const int n_hidden = seq_len * inter;

        if (!use_quantized) {
            ffn_.forward(normed_buf_.data_as<float>(),
                         ffn_gate_weight_[i].data_as<float>(),
                         ffn_up_weight_[i].data_as<float>(),
                         ffn_down_weight_[i].data_as<float>(),
                         ffn_output_buf_.data_as<float>(),
                         seq_len);
        } else {
            // Quantized FFN: gate + up projections with dispatch, SiLU gating,
            // then down projection with dispatch.

            // Step 1: gate = input * gate_proj
            matmul_dispatch(normed_buf_.data_as<float>(), ffn_gate_weight_[i],
                           ffn_.gate_buf(), seq_len, inter, dim,
                           ffn_gate_format_[i]);

            // Step 2: up = input * up_proj
            matmul_dispatch(normed_buf_.data_as<float>(), ffn_up_weight_[i],
                           ffn_.up_buf(), seq_len, inter, dim,
                           ffn_up_format_[i]);

            // Step 3: hidden = SiLU(gate) * up
            silu(ffn_.gate_buf(), ffn_.hidden_buf(), n_hidden);
            multiply(ffn_.hidden_buf(), ffn_.up_buf(), ffn_.gate_buf(), n_hidden);

            // Step 4: output = hidden * down_proj
            matmul_dispatch(ffn_.gate_buf(), ffn_down_weight_[i],
                           ffn_output_buf_.data_as<float>(), seq_len, dim, inter,
                           ffn_down_format_[i]);

        }

        // FUSED: hidden += ffn_out; normed_buf = RMSNorm(hidden, next_norm_weight)
        // For non-last layers: next norm is attn_norm[i+1] (pre-attention norm).
        // For last layer: next norm is output_norm (final normalization).
        const float* next_norm = (i < config_.n_layers - 1)
            ? attn_norm_weight_[i + 1].data_as<float>()
            : output_norm_weight_.data_as<float>();

        rmsnorm_residual(hidden.data_as<float>(),
                         ffn_output_buf_.data_as<float>(),
                         next_norm,
                         hidden.data_as<float>(),
                         normed_buf_.data_as<float>(),
                         seq_len, dim, config_.rms_norm_eps);
    }

    // normed_buf_ now contains RMSNorm(hidden, output_norm) from the
    // last layer's fused post-FFN call — no separate final RMSNorm needed.

    // ── LM Head: project to vocabulary logits ───────────────
    Tensor normed_final = normed_buf_.view(2, 0, seq_len);

    if (config_.tie_embeddings) {
        // Tied weights: embedding table is always FP32, use transposed-B matmul
        return lm_head_.forward(normed_final, token_embd_weight_);
    } else if (lm_head_format_ == WeightFormat::FP32) {
        return lm_head_.forward(normed_final, lm_head_weight_);
    } else {
        // Quantized LM head: use matmul_dispatch (B is [vocab, dim] quantized)
        Tensor logits({(int64_t)seq_len, (int64_t)config_.vocab_size});
        matmul_dispatch(normed_final.data_as<float>(), lm_head_weight_,
                       logits.data_as<float>(), seq_len, config_.vocab_size,
                       config_.dim, lm_head_format_);
        return logits;
    }
}

// ════════════════════════════════════════════════════════════════
// Text Generation
// ════════════════════════════════════════════════════════════════

std::string TransformerModel::generate(const std::string& prompt,
                                        int max_tokens,
                                        float temperature,
                                        int top_k) {
    // Reset for new generation
    reset();

    // Encode prompt
    std::vector<int32_t> tokens = tokenizer_->encode(prompt, true);

    if (tokens.empty()) {
        throw std::invalid_argument(
            "TransformerModel::generate: prompt produced empty token sequence");
    }

    const int prompt_len = static_cast<int>(tokens.size());

    // Prefill: process entire prompt
    Tensor logits = forward(tokens);

    // Sample first generated token from the last position's logits
    const int last_pos = static_cast<int>(tokens.size()) - 1;
    const float* last_logits = logits.data_as<float>() +
                               last_pos * config_.vocab_size;
    int32_t next_token;

    if (temperature <= 0.0f || top_k <= 0) {
        next_token = argmax_sample(last_logits, config_.vocab_size);
    } else {
        next_token = top_k_sample(last_logits, config_.vocab_size,
                                   top_k, temperature);
    }
    tokens.push_back(next_token);

    // Autoregressive decode loop
    for (int i = 1; i < max_tokens; ++i) {
        if (next_token == tokenizer_->eos_token_id()) break;

        // Single-token forward pass
        Tensor step_logits = forward({next_token});

        const float* step_last = step_logits.data_as<float>();
        if (temperature <= 0.0f || top_k <= 0) {
            next_token = argmax_sample(step_last, config_.vocab_size);
        } else {
            next_token = top_k_sample(step_last, config_.vocab_size,
                                       top_k, temperature);
        }
        tokens.push_back(next_token);
    }

    // Decode only the generated tokens (skip prompt)
    std::vector<int32_t> generated(tokens.begin() + prompt_len, tokens.end());
    return tokenizer_->decode(generated);
}

// ════════════════════════════════════════════════════════════════
// State Management
// ════════════════════════════════════════════════════════════════

void TransformerModel::reset() {
    kv_cache_->clear();
}

int TransformerModel::current_position() const {
    return kv_cache_->current_length();
}

// ════════════════════════════════════════════════════════════════
// Memory Report
// ════════════════════════════════════════════════════════════════

void TransformerModel::print_memory_report() const {
    // Weight memory
    size_t weight_bytes = 0;
    weight_bytes += token_embd_weight_.size_bytes();
    weight_bytes += output_norm_weight_.size_bytes();
    if (!config_.tie_embeddings) {
        weight_bytes += lm_head_weight_.size_bytes();
    }
    for (int i = 0; i < config_.n_layers; ++i) {
        weight_bytes += attn_norm_weight_[i].size_bytes();
        weight_bytes += attn_q_weight_[i].size_bytes();
        weight_bytes += attn_k_weight_[i].size_bytes();
        weight_bytes += attn_v_weight_[i].size_bytes();
        weight_bytes += attn_o_weight_[i].size_bytes();
        weight_bytes += ffn_norm_weight_[i].size_bytes();
        weight_bytes += ffn_gate_weight_[i].size_bytes();
        weight_bytes += ffn_up_weight_[i].size_bytes();
        weight_bytes += ffn_down_weight_[i].size_bytes();
    }

    // KV-cache memory
    size_t kv_bytes = kv_cache_->total_bytes();

    // Activation buffers
    size_t buf_bytes = normed_buf_.size_bytes()
                     + ffn_output_buf_.size_bytes()
                     + config_.max_seq_len * sizeof(int32_t)  // token_ids_gpu
                     + ffn_.temp_buffer_bytes();

    double weight_mb = weight_bytes / (1024.0 * 1024.0);
    double kv_mb     = kv_bytes    / (1024.0 * 1024.0);
    double buf_mb    = buf_bytes   / (1024.0 * 1024.0);
    double total_mb  = weight_mb + kv_mb + buf_mb;

    char buf[64];
    printf("┌──────────────────────────────────────────────────┐\n");
    printf("│ CuInfer — Memory Report                          │\n");
    printf("├──────────────────────────────────────────────────┤\n");
    snprintf(buf, sizeof(buf), "%.1f MB", weight_mb);
    printf("│ Weights:      %-34s │\n", buf);
    snprintf(buf, sizeof(buf), "%.1f MB (max seq %d)", kv_mb, config_.max_seq_len);
    printf("│ KV-Cache:     %-34s │\n", buf);
    snprintf(buf, sizeof(buf), "%.1f MB", buf_mb);
    printf("│ Buffers:      %-34s │\n", buf);
    snprintf(buf, sizeof(buf), "%.1f MB", total_mb);
    printf("│ Total:        %-34s │\n", buf);
    printf("└──────────────────────────────────────────────────┘\n");

    print_vram_usage("After model load");
}

} // namespace cuinfer
