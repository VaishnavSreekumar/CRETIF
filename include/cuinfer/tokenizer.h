#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuinfer {

// ── Token Types ─────────────────────────────────────────────
// Mirrors the GGUF tokenizer.ggml.token_type values used by
// Llama-family and GPT2-family models.

enum class TokenType : int32_t {
    NORMAL       = 0,
    UNKNOWN      = 1,
    CONTROL      = 2,  // BOS, EOS, PAD, etc.
    USER_DEFINED = 3,
    UNUSED       = 4,
    BYTE         = 5   // Byte-fallback tokens like <0x41>
};

// ── Tokenizer Config ────────────────────────────────────────
// Configuration extracted from GGUF metadata keys:
//   tokenizer.ggml.model, tokenizer.ggml.bos_token_id, etc.

struct TokenizerConfig {
    std::string model_type;       ///< Tokenizer family: "llama" or "gpt2"
    uint32_t bos_token_id = 1;    ///< Beginning-of-sequence token ID
    uint32_t eos_token_id = 2;    ///< End-of-sequence token ID
    uint32_t pad_token_id = 0;    ///< Padding token ID
    uint32_t unk_token_id = 0;    ///< Unknown token ID
    bool add_bos_token = true;    ///< Prepend BOS when encoding
    bool add_eos_token = false;   ///< Append EOS when encoding
};

// ── Tokenizer ───────────────────────────────────────────────
// Byte-Pair Encoding (BPE) tokenizer supporting both SentencePiece-style
// (score-based merges, used by Llama/SmolLM) and GPT2-style (explicit
// ordered merge list).
//
// This is a CPU-only component. During inference, tokenization happens
// once per prompt and is not the performance bottleneck -- clarity and
// correctness matter more than speed here.

class Tokenizer {
public:
    // ── Constructors ────────────────────────────────────────

    /// SentencePiece-style: merges are determined by per-token scores.
    /// Higher score = higher priority merge. This is the format used by
    /// Llama, SmolLM2, and most modern open-weight models.
    Tokenizer(
        std::vector<std::string> vocab,
        std::vector<float> scores,
        std::vector<TokenType> token_types,
        TokenizerConfig config = {}
    );

    /// GPT2-style: merges are given as an explicit ordered list.
    /// Earlier entries in the list have higher priority.
    Tokenizer(
        std::vector<std::string> vocab,
        std::vector<std::pair<std::string, std::string>> merges,
        std::vector<TokenType> token_types,
        TokenizerConfig config = {}
    );

    // ── Encoding ────────────────────────────────────────────

    /// Encode text into a sequence of token IDs.
    /// If add_special is true, prepends BOS and/or appends EOS
    /// according to the tokenizer config.
    std::vector<int32_t> encode(const std::string& text,
                                bool add_special = true) const;

    // ── Decoding ────────────────────────────────────────────

    /// Decode a sequence of token IDs back to text.
    /// Concatenates token strings and converts SentencePiece
    /// space markers back to regular spaces.
    std::string decode(const std::vector<int32_t>& tokens) const;

    /// Decode a single token ID to its string representation.
    /// Useful for streaming output one token at a time.
    std::string decode_token(int32_t token_id) const;

    // ── Vocabulary Info ─────────────────────────────────────

    /// Returns the total number of tokens in the vocabulary.
    size_t vocab_size() const;

    /// Returns the beginning-of-sequence token ID.
    int32_t bos_token_id() const;

    /// Returns the end-of-sequence token ID.
    int32_t eos_token_id() const;

    /// Returns the padding token ID.
    int32_t pad_token_id() const;

    // ── Debug ───────────────────────────────────────────────

    /// Prints a summary of the vocabulary to stdout: total size,
    /// special token IDs, and a sample of the first few entries.
    void print_vocab_summary() const;

private:
    // ── BPE Core ────────────────────────────────────────────

    /// The main BPE encoding loop. Converts pre-processed text
    /// (with SentencePiece space markers already applied) into
    /// a sequence of token IDs by iteratively merging the
    /// highest-priority adjacent pair.
    std::vector<int32_t> bpe_encode(const std::string& text) const;

    /// Convert a raw byte value to its corresponding token ID.
    /// For SentencePiece models, this looks up the byte-fallback
    /// token "<0xNN>". Returns -1 if no byte token exists.
    int32_t byte_to_token(uint8_t byte) const;

    /// Look up the merge priority for a candidate merged string.
    /// Returns the score (higher = merge first) or a sentinel
    /// value indicating "no merge possible".
    float merge_score(const std::string& merged) const;

    // ── Data ────────────────────────────────────────────────

    std::vector<std::string> vocab_;                         // id -> string
    std::unordered_map<std::string, int32_t> token_to_id_;   // string -> id
    std::vector<float> scores_;                              // merge priority per token
    std::vector<TokenType> token_types_;
    TokenizerConfig config_;

    // GPT2-style: explicit merge pairs mapped to priority (lower index = higher priority).
    // For SentencePiece-style, this remains empty and scores_ is used instead.
    std::unordered_map<std::string, int32_t> merge_ranks_;

    // Precomputed byte-to-token lookup table (256 entries).
    // byte_tokens_[b] is the token ID for byte value b, or -1 if none.
    std::vector<int32_t> byte_tokens_;

    // Flag: true if using GPT2-style merges, false for SentencePiece scores.
    bool use_merge_ranks_ = false;

    // GPT2 byte-level BPE: inverse mapping from Unicode codepoint back to
    // original byte value. Only populated for GPT2-style tokenizers.
    // Key: Unicode codepoint (e.g., 0x0120 for Ġ), Value: original byte (e.g., 0x20 = space).
    std::unordered_map<uint32_t, uint8_t> gpt2_unicode_to_byte_;

    // GPT2 byte-level BPE: forward mapping from byte value to UTF-8 string
    // of the corresponding Unicode character. Only populated for GPT2-style tokenizers.
    std::vector<std::string> gpt2_byte_to_utf8_;
};

} // namespace cuinfer
