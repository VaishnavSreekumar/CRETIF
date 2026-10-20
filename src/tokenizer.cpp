#include "cuinfer/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cuinfer {

// ── Constants ───────────────────────────────────────────────

// SentencePiece uses U+2581 (Lower One Eighth Block) as a space marker.
// In UTF-8 this encodes to the three bytes 0xE2 0x96 0x81.
static const std::string SPIECE_SPACE = "\xE2\x96\x81";

// Sentinel value: "no merge is possible for this pair".
static constexpr float NO_MERGE_SCORE = -1e18f;

// ── GPT2 Byte-Level BPE Helpers ─────────────────────────────
//
// GPT2 maps every byte (0x00-0xFF) to a printable Unicode character so
// the vocabulary contains only printable strings. Bytes in the ASCII
// printable range map to themselves; control chars / non-printable bytes
// are shifted up to the U+0100-U+0143 range.
//
// Reference: openai/gpt-2 encoder.py bytes_to_unicode()

/// Encode a Unicode codepoint to its UTF-8 string representation.
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

/// Decode one UTF-8 character from a string starting at position i.
/// Returns the codepoint and advances i past the character.
static uint32_t utf8_to_codepoint(const std::string& s, size_t& i) {
    uint8_t lead = static_cast<uint8_t>(s[i]);
    uint32_t cp = 0;

    if (lead < 0x80) {
        cp = lead;
        i += 1;
    } else if (lead < 0xE0) {
        cp = (lead & 0x1F) << 6;
        if (i + 1 < s.size())
            cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F);
        i += 2;
    } else if (lead < 0xF0) {
        cp = (lead & 0x0F) << 12;
        if (i + 1 < s.size())
            cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6;
        if (i + 2 < s.size())
            cp |= (static_cast<uint8_t>(s[i + 2]) & 0x3F);
        i += 3;
    } else {
        cp = (lead & 0x07) << 18;
        if (i + 1 < s.size())
            cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12;
        if (i + 2 < s.size())
            cp |= (static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6;
        if (i + 3 < s.size())
            cp |= (static_cast<uint8_t>(s[i + 3]) & 0x3F);
        i += 4;
    }
    return cp;
}

/// Build the GPT2 byte-to-unicode mapping.
/// Returns a vector of 256 entries: byte value -> Unicode codepoint.
static std::vector<uint32_t> gpt2_bytes_to_unicode() {
    std::vector<uint32_t> mapping(256, 0);

    // Printable byte ranges that map to themselves:
    //   '!' (33) through '~' (126)
    //   0xA1 (161) through 0xAC (172)
    //   0xAE (174) through 0xFF (255)
    uint32_t next_free = 256; // non-printable bytes get codepoints starting at U+0100

    for (int b = 0; b < 256; ++b) {
        if ((b >= 33 && b <= 126) ||
            (b >= 161 && b <= 172) ||
            (b >= 174 && b <= 255)) {
            mapping[b] = static_cast<uint32_t>(b);
        } else {
            mapping[b] = next_free++;
        }
    }
    return mapping;
}

/// Split input text into GPT2 word chunks for pre-tokenization.
/// Spaces attach to the BEGINNING of the following word:
///   "Once upon a time" -> ["Once", " upon", " a", " time"]
/// This is a simplified version of the GPT2 pre-tokenization regex.
static std::vector<std::string> gpt2_pre_tokenize(const std::string& text) {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;

    std::string current;
    size_t i = 0;

    while (i < text.size()) {
        char c = text[i];

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            // Whitespace: if we have accumulated a chunk, emit it
            if (!current.empty()) {
                chunks.push_back(std::move(current));
                current.clear();
            }
            // Attach space to the start of the next chunk
            current += c;
            ++i;
            // Absorb trailing whitespace into the same chunk only
            // if the next char is also whitespace (whitespace-only chunks)
            while (i < text.size() &&
                   (text[i] == ' ' || text[i] == '\t' ||
                    text[i] == '\n' || text[i] == '\r')) {
                // If there's no following non-whitespace, keep accumulating
                current += text[i];
                ++i;
            }
            // If we have whitespace but nothing follows, emit as its own chunk
            if (i >= text.size()) {
                chunks.push_back(std::move(current));
                current.clear();
            }
            // Otherwise, whitespace stays as prefix for the next chunk
        } else {
            current += c;
            ++i;
        }
    }

    if (!current.empty()) {
        chunks.push_back(std::move(current));
    }

    return chunks;
}

// ── UTF-8 Helpers ───────────────────────────────────────────

/// Returns the byte length of a UTF-8 character given its leading byte.
/// If the byte is not a valid UTF-8 lead byte, returns 1 (treat as raw byte).
static int utf8_char_length(uint8_t lead_byte) {
    if (lead_byte < 0x80) return 1;       // 0xxxxxxx — ASCII
    if (lead_byte < 0xC0) return 1;       // 10xxxxxx — continuation (invalid as lead)
    if (lead_byte < 0xE0) return 2;       // 110xxxxx
    if (lead_byte < 0xF0) return 3;       // 1110xxxx
    if (lead_byte < 0xF8) return 4;       // 11110xxx
    return 1;                              // Invalid lead byte — treat as single byte
}

// ── Byte-Fallback Token Parsing ─────────────────────────────

/// Checks if a token string is a byte-fallback token like "<0xNN>"
/// and returns the byte value. Returns -1 if not a byte token.
static int parse_byte_token(const std::string& token) {
    // Format: "<0xNN>" where NN is exactly two hex digits
    if (token.size() == 6 &&
        token[0] == '<' && token[1] == '0' && token[2] == 'x' &&
        token[5] == '>') {
        char hi = token[3];
        char lo = token[4];

        auto hex_val = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            return -1;
        };

        int h = hex_val(hi);
        int l = hex_val(lo);
        if (h >= 0 && l >= 0) {
            return (h << 4) | l;
        }
    }
    return -1;
}

// ── Constructor: SentencePiece-style (score-based merges) ───

Tokenizer::Tokenizer(
    std::vector<std::string> vocab,
    std::vector<float> scores,
    std::vector<TokenType> token_types,
    TokenizerConfig config)
    : vocab_(std::move(vocab))
    , scores_(std::move(scores))
    , token_types_(std::move(token_types))
    , config_(std::move(config))
    , use_merge_ranks_(false)
{
    if (scores_.size() != vocab_.size()) {
        throw std::invalid_argument(
            "Tokenizer: scores array size (" + std::to_string(scores_.size()) +
            ") must match vocab size (" + std::to_string(vocab_.size()) + ")");
    }
    if (token_types_.size() != vocab_.size()) {
        throw std::invalid_argument(
            "Tokenizer: token_types array size (" + std::to_string(token_types_.size()) +
            ") must match vocab size (" + std::to_string(vocab_.size()) + ")");
    }

    // Build the reverse lookup: token string -> token ID.
    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
    }

    // Build the byte-fallback lookup table.
    // For each byte value 0x00..0xFF, find the corresponding token ID.
    byte_tokens_.assign(256, -1);
    for (size_t i = 0; i < vocab_.size(); ++i) {
        if (token_types_[i] == TokenType::BYTE) {
            int byte_val = parse_byte_token(vocab_[i]);
            if (byte_val >= 0 && byte_val < 256) {
                byte_tokens_[byte_val] = static_cast<int32_t>(i);
            }
        }
    }
}

// ── Constructor: GPT2-style (explicit merge list) ───────────

Tokenizer::Tokenizer(
    std::vector<std::string> vocab,
    std::vector<std::pair<std::string, std::string>> merges,
    std::vector<TokenType> token_types,
    TokenizerConfig config)
    : vocab_(std::move(vocab))
    , token_types_(std::move(token_types))
    , config_(std::move(config))
    , use_merge_ranks_(true)
{
    if (token_types_.size() != vocab_.size()) {
        throw std::invalid_argument(
            "Tokenizer: token_types array size (" + std::to_string(token_types_.size()) +
            ") must match vocab size (" + std::to_string(vocab_.size()) + ")");
    }

    // For GPT2-style, scores_ is not used for merge priority.
    // Fill with zeros so the vector size matches vocab.
    scores_.assign(vocab_.size(), 0.0f);

    // Build the reverse lookup.
    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
    }

    // Build the merge rank map. The merge list is ordered by priority:
    // index 0 has the highest priority (smallest rank number).
    // We store rank as the index, so lower rank = higher priority.
    for (size_t i = 0; i < merges.size(); ++i) {
        std::string merged = merges[i].first + merges[i].second;
        merge_ranks_[merged] = static_cast<int32_t>(i);
    }

    // Build byte-fallback lookup table.
    byte_tokens_.assign(256, -1);
    for (size_t i = 0; i < vocab_.size(); ++i) {
        if (token_types_[i] == TokenType::BYTE) {
            int byte_val = parse_byte_token(vocab_[i]);
            if (byte_val >= 0 && byte_val < 256) {
                byte_tokens_[byte_val] = static_cast<int32_t>(i);
            }
        }
    }

    // Build GPT2 byte-level BPE lookup tables.
    auto b2u = gpt2_bytes_to_unicode();
    gpt2_byte_to_utf8_.resize(256);
    for (int b = 0; b < 256; ++b) {
        std::string utf8 = codepoint_to_utf8(b2u[b]);
        gpt2_byte_to_utf8_[b] = utf8;
        gpt2_unicode_to_byte_[b2u[b]] = static_cast<uint8_t>(b);
    }
}

// ── Merge Score Lookup ──────────────────────────────────────

float Tokenizer::merge_score(const std::string& merged) const {
    // First check: is this merged string even in the vocabulary?
    auto it = token_to_id_.find(merged);
    if (it == token_to_id_.end()) {
        return NO_MERGE_SCORE;
    }

    if (use_merge_ranks_) {
        // GPT2-style: look up the merge rank. Lower rank = higher priority.
        // We negate the rank so that "higher score = higher priority" still holds.
        auto rank_it = merge_ranks_.find(merged);
        if (rank_it == merge_ranks_.end()) {
            return NO_MERGE_SCORE;
        }
        // Negate so that rank 0 (highest priority) maps to the highest score.
        return -static_cast<float>(rank_it->second);
    } else {
        // SentencePiece-style: the score for the merged token IS its priority.
        return scores_[it->second];
    }
}

// ── Byte-to-Token Lookup ────────────────────────────────────

int32_t Tokenizer::byte_to_token(uint8_t byte) const {
    return byte_tokens_[byte];
}

// ── BPE Encoding ────────────────────────────────────────────
//
// The BPE algorithm works by:
//   1. Breaking the input text into an initial sequence of tokens
//      (individual UTF-8 characters or single bytes).
//   2. Repeatedly finding the adjacent pair that would produce the
//      highest-priority merge and combining them.
//   3. Stopping when no more merges are possible.
//
// This is the O(n^2) straightforward implementation. For short
// sequences (typical LLM prompts), this is fast enough. A production
// tokenizer would use a priority queue for O(n log n).

std::vector<int32_t> Tokenizer::bpe_encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    // Step 1: Split text into initial tokens.
    // Each UTF-8 character becomes one token string. If a character
    // is not in the vocabulary, we fall back to individual bytes.
    std::vector<std::string> tokens;

    size_t i = 0;
    while (i < text.size()) {
        int char_len = utf8_char_length(static_cast<uint8_t>(text[i]));
        // Clamp to remaining text length (handles truncated UTF-8).
        char_len = std::min(char_len, static_cast<int>(text.size() - i));

        std::string ch = text.substr(i, char_len);

        // If this character exists as a single token, use it directly.
        if (token_to_id_.count(ch)) {
            tokens.push_back(ch);
        } else {
            // Byte fallback: add each byte of this character individually.
            // Each byte maps to either a single-byte vocab entry or a
            // byte-fallback token like "<0xNN>".
            for (int b = 0; b < char_len; ++b) {
                uint8_t byte_val = static_cast<uint8_t>(text[i + b]);
                std::string byte_str(1, static_cast<char>(byte_val));

                if (token_to_id_.count(byte_str)) {
                    tokens.push_back(byte_str);
                } else {
                    // Use byte-fallback token "<0xNN>".
                    int32_t bt = byte_to_token(byte_val);
                    if (bt >= 0) {
                        tokens.push_back(vocab_[bt]);
                    } else {
                        // Last resort: if no byte token exists, use UNK.
                        tokens.push_back(vocab_[config_.unk_token_id]);
                    }
                }
            }
        }
        i += char_len;
    }

    // Step 2: Iteratively merge the best pair.
    while (tokens.size() >= 2) {
        // Scan all adjacent pairs to find the best merge.
        float best_score = NO_MERGE_SCORE;
        size_t best_pos = 0;

        for (size_t j = 0; j + 1 < tokens.size(); ++j) {
            std::string merged = tokens[j] + tokens[j + 1];
            float score = merge_score(merged);

            if (score > best_score) {
                best_score = score;
                best_pos = j;
            }
        }

        // No more merges possible.
        if (best_score <= NO_MERGE_SCORE) {
            break;
        }

        // Apply the merge: combine tokens[best_pos] and tokens[best_pos+1].
        tokens[best_pos] = tokens[best_pos] + tokens[best_pos + 1];
        tokens.erase(tokens.begin() + static_cast<ptrdiff_t>(best_pos) + 1);
    }

    // Step 3: Convert token strings to IDs.
    std::vector<int32_t> ids;
    ids.reserve(tokens.size());

    for (const auto& tok : tokens) {
        auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Should not happen after byte-fallback, but handle gracefully.
            ids.push_back(static_cast<int32_t>(config_.unk_token_id));
        }
    }

    return ids;
}

// ── Public Encode ───────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                        bool add_special) const {
    if (text.empty()) {
        std::vector<int32_t> result;
        if (add_special && config_.add_bos_token) {
            result.push_back(static_cast<int32_t>(config_.bos_token_id));
        }
        if (add_special && config_.add_eos_token) {
            result.push_back(static_cast<int32_t>(config_.eos_token_id));
        }
        return result;
    }

    std::vector<int32_t> ids;

    if (use_merge_ranks_) {
        // GPT2 path: pre-tokenize into word chunks, convert each byte
        // to its GPT2 unicode representation, then BPE-encode per chunk.
        auto chunks = gpt2_pre_tokenize(text);

        for (const auto& chunk : chunks) {
            // Map each byte through bytes_to_unicode.
            std::string unicode_chunk;
            unicode_chunk.reserve(chunk.size() * 2);
            for (size_t ci = 0; ci < chunk.size(); ++ci) {
                uint8_t byte_val = static_cast<uint8_t>(chunk[ci]);
                unicode_chunk += gpt2_byte_to_utf8_[byte_val];
            }
            // BPE-encode this chunk independently.
            auto chunk_ids = bpe_encode(unicode_chunk);
            ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
        }
    } else {
        // SentencePiece preprocessing: replace leading space (and all spaces)
        // with the special space marker character U+2581.
        std::string processed;
        processed.reserve(text.size() + 3);

        // Prepend the space marker (SentencePiece convention: text always
        // starts with a space marker to indicate "beginning of word").
        processed += SPIECE_SPACE;

        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == ' ') {
                processed += SPIECE_SPACE;
            } else {
                processed += text[i];
            }
        }

        ids = bpe_encode(processed);
    }

    // Optionally add special tokens.
    if (add_special) {
        if (config_.add_bos_token) {
            ids.insert(ids.begin(), static_cast<int32_t>(config_.bos_token_id));
        }
        if (config_.add_eos_token) {
            ids.push_back(static_cast<int32_t>(config_.eos_token_id));
        }
    }

    return ids;
}

// ── Decode ──────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;

    for (int32_t id : tokens) {
        // Skip special control tokens (BOS, EOS, PAD) during decode.
        if (id >= 0 && static_cast<size_t>(id) < token_types_.size() &&
            token_types_[id] == TokenType::CONTROL) {
            continue;
        }
        result += decode_token(id);
    }

    return result;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_.size()) {
        return "";  // Invalid token ID -- return empty rather than crash.
    }

    const std::string& token_str = vocab_[token_id];

    // Check if this is a byte-fallback token "<0xNN>".
    int byte_val = parse_byte_token(token_str);
    if (byte_val >= 0) {
        return std::string(1, static_cast<char>(byte_val));
    }

    if (use_merge_ranks_ && !gpt2_unicode_to_byte_.empty()) {
        // GPT2 path: each Unicode character in the token string represents
        // one original byte via the GPT2 bytes_to_unicode() mapping.
        // Decode each codepoint back to its original byte value.
        std::string result;
        result.reserve(token_str.size());

        size_t i = 0;
        while (i < token_str.size()) {
            uint32_t cp = utf8_to_codepoint(token_str, i);
            auto it = gpt2_unicode_to_byte_.find(cp);
            if (it != gpt2_unicode_to_byte_.end()) {
                result += static_cast<char>(it->second);
            } else {
                // Not in mapping — pass through as UTF-8.
                result += codepoint_to_utf8(cp);
            }
        }
        return result;
    }

    // SentencePiece path: replace U+2581 space markers with actual spaces.
    std::string result;
    result.reserve(token_str.size());

    size_t i = 0;
    while (i < token_str.size()) {
        // Check for the 3-byte UTF-8 sequence of U+2581.
        if (i + 2 < token_str.size() &&
            static_cast<uint8_t>(token_str[i])     == 0xE2 &&
            static_cast<uint8_t>(token_str[i + 1]) == 0x96 &&
            static_cast<uint8_t>(token_str[i + 2]) == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += token_str[i];
            ++i;
        }
    }

    return result;
}

// ── Vocabulary Info ─────────────────────────────────────────

size_t Tokenizer::vocab_size() const {
    return vocab_.size();
}

int32_t Tokenizer::bos_token_id() const {
    return static_cast<int32_t>(config_.bos_token_id);
}

int32_t Tokenizer::eos_token_id() const {
    return static_cast<int32_t>(config_.eos_token_id);
}

int32_t Tokenizer::pad_token_id() const {
    return static_cast<int32_t>(config_.pad_token_id);
}

// ── Debug ───────────────────────────────────────────────────

void Tokenizer::print_vocab_summary() const {
    printf("Tokenizer Vocabulary Summary\n");
    printf("  Model type:    %s\n", config_.model_type.c_str());
    printf("  Vocab size:    %zu\n", vocab_.size());
    printf("  BOS token ID:  %u\n", config_.bos_token_id);
    printf("  EOS token ID:  %u\n", config_.eos_token_id);
    printf("  PAD token ID:  %u\n", config_.pad_token_id);
    printf("  UNK token ID:  %u\n", config_.unk_token_id);
    printf("  Add BOS:       %s\n", config_.add_bos_token ? "yes" : "no");
    printf("  Add EOS:       %s\n", config_.add_eos_token ? "yes" : "no");

    // Count token types.
    int num_normal = 0, num_byte = 0, num_control = 0, num_other = 0;
    for (const auto& tt : token_types_) {
        switch (tt) {
            case TokenType::NORMAL:  ++num_normal;  break;
            case TokenType::BYTE:    ++num_byte;    break;
            case TokenType::CONTROL: ++num_control; break;
            default:                 ++num_other;   break;
        }
    }
    printf("  Normal tokens: %d\n", num_normal);
    printf("  Byte tokens:   %d\n", num_byte);
    printf("  Control tokens:%d\n", num_control);

    // Print first few entries.
    size_t preview = std::min<size_t>(vocab_.size(), 10);
    printf("  First %zu tokens:\n", preview);
    for (size_t i = 0; i < preview; ++i) {
        printf("    [%3zu] \"%s\" (score=%.4f, type=%d)\n",
               i, vocab_[i].c_str(), scores_[i],
               static_cast<int>(token_types_[i]));
    }
}

} // namespace cuinfer
