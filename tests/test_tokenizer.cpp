#include <gtest/gtest.h>
#include "cuinfer/tokenizer.h"

#include <string>
#include <utility>
#include <vector>

using namespace cuinfer;

// ── Test Fixture: Synthetic Vocabulary ──────────────────────
//
// We build a small vocabulary by hand so that tests are self-contained
// and don't require loading a real model file.
//
// Vocabulary (SentencePiece-style):
//   ID  Token          Type      Score   Notes
//   --  -----          ----      -----   -----
//    0  "<unk>"        UNKNOWN   0.0     Unknown fallback
//    1  "<s>"          CONTROL   0.0     BOS
//    2  "</s>"         CONTROL   0.0     EOS
//    3  "\xe2\x96\x81" NORMAL   -1.0     Space marker (lowest merge priority)
//    4  "h"            NORMAL   -5.0
//    5  "e"            NORMAL   -5.0
//    6  "l"            NORMAL   -5.0
//    7  "o"            NORMAL   -5.0
//    8  "w"            NORMAL   -5.0
//    9  "r"            NORMAL   -5.0
//   10  "d"            NORMAL   -5.0
//   11  "he"           NORMAL    1.0
//   12  "ll"           NORMAL    2.0
//   13  "lo"           NORMAL    0.5
//   14  "hel"          NORMAL    3.0
//   15  "hell"         NORMAL    4.0
//   16  "hello"        NORMAL    5.0
//   17  "\xe2\x96\x81he" NORMAL  0.8
//   18  "\xe2\x96\x81hello" NORMAL 6.0
//   19  "or"           NORMAL    0.3
//   20  "ld"           NORMAL    0.4
//   21  "world"        NORMAL    3.5
//   22  "orld"         NORMAL    2.5
//   23  "\xe2\x96\x81world" NORMAL 5.5
//   24  "<0x00>"       BYTE      0.0     Byte-fallback for 0x00
//   25  "<0x41>"       BYTE      0.0     Byte-fallback for 'A' (0x41)
//   26  "<0xFF>"       BYTE      0.0     Byte-fallback for 0xFF

// The SentencePiece space marker as a C++ string.
static const std::string SP = "\xE2\x96\x81";

/// Helper: build the synthetic SentencePiece-style tokenizer.
static Tokenizer make_test_tokenizer(bool add_bos = true, bool add_eos = false) {
    std::vector<std::string> vocab = {
        "<unk>",              //  0
        "<s>",                //  1
        "</s>",               //  2
        SP,                   //  3  space marker
        "h",                  //  4
        "e",                  //  5
        "l",                  //  6
        "o",                  //  7
        "w",                  //  8
        "r",                  //  9
        "d",                  // 10
        "he",                 // 11
        "ll",                 // 12
        "lo",                 // 13
        "hel",                // 14
        "hell",               // 15
        "hello",              // 16
        SP + "he",            // 17
        SP + "hello",         // 18
        "or",                 // 19
        "ld",                 // 20
        "world",              // 21
        "orld",               // 22
        SP + "world",         // 23
        "<0x00>",             // 24
        "<0x41>",             // 25 — byte 0x41 = 'A'
        "<0xFF>",             // 26
    };

    std::vector<float> scores = {
         0.0f,   //  0  <unk>
         0.0f,   //  1  <s>
         0.0f,   //  2  </s>
        -1.0f,   //  3  SP
        -5.0f,   //  4  h
        -5.0f,   //  5  e
        -5.0f,   //  6  l
        -5.0f,   //  7  o
        -5.0f,   //  8  w
        -5.0f,   //  9  r
        -5.0f,   // 10  d
         1.0f,   // 11  he
         2.0f,   // 12  ll
         0.5f,   // 13  lo
         3.0f,   // 14  hel
         4.0f,   // 15  hell
         5.0f,   // 16  hello
         0.8f,   // 17  SP+he
         6.0f,   // 18  SP+hello
         0.3f,   // 19  or
         0.4f,   // 20  ld
         3.5f,   // 21  world
         2.5f,   // 22  orld
         5.5f,   // 23  SP+world
         0.0f,   // 24  <0x00>
         0.0f,   // 25  <0x41>
         0.0f,   // 26  <0xFF>
    };

    std::vector<TokenType> types = {
        TokenType::UNKNOWN,   //  0
        TokenType::CONTROL,   //  1
        TokenType::CONTROL,   //  2
        TokenType::NORMAL,    //  3
        TokenType::NORMAL,    //  4
        TokenType::NORMAL,    //  5
        TokenType::NORMAL,    //  6
        TokenType::NORMAL,    //  7
        TokenType::NORMAL,    //  8
        TokenType::NORMAL,    //  9
        TokenType::NORMAL,    // 10
        TokenType::NORMAL,    // 11
        TokenType::NORMAL,    // 12
        TokenType::NORMAL,    // 13
        TokenType::NORMAL,    // 14
        TokenType::NORMAL,    // 15
        TokenType::NORMAL,    // 16
        TokenType::NORMAL,    // 17
        TokenType::NORMAL,    // 18
        TokenType::NORMAL,    // 19
        TokenType::NORMAL,    // 20
        TokenType::NORMAL,    // 21
        TokenType::NORMAL,    // 22
        TokenType::NORMAL,    // 23
        TokenType::BYTE,      // 24
        TokenType::BYTE,      // 25
        TokenType::BYTE,      // 26
    };

    TokenizerConfig config;
    config.model_type = "llama";
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    config.pad_token_id = 0;
    config.unk_token_id = 0;
    config.add_bos_token = add_bos;
    config.add_eos_token = add_eos;

    return Tokenizer(vocab, scores, types, config);
}

// ── Test: Vocabulary Size and Special Token IDs ─────────────

TEST(Tokenizer, VocabSizeAndSpecialTokens) {
    Tokenizer tok = make_test_tokenizer();

    EXPECT_EQ(tok.vocab_size(), 27u);
    EXPECT_EQ(tok.bos_token_id(), 1);
    EXPECT_EQ(tok.eos_token_id(), 2);
    EXPECT_EQ(tok.pad_token_id(), 0);
}

// ── Test: Encode "hello" ────────────────────────────────────
// Input: "hello"
// After SentencePiece preprocessing: SP + "hello"
// The BPE algorithm should merge this to the single token
// SP+"hello" (id=18, score=6.0) since it has the highest score.

TEST(Tokenizer, EncodeHello) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    auto ids = tok.encode("hello", /*add_special=*/false);

    // Should produce a single merged token: SP+"hello" = id 18.
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 18);
}

// ── Test: Encode "hello world" ──────────────────────────────
// After preprocessing: SP+"hello"+SP+"world"
// Should merge to: [SP+"hello" (18), SP+"world" (23)]

TEST(Tokenizer, EncodeHelloWorld) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    auto ids = tok.encode("hello world", /*add_special=*/false);

    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 18);  // SP+"hello"
    EXPECT_EQ(ids[1], 23);  // SP+"world"
}

// ── Test: BOS Token Prepended ───────────────────────────────

TEST(Tokenizer, BOSTokenPrepended) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/true, /*add_eos=*/false);

    auto ids = tok.encode("hello", /*add_special=*/true);

    ASSERT_GE(ids.size(), 2u);
    EXPECT_EQ(ids[0], 1);   // BOS
    EXPECT_EQ(ids[1], 18);  // SP+"hello"
}

// ── Test: EOS Token Appended ────────────────────────────────

TEST(Tokenizer, EOSTokenAppended) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/true);

    auto ids = tok.encode("hello", /*add_special=*/true);

    ASSERT_GE(ids.size(), 2u);
    EXPECT_EQ(ids[0], 18);                      // SP+"hello"
    EXPECT_EQ(ids[ids.size() - 1], 2);           // EOS
}

// ── Test: Both BOS and EOS ──────────────────────────────────

TEST(Tokenizer, BOSAndEOS) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/true, /*add_eos=*/true);

    auto ids = tok.encode("hello", /*add_special=*/true);

    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 1);   // BOS
    EXPECT_EQ(ids[1], 18);  // SP+"hello"
    EXPECT_EQ(ids[2], 2);   // EOS
}

// ── Test: No Special Tokens When add_special=false ──────────

TEST(Tokenizer, NoSpecialTokensWhenDisabled) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/true, /*add_eos=*/true);

    auto ids = tok.encode("hello", /*add_special=*/false);

    // Even though config says add_bos and add_eos, add_special=false overrides.
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 18);  // SP+"hello"
}

// ── Test: Decode Single Token ───────────────────────────────

TEST(Tokenizer, DecodeSingleToken) {
    Tokenizer tok = make_test_tokenizer();

    // SP+"hello" should decode to " hello" (space + hello).
    std::string decoded = tok.decode_token(18);
    EXPECT_EQ(decoded, " hello");

    // "hello" (no space marker) should decode to "hello".
    decoded = tok.decode_token(16);
    EXPECT_EQ(decoded, "hello");

    // Plain "h" should decode to "h".
    decoded = tok.decode_token(4);
    EXPECT_EQ(decoded, "h");
}

// ── Test: Decode Sequence ───────────────────────────────────

TEST(Tokenizer, DecodeSequence) {
    Tokenizer tok = make_test_tokenizer();

    // Decode [SP+"hello", SP+"world"] should give " hello world".
    std::string decoded = tok.decode({18, 23});
    EXPECT_EQ(decoded, " hello world");
}

// ── Test: Decode Skips Control Tokens ───────────────────────

TEST(Tokenizer, DecodeSkipsControlTokens) {
    Tokenizer tok = make_test_tokenizer();

    // BOS (1) and EOS (2) are CONTROL tokens and should be skipped.
    std::string decoded = tok.decode({1, 18, 23, 2});
    EXPECT_EQ(decoded, " hello world");
}

// ── Test: Encode-Decode Roundtrip ───────────────────────────
// Encoding then decoding should reconstruct the original text
// (modulo a leading space from SentencePiece convention).

TEST(Tokenizer, EncodeDecodeRoundtrip) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    std::string original = "hello world";
    auto ids = tok.encode(original, /*add_special=*/false);
    std::string decoded = tok.decode(ids);

    // SentencePiece prepends a space marker, so decoded text has a leading space.
    EXPECT_EQ(decoded, " " + original);
}

// ── Test: Encode-Decode Roundtrip with Special Tokens ───────

TEST(Tokenizer, EncodeDecodeRoundtripWithSpecial) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/true, /*add_eos=*/true);

    std::string original = "hello world";
    auto ids = tok.encode(original, /*add_special=*/true);
    std::string decoded = tok.decode(ids);

    // Control tokens (BOS, EOS) are skipped during decode.
    EXPECT_EQ(decoded, " " + original);
}

// ── Test: Empty String ──────────────────────────────────────

TEST(Tokenizer, EmptyString) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    auto ids = tok.encode("", /*add_special=*/false);
    EXPECT_TRUE(ids.empty());

    // With special tokens, empty input should still get BOS/EOS if configured.
    Tokenizer tok2 = make_test_tokenizer(/*add_bos=*/true, /*add_eos=*/true);
    auto ids2 = tok2.encode("", /*add_special=*/true);
    ASSERT_EQ(ids2.size(), 2u);
    EXPECT_EQ(ids2[0], 1);  // BOS
    EXPECT_EQ(ids2[1], 2);  // EOS
}

// ── Test: Decode Byte-Fallback Tokens ───────────────────────

TEST(Tokenizer, DecodeByteTokens) {
    Tokenizer tok = make_test_tokenizer();

    // <0x41> should decode to "A" (0x41 is ASCII 'A').
    std::string decoded = tok.decode_token(25);
    EXPECT_EQ(decoded, "A");

    // <0x00> should decode to a null byte.
    decoded = tok.decode_token(24);
    EXPECT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0], '\0');

    // <0xFF> should decode to byte 0xFF.
    decoded = tok.decode_token(26);
    EXPECT_EQ(decoded.size(), 1u);
    EXPECT_EQ(static_cast<uint8_t>(decoded[0]), 0xFF);
}

// ── Test: Invalid Token ID Decoding ─────────────────────────

TEST(Tokenizer, DecodeInvalidTokenId) {
    Tokenizer tok = make_test_tokenizer();

    // Out-of-range token IDs should return empty string.
    EXPECT_EQ(tok.decode_token(-1), "");
    EXPECT_EQ(tok.decode_token(999), "");
}

// ── Test: Single Character ──────────────────────────────────

TEST(Tokenizer, SingleCharacter) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    // "h" -> after preprocessing: SP + "h"
    // SP has id=3, "h" has id=4. No merge exists for SP+"h" as a single token
    // except SP+"he" (17), but we only have "h" not "he", so:
    // Wait -- SP+"he" exists as token 17, but we have SP and h, not SP+h+e.
    // The input is SP + "h". These can't merge to SP+"he" because there's no "e".
    // So result should be [SP(3), h(4)].
    auto ids = tok.encode("h", /*add_special=*/false);
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 3);  // SP
    EXPECT_EQ(ids[1], 4);  // h
}

// ── Test: Decode Individual Tokens Match Full Decode ────────
// Decoding tokens one-by-one and concatenating should equal
// decoding the full sequence at once.

TEST(Tokenizer, DecodeTokenByTokenMatchesFullDecode) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    auto ids = tok.encode("hello world", /*add_special=*/false);

    // Full decode.
    std::string full = tok.decode(ids);

    // Token-by-token decode (skip control tokens to match decode() behavior).
    std::string piecewise;
    for (int32_t id : ids) {
        piecewise += tok.decode_token(id);
    }

    EXPECT_EQ(piecewise, full);
}

// ── Test: Merge Priority Ordering ───────────────────────────
// Verify that the BPE algorithm picks the highest-score merge
// at each step. For "hello":
//   Initial: SP, h, e, l, l, o
//   "hello" (score=5.0) is the highest scoring vocab entry that
//   can be formed, but it can only be formed after intermediate merges.
//   The algorithm should merge step-by-step:
//     he (1.0) -> ll (2.0) -> hel (3.0) -> hell (4.0) -> hello (5.0)
//   Then SP + hello -> SP+hello (6.0)
//
// This test verifies the final result matches expectations.

TEST(Tokenizer, MergePriorityOrdering) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    auto ids = tok.encode("hello", /*add_special=*/false);

    // The final merged token should be SP+"hello" (id=18).
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 18);
}

// ── Test: Repeated Characters ───────────────────────────────
// "ll" should merge to token 12 (score=2.0).
// "lll" -> first "ll" merges (leftmost), then "ll"+"l" can't merge further.

TEST(Tokenizer, RepeatedCharacters) {
    Tokenizer tok = make_test_tokenizer(/*add_bos=*/false, /*add_eos=*/false);

    // "ll" -> after preprocessing: SP + "l" + "l"
    // SP stays separate, "ll" merges to token 12.
    auto ids = tok.encode("ll", /*add_special=*/false);
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 3);   // SP
    EXPECT_EQ(ids[1], 12);  // ll
}

// ── Test: GPT2-Style Merges ─────────────────────────────────
// Verify that the GPT2 constructor (explicit merge pairs) works.

TEST(Tokenizer, GPT2StyleMerges) {
    // Minimal GPT2-style vocabulary.
    // GPT2 uses byte-to-unicode: space (0x20) -> Ġ (U+0120, UTF-8: \xC4\xA0).
    // Regular ASCII letters (a-z) map to themselves.
    const std::string G_SPACE = "\xC4\xA0"; // Ġ = GPT2 encoding of space byte

    std::vector<std::string> vocab = {
        "<unk>",           // 0
        "<s>",             // 1
        "</s>",            // 2
        G_SPACE,           // 3 — GPT2 space character
        "a",               // 4
        "b",               // 5
        "c",               // 6
        "ab",              // 7
        "abc",             // 8
        G_SPACE + "a",     // 9
        G_SPACE + "abc",   // 10
    };

    // Merge rules in priority order (first = highest priority).
    std::vector<std::pair<std::string, std::string>> merges = {
        {"a", "b"},                      // Priority 0 (highest) -> "ab"
        {"ab", "c"},                     // Priority 1 -> "abc"
        {G_SPACE, "a"},                  // Priority 2 -> Ġ+"a"
        {G_SPACE + "a", "bc"},           // Priority 3 -> Ġ+"abc"
    };

    std::vector<TokenType> types = {
        TokenType::UNKNOWN,
        TokenType::CONTROL,
        TokenType::CONTROL,
        TokenType::NORMAL,
        TokenType::NORMAL,
        TokenType::NORMAL,
        TokenType::NORMAL,
        TokenType::NORMAL,
        TokenType::NORMAL,
        TokenType::NORMAL,
        TokenType::NORMAL,
    };

    TokenizerConfig config;
    config.model_type = "gpt2";
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    config.add_bos_token = false;
    config.add_eos_token = false;

    Tokenizer tok(vocab, merges, types, config);

    // "abc" — GPT2 pre-tokenization produces one chunk: "abc"
    // bytes_to_unicode maps: 'a'->a, 'b'->b, 'c'->c (ASCII letters map to themselves)
    // BPE merges: a+b -> ab (rank 0), ab+c -> abc (rank 1)
    auto ids = tok.encode("abc", /*add_special=*/false);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 8);  // "abc"

    // " abc" — pre-tokenization: [" abc"] (space attaches to following word)
    // bytes_to_unicode maps: ' '->Ġ, 'a'->a, 'b'->b, 'c'->c
    // So the BPE input is "Ġabc"
    // Merges: a+b -> ab (rank 0), ab+c -> abc (rank 1), Ġ+a -> Ġa (rank 2), Ġa+bc -> Ġabc (rank 3)
    auto ids2 = tok.encode(" abc", /*add_special=*/false);
    ASSERT_EQ(ids2.size(), 1u);
    EXPECT_EQ(ids2[0], 10);  // Ġ+"abc"

    // Verify decode round-trip: token 10 = Ġ+"abc" should decode back to " abc"
    std::string decoded = tok.decode({10});
    EXPECT_EQ(decoded, " abc");
}

// ── Test: Constructor Validation ────────────────────────────

TEST(Tokenizer, ConstructorValidation) {
    std::vector<std::string> vocab = {"a", "b", "c"};
    std::vector<TokenType> types = {TokenType::NORMAL, TokenType::NORMAL, TokenType::NORMAL};

    // Mismatched scores size should throw.
    EXPECT_THROW(
        Tokenizer(vocab, std::vector<float>{1.0f, 2.0f}, types),
        std::invalid_argument
    );

    // Mismatched token_types size should throw.
    EXPECT_THROW(
        Tokenizer(vocab, std::vector<float>{1.0f, 2.0f, 3.0f},
                  std::vector<TokenType>{TokenType::NORMAL}),
        std::invalid_argument
    );
}

// ── Test: Print Vocab Summary (Smoke Test) ──────────────────
// Just verify it doesn't crash.

TEST(Tokenizer, PrintVocabSummary) {
    Tokenizer tok = make_test_tokenizer();
    EXPECT_NO_THROW(tok.print_vocab_summary());
}
