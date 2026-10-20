// Unit tests for the GGUF binary format parser.
//
// Tests create synthetic GGUF files with known contents, write them to
// temporary files, parse them, and verify every field. This avoids any
// dependency on real model files (which are multi-GB).

#include <gtest/gtest.h>
#include "cuinfer/gguf_parser.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <io.h>       // _mktemp_s
    #include <windows.h>
#else
    #include <unistd.h>   // mkstemp
#endif

using namespace cuinfer;

// ── Helpers: Binary Writer ──────────────────────────────────
// Builds a GGUF file byte-by-byte in a std::vector<uint8_t>.

class GGUFWriter {
public:
    // Write a raw value
    template <typename T>
    void write(T val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        data_.insert(data_.end(), p, p + sizeof(T));
    }

    // Write a GGUF-format string (uint64 length + chars)
    void write_string(const std::string& s) {
        write<uint64_t>(s.size());
        data_.insert(data_.end(), s.begin(), s.end());
    }

    // Write raw bytes
    void write_bytes(const void* ptr, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(ptr);
        data_.insert(data_.end(), p, p + len);
    }

    // Pad to alignment boundary
    void align_to(size_t alignment) {
        size_t remainder = data_.size() % alignment;
        if (remainder != 0) {
            size_t padding = alignment - remainder;
            data_.insert(data_.end(), padding, 0);
        }
    }

    size_t size() const { return data_.size(); }
    const std::vector<uint8_t>& data() const { return data_; }

private:
    std::vector<uint8_t> data_;
};

// ── Helpers: Temp File ──────────────────────────────────────
// Creates a temporary file, writes the given data, and provides the path.
// The file is deleted in the destructor.

class TempFile {
public:
    explicit TempFile(const std::vector<uint8_t>& content) {
#ifdef _WIN32
        char temp_dir[MAX_PATH];
        GetTempPathA(MAX_PATH, temp_dir);
        char temp_path[MAX_PATH];
        GetTempFileNameA(temp_dir, "gguf", 0, temp_path);
        path_ = temp_path;
#else
        char templ[] = "/tmp/gguf_test_XXXXXX";
        int fd = mkstemp(templ);
        if (fd < 0) {
            throw std::runtime_error("Failed to create temp file");
        }
        ::close(fd);
        path_ = templ;
#endif
        std::ofstream ofs(path_, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open temp file for writing: " + path_);
        }
        ofs.write(reinterpret_cast<const char*>(content.data()),
                  static_cast<std::streamsize>(content.size()));
        ofs.close();
    }

    ~TempFile() {
        std::remove(path_.c_str());
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

// ── Helper: Build a GGUF file from callbacks ────────────────
// The callbacks write metadata, tensor info, and tensor data sections
// independently. This gives tests full control over the binary layout.

std::vector<uint8_t> build_simple_gguf(
    uint32_t version,
    uint64_t metadata_count,
    uint64_t tensor_count,
    std::function<void(GGUFWriter&)> write_metadata,
    std::function<void(GGUFWriter&)> write_tensor_info,
    std::function<void(GGUFWriter&)> write_tensor_data)
{
    GGUFWriter w;

    // Header
    w.write<uint32_t>(GGUF_MAGIC);
    w.write<uint32_t>(version);
    w.write<uint64_t>(tensor_count);
    w.write<uint64_t>(metadata_count);

    // Metadata
    if (write_metadata) write_metadata(w);

    // Tensor info
    if (write_tensor_info) write_tensor_info(w);

    // Alignment padding before data section
    w.align_to(GGUF_DEFAULT_ALIGNMENT);

    // Tensor data
    if (write_tensor_data) write_tensor_data(w);

    return w.data();
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

// ── Header Parsing ──────────────────────────────────────────

TEST(GGUFParser, ParseMinimalFile) {
    // The simplest valid GGUF file: no metadata, no tensors.
    auto data = build_simple_gguf(
        3, 0, 0, nullptr, nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_EQ(parser.version(), 3u);
    EXPECT_EQ(parser.tensor_count(), 0u);
    EXPECT_EQ(parser.metadata_count(), 0u);
    EXPECT_TRUE(parser.metadata().empty());
    EXPECT_TRUE(parser.tensors().empty());
}

TEST(GGUFParser, ParseVersion2) {
    // GGUF version 2 should also be supported
    auto data = build_simple_gguf(
        2, 0, 0, nullptr, nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_EQ(parser.version(), 2u);
}

// ── Metadata Parsing ────────────────────────────────────────

TEST(GGUFParser, ParseStringMetadata) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            // Key: "general.architecture"
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_EQ(parser.metadata_count(), 1u);
    EXPECT_TRUE(parser.has_metadata("general.architecture"));
    EXPECT_EQ(parser.get_string("general.architecture"), "llama");
}

TEST(GGUFParser, ParseUint32Metadata) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("llama.context_length");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(4096);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_EQ(parser.get_uint32("llama.context_length"), 4096u);
}

TEST(GGUFParser, ParseFloat32Metadata) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("llama.rope.freq_base");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
            w.write<float>(10000.0f);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_FLOAT_EQ(parser.get_float32("llama.rope.freq_base"), 10000.0f);
}

TEST(GGUFParser, ParseBoolMetadata) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("general.quantized");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::BOOL));
            w.write<uint8_t>(1); // true
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_TRUE(parser.has_metadata("general.quantized"));
    auto it = parser.metadata().find("general.quantized");
    ASSERT_NE(it, parser.metadata().end());
    EXPECT_TRUE(it->second.as_bool());
}

TEST(GGUFParser, ParseMultipleMetadata) {
    // A realistic set of metadata entries
    auto data = build_simple_gguf(3, 4, 0,
        [](GGUFWriter& w) {
            // String: architecture
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");

            // uint32: context length
            w.write_string("llama.context_length");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(2048);

            // uint32: embedding length
            w.write_string("llama.embedding_length");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(576);

            // float: rope freq base
            w.write_string("llama.rope.freq_base");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
            w.write<float>(10000.0f);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_EQ(parser.metadata_count(), 4u);
    EXPECT_EQ(parser.architecture(), "llama");
    EXPECT_EQ(parser.context_length(), 2048u);
    EXPECT_EQ(parser.embedding_length(), 576u);
    EXPECT_FLOAT_EQ(parser.rope_freq_base(), 10000.0f);
}

TEST(GGUFParser, ParseInt32Metadata) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("test.negative");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::INT32));
            w.write<int32_t>(-42);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_EQ(parser.get_int32("test.negative"), -42);
}

TEST(GGUFParser, ParseUint64Metadata) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("test.big_number");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT64));
            w.write<uint64_t>(0xDEADBEEFCAFE0000ULL);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_EQ(parser.get_uint64("test.big_number"), 0xDEADBEEFCAFE0000ULL);
}

TEST(GGUFParser, ParseArrayMetadata) {
    // Array of uint32 values
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("test.array");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
            // Array header: element type + count
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint64_t>(3); // 3 elements
            w.write<uint32_t>(10);
            w.write<uint32_t>(20);
            w.write<uint32_t>(30);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    auto it = parser.metadata().find("test.array");
    ASSERT_NE(it, parser.metadata().end());
    EXPECT_EQ(it->second.type, GGUFValueType::ARRAY);

    const auto& arr = it->second.as_array();
    ASSERT_EQ(arr.size(), 3u);
    EXPECT_EQ(arr[0].as_uint32(), 10u);
    EXPECT_EQ(arr[1].as_uint32(), 20u);
    EXPECT_EQ(arr[2].as_uint32(), 30u);
}

// ── Tensor Parsing ──────────────────────────────────────────

TEST(GGUFParser, ParseSingleF32Tensor) {
    // Create a 4x4 F32 tensor with known values
    constexpr size_t ROWS = 4, COLS = 4;
    constexpr size_t N = ROWS * COLS;
    std::vector<float> tensor_values(N);
    for (size_t i = 0; i < N; ++i) {
        tensor_values[i] = static_cast<float>(i) * 1.5f;
    }
    size_t tensor_bytes = N * sizeof(float);

    auto data = build_simple_gguf(3, 0, 1,
        nullptr,
        [&](GGUFWriter& w) {
            // Tensor info: name + ndims + dims + type + offset
            w.write_string("weight_matrix");
            w.write<uint32_t>(2); // 2 dimensions
            w.write<uint64_t>(ROWS);
            w.write<uint64_t>(COLS);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0); // offset 0 in data section
        },
        [&](GGUFWriter& w) {
            w.write_bytes(tensor_values.data(), tensor_bytes);
        });

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_EQ(parser.tensor_count(), 1u);
    ASSERT_EQ(parser.tensors().size(), 1u);

    const auto& t = parser.tensors()[0];
    EXPECT_EQ(t.name, "weight_matrix");
    ASSERT_EQ(t.dimensions.size(), 2u);
    EXPECT_EQ(t.dimensions[0], ROWS);
    EXPECT_EQ(t.dimensions[1], COLS);
    EXPECT_EQ(t.type, GGMLType::F32);
    EXPECT_EQ(t.element_count(), N);
    EXPECT_EQ(t.byte_size(), tensor_bytes);

    // Verify tensor data can be accessed and matches
    const float* parsed_data = static_cast<const float*>(parser.tensor_data(t));
    ASSERT_NE(parsed_data, nullptr);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(parsed_data[i], tensor_values[i])
            << "Mismatch at element " << i;
    }
}

TEST(GGUFParser, ParseMultipleTensors) {
    // Two tensors: a 4x4 F32 and an 8x1 F32 (using F32 for both to simplify
    // data verification — F16 would need half-float encoding)
    constexpr size_t N1 = 16; // 4x4
    constexpr size_t N2 = 8;  // 8x1
    std::vector<float> data1(N1), data2(N2);
    for (size_t i = 0; i < N1; ++i) data1[i] = static_cast<float>(i);
    for (size_t i = 0; i < N2; ++i) data2[i] = static_cast<float>(i + 100);

    size_t bytes1 = N1 * sizeof(float);
    size_t bytes2 = N2 * sizeof(float);

    // Tensor 2 offset: after tensor 1, aligned to 32 bytes
    uint64_t offset2 = ((bytes1 + GGUF_DEFAULT_ALIGNMENT - 1)
                        / GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;

    auto file_data = build_simple_gguf(3, 0, 2,
        nullptr,
        [&](GGUFWriter& w) {
            // Tensor 1 info
            w.write_string("layer.0.weight");
            w.write<uint32_t>(2);
            w.write<uint64_t>(4); w.write<uint64_t>(4);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0);

            // Tensor 2 info
            w.write_string("layer.0.bias");
            w.write<uint32_t>(1);
            w.write<uint64_t>(8);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(offset2);
        },
        [&](GGUFWriter& w) {
            // Tensor 1 data at offset 0
            w.write_bytes(data1.data(), bytes1);

            // Pad to tensor 2 offset
            size_t current = bytes1;
            while (current < static_cast<size_t>(offset2)) {
                w.write<uint8_t>(0);
                current++;
            }

            // Tensor 2 data
            w.write_bytes(data2.data(), bytes2);
        });

    TempFile tmp(file_data);
    GGUFParser parser(tmp.path());

    ASSERT_EQ(parser.tensors().size(), 2u);

    // Verify tensor 1
    const auto* t1 = parser.find_tensor("layer.0.weight");
    ASSERT_NE(t1, nullptr);
    EXPECT_EQ(t1->dimensions.size(), 2u);
    EXPECT_EQ(t1->element_count(), N1);

    const float* p1 = static_cast<const float*>(parser.tensor_data(*t1));
    for (size_t i = 0; i < N1; ++i) {
        EXPECT_FLOAT_EQ(p1[i], data1[i]);
    }

    // Verify tensor 2
    const auto* t2 = parser.find_tensor("layer.0.bias");
    ASSERT_NE(t2, nullptr);
    EXPECT_EQ(t2->dimensions.size(), 1u);
    EXPECT_EQ(t2->dimensions[0], 8u);

    const float* p2 = static_cast<const float*>(parser.tensor_data(*t2));
    for (size_t i = 0; i < N2; ++i) {
        EXPECT_FLOAT_EQ(p2[i], data2[i]);
    }
}

TEST(GGUFParser, TensorDataByName) {
    // Verify that tensor_data(string) works
    constexpr size_t N = 4;
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

    auto data = build_simple_gguf(3, 0, 1,
        nullptr,
        [](GGUFWriter& w) {
            w.write_string("my_tensor");
            w.write<uint32_t>(1);
            w.write<uint64_t>(4);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0);
        },
        [&](GGUFWriter& w) {
            w.write_bytes(values.data(), N * sizeof(float));
        });

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    const float* p = static_cast<const float*>(parser.tensor_data("my_tensor"));
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[3], 4.0f);
}

TEST(GGUFParser, FindTensorNotFound) {
    auto data = build_simple_gguf(3, 0, 0, nullptr, nullptr, nullptr);
    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_EQ(parser.find_tensor("nonexistent"), nullptr);
}

TEST(GGUFParser, TensorDataByNameThrowsOnMissing) {
    auto data = build_simple_gguf(3, 0, 0, nullptr, nullptr, nullptr);
    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_THROW(parser.tensor_data("nonexistent"), std::runtime_error);
}

// ── Metadata + Tensor Combined ──────────────────────────────

TEST(GGUFParser, FullModelLikeFile) {
    // Simulates a tiny model file with metadata and tensors
    constexpr size_t EMB_DIM = 8;
    constexpr size_t VOCAB = 32;
    constexpr size_t N_EMB = VOCAB * EMB_DIM; // 256 elements
    std::vector<float> emb_data(N_EMB);
    for (size_t i = 0; i < N_EMB; ++i) {
        emb_data[i] = static_cast<float>(i) * 0.01f;
    }

    constexpr size_t NORM_DIM = 8;
    std::vector<float> norm_data(NORM_DIM);
    for (size_t i = 0; i < NORM_DIM; ++i) {
        norm_data[i] = 1.0f;
    }

    size_t emb_bytes = N_EMB * sizeof(float);
    size_t norm_bytes = NORM_DIM * sizeof(float);

    uint64_t norm_offset = ((emb_bytes + GGUF_DEFAULT_ALIGNMENT - 1)
                            / GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;

    auto file_data = build_simple_gguf(3, 7, 2,
        [](GGUFWriter& w) {
            // general.architecture = "llama"
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");

            // llama.context_length = 512
            w.write_string("llama.context_length");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(512);

            // llama.embedding_length = 8
            w.write_string("llama.embedding_length");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(8);

            // llama.block_count = 2
            w.write_string("llama.block_count");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(2);

            // llama.attention.head_count = 4
            w.write_string("llama.attention.head_count");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(4);

            // llama.attention.head_count_kv = 2
            w.write_string("llama.attention.head_count_kv");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(2);

            // llama.feed_forward_length = 32
            w.write_string("llama.feed_forward_length");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(32);
        },
        [&](GGUFWriter& w) {
            // token_embd.weight: [32, 8] F32
            w.write_string("token_embd.weight");
            w.write<uint32_t>(2);
            w.write<uint64_t>(VOCAB);
            w.write<uint64_t>(EMB_DIM);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0);

            // output_norm.weight: [8] F32
            w.write_string("output_norm.weight");
            w.write<uint32_t>(1);
            w.write<uint64_t>(NORM_DIM);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(norm_offset);
        },
        [&](GGUFWriter& w) {
            // Embedding data
            w.write_bytes(emb_data.data(), emb_bytes);

            // Pad to norm offset
            size_t current = emb_bytes;
            while (current < static_cast<size_t>(norm_offset)) {
                w.write<uint8_t>(0);
                current++;
            }

            // Norm data
            w.write_bytes(norm_data.data(), norm_bytes);
        });

    TempFile tmp(file_data);
    GGUFParser parser(tmp.path());

    // Verify model architecture helpers
    EXPECT_EQ(parser.architecture(), "llama");
    EXPECT_EQ(parser.context_length(), 512u);
    EXPECT_EQ(parser.embedding_length(), 8u);
    EXPECT_EQ(parser.block_count(), 2u);
    EXPECT_EQ(parser.head_count(), 4u);
    EXPECT_EQ(parser.head_count_kv(), 2u);
    EXPECT_EQ(parser.feed_forward_length(), 32u);
    EXPECT_FLOAT_EQ(parser.rope_freq_base(), 10000.0f); // default

    // Verify tensors
    ASSERT_EQ(parser.tensors().size(), 2u);

    const auto* emb = parser.find_tensor("token_embd.weight");
    ASSERT_NE(emb, nullptr);
    EXPECT_EQ(emb->type, GGMLType::F32);
    EXPECT_EQ(emb->element_count(), N_EMB);

    const float* emb_ptr = static_cast<const float*>(parser.tensor_data(*emb));
    EXPECT_FLOAT_EQ(emb_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(emb_ptr[1], 0.01f);
    EXPECT_FLOAT_EQ(emb_ptr[N_EMB - 1], static_cast<float>(N_EMB - 1) * 0.01f);

    const auto* norm = parser.find_tensor("output_norm.weight");
    ASSERT_NE(norm, nullptr);
    EXPECT_EQ(norm->type, GGMLType::F32);

    const float* norm_ptr = static_cast<const float*>(parser.tensor_data(*norm));
    for (size_t i = 0; i < NORM_DIM; ++i) {
        EXPECT_FLOAT_EQ(norm_ptr[i], 1.0f);
    }
}

// ── Move Semantics ──────────────────────────────────────────

TEST(GGUFParser, MoveConstruction) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("test.key");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("value");
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser1(tmp.path());
    GGUFParser parser2(std::move(parser1));

    EXPECT_EQ(parser2.version(), 3u);
    EXPECT_EQ(parser2.get_string("test.key"), "value");
}

TEST(GGUFParser, MoveAssignment) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("hello");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(42);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser1(tmp.path());

    // Create a second empty-ish parser via another temp file
    auto data2 = build_simple_gguf(3, 0, 0, nullptr, nullptr, nullptr);
    TempFile tmp2(data2);
    GGUFParser parser2(tmp2.path());

    parser2 = std::move(parser1);
    EXPECT_EQ(parser2.get_uint32("hello"), 42u);
}

// ── Error Cases ─────────────────────────────────────────────

TEST(GGUFParser, InvalidMagic) {
    GGUFWriter w;
    w.write<uint32_t>(0xDEADBEEF); // wrong magic
    w.write<uint32_t>(3);
    w.write<uint64_t>(0);
    w.write<uint64_t>(0);

    TempFile tmp(w.data());
    EXPECT_THROW(GGUFParser parser(tmp.path()), std::runtime_error);
}

TEST(GGUFParser, UnsupportedVersion) {
    GGUFWriter w;
    w.write<uint32_t>(GGUF_MAGIC);
    w.write<uint32_t>(99); // unsupported version
    w.write<uint64_t>(0);
    w.write<uint64_t>(0);

    TempFile tmp(w.data());
    EXPECT_THROW(GGUFParser parser(tmp.path()), std::runtime_error);
}

TEST(GGUFParser, TruncatedHeader) {
    // Only 8 bytes — not enough for a full header
    GGUFWriter w;
    w.write<uint32_t>(GGUF_MAGIC);
    w.write<uint32_t>(3);
    // Missing tensor_count and metadata_count

    TempFile tmp(w.data());
    EXPECT_THROW(GGUFParser parser(tmp.path()), std::runtime_error);
}

TEST(GGUFParser, MissingMetadataKey) {
    auto data = build_simple_gguf(3, 0, 0, nullptr, nullptr, nullptr);
    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    EXPECT_FALSE(parser.has_metadata("nonexistent.key"));
    EXPECT_THROW(parser.get_string("nonexistent.key"), std::runtime_error);
    EXPECT_THROW(parser.get_uint32("nonexistent.key"), std::runtime_error);
}

TEST(GGUFParser, MetadataTypeMismatch) {
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("test.value");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(100);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    // Value is UINT32, but we try to read it as STRING
    EXPECT_THROW(parser.get_string("test.value"), std::runtime_error);
    // Should succeed as uint32
    EXPECT_EQ(parser.get_uint32("test.value"), 100u);
}

TEST(GGUFParser, FileNotFound) {
    EXPECT_THROW(
        GGUFParser parser("/tmp/nonexistent_gguf_file_abc123.gguf"),
        std::runtime_error);
}

// ── GGMLType Utilities ──────────────────────────────────────

TEST(GGMLType, TypeSizes) {
    EXPECT_EQ(ggml_type_size(GGMLType::F32), 4u);
    EXPECT_EQ(ggml_type_size(GGMLType::F16), 2u);
    EXPECT_EQ(ggml_type_size(GGMLType::Q4_0), 18u);
    EXPECT_EQ(ggml_type_size(GGMLType::Q8_0), 34u);
}

TEST(GGMLType, BlockSizes) {
    EXPECT_EQ(ggml_block_size(GGMLType::F32), 1u);
    EXPECT_EQ(ggml_block_size(GGMLType::F16), 1u);
    EXPECT_EQ(ggml_block_size(GGMLType::Q4_0), 32u);
    EXPECT_EQ(ggml_block_size(GGMLType::Q8_0), 32u);
    EXPECT_EQ(ggml_block_size(GGMLType::Q4_K), 256u);
}

TEST(GGMLType, TensorByteSize) {
    // F32 tensor: 16 elements = 64 bytes
    GGUFTensorInfo info;
    info.name = "test";
    info.dimensions = {4, 4};
    info.type = GGMLType::F32;
    EXPECT_EQ(info.byte_size(), 64u);
    EXPECT_EQ(info.element_count(), 16u);

    // Q4_0 tensor: 256 elements (8 blocks of 32) = 8 * 18 = 144 bytes
    GGUFTensorInfo q4_info;
    q4_info.name = "q4_test";
    q4_info.dimensions = {256};
    q4_info.type = GGMLType::Q4_0;
    EXPECT_EQ(q4_info.byte_size(), 144u);

    // Q8_0 tensor: 64 elements (2 blocks of 32) = 2 * 34 = 68 bytes
    GGUFTensorInfo q8_info;
    q8_info.name = "q8_test";
    q8_info.dimensions = {64};
    q8_info.type = GGMLType::Q8_0;
    EXPECT_EQ(q8_info.byte_size(), 68u);
}

TEST(GGMLType, TypeNames) {
    EXPECT_STREQ(ggml_type_name(GGMLType::F32), "F32");
    EXPECT_STREQ(ggml_type_name(GGMLType::F16), "F16");
    EXPECT_STREQ(ggml_type_name(GGMLType::Q4_0), "Q4_0");
    EXPECT_STREQ(ggml_type_name(GGMLType::Q8_0), "Q8_0");
    EXPECT_STREQ(ggml_type_name(GGMLType::Q4_K), "Q4_K");
}

// ── GGUFValue Type Accessors ────────────────────────────────

TEST(GGUFValue, TypeMismatchThrows) {
    GGUFValue val;
    val.type = GGUFValueType::UINT32;
    val.value = uint32_t(42);

    // Correct type works
    EXPECT_EQ(val.as_uint32(), 42u);

    // Wrong type throws
    EXPECT_THROW(val.as_string(), std::runtime_error);
    EXPECT_THROW(val.as_float32(), std::runtime_error);
    EXPECT_THROW(val.as_int32(), std::runtime_error);
    EXPECT_THROW(val.as_bool(), std::runtime_error);
}

// ── Print Info (smoke test) ─────────────────────────────────

TEST(GGUFParser, PrintInfoDoesNotCrash) {
    constexpr size_t N = 4;
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

    auto data = build_simple_gguf(3, 1, 1,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("test");
        },
        [](GGUFWriter& w) {
            w.write_string("layer.0.weight");
            w.write<uint32_t>(1);
            w.write<uint64_t>(4);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0);
        },
        [&](GGUFWriter& w) {
            w.write_bytes(values.data(), N * sizeof(float));
        });

    TempFile tmp(data);
    GGUFParser parser(tmp.path());

    // Should not crash or throw
    EXPECT_NO_THROW(parser.print_info());
}

// ── Rope Freq Base Default ──────────────────────────────────

TEST(GGUFParser, RopeFreqBaseDefault) {
    // When rope.freq_base is not in metadata, should return 10000.0
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_FLOAT_EQ(parser.rope_freq_base(), 10000.0f);
}

// ── rms_norm_eps() ──────────────────────────────────────────

TEST(GGUFParser, RmsNormEpsExplicit) {
    // When the metadata key is present, return its value
    auto data = build_simple_gguf(3, 2, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");

            w.write_string("llama.attention.layer_norm_rms_epsilon");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::FLOAT32));
            w.write<float>(1e-6f);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_FLOAT_EQ(parser.rms_norm_eps(), 1e-6f);
}

TEST(GGUFParser, RmsNormEpsDefault) {
    // When the key is absent, should return default 1e-5
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_FLOAT_EQ(parser.rms_norm_eps(), 1e-5f);
}

// ── rope_dimension_count() ──────────────────────────────────

TEST(GGUFParser, RopeDimensionCount) {
    auto data = build_simple_gguf(3, 2, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");

            w.write_string("llama.rope.dimension_count");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(64);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_EQ(parser.rope_dimension_count(), 64u);
}

// ── vocab_size() ────────────────────────────────────────────

TEST(GGUFParser, VocabSizeFromMetadata) {
    // When {arch}.vocab_size is present, use it
    auto data = build_simple_gguf(3, 2, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");

            w.write_string("llama.vocab_size");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::UINT32));
            w.write<uint32_t>(49152);
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_EQ(parser.vocab_size(), 49152u);
}

TEST(GGUFParser, VocabSizeFromTokenizerArray) {
    // When {arch}.vocab_size is absent, fall back to tokenizer.ggml.tokens length
    auto data = build_simple_gguf(3, 2, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");

            // Array of 5 token strings
            w.write_string("tokenizer.ggml.tokens");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::ARRAY));
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write<uint64_t>(5); // 5 elements
            w.write_string("hello");
            w.write_string("world");
            w.write_string("foo");
            w.write_string("bar");
            w.write_string("baz");
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_EQ(parser.vocab_size(), 5u);
}

TEST(GGUFParser, VocabSizeThrowsWhenMissing) {
    // When neither source is available, should throw
    auto data = build_simple_gguf(3, 1, 0,
        [](GGUFWriter& w) {
            w.write_string("general.architecture");
            w.write<uint32_t>(static_cast<uint32_t>(GGUFValueType::STRING));
            w.write_string("llama");
        },
        nullptr, nullptr);

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_THROW(parser.vocab_size(), std::runtime_error);
}

// ── tie_word_embeddings() ───────────────────────────────────

TEST(GGUFParser, TieWordEmbeddingsWhenOutputAbsent) {
    // No output.weight tensor -> embeddings are tied
    constexpr size_t N = 4;
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

    auto data = build_simple_gguf(3, 0, 1,
        nullptr,
        [](GGUFWriter& w) {
            w.write_string("token_embd.weight");
            w.write<uint32_t>(1);
            w.write<uint64_t>(4);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0);
        },
        [&](GGUFWriter& w) {
            w.write_bytes(values.data(), N * sizeof(float));
        });

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_TRUE(parser.tie_word_embeddings());
}

TEST(GGUFParser, TieWordEmbeddingsFalseWhenOutputPresent) {
    // output.weight tensor exists -> embeddings are NOT tied
    constexpr size_t N = 4;
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t bytes = N * sizeof(float);

    uint64_t offset2 = ((bytes + GGUF_DEFAULT_ALIGNMENT - 1)
                        / GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;

    auto data = build_simple_gguf(3, 0, 2,
        nullptr,
        [&](GGUFWriter& w) {
            w.write_string("token_embd.weight");
            w.write<uint32_t>(1);
            w.write<uint64_t>(4);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(0);

            w.write_string("output.weight");
            w.write<uint32_t>(1);
            w.write<uint64_t>(4);
            w.write<uint32_t>(static_cast<uint32_t>(GGMLType::F32));
            w.write<uint64_t>(offset2);
        },
        [&](GGUFWriter& w) {
            w.write_bytes(values.data(), bytes);
            // Pad to second tensor
            size_t current = bytes;
            while (current < static_cast<size_t>(offset2)) {
                w.write<uint8_t>(0);
                current++;
            }
            w.write_bytes(values.data(), bytes);
        });

    TempFile tmp(data);
    GGUFParser parser(tmp.path());
    EXPECT_FALSE(parser.tie_word_embeddings());
}
