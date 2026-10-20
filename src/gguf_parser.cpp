// GGUF Binary Format Parser — Implementation
//
// This file implements a zero-copy GGUF parser that memory-maps the model file
// and reads header, metadata, and tensor descriptors sequentially. Tensor data
// is accessed directly through the mapped region, avoiding any data copies.
//
// The parsing strategy is streaming: we maintain a read cursor that advances
// through the file as we decode each section. This matches the GGUF layout
// where fields are packed sequentially without a central directory.

#include "cuinfer/gguf_parser.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sstream>

// ── Platform-specific memory mapping ────────────────────────
// POSIX (Linux/WSL2): mmap/munmap
// Windows: CreateFileMapping/MapViewOfFile

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace cuinfer {

// ── Utility Functions ───────────────────────────────────────

const char* gguf_value_type_name(GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:   return "UINT8";
        case GGUFValueType::INT8:    return "INT8";
        case GGUFValueType::UINT16:  return "UINT16";
        case GGUFValueType::INT16:   return "INT16";
        case GGUFValueType::UINT32:  return "UINT32";
        case GGUFValueType::INT32:   return "INT32";
        case GGUFValueType::FLOAT32: return "FLOAT32";
        case GGUFValueType::BOOL:    return "BOOL";
        case GGUFValueType::STRING:  return "STRING";
        case GGUFValueType::ARRAY:   return "ARRAY";
        case GGUFValueType::UINT64:  return "UINT64";
        case GGUFValueType::INT64:   return "INT64";
        case GGUFValueType::FLOAT64: return "FLOAT64";
        default:                     return "UNKNOWN";
    }
}

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        default:             return "UNKNOWN";
    }
}

size_t ggml_type_size(GGMLType type) {
    // Returns the byte size of one block. For unquantized types, a "block"
    // is a single element. For quantized types, each block packs multiple
    // elements (see ggml_block_size).
    switch (type) {
        case GGMLType::F32:  return 4;   // 1 float32
        case GGMLType::F16:  return 2;   // 1 float16
        case GGMLType::Q4_0: return 18;  // 1 fp16 scale + 16 bytes (32 x 4-bit values)
        case GGMLType::Q4_1: return 20;  // 1 fp16 scale + 1 fp16 min + 16 bytes
        case GGMLType::Q5_0: return 22;  // 1 fp16 scale + 4 bytes high-bits + 16 bytes low-nibbles
        case GGMLType::Q5_1: return 24;  // 1 fp16 scale + 1 fp16 min + 4 bytes high-bits + 16 bytes
        case GGMLType::Q8_0: return 34;  // 1 fp16 scale + 32 int8 values
        case GGMLType::Q8_1: return 36;  // 1 fp16 scale + 1 fp16 sum + 32 int8 values
        // K-quant types have more complex block structures
        case GGMLType::Q2_K: return 256; // super-block of 256 elements
        case GGMLType::Q3_K: return 256; // varies — using approximate size
        case GGMLType::Q4_K: return 144; // super-block of 256 elements
        case GGMLType::Q5_K: return 176; // super-block of 256 elements
        case GGMLType::Q6_K: return 210; // super-block of 256 elements
        default:
            throw std::runtime_error(
                std::string("ggml_type_size: unsupported type ") +
                std::to_string(static_cast<uint32_t>(type)));
    }
}

size_t ggml_block_size(GGMLType type) {
    // Number of elements packed into one block.
    switch (type) {
        case GGMLType::F32:  return 1;
        case GGMLType::F16:  return 1;
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q8_1: return 32;
        case GGMLType::Q2_K: return 256;
        case GGMLType::Q3_K: return 256;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        default:
            throw std::runtime_error(
                std::string("ggml_block_size: unsupported type ") +
                std::to_string(static_cast<uint32_t>(type)));
    }
}

// ── GGUFValue Accessors ─────────────────────────────────────
// Each accessor checks the type tag before extracting from the variant,
// providing clear error messages on mismatch.

#define GGUF_VALUE_ACCESSOR(method_name, cpp_type, enum_val, variant_index) \
    cpp_type GGUFValue::method_name() const {                               \
        if (type != GGUFValueType::enum_val) {                              \
            throw std::runtime_error(                                       \
                std::string("GGUFValue type mismatch: expected ") +         \
                #enum_val + ", got " + gguf_value_type_name(type));          \
        }                                                                   \
        return std::get<cpp_type>(value);                                   \
    }

GGUF_VALUE_ACCESSOR(as_uint8,   uint8_t,     UINT8,   0)
GGUF_VALUE_ACCESSOR(as_int8,    int8_t,      INT8,    1)
GGUF_VALUE_ACCESSOR(as_uint16,  uint16_t,    UINT16,  2)
GGUF_VALUE_ACCESSOR(as_int16,   int16_t,     INT16,   3)
GGUF_VALUE_ACCESSOR(as_uint32,  uint32_t,    UINT32,  4)
GGUF_VALUE_ACCESSOR(as_int32,   int32_t,     INT32,   5)
GGUF_VALUE_ACCESSOR(as_float32, float,       FLOAT32, 6)
GGUF_VALUE_ACCESSOR(as_bool,    bool,        BOOL,    7)
GGUF_VALUE_ACCESSOR(as_uint64,  uint64_t,    UINT64,  10)
GGUF_VALUE_ACCESSOR(as_int64,   int64_t,     INT64,   11)
GGUF_VALUE_ACCESSOR(as_float64, double,      FLOAT64, 12)

#undef GGUF_VALUE_ACCESSOR

const std::string& GGUFValue::as_string() const {
    if (type != GGUFValueType::STRING) {
        throw std::runtime_error(
            std::string("GGUFValue type mismatch: expected STRING, got ") +
            gguf_value_type_name(type));
    }
    return std::get<std::string>(value);
}

const std::vector<GGUFValue>& GGUFValue::as_array() const {
    if (type != GGUFValueType::ARRAY) {
        throw std::runtime_error(
            std::string("GGUFValue type mismatch: expected ARRAY, got ") +
            gguf_value_type_name(type));
    }
    return std::get<std::vector<GGUFValue>>(value);
}

// ── GGUFTensorInfo ──────────────────────────────────────────

uint64_t GGUFTensorInfo::element_count() const {
    if (dimensions.empty()) return 0;
    uint64_t count = 1;
    for (auto d : dimensions) {
        count *= d;
    }
    return count;
}

size_t GGUFTensorInfo::byte_size() const {
    uint64_t n_elements = element_count();
    size_t block_sz = ggml_block_size(type);
    size_t type_sz = ggml_type_size(type);

    // For quantized types, elements must be aligned to block boundaries
    if (block_sz > 1 && (n_elements % block_sz != 0)) {
        throw std::runtime_error(
            "Tensor '" + name + "' has " + std::to_string(n_elements) +
            " elements, not divisible by block size " + std::to_string(block_sz));
    }

    // total_bytes = (n_elements / block_size) * type_size
    return static_cast<size_t>((n_elements / block_sz) * type_sz);
}

// ── StreamReader ────────────────────────────────────────────
// A lightweight helper that wraps a memory region with bounds-checked
// sequential reads. Advances an internal cursor after each read.

class StreamReader {
public:
    StreamReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    size_t position() const { return pos_; }
    size_t remaining() const { return size_ - pos_; }

    void seek(size_t pos) {
        if (pos > size_) {
            throw std::runtime_error(
                "StreamReader: seek to " + std::to_string(pos) +
                " exceeds file size " + std::to_string(size_));
        }
        pos_ = pos;
    }

    // Read a fixed-size value and advance the cursor
    template <typename T>
    T read() {
        check_bounds(sizeof(T));
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    // Read a GGUF-format string: uint64 length prefix + raw chars (no null terminator)
    std::string read_string() {
        uint64_t len = read<uint64_t>();
        check_bounds(static_cast<size_t>(len));
        std::string s(reinterpret_cast<const char*>(data_ + pos_),
                      static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return s;
    }

    // Read a GGUF metadata value based on its type tag
    GGUFValue read_value(GGUFValueType type) {
        GGUFValue val;
        val.type = type;

        switch (type) {
            case GGUFValueType::UINT8:
                val.value = read<uint8_t>();
                break;
            case GGUFValueType::INT8:
                val.value = read<int8_t>();
                break;
            case GGUFValueType::UINT16:
                val.value = read<uint16_t>();
                break;
            case GGUFValueType::INT16:
                val.value = read<int16_t>();
                break;
            case GGUFValueType::UINT32:
                val.value = read<uint32_t>();
                break;
            case GGUFValueType::INT32:
                val.value = read<int32_t>();
                break;
            case GGUFValueType::FLOAT32:
                val.value = read<float>();
                break;
            case GGUFValueType::BOOL: {
                // GGUF bools are stored as uint8 (0 = false, nonzero = true)
                uint8_t b = read<uint8_t>();
                val.value = (b != 0);
                break;
            }
            case GGUFValueType::STRING:
                val.value = read_string();
                break;
            case GGUFValueType::ARRAY: {
                // Arrays have: element_type (uint32) + count (uint64) + elements
                auto element_type = static_cast<GGUFValueType>(read<uint32_t>());
                uint64_t count = read<uint64_t>();
                std::vector<GGUFValue> elements;
                elements.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; ++i) {
                    elements.push_back(read_value(element_type));
                }
                val.value = std::move(elements);
                break;
            }
            case GGUFValueType::UINT64:
                val.value = read<uint64_t>();
                break;
            case GGUFValueType::INT64:
                val.value = read<int64_t>();
                break;
            case GGUFValueType::FLOAT64:
                val.value = read<double>();
                break;
            default:
                throw std::runtime_error(
                    "Unknown GGUF value type: " +
                    std::to_string(static_cast<uint32_t>(type)));
        }
        return val;
    }

private:
    void check_bounds(size_t bytes) const {
        if (pos_ + bytes > size_) {
            throw std::runtime_error(
                "StreamReader: read of " + std::to_string(bytes) +
                " bytes at position " + std::to_string(pos_) +
                " exceeds file size " + std::to_string(size_));
        }
    }

    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

// ── MemoryMappedFile ────────────────────────────────────────
// RAII wrapper around platform-specific memory mapping.

class MemoryMappedFile {
public:
    MemoryMappedFile() = default;

    explicit MemoryMappedFile(const std::string& path) {
#ifdef _WIN32
        // Windows: CreateFile -> CreateFileMapping -> MapViewOfFile
        file_handle_ = CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle_, &file_size)) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Failed to get file size: " + path);
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        if (size_ == 0) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("File is empty: " + path);
        }

        mapping_handle_ = CreateFileMappingA(
            file_handle_,
            nullptr,
            PAGE_READONLY,
            0, 0,  // Map entire file
            nullptr);

        if (mapping_handle_ == nullptr) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Failed to create file mapping: " + path);
        }

        data_ = static_cast<uint8_t*>(
            MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));

        if (data_ == nullptr) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Failed to map file into memory: " + path);
        }
#else
        // POSIX: open -> fstat -> mmap
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to stat file: " + path);
        }
        size_ = static_cast<size_t>(st.st_size);

        if (size_ == 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("File is empty: " + path);
        }

        void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to mmap file: " + path);
        }
        data_ = static_cast<uint8_t*>(mapped);
#endif
    }

    ~MemoryMappedFile() {
        unmap();
    }

    // Non-copyable
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    // Movable
    MemoryMappedFile(MemoryMappedFile&& other) noexcept
        : data_(other.data_), size_(other.size_)
#ifdef _WIN32
        , file_handle_(other.file_handle_)
        , mapping_handle_(other.mapping_handle_)
#else
        , fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }

    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept {
        if (this != &other) {
            unmap();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_handle_ = other.file_handle_;
            mapping_handle_ = other.mapping_handle_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_handle_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void unmap() {
#ifdef _WIN32
        if (data_) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_handle_) {
            CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_ && size_ > 0) {
            ::munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
    }

    uint8_t* data_ = nullptr;
    size_t size_ = 0;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ── GGUFParser::Impl ────────────────────────────────────────
// The private implementation holds the memory-mapped file and all
// parsed data structures.

struct GGUFParser::Impl {
    MemoryMappedFile mmap;

    // Header fields
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;

    // Parsed data
    std::unordered_map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;

    // Byte offset where the tensor data section begins (after alignment padding)
    size_t data_section_offset = 0;
};

// ── GGUFParser ──────────────────────────────────────────────

GGUFParser::GGUFParser(const std::string& file_path)
    : impl_(std::make_unique<Impl>())
{
    // Step 1: Memory-map the file
    impl_->mmap = MemoryMappedFile(file_path);

    const size_t file_size = impl_->mmap.size();
    StreamReader reader(impl_->mmap.data(), file_size);

    // Step 2: Parse and validate the header
    //   Bytes 0-3:   magic number ("GGUF" = 0x46475547 little-endian)
    //   Bytes 4-7:   format version (uint32)
    //   Bytes 8-15:  tensor count (uint64)
    //   Bytes 16-23: metadata kv count (uint64)

    constexpr size_t HEADER_SIZE = 4 + 4 + 8 + 8; // 24 bytes
    if (file_size < HEADER_SIZE) {
        throw std::runtime_error(
            "GGUF file too small for header (" + std::to_string(file_size) +
            " bytes, need at least " + std::to_string(HEADER_SIZE) + ")");
    }

    uint32_t magic = reader.read<uint32_t>();
    if (magic != GGUF_MAGIC) {
        std::ostringstream oss;
        oss << "Invalid GGUF magic: expected 0x" << std::hex << GGUF_MAGIC
            << ", got 0x" << magic;
        throw std::runtime_error(oss.str());
    }

    impl_->version = reader.read<uint32_t>();
    if (impl_->version < 2 || impl_->version > 3) {
        throw std::runtime_error(
            "Unsupported GGUF version: " + std::to_string(impl_->version) +
            " (supported: 2, 3)");
    }

    impl_->tensor_count = reader.read<uint64_t>();
    impl_->metadata_kv_count = reader.read<uint64_t>();

    // Step 3: Parse metadata key-value pairs
    // Each entry is: gguf_string key + uint32 value_type + value
    for (uint64_t i = 0; i < impl_->metadata_kv_count; ++i) {
        std::string key = reader.read_string();
        auto value_type = static_cast<GGUFValueType>(reader.read<uint32_t>());
        GGUFValue value = reader.read_value(value_type);
        impl_->metadata[std::move(key)] = std::move(value);
    }

    // Step 4: Parse tensor info entries
    // Each entry is: gguf_string name + uint32 n_dims + uint64[n_dims] dims
    //              + uint32 type + uint64 offset_in_data_section
    impl_->tensors.reserve(static_cast<size_t>(impl_->tensor_count));

    for (uint64_t i = 0; i < impl_->tensor_count; ++i) {
        GGUFTensorInfo info;
        info.name = reader.read_string();

        uint32_t n_dims = reader.read<uint32_t>();
        info.dimensions.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.dimensions[d] = reader.read<uint64_t>();
        }

        info.type = static_cast<GGMLType>(reader.read<uint32_t>());
        info.offset = reader.read<uint64_t>();

        // data_offset will be computed after we know where the data section starts
        info.data_offset = 0;

        impl_->tensors.push_back(std::move(info));
    }

    // Step 5: Calculate data section start
    // The data section is aligned to GGUF_DEFAULT_ALIGNMENT (32 bytes) after
    // all the tensor info entries.
    size_t cursor = reader.position();
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;

    // Check if a custom alignment is specified in metadata
    auto it = impl_->metadata.find("general.alignment");
    if (it != impl_->metadata.end() && it->second.type == GGUFValueType::UINT32) {
        alignment = it->second.as_uint32();
        if (alignment == 0) alignment = GGUF_DEFAULT_ALIGNMENT;
    }

    // Round up to alignment boundary
    size_t data_start = ((cursor + alignment - 1) / alignment) * alignment;
    impl_->data_section_offset = data_start;

    // Step 6: Compute absolute file offsets for each tensor
    for (auto& t : impl_->tensors) {
        t.data_offset = data_start + t.offset;
    }
}

GGUFParser::~GGUFParser() = default;

GGUFParser::GGUFParser(GGUFParser&&) noexcept = default;
GGUFParser& GGUFParser::operator=(GGUFParser&&) noexcept = default;

// ── Header Accessors ────────────────────────────────────────

uint32_t GGUFParser::version() const {
    return impl_->version;
}

uint64_t GGUFParser::tensor_count() const {
    return impl_->tensor_count;
}

uint64_t GGUFParser::metadata_count() const {
    return impl_->metadata_kv_count;
}

// ── Metadata Access ─────────────────────────────────────────

const std::unordered_map<std::string, GGUFValue>& GGUFParser::metadata() const {
    return impl_->metadata;
}

bool GGUFParser::has_metadata(const std::string& key) const {
    return impl_->metadata.count(key) > 0;
}

std::string GGUFParser::get_string(const std::string& key) const {
    auto it = impl_->metadata.find(key);
    if (it == impl_->metadata.end()) {
        throw std::runtime_error("GGUF metadata key not found: " + key);
    }
    return it->second.as_string();
}

uint32_t GGUFParser::get_uint32(const std::string& key) const {
    auto it = impl_->metadata.find(key);
    if (it == impl_->metadata.end()) {
        throw std::runtime_error("GGUF metadata key not found: " + key);
    }
    return it->second.as_uint32();
}

int32_t GGUFParser::get_int32(const std::string& key) const {
    auto it = impl_->metadata.find(key);
    if (it == impl_->metadata.end()) {
        throw std::runtime_error("GGUF metadata key not found: " + key);
    }
    return it->second.as_int32();
}

uint64_t GGUFParser::get_uint64(const std::string& key) const {
    auto it = impl_->metadata.find(key);
    if (it == impl_->metadata.end()) {
        throw std::runtime_error("GGUF metadata key not found: " + key);
    }
    return it->second.as_uint64();
}

float GGUFParser::get_float32(const std::string& key) const {
    auto it = impl_->metadata.find(key);
    if (it == impl_->metadata.end()) {
        throw std::runtime_error("GGUF metadata key not found: " + key);
    }
    return it->second.as_float32();
}

// ── Model Architecture Helpers ──────────────────────────────
// These read standardized GGUF keys that model converters populate.
// The pattern is: "general.architecture" gives the arch name (e.g., "llama"),
// then arch-specific keys follow the pattern "{arch}.{property}".

std::string GGUFParser::architecture() const {
    return get_string("general.architecture");
}

uint32_t GGUFParser::context_length() const {
    return get_uint32(architecture() + ".context_length");
}

uint32_t GGUFParser::embedding_length() const {
    return get_uint32(architecture() + ".embedding_length");
}

uint32_t GGUFParser::block_count() const {
    return get_uint32(architecture() + ".block_count");
}

uint32_t GGUFParser::head_count() const {
    return get_uint32(architecture() + ".attention.head_count");
}

uint32_t GGUFParser::head_count_kv() const {
    return get_uint32(architecture() + ".attention.head_count_kv");
}

uint32_t GGUFParser::feed_forward_length() const {
    return get_uint32(architecture() + ".feed_forward_length");
}

float GGUFParser::rope_freq_base() const {
    std::string key = architecture() + ".rope.freq_base";
    if (!has_metadata(key)) {
        return 10000.0f; // Default RoPE frequency base
    }
    return get_float32(key);
}

float GGUFParser::rms_norm_eps() const {
    std::string key = architecture() + ".attention.layer_norm_rms_epsilon";
    if (!has_metadata(key)) {
        return 1e-5f; // Standard default for LLaMA-family models
    }
    return get_float32(key);
}

uint32_t GGUFParser::rope_dimension_count() const {
    return get_uint32(architecture() + ".rope.dimension_count");
}

uint32_t GGUFParser::vocab_size() const {
    // Primary: explicit metadata key set by the converter
    std::string key = architecture() + ".vocab_size";
    if (has_metadata(key)) {
        return get_uint32(key);
    }

    // Fallback: derive from the tokenizer vocabulary array length
    const std::string tokens_key = "tokenizer.ggml.tokens";
    if (has_metadata(tokens_key)) {
        auto it = impl_->metadata.find(tokens_key);
        return static_cast<uint32_t>(it->second.as_array().size());
    }

    throw std::runtime_error(
        "Cannot determine vocab_size: neither '" + key +
        "' nor 'tokenizer.ggml.tokens' found in metadata");
}

bool GGUFParser::tie_word_embeddings() const {
    // When the converter omits "output.weight", the LM head reuses
    // token_embd.weight — i.e., embeddings are tied.
    return find_tensor("output.weight") == nullptr;
}

// ── Tensor Access ───────────────────────────────────────────

const std::vector<GGUFTensorInfo>& GGUFParser::tensors() const {
    return impl_->tensors;
}

const GGUFTensorInfo* GGUFParser::find_tensor(const std::string& name) const {
    // Linear scan is fine for typical model sizes (hundreds of tensors, not millions).
    // If this becomes a bottleneck, we can add a name->index map.
    for (const auto& t : impl_->tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

const void* GGUFParser::tensor_data(const GGUFTensorInfo& info) const {
    // Bounds check: ensure the tensor's data fits within the mapped region
    if (info.data_offset + info.byte_size() > impl_->mmap.size()) {
        throw std::runtime_error(
            "Tensor '" + info.name + "' data extends beyond file boundary "
            "(offset=" + std::to_string(info.data_offset) +
            ", size=" + std::to_string(info.byte_size()) +
            ", file_size=" + std::to_string(impl_->mmap.size()) + ")");
    }
    return impl_->mmap.data() + info.data_offset;
}

const void* GGUFParser::tensor_data(const std::string& name) const {
    const GGUFTensorInfo* info = find_tensor(name);
    if (!info) {
        throw std::runtime_error("GGUFParser::tensor_data: tensor not found: '" + name + "'");
    }
    return tensor_data(*info);
}

// ── Debug Output ────────────────────────────────────────────

void GGUFParser::print_info() const {
    printf("=== GGUF File Info ===\n");
    printf("Version:        %u\n", impl_->version);
    printf("Tensor count:   %llu\n",
           static_cast<unsigned long long>(impl_->tensor_count));
    printf("Metadata count: %llu\n",
           static_cast<unsigned long long>(impl_->metadata_kv_count));
    printf("Data offset:    0x%zx (%zu bytes)\n",
           impl_->data_section_offset, impl_->data_section_offset);
    printf("\n");

    printf("--- Metadata ---\n");
    for (const auto& [key, val] : impl_->metadata) {
        printf("  %-40s [%s]", key.c_str(), gguf_value_type_name(val.type));
        // Print scalar values inline for quick inspection
        switch (val.type) {
            case GGUFValueType::STRING:
                printf(" = \"%s\"", val.as_string().c_str());
                break;
            case GGUFValueType::UINT32:
                printf(" = %u", val.as_uint32());
                break;
            case GGUFValueType::INT32:
                printf(" = %d", val.as_int32());
                break;
            case GGUFValueType::UINT64:
                printf(" = %llu",
                       static_cast<unsigned long long>(val.as_uint64()));
                break;
            case GGUFValueType::FLOAT32:
                printf(" = %f", val.as_float32());
                break;
            case GGUFValueType::BOOL:
                printf(" = %s", val.as_bool() ? "true" : "false");
                break;
            case GGUFValueType::ARRAY:
                printf(" (%zu elements)",
                       val.as_array().size());
                break;
            default:
                break;
        }
        printf("\n");
    }
    printf("\n");

    printf("--- Tensors ---\n");
    size_t total_bytes = 0;
    for (const auto& t : impl_->tensors) {
        printf("  %-50s [%s] ", t.name.c_str(), ggml_type_name(t.type));
        // Print shape like [4096, 4096]
        printf("[");
        for (size_t d = 0; d < t.dimensions.size(); ++d) {
            if (d > 0) printf(", ");
            printf("%llu", static_cast<unsigned long long>(t.dimensions[d]));
        }
        printf("]");
        size_t bytes = t.byte_size();
        total_bytes += bytes;
        printf("  (%.2f MB)\n", bytes / (1024.0 * 1024.0));
    }
    printf("\nTotal tensor data: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
}

} // namespace cuinfer
