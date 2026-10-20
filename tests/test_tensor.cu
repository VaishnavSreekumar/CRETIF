#include <gtest/gtest.h>
#include "cuinfer/tensor.h"

#include <cstring>
#include <numeric>
#include <vector>

using namespace cuinfer;

// ── CreateAndShape ──────────────────────────────────────────

TEST(Tensor, CreateAndShape) {
    // 2D tensor: 32 x 768
    Tensor t({32, 768});

    // Shape should be right-aligned into 4D: {1, 1, 32, 768}.
    EXPECT_EQ(t.shape(0), 1);
    EXPECT_EQ(t.shape(1), 1);
    EXPECT_EQ(t.shape(2), 32);
    EXPECT_EQ(t.shape(3), 768);

    EXPECT_EQ(t.numel(), 32 * 768);
    EXPECT_EQ(t.size_bytes(), static_cast<size_t>(32 * 768 * sizeof(float)));
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.dtype(), DType::FP32);
    EXPECT_TRUE(t.is_owning());
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_NE(t.data(), nullptr);

    // Verify strides are row-major.
    EXPECT_EQ(t.stride(3), 1);
    EXPECT_EQ(t.stride(2), 768);
    EXPECT_EQ(t.stride(1), 32 * 768);
    EXPECT_EQ(t.stride(0), 1 * 32 * 768);
}

// ── HostRoundtrip ───────────────────────────────────────────

TEST(Tensor, HostRoundtrip) {
    const int N = 256;
    std::vector<float> src(N);
    for (int i = 0; i < N; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
    }

    Tensor t = Tensor::from_host(src.data(), {N});

    std::vector<float> dst(N, -1.0f);
    t.copy_to_host(dst.data(), N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i])
            << "Mismatch at index " << i;
    }
}

// ── Zeros ───────────────────────────────────────────────────

TEST(Tensor, Zeros) {
    const int ROWS = 16;
    const int COLS = 64;
    Tensor t = Tensor::zeros({ROWS, COLS});

    std::vector<float> host(ROWS * COLS, 999.0f);
    t.copy_to_host(host.data(), t.size_bytes());

    for (int i = 0; i < ROWS * COLS; ++i) {
        EXPECT_EQ(host[i], 0.0f)
            << "Non-zero at index " << i;
    }
}

// ── Clone ───────────────────────────────────────────────────

TEST(Tensor, Clone) {
    const int N = 128;
    std::vector<float> src(N);
    for (int i = 0; i < N; ++i) {
        src[i] = static_cast<float>(i);
    }

    Tensor original = Tensor::from_host(src.data(), {N});
    Tensor copy = original.clone();

    // Cloned tensor must be owning and have a different device pointer.
    EXPECT_TRUE(copy.is_owning());
    EXPECT_NE(copy.data(), original.data());
    EXPECT_EQ(copy.numel(), original.numel());
    EXPECT_EQ(copy.dtype(), original.dtype());

    // Data content must match.
    std::vector<float> orig_host(N);
    std::vector<float> copy_host(N);
    original.copy_to_host(orig_host.data(), N * sizeof(float));
    copy.copy_to_host(copy_host.data(), N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(copy_host[i], orig_host[i])
            << "Mismatch at index " << i;
    }
}

// ── MoveSemantics ───────────────────────────────────────────

TEST(Tensor, MoveSemantics) {
    Tensor original({64, 64});
    const void* orig_ptr = original.data();
    const int64_t orig_numel = original.numel();

    ASSERT_NE(orig_ptr, nullptr);

    // Move-construct a new tensor.
    Tensor moved(std::move(original));

    // The moved-to tensor should hold the original's pointer and metadata.
    EXPECT_EQ(moved.data(), orig_ptr);
    EXPECT_EQ(moved.numel(), orig_numel);
    EXPECT_TRUE(moved.is_owning());

    // The moved-from tensor should be empty / null.
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_FALSE(original.is_owning());
    EXPECT_EQ(original.numel(), 1); // all dims reset to 1
}

// ── View ────────────────────────────────────────────────────

TEST(Tensor, View) {
    // Create a 4 x 8 tensor filled with sequential values.
    const int ROWS = 4;
    const int COLS = 8;
    std::vector<float> src(ROWS * COLS);
    for (int i = 0; i < ROWS * COLS; ++i) {
        src[i] = static_cast<float>(i);
    }
    Tensor t = Tensor::from_host(src.data(), {ROWS, COLS});

    // Slice rows [1, 3) — should give a 2 x 8 view.
    // dim 2 in the 4D layout corresponds to the "ROWS" axis
    // because {4,8} is stored as {1,1,4,8}.
    Tensor v = t.view(/*dim=*/2, /*start=*/1, /*end=*/3);

    EXPECT_FALSE(v.is_owning());
    EXPECT_EQ(v.shape(2), 2);
    EXPECT_EQ(v.shape(3), 8);
    EXPECT_EQ(v.numel(), 2 * 8);

    // The view's data pointer should be offset by 1 row (8 floats).
    const auto* base = static_cast<const char*>(t.data());
    const auto* view_base = static_cast<const char*>(v.data());
    EXPECT_EQ(view_base - base, static_cast<ptrdiff_t>(1 * COLS * sizeof(float)));

    // Verify contents: rows 1 and 2 of the original.
    std::vector<float> host(2 * COLS);
    v.copy_to_host(host.data(), v.size_bytes());

    for (int i = 0; i < 2 * COLS; ++i) {
        EXPECT_FLOAT_EQ(host[i], src[1 * COLS + i])
            << "View mismatch at index " << i;
    }
}

// ── Reshape ─────────────────────────────────────────────────

TEST(Tensor, Reshape) {
    const int ROWS = 4;
    const int COLS = 8;
    std::vector<float> src(ROWS * COLS);
    for (int i = 0; i < ROWS * COLS; ++i) {
        src[i] = static_cast<float>(i);
    }
    Tensor t = Tensor::from_host(src.data(), {ROWS, COLS});

    // Reshape to 1D: {32}.
    Tensor flat = t.reshape({ROWS * COLS});
    EXPECT_EQ(flat.numel(), ROWS * COLS);
    EXPECT_EQ(flat.shape(3), ROWS * COLS); // right-aligned 1D
    EXPECT_FALSE(flat.is_owning());

    // Data pointer must be identical (no copy).
    EXPECT_EQ(flat.data(), t.data());

    // Verify data survives the reshape unchanged.
    std::vector<float> host(ROWS * COLS);
    flat.copy_to_host(host.data(), flat.size_bytes());
    for (int i = 0; i < ROWS * COLS; ++i) {
        EXPECT_FLOAT_EQ(host[i], src[i]);
    }

    // Reshape to 3D: {2, 2, 8} — same total elements.
    Tensor r3d = t.reshape({2, 2, 8});
    EXPECT_EQ(r3d.numel(), ROWS * COLS);
    EXPECT_EQ(r3d.shape(1), 2);
    EXPECT_EQ(r3d.shape(2), 2);
    EXPECT_EQ(r3d.shape(3), 8);
}

// ── DTypes ──────────────────────────────────────────────────

TEST(Tensor, DTypes) {
    const int N = 512;

    // FP16: 2 bytes per element.
    {
        Tensor fp16 = Tensor::empty({N}, DType::FP16);
        EXPECT_EQ(fp16.dtype(), DType::FP16);
        EXPECT_EQ(fp16.size_bytes(), static_cast<size_t>(N * 2));
        EXPECT_EQ(fp16.numel(), N);
        EXPECT_TRUE(fp16.is_owning());
    }

    // INT8: 1 byte per element.
    {
        Tensor int8 = Tensor::empty({N}, DType::INT8);
        EXPECT_EQ(int8.dtype(), DType::INT8);
        EXPECT_EQ(int8.size_bytes(), static_cast<size_t>(N * 1));
        EXPECT_EQ(int8.numel(), N);
        EXPECT_TRUE(int8.is_owning());
    }

    // Verify dtype_size helper directly.
    EXPECT_EQ(dtype_size(DType::FP32), 4u);
    EXPECT_EQ(dtype_size(DType::FP16), 2u);
    EXPECT_EQ(dtype_size(DType::INT8), 1u);
}
