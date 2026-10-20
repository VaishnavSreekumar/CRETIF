// CUDA NVPTX Builtin Mapping for Clang NVPTX Target
extern "C" {
    int __nvvm_read_ptx_sreg_tid_x();
    int __nvvm_read_ptx_sreg_tid_y();
    int __nvvm_read_ptx_sreg_ctaid_x();
    int __nvvm_read_ptx_sreg_ctaid_y();
    void __nvvm_barrier0();
}

struct uint16_t_half {
    unsigned short val;
};

// FP16 to FP32 conversion via PTX intrinsic
inline float half_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

struct BlockQ8_0 {
    unsigned short scale;
    signed char qs[32];
};

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

#ifdef PAD_TILE
constexpr int TILE_N_PAD = TILE_N + 1; // 33
#else
constexpr int TILE_N_PAD = TILE_N;     // 32 (unpadded baseline)
#endif

// Shared memory allocations in address_space(3)
__attribute__((address_space(3))) float tile_a[TILE_M][TILE_K];
__attribute__((address_space(3))) float tile_b[TILE_K][TILE_N_PAD];

extern "C" __attribute__((nvptx_kernel)) void kernel_matmul_q8_0(
    const float* __restrict__ A,
    const BlockQ8_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int threadIdx_x = __nvvm_read_ptx_sreg_tid_x();
    const int threadIdx_y = __nvvm_read_ptx_sreg_tid_y();
    const int blockIdx_x = __nvvm_read_ptx_sreg_ctaid_x();
    const int blockIdx_y = __nvvm_read_ptx_sreg_ctaid_y();

    const int row = blockIdx_y * TILE_M + threadIdx_y;
    const int col = blockIdx_x * TILE_N + threadIdx_x;

    float sum = 0.0f;
    const int num_tiles = K / TILE_K;
    const int blocks_per_row = K / 32;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_K + threadIdx_x;
        if (row < M && a_col < K) {
            tile_a[threadIdx_y][threadIdx_x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx_y][threadIdx_x] = 0.0f;
        }

        const int b_row = blockIdx_x * TILE_N + threadIdx_y;
        if (b_row < N) {
            const BlockQ8_0& blk = B[b_row * blocks_per_row + t];
            float scale = half_to_float(blk.scale);
            tile_b[threadIdx_x][threadIdx_y] = scale * static_cast<float>(blk.qs[threadIdx_x]);
        } else {
            tile_b[threadIdx_x][threadIdx_y] = 0.0f;
        }

        __nvvm_barrier0();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += tile_a[threadIdx_y][k] * tile_b[k][threadIdx_x];
        }

        __nvvm_barrier0();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
