#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace cuinfer {

// ── CUDA Error Checking ──────────────────────────────────────

#define CUINFER_CHECK_CUDA(expr)                                              \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + " — " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

#define CUINFER_CHECK_KERNEL()                                                 \
    do {                                                                       \
        CUINFER_CHECK_CUDA(cudaGetLastError());                                \
        CUINFER_CHECK_CUDA(cudaDeviceSynchronize());                           \
    } while (0)

// Non-synchronizing kernel check (for production — errors caught at next sync)
#define CUINFER_CHECK_KERNEL_ASYNC()                                           \
    CUINFER_CHECK_CUDA(cudaGetLastError())

// ── Device Info ──────────────────────────────────────────────

inline void print_device_info() {
    int device;
    CUINFER_CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp props;
    CUINFER_CHECK_CUDA(cudaGetDeviceProperties(&props, device));

    // Use a wide enough box to fit long GPU names (e.g., "NVIDIA GeForce RTX 3060 Laptop GPU")
    char buf[64];
    printf("┌──────────────────────────────────────────────────┐\n");
    printf("│ CuInfer — Device Info                            │\n");
    printf("├──────────────────────────────────────────────────┤\n");
    printf("│ GPU:          %-34s │\n", props.name);
    snprintf(buf, sizeof(buf), "SM %d.%d", props.major, props.minor);
    printf("│ Compute:      %-34s │\n", buf);
    snprintf(buf, sizeof(buf), "%.1f GB", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("│ VRAM:         %-34s │\n", buf);
    printf("│ SMs:          %-34d │\n", props.multiProcessorCount);
    printf("│ Max threads:  %-34d │\n", props.maxThreadsPerBlock);
    snprintf(buf, sizeof(buf), "%.1f KB / block", props.sharedMemPerBlock / 1024.0);
    printf("│ Shared mem:   %-34s │\n", buf);
    printf("│ Warp size:    %-34d │\n", props.warpSize);
    printf("└──────────────────────────────────────────────────┘\n");
}

// ── VRAM Reporting ───────────────────────────────────────────

inline void print_vram_usage(const char* label = nullptr) {
    size_t free_bytes, total_bytes;
    CUINFER_CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));

    double used_gb = (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);
    double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);

    if (label) {
        printf("[VRAM] %s: %.2f / %.2f GB (%.1f%%)\n",
               label, used_gb, total_gb, 100.0 * used_gb / total_gb);
    } else {
        printf("[VRAM] %.2f / %.2f GB (%.1f%%)\n",
               used_gb, total_gb, 100.0 * used_gb / total_gb);
    }
}

// ── Timer (CUDA Events) ─────────────────────────────────────

class CudaTimer {
public:
    CudaTimer() {
        CUINFER_CHECK_CUDA(cudaEventCreate(&start_));
        CUINFER_CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream = 0) {
        CUINFER_CHECK_CUDA(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUINFER_CHECK_CUDA(cudaEventRecord(stop_, stream));
        CUINFER_CHECK_CUDA(cudaEventSynchronize(stop_));
    }

    /// Returns elapsed time in milliseconds
    float elapsed_ms() const {
        float ms = 0.0f;
        CUINFER_CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

} // namespace cuinfer
