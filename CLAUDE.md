# CLAUDE.md — CuInfer Project Instructions

## Project Overview

CuInfer is a from-scratch LLM inference engine in C++17/CUDA. The goal is to implement transformer inference with custom CUDA kernels, load real pretrained models (GGUF format), and benchmark against PyTorch on an RTX 3060 (6 GB VRAM).

**This is a portfolio/showcase project** — code clarity, documentation, and benchmarks matter as much as raw performance. Every design decision should be explainable and well-commented.

## Development Environment

- **OS**: Windows 10 Pro → WSL2 Ubuntu 24.04 (primary dev environment for this project)
- **GPU**: NVIDIA RTX 3060, 6 GB VRAM, Compute Capability 8.6 (Ampere)
- **CUDA**: 12.8
- **Compiler**: GCC 11+ (WSL2), or MSVC 2022 (Windows fallback)
- **Build**: CMake 3.24+
- **VRAM Budget**: Keep peak usage under 5.5 GB to avoid OOM (reserve headroom for OS/display)

## Build Commands

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

# Build
cmake --build build --config Release -j$(nproc)

# Build debug
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-debug -j$(nproc)

# Run tests
cd build && ctest --output-on-failure

# Run specific test
./build/tests/test_kernels

# Run benchmarks
./build/benchmarks/bench_matmul
./build/benchmarks/bench_e2e --model models/smollm2-135m.gguf
```

## Code Standards

### C++ Style
- **Standard**: C++17 (no C++20 modules — keep broad compatibility)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes/structs, `UPPER_CASE` for constants and macros
- **Namespaces**: All code under `namespace cuinfer { }`
- **Headers**: Use `#pragma once`, include what you use
- **RAII**: All GPU memory managed via RAII wrappers (Tensor class owns its memory)
- **Error handling**: CUDA errors checked via macro `CUINFER_CHECK_CUDA(expr)`, throw on failure
- **No raw `new`/`delete`** — use smart pointers or RAII containers
- **Comments**: Explain *why*, not *what*. Document kernel launch configurations and memory access patterns

### CUDA Style
- **Kernel naming**: `kernel_<operation>` (e.g., `kernel_matmul_tiled`, `kernel_rmsnorm`)
- **Block sizes**: Default 256 threads unless profiling shows otherwise; use `constexpr` for tile dimensions
- **Shared memory**: Always declare tile sizes as `constexpr`; comment memory usage per block
- **Error checking**: Every kernel launch followed by `CUINFER_CHECK_CUDA(cudaGetLastError())`
- **Streams**: Support async execution via CUDA streams from the start (default stream acceptable initially)
- **Precision**: FP32 as baseline, FP16 as optimization, INT8/INT4 for quantized path
- **Architecture targets**: SM 8.6 primary (RTX 3060), SM 7.5 secondary (Turing compatibility)

### Testing
- Framework: Google Test (fetched via CMake FetchContent)
- Test naming: `TEST(KernelName, OperationDescription)`
- Every kernel must have a correctness test comparing against CPU reference implementation
- Tolerance: `1e-5` for FP32, `1e-2` for FP16, `5e-2` for quantized
- Tests should be runnable without a model file (use synthetic data)

### Benchmarking
- Use CUDA events for timing (not wall clock)
- Report: median of 100 runs after 10 warmup iterations
- Always report VRAM usage alongside throughput
- Compare against PyTorch eager mode baseline (script in `scripts/benchmark_pytorch.py`)

## Architecture Decisions

### Why GGUF?
- De facto standard for quantized model distribution
- Well-documented binary format, parseable without dependencies
- Models readily available on HuggingFace
- Avoids needing Python/HuggingFace as runtime dependency

### Why not use cuBLAS/cuDNN?
- This is a **showcase project** — writing kernels from scratch demonstrates understanding
- cuBLAS can be added later as a comparison baseline (Phase 4 stretch goal)
- The performance gap itself is educational and worth documenting

### CUB / CCCL Strategy (Middle Ground)
- Phase 1-2: all kernels hand-written from scratch (proves understanding of reductions, shared memory, warp primitives)
- Phase 3: introduce CUB-accelerated variants (`cub::BlockReduce`, `cub::WarpReduce`) for Softmax and RMSNorm alongside the originals; benchmark both to quantify the gap
- CUB is header-only and ships with the CUDA toolkit — it's not an external dependency like cuBLAS
- This tells a stronger portfolio story: "I wrote it from scratch, then compared against the standard library"
- Phase 4 stretch: broader CCCL exploration (Thrust, libcudacxx) if time permits

### Memory Strategy
- Model weights: loaded once, read-only, resident on GPU
- KV-cache: pre-allocated for max sequence length, reused across tokens
- Activations: allocated once at max batch size, reused across layers
- No dynamic allocation during generation loop

### Tensor Design
- Simple `Tensor` class wrapping `float*` (or `half*`, `int8_t*`) + shape + stride
- Device-only tensors (no automatic host↔device sync — explicit transfers only)
- Shape: `std::array<int64_t, 4>` supporting up to 4D (batch, seq, heads, dim)
- Non-owning "view" variant for slicing without copies

## File Organization

- `include/cuinfer/` — public headers only
- `src/kernels/` — CUDA kernel implementations (`.cu`)
- `src/` — C++ implementation files (`.cpp`)
- `tests/` — Google Test files
- `benchmarks/` — standalone benchmark executables
- `scripts/` — utility scripts (Python benchmarks, model download, plotting)
- `docs/` — detailed design documents and benchmark reports
- `models/` — gitignored, downloaded model files

## Dependencies (Minimal)

- CUDA Toolkit 12.x (system)
- CMake 3.24+ (system)
- Google Test (FetchContent)
- nlohmann/json (FetchContent, for config files — optional)
- No Boost, no OpenCV, no heavy frameworks

## Current Phase

Check PROJECT.md for the current phase and task breakdown. Always verify which phase we're working on before starting a task.

## Key Reminders

1. **Portfolio first**: Clean code and good README updates matter more than squeezing 5% perf
2. **VRAM constraint**: Always test on RTX 3060 6 GB — if it doesn't fit, it doesn't ship
3. **Incremental progress**: Each commit should compile and pass tests
4. **Benchmark everything**: No optimization without measurement
5. **Document design decisions**: Future employers will read the code and docs
