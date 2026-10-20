#!/usr/bin/env python3
"""PyTorch eager-mode baseline benchmark for the CuInfer project.

Benchmarks the same operations that CuInfer implements with custom CUDA kernels,
using identical methodology (CUDA events for timing, 100 runs median after 10
warmup iterations) so results are directly comparable.

Operations benchmarked:
  - Matrix multiplication (torch.matmul) with dimension sweep
  - Elementwise kernels: RMSNorm, SiLU, GeLU, Softmax, RoPE, Add, Multiply

Usage:
  python benchmark_pytorch.py --all              # Run everything
  python benchmark_pytorch.py --matmul           # Matmul sweep only
  python benchmark_pytorch.py --elementwise      # Elementwise kernels only
  python benchmark_pytorch.py --all -o out.json  # Custom output path

Results are saved to scripts/pytorch_baseline_results.json by default.
"""

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration — matches CuInfer's benchmark methodology
# ---------------------------------------------------------------------------

NUM_WARMUP = 10
NUM_RUNS = 100

# Matmul dimension sweep (square matrices: M=N=K)
MATMUL_SIZES = [128, 512, 1024, 2048, 4096]

# Elementwise dimension sweep
ELEMENTWISE_ROWS = [1, 32, 128, 512]
ELEMENTWISE_COLS = [768, 2048, 4096, 8192]

# RoPE parameters matching CuInfer defaults
ROPE_THETA_BASE = 10000.0
ROPE_N_HEADS_LIST = [1, 8, 32]
ROPE_HEAD_DIM_LIST = [64, 128]


# ---------------------------------------------------------------------------
# Timing utility using CUDA events (NOT wall clock)
# ---------------------------------------------------------------------------

def benchmark_cuda(
    fn: Callable[[], Any],
    num_warmup: int = NUM_WARMUP,
    num_runs: int = NUM_RUNS,
) -> Dict[str, float]:
    """Time a GPU operation using torch.cuda.Event for accurate measurement.

    Returns a dict with 'median_ms', 'min_ms', 'max_ms', and 'mean_ms'.
    All values are in milliseconds.
    """
    # Warmup — let CUDA JIT compile and caches settle
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    timings_ms: List[float] = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "mean_ms": statistics.mean(timings_ms),
    }


# ---------------------------------------------------------------------------
# VRAM reporting
# ---------------------------------------------------------------------------

def get_vram_info() -> Dict[str, float]:
    """Return current and peak VRAM usage in MB."""
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def reset_vram_stats() -> None:
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Matmul benchmark
# ---------------------------------------------------------------------------

def compute_matmul_gflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute GFLOPS for a matrix multiply C[M,N] = A[M,K] @ B[K,N].

    FLOPs = 2 * M * N * K  (multiply + accumulate per output element).
    """
    flops = 2.0 * M * N * K
    seconds = time_ms / 1000.0
    if seconds == 0:
        return 0.0
    return flops / seconds / 1e9


def bench_matmul(sizes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Benchmark torch.matmul across a sweep of square matrix dimensions.

    Returns a list of result dicts, one per size.
    """
    if sizes is None:
        sizes = MATMUL_SIZES

    results = []

    print("\n" + "=" * 72)
    print("  MATRIX MULTIPLICATION BENCHMARK  (torch.matmul, FP32)")
    print("=" * 72)
    print(f"  {'Size (M=N=K)':>12}  {'Median (ms)':>12}  {'GFLOPS':>10}  "
          f"{'VRAM (MB)':>10}")
    print("-" * 72)

    for size in sizes:
        M = N = K = size
        reset_vram_stats()

        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        timing = benchmark_cuda(lambda: torch.matmul(A, B))
        vram = get_vram_info()
        gflops = compute_matmul_gflops(M, N, K, timing["median_ms"])

        result = {
            "operation": "matmul",
            "M": M,
            "N": N,
            "K": K,
            "dtype": "float32",
            **timing,
            "gflops": gflops,
            **{f"vram_{k}": v for k, v in vram.items()},
        }
        results.append(result)

        print(f"  {size:>12}  {timing['median_ms']:>12.3f}  {gflops:>10.1f}  "
              f"{vram['max_allocated_mb']:>10.1f}")

        # Free tensors to keep VRAM tidy between sizes
        del A, B
        torch.cuda.empty_cache()

    print("-" * 72)
    return results


# ---------------------------------------------------------------------------
# Elementwise kernel benchmarks
# ---------------------------------------------------------------------------

def compute_bandwidth_gbs(
    num_bytes_read: int,
    num_bytes_written: int,
    time_ms: float,
) -> float:
    """Compute effective memory bandwidth in GB/s."""
    total_bytes = num_bytes_read + num_bytes_written
    seconds = time_ms / 1000.0
    if seconds == 0:
        return 0.0
    return total_bytes / seconds / 1e9


# ---- Individual operation wrappers ----------------------------------------
# Each returns (callable, bytes_read, bytes_written, extra_info_dict).


def make_rmsnorm(
    rows: int, cols: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """RMSNorm: x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps).

    Matches CuInfer's kernel_rmsnorm which takes input[rows x n] and
    weight[n] and computes output[i] = input[i] * weight[i] * norm_factor.
    """
    eps = 1e-5
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32)

    def fn() -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        return normed * weight

    # Reads: input (rows*cols) + weight (cols, broadcast across rows)
    # Writes: output (rows*cols)
    # In practice the intermediate allocations add reads/writes, but we report
    # the logical I/O to match CuInfer's fused single-kernel accounting.
    bytes_read = (rows * cols + cols) * 4
    bytes_written = rows * cols * 4
    return fn, bytes_read, bytes_written, {"eps": eps}


def make_silu(
    rows: int, cols: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """SiLU: x * sigmoid(x). Matches CuInfer's kernel_silu."""
    n = rows * cols
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    def fn() -> torch.Tensor:
        return F.silu(x)

    bytes_read = n * 4
    bytes_written = n * 4
    return fn, bytes_read, bytes_written, {}


def make_gelu(
    rows: int, cols: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """GeLU (tanh approximation). Matches CuInfer's kernel_gelu."""
    n = rows * cols
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    def fn() -> torch.Tensor:
        return F.gelu(x, approximate="tanh")

    bytes_read = n * 4
    bytes_written = n * 4
    return fn, bytes_read, bytes_written, {"approximate": "tanh"}


def make_softmax(
    rows: int, cols: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """Softmax along last dimension. Matches CuInfer's kernel_softmax."""
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    def fn() -> torch.Tensor:
        return F.softmax(x, dim=-1)

    bytes_read = rows * cols * 4
    bytes_written = rows * cols * 4
    return fn, bytes_read, bytes_written, {}


def make_rope(
    n_heads: int, head_dim: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """RoPE (Rotary Positional Embeddings).

    Matches CuInfer's kernel_rope: operates on [n_heads x dim], applies
    rotation to pairs of dimensions for a single token position.

    The standard formula for dimension pair (2i, 2i+1) at position pos:
      freq_i = 1 / (theta_base ^ (2i / dim))
      theta  = pos * freq_i
      x_out[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
      x_out[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)
    """
    pos = 42  # Arbitrary position for benchmarking
    theta_base = ROPE_THETA_BASE

    # Input: [n_heads, head_dim]
    x = torch.randn(n_heads, head_dim, device="cuda", dtype=torch.float32)

    # Precompute frequency table (this matches CuInfer's inline computation)
    half_dim = head_dim // 2
    freq_exponents = torch.arange(0, half_dim, device="cuda", dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (2.0 * freq_exponents / head_dim))
    theta = pos * freqs  # [half_dim]
    cos_theta = torch.cos(theta)  # [half_dim]
    sin_theta = torch.sin(theta)  # [half_dim]

    def fn() -> torch.Tensor:
        x_pairs = x.view(n_heads, half_dim, 2)
        x0 = x_pairs[..., 0]  # [n_heads, half_dim]
        x1 = x_pairs[..., 1]  # [n_heads, half_dim]
        out0 = x0 * cos_theta - x1 * sin_theta
        out1 = x0 * sin_theta + x1 * cos_theta
        return torch.stack([out0, out1], dim=-1).view(n_heads, head_dim)

    # RoPE is in-place in CuInfer; logical I/O is read + write of the full buffer
    total_elements = n_heads * head_dim
    bytes_read = total_elements * 4
    bytes_written = total_elements * 4
    return fn, bytes_read, bytes_written, {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "pos": pos,
        "theta_base": theta_base,
    }


def make_add(
    rows: int, cols: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """Element-wise add. Matches CuInfer's kernel_add."""
    n = rows * cols
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    def fn() -> torch.Tensor:
        return torch.add(a, b)

    bytes_read = 2 * n * 4
    bytes_written = n * 4
    return fn, bytes_read, bytes_written, {}


def make_multiply(
    rows: int, cols: int
) -> Tuple[Callable[[], Any], int, int, Dict[str, Any]]:
    """Element-wise multiply. Matches CuInfer's kernel_multiply."""
    n = rows * cols
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    def fn() -> torch.Tensor:
        return torch.mul(a, b)

    bytes_read = 2 * n * 4
    bytes_written = n * 4
    return fn, bytes_read, bytes_written, {}


# ---- Elementwise benchmark runner -----------------------------------------

def bench_elementwise() -> List[Dict[str, Any]]:
    """Benchmark all elementwise operations across the dimension sweep.

    Returns a list of result dicts.
    """
    results = []

    # Operations that work on a (rows x cols) grid
    grid_ops = [
        ("RMSNorm", make_rmsnorm),
        ("SiLU", make_silu),
        ("GeLU", make_gelu),
        ("Softmax", make_softmax),
        ("Add", make_add),
        ("Multiply", make_multiply),
    ]

    for op_name, make_fn in grid_ops:
        print(f"\n{'=' * 72}")
        print(f"  {op_name.upper()} BENCHMARK  (PyTorch, FP32)")
        print(f"{'=' * 72}")
        print(f"  {'Rows':>6}  {'Cols':>6}  {'Elements':>10}  "
              f"{'Median (ms)':>12}  {'GB/s':>8}  {'VRAM (MB)':>10}")
        print("-" * 72)

        for rows in ELEMENTWISE_ROWS:
            for cols in ELEMENTWISE_COLS:
                reset_vram_stats()

                fn, bytes_r, bytes_w, extra = make_fn(rows, cols)
                timing = benchmark_cuda(fn)
                vram = get_vram_info()
                bw_gbs = compute_bandwidth_gbs(bytes_r, bytes_w,
                                               timing["median_ms"])

                result = {
                    "operation": op_name.lower(),
                    "rows": rows,
                    "cols": cols,
                    "elements": rows * cols,
                    "dtype": "float32",
                    **timing,
                    "bandwidth_gbs": bw_gbs,
                    "bytes_read": bytes_r,
                    "bytes_written": bytes_w,
                    **{f"vram_{k}": v for k, v in vram.items()},
                    **extra,
                }
                results.append(result)

                print(f"  {rows:>6}  {cols:>6}  {rows * cols:>10}  "
                      f"{timing['median_ms']:>12.4f}  {bw_gbs:>8.1f}  "
                      f"{vram['max_allocated_mb']:>10.2f}")

                # Clean up between runs
                torch.cuda.empty_cache()

        print("-" * 72)

    # RoPE has a different parameter space (n_heads x head_dim, not rows x cols)
    print(f"\n{'=' * 72}")
    print(f"  ROPE BENCHMARK  (PyTorch manual, FP32)")
    print(f"{'=' * 72}")
    print(f"  {'Heads':>6}  {'HeadDim':>8}  {'Pairs':>8}  "
          f"{'Median (ms)':>12}  {'GB/s':>8}  {'VRAM (MB)':>10}")
    print("-" * 72)

    for n_heads in ROPE_N_HEADS_LIST:
        for head_dim in ROPE_HEAD_DIM_LIST:
            reset_vram_stats()

            fn, bytes_r, bytes_w, extra = make_rope(n_heads, head_dim)
            timing = benchmark_cuda(fn)
            vram = get_vram_info()
            bw_gbs = compute_bandwidth_gbs(bytes_r, bytes_w,
                                           timing["median_ms"])
            total_pairs = n_heads * (head_dim // 2)

            result = {
                "operation": "rope",
                "n_heads": n_heads,
                "head_dim": head_dim,
                "total_pairs": total_pairs,
                "dtype": "float32",
                **timing,
                "bandwidth_gbs": bw_gbs,
                "bytes_read": bytes_r,
                "bytes_written": bytes_w,
                **{f"vram_{k}": v for k, v in vram.items()},
                **extra,
            }
            results.append(result)

            print(f"  {n_heads:>6}  {head_dim:>8}  {total_pairs:>8}  "
                  f"{timing['median_ms']:>12.4f}  {bw_gbs:>8.2f}  "
                  f"{vram['max_allocated_mb']:>10.2f}")

            torch.cuda.empty_cache()

    print("-" * 72)
    return results


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

def print_system_info() -> Dict[str, str]:
    """Print and return system/GPU information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "gpu_total_memory_mb": f"{torch.cuda.get_device_properties(0).total_mem / (1024 ** 2):.0f}",
        "python_version": sys.version.split()[0],
        "timestamp": datetime.now().isoformat(),
    }

    print("=" * 72)
    print("  CuInfer PyTorch Baseline Benchmark")
    print("=" * 72)
    print(f"  PyTorch version   : {info['pytorch_version']}")
    print(f"  CUDA version      : {info['cuda_version']}")
    print(f"  cuDNN version     : {info['cudnn_version']}")
    print(f"  GPU               : {info['gpu_name']}")
    print(f"  Compute capability: {info['gpu_compute_capability']}")
    print(f"  GPU memory        : {info['gpu_total_memory_mb']} MB")
    print(f"  Python version    : {info['python_version']}")
    print(f"  Timestamp         : {info['timestamp']}")
    print(f"  Methodology       : {NUM_RUNS} runs, median, {NUM_WARMUP} warmup")
    print(f"  Timing            : torch.cuda.Event (not wall clock)")
    print("=" * 72)

    return info


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_results(
    system_info: Dict[str, str],
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Save benchmark results to a JSON file for later plotting."""
    payload = {
        "system_info": system_info,
        "methodology": {
            "num_warmup": NUM_WARMUP,
            "num_runs": NUM_RUNS,
            "timing": "torch.cuda.Event",
            "aggregation": "median",
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch eager-mode baseline benchmark for CuInfer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark_pytorch.py --all\n"
            "  python benchmark_pytorch.py --matmul\n"
            "  python benchmark_pytorch.py --elementwise\n"
            "  python benchmark_pytorch.py --all -o custom_results.json\n"
        ),
    )

    group = parser.add_argument_group("benchmark selection")
    group.add_argument(
        "--all", action="store_true",
        help="Run all benchmarks (matmul + elementwise).",
    )
    group.add_argument(
        "--matmul", action="store_true",
        help="Run matrix multiplication benchmark.",
    )
    group.add_argument(
        "--elementwise", action="store_true",
        help="Run elementwise kernel benchmarks (RMSNorm, SiLU, GeLU, "
             "Softmax, RoPE, Add, Multiply).",
    )

    parser.add_argument(
        "-o", "--output", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pytorch_baseline_results.json"),
        help="Path for JSON results file "
             "(default: scripts/pytorch_baseline_results.json).",
    )

    args = parser.parse_args()

    # Default to --all if nothing selected
    if not (args.all or args.matmul or args.elementwise):
        args.all = True

    # Verify CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    system_info = print_system_info()
    all_results: List[Dict[str, Any]] = []

    if args.all or args.matmul:
        all_results.extend(bench_matmul())

    if args.all or args.elementwise:
        all_results.extend(bench_elementwise())

    # Final VRAM summary
    print(f"\n{'=' * 72}")
    print("  VRAM SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Current allocated : "
          f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB")
    print(f"  Peak allocated    : "
          f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB")
    print(f"{'=' * 72}")

    save_results(system_info, all_results, args.output)

    print(f"\nBenchmark complete. {len(all_results)} measurements recorded.")


if __name__ == "__main__":
    main()
