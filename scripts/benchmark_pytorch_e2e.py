#!/usr/bin/env python3
"""PyTorch end-to-end text generation baseline for the CuInfer project.

Loads SmolLM2-135M-Instruct via HuggingFace transformers and runs the same
prompts as the CuInfer executable, using CUDA events for timing.  This gives
an apples-to-apples comparison of tokens/sec and VRAM usage.

Usage:
  # Using a venv on E: drive (recommended):
  .venv/Scripts/python.exe scripts/benchmark_pytorch_e2e.py   # Windows
  .venv/bin/python scripts/benchmark_pytorch_e2e.py           # Linux/WSL

Environment:
  Requires torch (with CUDA), transformers, and accelerate.
  Install into a venv on E: drive (never install to C: which has no space):
    python -m venv .venv
    .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv/Scripts/pip install transformers accelerate
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_NEW_TOKENS = 50
NUM_ITERATIONS = 5  # median of 5 iterations per prompt

PROMPTS = [
    "Hi",
    "The capital of France is",
    ("Once upon a time in a land far far away there lived a young princess "
     "who dreamed of adventure beyond the castle walls"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_prompt_tokens(tokenizer, prompt: str) -> int:
    """Return the number of tokens in a prompt."""
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def measure_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[int, float, str]:
    """Generate text and return (num_generated_tokens, elapsed_ms, text).

    Uses torch.cuda.Event for precise GPU timing, NOT wall-clock.
    Uses greedy decoding (temperature=0, do_sample=False) to match CuInfer's
    argmax decoding.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Synchronize before starting
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy decoding (argmax)
            temperature=1.0,       # ignored when do_sample=False
            top_p=1.0,             # ignored when do_sample=False
            use_cache=True,        # KV-cache enabled
        )

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    generated_ids = outputs[0][input_len:]
    num_generated = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return num_generated, elapsed_ms, text


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark() -> Dict[str, Any]:
    """Run the full benchmark suite and return structured results."""

    print("=" * 72)
    print("  CuInfer PyTorch E2E Baseline — SmolLM2-135M-Instruct")
    print("=" * 72)

    # System info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  GPU             : {gpu_name}")
    print(f"  GPU memory      : {gpu_mem_mb:.0f} MB")
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Decoding        : greedy (argmax)")
    print(f"  Max new tokens  : {MAX_NEW_TOKENS}")
    print(f"  Iterations      : {NUM_ITERATIONS} (report median)")
    print(f"  Timing          : torch.cuda.Event (NOT wall clock)")
    print("=" * 72)

    # Reset VRAM stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,  # FP16 to match CuInfer's GGUF F16
        device_map="cuda",
    )
    model.eval()

    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s")

    vram_after_load = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  VRAM after model load: {vram_after_load:.1f} MB")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Report model config
    config = model.config
    print(f"\n  Architecture config:")
    print(f"    Hidden dim   : {config.hidden_size}")
    print(f"    Layers       : {config.num_hidden_layers}")
    print(f"    Heads        : {config.num_attention_heads}")
    kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    print(f"    KV heads     : {kv_heads}")
    print(f"    FFN dim      : {config.intermediate_size}")
    print(f"    Vocab size   : {config.vocab_size}")

    # Run prompts
    results = []

    print(f"\n{'=' * 72}")
    print(f"  {'Prompt':<50} {'Tokens':>6} {'Prompt':>6} {'Tok/s':>8} "
          f"{'Time ms':>8} {'VRAM MB':>8}")
    print(f"  {'':50} {'gen':>6} {'toks':>6} {'median':>8} "
          f"{'median':>8} {'peak':>8}")
    print("-" * 72)

    for prompt in PROMPTS:
        prompt_tokens = count_prompt_tokens(tokenizer, prompt)
        display_prompt = prompt[:47] + "..." if len(prompt) > 50 else prompt

        iter_results = []
        sample_text = ""

        for i in range(NUM_ITERATIONS):
            # Reset peak memory for this iteration
            torch.cuda.reset_peak_memory_stats()

            num_gen, elapsed_ms, text = measure_generation(
                model, tokenizer, prompt, MAX_NEW_TOKENS
            )

            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

            # total_tokens includes prompt + generated for throughput
            # but CuInfer reports based on all tokens processed
            # For fair comparison we report tokens/sec on generated tokens only
            # (decode throughput), since that's what matters for user experience
            tok_per_sec = num_gen / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0

            iter_results.append({
                "num_generated": num_gen,
                "elapsed_ms": elapsed_ms,
                "tok_per_sec": tok_per_sec,
                "vram_peak_mb": vram_peak,
            })

            if i == 0:
                sample_text = text

        # Compute medians
        median_tok_s = statistics.median([r["tok_per_sec"] for r in iter_results])
        median_ms = statistics.median([r["elapsed_ms"] for r in iter_results])
        median_gen = int(statistics.median([r["num_generated"] for r in iter_results]))
        max_vram = max(r["vram_peak_mb"] for r in iter_results)

        result = {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "num_generated_median": median_gen,
            "median_tok_per_sec": round(median_tok_s, 1),
            "median_elapsed_ms": round(median_ms, 1),
            "peak_vram_mb": round(max_vram, 1),
            "sample_output": sample_text[:200],
            "all_iterations": iter_results,
        }
        results.append(result)

        print(f"  {display_prompt:<50} {median_gen:>6} {prompt_tokens:>6} "
              f"{median_tok_s:>8.1f} {median_ms:>8.1f} {max_vram:>8.1f}")

    print("-" * 72)

    # Final VRAM summary
    print(f"\n  Final VRAM allocated : "
          f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB")
    print(f"  Peak VRAM allocated : "
          f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB")

    # Compose output
    output = {
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
            "gpu_name": gpu_name,
            "gpu_memory_mb": f"{gpu_mem_mb:.0f}",
            "model_name": MODEL_NAME,
            "model_dtype": "float16",
            "timestamp": datetime.now().isoformat(),
        },
        "methodology": {
            "timing": "torch.cuda.Event",
            "decoding": "greedy (do_sample=False)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_iterations": NUM_ITERATIONS,
            "aggregation": "median",
            "kv_cache": True,
        },
        "vram_after_model_load_mb": round(vram_after_load, 1),
        "results": results,
    }

    return output


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    output = run_benchmark()

    # Save JSON results
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pytorch_e2e_results.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
