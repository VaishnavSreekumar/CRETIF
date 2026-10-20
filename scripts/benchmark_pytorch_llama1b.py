#!/usr/bin/env python3
"""PyTorch end-to-end baseline for Llama 3.2 1B-Instruct.

Matches CuInfer's Phase 3 protocol: 100 generated tokens, greedy decoding,
CUDA event timing, median of 5 iterations.

Usage:
  .venv/Scripts/python.exe scripts/benchmark_pytorch_llama1b.py   # Windows
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 100
NUM_ITERATIONS = 5

PROMPT = "The key properties of transformer attention are"


def measure_generation(
    model, tokenizer, prompt: str, max_new_tokens: int,
) -> Tuple[int, float, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return len(generated_ids), elapsed_ms, text


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print("=" * 72)
    print("  PyTorch E2E Baseline — Llama 3.2 1B-Instruct (FP16)")
    print("=" * 72)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.version.cuda}")
    print(f"  GPU      : {gpu_name} ({gpu_mem_mb:.0f} MB)")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Tokens   : {MAX_NEW_TOKENS}, greedy, median of {NUM_ITERATIONS}")
    print("=" * 72)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  VRAM after load: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    config = model.config
    kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    print(f"  Arch: {config.hidden_size}d, {config.num_hidden_layers}L, "
          f"{config.num_attention_heads}H/{kv_heads}KV, vocab {config.vocab_size}")

    print(f"\nBenchmarking: \"{PROMPT}\"")
    print("-" * 72)

    iter_results = []
    sample_text = ""

    for i in range(NUM_ITERATIONS):
        torch.cuda.reset_peak_memory_stats()
        num_gen, elapsed_ms, text = measure_generation(
            model, tokenizer, PROMPT, MAX_NEW_TOKENS
        )
        tok_s = num_gen / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
        vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        iter_results.append({
            "num_generated": num_gen,
            "elapsed_ms": elapsed_ms,
            "tok_per_sec": tok_s,
            "vram_peak_mb": vram,
        })
        if i == 0:
            sample_text = text
        print(f"  Iter {i+1}: {num_gen} tokens, {elapsed_ms:.1f} ms, "
              f"{tok_s:.1f} tok/s, {vram:.1f} MB peak")

    median_tok_s = statistics.median([r["tok_per_sec"] for r in iter_results])
    median_ms = statistics.median([r["elapsed_ms"] for r in iter_results])
    max_vram = max(r["vram_peak_mb"] for r in iter_results)

    print("-" * 72)
    print(f"  MEDIAN: {median_tok_s:.1f} tok/s, {median_ms:.1f} ms, "
          f"{max_vram:.1f} MB peak VRAM")
    print(f"\n  Sample output: {sample_text[:200]}")

    output = {
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
            "gpu_name": gpu_name,
            "model_name": MODEL_NAME,
            "model_dtype": "float16",
            "timestamp": datetime.now().isoformat(),
        },
        "methodology": {
            "timing": "torch.cuda.Event",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_iterations": NUM_ITERATIONS,
            "prompt": PROMPT,
        },
        "median_tok_per_sec": round(median_tok_s, 1),
        "median_elapsed_ms": round(median_ms, 1),
        "peak_vram_mb": round(max_vram, 1),
        "iterations": iter_results,
        "sample_output": sample_text[:500],
    }

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pytorch_llama1b_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
