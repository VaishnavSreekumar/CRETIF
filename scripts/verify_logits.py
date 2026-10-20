#!/usr/bin/env python3
"""Verify CuInfer logits against PyTorch reference for SmolLM2-135M.

This script has two modes:

  1. Generate reference logits:
       python verify_logits.py
     Loads SmolLM2-135M-Instruct from HuggingFace, runs a test prompt,
     and saves the top-20 logits to scripts/logits_reference.json.

  2. Compare CuInfer output against the reference:
       python verify_logits.py <cuinfer_logits.json>
     Loads both the reference and CuInfer dump, then reports:
       - Max absolute difference in top-20 logit values
       - Whether top-5 predicted tokens match
       - Pass/fail based on tolerance (configurable, default 1e-3)

Environment:
  pip install torch transformers
  (Or: pip install --target /mnt/e/Repos/CuInfer/.pylibs torch transformers
   and set PYTHONPATH=/mnt/e/Repos/CuInfer/.pylibs)

Usage:
  # Step 1: Generate reference
  python scripts/verify_logits.py

  # Step 2: Run CuInfer with --dump-logits
  ./build/cuinfer models/SmolLM2-135M-Instruct-f16.gguf "The capital of France is" --dump-logits

  # Step 3: Compare
  python scripts/verify_logits.py cuinfer_logits.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure stdout can handle Unicode on Windows (cp1251/cp1252 consoles)
if sys.platform == "win32":
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEST_PROMPT = "The capital of France is"
TOP_K = 20
TOLERANCE = 1e-3  # Max acceptable absolute logit difference for FP16 weights

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = SCRIPT_DIR / "logits_reference.json"


# ---------------------------------------------------------------------------
# Mode 1: Generate Reference
# ---------------------------------------------------------------------------

def generate_reference() -> None:
    """Load SmolLM2-135M from HuggingFace and save reference logits."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.")
        print("  pip install torch transformers")
        print("  (Use --target on E: drive if C: is low on space)")
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.")
        print("  pip install transformers")
        sys.exit(1)

    print("=" * 60)
    print("  CuInfer Logit Verification — Reference Generation")
    print("=" * 60)
    print(f"  Model  : {MODEL_ID}")
    print(f"  Prompt : \"{TEST_PROMPT}\"")
    print(f"  Top-K  : {TOP_K}")
    print()

    # Load model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load model in FP16 first, then cast to FP32. This matches what
    # CuInfer does: load FP16 GGUF weights, dequantize to FP32 on the host,
    # then run all computation in FP32. Loading directly in FP32 from HF
    # would give slightly different weights due to FP16 quantization loss,
    # making the comparison unfair.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model (FP16 -> FP32, matching CuInfer's dequant path)...")
    print(f"  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    # Cast to FP32 for the forward pass (same as CuInfer's dequantized path)
    model = model.float().to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print()

    # Tokenize prompt
    encoding = tokenizer(TEST_PROMPT, return_tensors="pt").to(device)
    input_ids = encoding["input_ids"]  # [1, seq_len]
    input_token_ids = input_ids[0].tolist()
    token_strings = [tokenizer.decode([tid]) for tid in input_token_ids]

    print(f"Input tokens ({len(input_token_ids)}): {input_token_ids}")
    print(f"Token strings: {token_strings}")
    print()

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Extract logits at the last position (prediction for next token)
    last_logits = logits[0, -1, :]  # [vocab_size]

    # Get top-K logits
    top_values, top_indices = torch.topk(last_logits, TOP_K)

    top_entries = []
    print(f"Top-{TOP_K} logits at last position (next-token prediction):")
    for rank, (val, idx) in enumerate(
        zip(top_values.tolist(), top_indices.tolist())
    ):
        token_str = tokenizer.decode([idx])
        entry = {
            "rank": rank,
            "token_id": idx,
            "token_string": token_str,
            "logit_value": val,
        }
        top_entries.append(entry)
        if rank < 10:
            print(f"  #{rank+1:>2}: [{idx:>6}] {val:>10.4f}  \"{token_str}\"")
    if TOP_K > 10:
        print(f"  ... ({TOP_K - 10} more entries in output file)")
    print()

    # Logit statistics
    logit_stats = {
        "min": float(last_logits.min()),
        "max": float(last_logits.max()),
        "mean": float(last_logits.mean()),
        "std": float(last_logits.std()),
    }
    print(f"Logit stats: min={logit_stats['min']:.4f}, max={logit_stats['max']:.4f}, "
          f"mean={logit_stats['mean']:.4f}, std={logit_stats['std']:.4f}")

    # Also get greedy token for verification
    greedy_id = int(last_logits.argmax())
    greedy_token = tokenizer.decode([greedy_id])
    print(f"Greedy next token: [{greedy_id}] \"{greedy_token}\"")
    print()

    # Build reference data
    reference = {
        "model_id": MODEL_ID,
        "prompt": TEST_PROMPT,
        "input_token_ids": input_token_ids,
        "input_token_strings": token_strings,
        "num_input_tokens": len(input_token_ids),
        "logits_position": len(input_token_ids) - 1,
        "logit_stats": logit_stats,
        "top_k": TOP_K,
        "top_logits": top_entries,
        "greedy_token_id": greedy_id,
        "greedy_token_string": greedy_token,
        "torch_dtype": "float32",
        "device": device,
    }

    # Save
    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)

    print(f"Reference saved to: {REFERENCE_PATH}")
    print(f"File size: {REFERENCE_PATH.stat().st_size / 1024:.1f} KB")
    print()
    print("Next step: run CuInfer with --dump-logits and then compare:")
    print(f"  ./build/cuinfer models/SmolLM2-135M-Instruct-f16.gguf "
          f"\"{TEST_PROMPT}\" --dump-logits")
    print(f"  python scripts/verify_logits.py cuinfer_logits.json")


# ---------------------------------------------------------------------------
# Mode 2: Compare CuInfer Output Against Reference
# ---------------------------------------------------------------------------

def compare_logits(cuinfer_path: str) -> bool:
    """Compare CuInfer logits dump against PyTorch reference.

    Returns True if the comparison passes, False otherwise.
    """
    print("=" * 60)
    print("  CuInfer Logit Verification — Comparison")
    print("=" * 60)
    print()

    # Load reference
    if not REFERENCE_PATH.exists():
        print(f"ERROR: Reference file not found: {REFERENCE_PATH}")
        print("Run this script without arguments first to generate the reference.")
        return False

    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        reference = json.load(f)

    # Load CuInfer output
    cuinfer_file = Path(cuinfer_path)
    if not cuinfer_file.exists():
        print(f"ERROR: CuInfer logits file not found: {cuinfer_file}")
        return False

    with open(cuinfer_file, "r", encoding="utf-8") as f:
        cuinfer = json.load(f)

    print(f"Reference : {REFERENCE_PATH.name}")
    print(f"  Model   : {reference['model_id']}")
    print(f"  Prompt  : \"{reference['prompt']}\"")
    print(f"  Tokens  : {reference['num_input_tokens']}")
    print(f"  Top-K   : {reference['top_k']}")
    print()

    print(f"CuInfer   : {cuinfer_file.name}")
    print(f"  Prompt  : \"{cuinfer.get('prompt', 'N/A')}\"")
    print(f"  Tokens  : {cuinfer.get('num_input_tokens', 'N/A')}")
    print(f"  Top-K   : {cuinfer.get('top_k', 'N/A')}")
    print()

    # Sanity check: prompts should match
    if reference["prompt"] != cuinfer.get("prompt", ""):
        print("WARNING: Prompts do not match!")
        print(f"  Reference: \"{reference['prompt']}\"")
        print(f"  CuInfer  : \"{cuinfer.get('prompt', '')}\"")
        print()

    # Compare token IDs
    ref_tokens = reference["input_token_ids"]
    cu_tokens = cuinfer.get("input_token_ids", [])
    if ref_tokens != cu_tokens:
        print("WARNING: Input token IDs differ!")
        print(f"  Reference: {ref_tokens}")
        print(f"  CuInfer  : {cu_tokens}")
        print()

    # Extract top logit entries
    ref_logits = reference["top_logits"]
    cu_logits = cuinfer.get("top_logits", [])

    if not cu_logits:
        print("ERROR: CuInfer output has no top_logits entries.")
        return False

    # Compare top-K logit values
    n_compare = min(len(ref_logits), len(cu_logits))
    print("-" * 60)
    print(f"{'Rank':>4}  {'Token':>6}  {'PyTorch':>10}  {'CuInfer':>10}  {'Diff':>10}  Token")
    print("-" * 60)

    max_diff = 0.0
    diffs = []
    for i in range(n_compare):
        ref_val = ref_logits[i]["logit_value"]
        cu_val = cu_logits[i]["logit_value"]
        ref_id = ref_logits[i]["token_id"]
        cu_id = cu_logits[i]["token_id"]
        diff = abs(ref_val - cu_val)
        max_diff = max(max_diff, diff)
        diffs.append(diff)

        # Show the comparison (use ref token for display)
        ref_str = ref_logits[i].get("token_string", "?")
        status = " " if diff < TOLERANCE else "*"
        print(f"  #{i+1:>2}: [{ref_id:>6}] {ref_val:>10.4f}  "
              f"[{cu_id:>6}] {cu_val:>10.4f}  {diff:>10.6f}{status} "
              f"\"{ref_str}\"")

    print("-" * 60)
    print()

    # Compare top-5 token agreement
    ref_top5_ids = [entry["token_id"] for entry in ref_logits[:5]]
    cu_top5_ids = [entry["token_id"] for entry in cu_logits[:5]]

    # Check exact position match
    top5_exact = ref_top5_ids == cu_top5_ids

    # Check set overlap (same tokens, maybe different order)
    top5_overlap = set(ref_top5_ids) == set(cu_top5_ids)

    # Check top-1 match
    top1_match = (len(ref_logits) > 0 and len(cu_logits) > 0 and
                  ref_logits[0]["token_id"] == cu_logits[0]["token_id"])

    # Report
    print("=== Results ===")
    print()
    print(f"  Max absolute difference : {max_diff:.6f}")
    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Tolerance               : {TOLERANCE}")
    print()
    print(f"  Top-1 token match       : {'PASS' if top1_match else 'FAIL'}")
    print(f"    PyTorch: [{ref_logits[0]['token_id']}] "
          f"\"{ref_logits[0].get('token_string', '?')}\"")
    if cu_logits:
        print(f"    CuInfer: [{cu_logits[0]['token_id']}] "
              f"\"{cu_logits[0].get('token_string', '?')}\"")
    print()
    print(f"  Top-5 exact order match : {'PASS' if top5_exact else 'FAIL'}")
    print(f"    PyTorch: {ref_top5_ids}")
    print(f"    CuInfer: {cu_top5_ids}")
    if not top5_exact and top5_overlap:
        print(f"    (Same tokens, different ordering)")
    print()
    print(f"  Top-5 set overlap       : {'PASS' if top5_overlap else 'FAIL'}")
    print()

    # Overall pass/fail
    value_pass = max_diff < TOLERANCE
    overall_pass = value_pass and top1_match

    if overall_pass:
        print(">>> OVERALL: PASS <<<")
        print(f"  Logit values within tolerance ({TOLERANCE}) and top-1 token matches.")
    else:
        print(">>> OVERALL: FAIL <<<")
        reasons = []
        if not value_pass:
            reasons.append(f"max diff {max_diff:.6f} > tolerance {TOLERANCE}")
        if not top1_match:
            reasons.append("top-1 token mismatch")
        print(f"  Reasons: {'; '.join(reasons)}")
        print()
        print("Note: Some numerical differences are expected due to:")
        print("  - FP16 weight dequantization (CuInfer loads F16 GGUF, dequants to FP32)")
        print("  - Different matmul implementations (custom tiled kernels vs cuBLAS)")
        print("  - Floating-point operation order differences")
        if max_diff < 0.1:
            print()
            print("  The difference is small enough that the model likely produces")
            print("  equivalent text output. Consider relaxing the tolerance.")

    return overall_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        # Mode 2: Compare
        cuinfer_path = sys.argv[1]
        success = compare_logits(cuinfer_path)
        sys.exit(0 if success else 1)
    else:
        # Mode 1: Generate reference
        generate_reference()


if __name__ == "__main__":
    main()
