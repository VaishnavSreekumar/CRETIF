#!/usr/bin/env python3
"""Generate reference outputs from SmolLM2-135M for validating the CuInfer C++ inference engine.

This script loads the SmolLM2-135M-Instruct model from HuggingFace using the
transformers library, runs a set of test prompts through the model in FP32
precision, and saves the results (token IDs, logits, decoded text) to a JSON
file that the CuInfer test suite can use for correctness validation.

For each test prompt, we capture:
  1. The tokenized input (token IDs from the HuggingFace tokenizer)
  2. First-token logits: the top-10 logit values and their token IDs
     (for the first generated position, i.e., the logits at the last input position)
  3. Greedy-decoded output (50 tokens of autoregressive generation)

The reference data is saved to tests/reference_data/smollm2_135m_reference.json.

This data serves as ground truth for verifying that the C++ forward pass
produces numerically equivalent results (within FP32 tolerance).

Usage:
  python reference_logits.py

Dependencies:
  pip install torch transformers

Note: This downloads the model from HuggingFace (~270 MB for the FP32 weights).
The model is cached by the transformers library in the HuggingFace cache directory.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def check_dependencies() -> None:
    """Verify that torch and transformers are installed, with helpful errors if not."""
    missing = []

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print()
        print("Install them with:")
        print("  pip install torch transformers")
        print()
        print("For CUDA-enabled PyTorch, see: https://pytorch.org/get-started/locally/")
        sys.exit(1)


def get_project_root() -> Path:
    """Return the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

TEST_PROMPTS = [
    "Hello",
    "The capital of France is",
    "Once upon a time",
]

NUM_GENERATE_TOKENS = 50
TOP_K_LOGITS = 10


# ---------------------------------------------------------------------------
# Reference generation
# ---------------------------------------------------------------------------

def generate_reference(
    model_id: str,
    prompts: List[str],
    num_tokens: int,
    top_k: int,
) -> Dict[str, Any]:
    """Generate reference outputs for the given prompts.

    Args:
        model_id: HuggingFace model identifier.
        prompts: List of text prompts to process.
        num_tokens: Number of tokens to generate via greedy decoding.
        top_k: Number of top logits to save per prompt.

    Returns:
        Dictionary with model info and per-prompt results.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    print(f"  Precision: torch.float32 (FP32)")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load model in FP32 for maximum accuracy
    # Use CPU if CUDA is not available — the model is small enough
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded. Parameters: {num_params:,}")
    print()

    # Collect model config info for the reference file
    config = model.config
    model_info = {
        "model_id": model_id,
        "num_parameters": num_params,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "float32",
        "device": device,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Process each prompt
    results = []
    for i, prompt in enumerate(prompts):
        print(f"--- Prompt {i + 1}/{len(prompts)}: \"{prompt}\" ---")
        result = process_prompt(model, tokenizer, prompt, num_tokens, top_k, device)
        results.append(result)
        print()

    return {
        "model_info": model_info,
        "generation_config": {
            "num_generate_tokens": num_tokens,
            "top_k_logits": top_k,
            "decoding": "greedy",
            "temperature": 1.0,
        },
        "prompts": results,
    }


def process_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_tokens: int,
    top_k: int,
    device: str,
) -> Dict[str, Any]:
    """Process a single prompt and return reference data.

    Args:
        model: The loaded HuggingFace model.
        tokenizer: The loaded HuggingFace tokenizer.
        prompt: The text prompt to process.
        num_tokens: Number of tokens to generate.
        top_k: Number of top logits to capture.
        device: Device string ("cuda" or "cpu").

    Returns:
        Dictionary with tokenization, logits, and generation results.
    """
    import torch

    # Tokenize the prompt
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoding["input_ids"]  # [1, seq_len]
    input_token_ids = input_ids[0].tolist()

    # Decode each token individually to show the tokenization
    token_strings = [tokenizer.decode([tid]) for tid in input_token_ids]

    print(f"  Input tokens ({len(input_token_ids)}): {input_token_ids}")
    print(f"  Token strings: {token_strings}")

    # Forward pass to get logits at the last position
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Extract logits at the last input position (the prediction position)
    last_logits = logits[0, -1, :]  # [vocab_size]

    # Get top-k logits
    top_values, top_indices = torch.topk(last_logits, top_k)
    top_logits = []
    for val, idx in zip(top_values.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        top_logits.append({
            "token_id": idx,
            "token_string": token_str,
            "logit_value": val,
        })

    print(f"  Top-{top_k} logits at last position:")
    for entry in top_logits:
        print(f"    [{entry['token_id']:>6}] {entry['logit_value']:>10.4f}  "
              f"\"{entry['token_string']}\"")

    # Greedy decoding: generate num_tokens tokens
    # We use model.generate with do_sample=False for deterministic greedy decoding
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,       # Greedy decoding
            temperature=1.0,
            top_p=1.0,
            num_beams=1,
        )

    # generated includes the input tokens, so slice them off
    generated_ids = generated[0].tolist()
    new_token_ids = generated_ids[len(input_token_ids):]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print(f"  Generated {len(new_token_ids)} new tokens")
    print(f"  Generated text: \"{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}\"")

    # Also capture logit statistics for debugging
    logit_stats = {
        "min": float(last_logits.min()),
        "max": float(last_logits.max()),
        "mean": float(last_logits.mean()),
        "std": float(last_logits.std()),
    }

    return {
        "prompt": prompt,
        "input_token_ids": input_token_ids,
        "input_token_strings": token_strings,
        "num_input_tokens": len(input_token_ids),
        "logits_position": len(input_token_ids) - 1,
        "logit_stats": logit_stats,
        "top_logits": top_logits,
        "generated_token_ids": new_token_ids,
        "generated_text": generated_text,
        "full_output_token_ids": generated_ids,
        "full_output_text": full_text,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Check dependencies first
    check_dependencies()

    print("=" * 72)
    print("  CuInfer Reference Logits Generator")
    print("=" * 72)
    print(f"  Model     : {MODEL_ID}")
    print(f"  Prompts   : {len(TEST_PROMPTS)}")
    print(f"  Tokens    : {NUM_GENERATE_TOKENS} per prompt (greedy)")
    print(f"  Top-k     : {TOP_K_LOGITS} logits saved per prompt")
    print(f"  Precision : FP32 (torch.float32)")
    print("=" * 72)
    print()

    # Generate reference data
    reference = generate_reference(
        model_id=MODEL_ID,
        prompts=TEST_PROMPTS,
        num_tokens=NUM_GENERATE_TOKENS,
        top_k=TOP_K_LOGITS,
    )

    # Determine output path
    project_root = get_project_root()
    output_dir = project_root / "tests" / "reference_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smollm2_135m_reference.json"

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("  Results Summary")
    print("=" * 72)
    print(f"  Output file: {output_path}")
    print(f"  File size  : {output_path.stat().st_size / 1024:.1f} KB")
    print()

    for i, result in enumerate(reference["prompts"]):
        print(f"  Prompt {i + 1}: \"{result['prompt']}\"")
        print(f"    Input tokens  : {result['num_input_tokens']}")
        print(f"    Top-1 predict : [{result['top_logits'][0]['token_id']}] "
              f"\"{result['top_logits'][0]['token_string']}\" "
              f"(logit={result['top_logits'][0]['logit_value']:.4f})")
        print(f"    Generated     : {len(result['generated_token_ids'])} tokens")
        gen_preview = result['generated_text'][:60]
        ellipsis = "..." if len(result['generated_text']) > 60 else ""
        print(f"    Output preview: \"{gen_preview}{ellipsis}\"")
        print()

    print("=" * 72)
    print("  Reference data saved successfully.")
    print("  Use this file to validate the CuInfer C++ inference engine.")
    print("=" * 72)


if __name__ == "__main__":
    main()
