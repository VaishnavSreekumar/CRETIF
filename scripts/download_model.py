#!/usr/bin/env python3
"""Download SmolLM2-135M-Instruct GGUF model files from HuggingFace.

This script downloads quantized GGUF model files for use with the CuInfer
inference engine. Files are downloaded to the models/ directory at the project
root using the huggingface_hub library.

Supported quantization variants:
  - f32   : Full FP32 precision (~540 MB) — maximum accuracy, largest file
  - f16   : FP16 half precision (~270 MB) — good balance for first integration test
  - q8_0  : INT8 quantized (~143 MB)      — smallest, slight quality loss

Source: https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF

Usage:
  python download_model.py              # Download F16 (default)
  python download_model.py --quant f32  # Download FP32
  python download_model.py --quant q8_0 # Download Q8_0

Note: All downloads go to E:\\Repos\\CuInfer\\models\\ — never to C: drive
(disk space constraint on this development machine).
"""

import argparse
import os
import sys
from pathlib import Path


def check_dependencies() -> None:
    """Verify that huggingface_hub is installed, with a helpful error if not."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: 'huggingface_hub' is not installed.")
        print()
        print("Install it with:")
        print("  pip install huggingface_hub")
        print()
        print("This is the only dependency needed for downloading models.")
        sys.exit(1)


def get_project_root() -> Path:
    """Return the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# GGUF file mapping — maps quantization names to actual filenames on HF
# ---------------------------------------------------------------------------

REPO_ID = "bartowski/SmolLM2-135M-Instruct-GGUF"

# Filenames follow bartowski's naming convention on HuggingFace.
# These map our friendly --quant flag values to the actual GGUF filenames.
QUANT_FILES = {
    "f32": "SmolLM2-135M-Instruct-f32.gguf",
    "f16": "SmolLM2-135M-Instruct-f16.gguf",
    "q8_0": "SmolLM2-135M-Instruct-Q8_0.gguf",
    "q4_k_m": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
}

# Approximate file sizes for user information (in MB)
QUANT_SIZES_MB = {
    "f32": 540,
    "f16": 270,
    "q8_0": 143,
    "q4_k_m": 100,
}


def download_model(quant: str, models_dir: Path) -> Path:
    """Download a specific GGUF quantization variant.

    Args:
        quant: Quantization variant key (f32, f16, q8_0).
        models_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    filename = QUANT_FILES[quant]
    dest_path = models_dir / filename

    # Skip if already downloaded
    if dest_path.exists():
        file_size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"File already exists: {dest_path}")
        print(f"  Size: {file_size_mb:.1f} MB")
        print()
        print("To re-download, delete the file first and run again.")
        return dest_path

    print(f"Downloading {filename} from {REPO_ID}")
    print(f"  Quantization : {quant.upper()}")
    print(f"  Expected size: ~{QUANT_SIZES_MB[quant]} MB")
    print(f"  Destination  : {models_dir}")
    print()

    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download with progress bar (huggingface_hub shows progress by default)
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )

    # Verify the downloaded file exists at the expected location
    result_path = Path(downloaded_path)
    if not result_path.exists():
        print(f"ERROR: Download reported success but file not found at {result_path}")
        sys.exit(1)

    file_size_mb = result_path.stat().st_size / (1024 * 1024)

    print()
    print("Download complete!")
    print(f"  File: {result_path}")
    print(f"  Size: {file_size_mb:.1f} MB")

    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SmolLM2-135M-Instruct GGUF model files from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python download_model.py              # Download F16 (default)\n"
            "  python download_model.py --quant f32  # Download FP32\n"
            "  python download_model.py --quant q8_0 # Download Q8_0\n"
            "\n"
            "Files are saved to the models/ directory at the project root.\n"
            f"Source: https://huggingface.co/{REPO_ID}\n"
        ),
    )

    parser.add_argument(
        "--quant",
        type=str,
        choices=list(QUANT_FILES.keys()),
        default="f16",
        help=(
            "Quantization variant to download. "
            "Choices: f32 (~540 MB), f16 (~270 MB, default), q8_0 (~143 MB)."
        ),
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=(
            "Override the models directory path. "
            "Default: <project_root>/models/"
        ),
    )

    args = parser.parse_args()

    # Check dependencies before doing anything else
    check_dependencies()

    # Determine models directory
    if args.models_dir:
        models_dir = Path(args.models_dir).resolve()
    else:
        models_dir = get_project_root() / "models"

    # Safety check: warn if downloading to C: drive (disk space constraint)
    models_dir_str = str(models_dir)
    if os.name == "nt" and models_dir_str.upper().startswith("C:"):
        print("WARNING: Downloading to C: drive. This machine has limited C: disk space.")
        print(f"  Consider using --models-dir to specify E: or D: drive instead.")
        print()

    print("=" * 60)
    print("  CuInfer Model Downloader")
    print("=" * 60)
    print(f"  Repository   : {REPO_ID}")
    print(f"  Quantization : {args.quant.upper()}")
    print(f"  Filename     : {QUANT_FILES[args.quant]}")
    print(f"  Models dir   : {models_dir}")
    print("=" * 60)
    print()

    download_model(args.quant, models_dir)


if __name__ == "__main__":
    main()
