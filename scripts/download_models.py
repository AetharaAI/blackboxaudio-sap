#!/usr/bin/env python3
"""Pre-download models needed by SAP services.

Checks ~/.cache/huggingface/hub for already-cached models
and only downloads what's missing.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir /path/to/models
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REQUIRED_MODELS = {
    "large-v3": {
        "cache_dir_name": "models--Systran--faster-whisper-large-v3",
        "purpose": "ASR final pass (high accuracy)",
        "size": "~3.1 GB",
    },
    "distil-large-v3": {
        "cache_dir_name": "models--Systran--faster-whisper-distil-large-v3",
        "purpose": "ASR streaming pass (low latency)",
        "size": "~1.5 GB",
    },
}


def check_cached(hub_dir: Path) -> dict[str, bool]:
    """Check which models are already cached."""
    results = {}
    for name, info in REQUIRED_MODELS.items():
        model_dir = hub_dir / info["cache_dir_name"]
        results[name] = model_dir.exists() and any(model_dir.iterdir())
    return results


def main():
    parser = argparse.ArgumentParser(description="Download SAP models")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    if args.model_dir:
        os.environ["HF_HOME"] = args.model_dir
        hub_dir = Path(args.model_dir) / "hub"
    else:
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub"

    print(f"HuggingFace hub cache: {hub_dir}\n")

    # Check what's already there
    cached = check_cached(hub_dir)

    print("Required models:")
    for name, info in REQUIRED_MODELS.items():
        status = "CACHED" if cached[name] else "MISSING"
        print(f"  [{status}] {name} ({info['size']}) — {info['purpose']}")

    # Show other useful cached models
    print("\nOther cached models found:")
    if hub_dir.exists():
        for d in sorted(hub_dir.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                model_name = d.name.replace("models--", "").replace("--", "/")
                if not any(d.name == info["cache_dir_name"] for info in REQUIRED_MODELS.values()):
                    print(f"  {model_name}")

    # Download missing models
    missing = [name for name, is_cached in cached.items() if not is_cached]

    if not missing:
        print("\nAll required models are cached. Nothing to download.")
        return

    print(f"\nDownloading {len(missing)} missing model(s)...\n")

    from faster_whisper import WhisperModel

    for name in missing:
        info = REQUIRED_MODELS[name]
        print(f"Downloading {name} ({info['size']})...")
        try:
            model = WhisperModel(name, device="cpu", compute_type="int8")
            print(f"  {name} — downloaded successfully")
            del model
        except Exception as e:
            print(f"  WARNING: Failed to download {name}: {e}")
            sys.exit(1)

    print("\nDone. Re-run to verify.")


if __name__ == "__main__":
    main()
