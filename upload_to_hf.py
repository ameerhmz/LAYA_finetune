#!/usr/bin/env python3
"""
Hugging Face Release Uploader for Fine-Tuned LAYA Decision Model (90.61% Val Acc).

Uploads the trained ModernBERT-large LAYA decision engine, including PyTorch
weights, Apple Silicon MLX weights, tokenizer, config files, inference engines,
and comprehensive model card to the Hugging Face Hub.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo


REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "rl_agent_config.json",
    "rl_agent_api.py",
    "rl_common.py",
    "README.md",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
    "mlx/model.safetensors",
    "mlx/mlx_config.json",
]


def validate_model_dir(model_dir: Path):
    """Verify that all essential files exist and display sizes."""
    if not model_dir.is_dir():
        print(f"❌ Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)

    print("=" * 75)
    print(f"🔍 Validating LAYA Checkpoint Files in: {model_dir}")
    print("=" * 75)

    all_present = True
    total_size = 0
    for rel_path in REQUIRED_FILES:
        target = model_dir / rel_path
        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            total_size += target.stat().st_size
            print(f"  ✓ {rel_path:<35} ({size_mb:>8.2f} MB)")
        else:
            print(f"  ✗ MISSING: {rel_path}")
            all_present = False

    total_mb = total_size / (1024 * 1024)
    print("-" * 75)
    print(f"  Total Package Size: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
    print("-" * 75)

    if not all_present:
        print("\n⚠️ Some required files are missing! Please verify checkpoint contents.")
        sys.exit(1)
    print("✅ All required model files verified successfully.\n")


def upload_model(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    token: str = None,
    commit_message: str = "Upload fine-tuned LAYA ModernBERT 90.61% decision engine",
    dry_run: bool = False,
):
    model_path = Path(model_dir).resolve()
    validate_model_dir(model_path)

    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("username")
        print(f"👤 Authenticated with Hugging Face as: \033[1;32m{username}\033[0m")
    except Exception as e:
        print(f"❌ Failed to authenticate with Hugging Face Hub: {e}")
        print("💡 Run 'hf auth login' or pass --token <your_hf_token>")
        sys.exit(1)

    print(f"🎯 Target Repository : \033[1;34m{repo_id}\033[0m")
    print(f"🔒 Visibility        : {'Private' if private else 'Public'}")
    print(f"💬 Commit Message    : {commit_message}")

    if dry_run:
        print("\n🔎 DRY RUN MODE: Pre-upload verification passed. No changes were made.")
        return

    # Create repository if it doesn't already exist
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"📦 Repository confirmed: {repo_url}")
    except Exception as e:
        print(f"❌ Failed to create/verify repository '{repo_id}': {e}")
        sys.exit(1)

    # Upload folder
    print(f"\n🚀 Uploading files from {model_path} to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.DS_Store", "__pycache__/*", "*.tmp"],
        )
        print("\n" + "=" * 75)
        print("🎉 SUCCESS! LAYA model has been successfully published to Hugging Face:")
        print(f"   👉 https://huggingface.co/{repo_id}")
        print("=" * 75)
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned LAYA ModernBERT decision model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ameerhmz5/laya-modernbert-decision-90pct",
        help="Target Hugging Face model repository ID (e.g., ameerhmz5/laya-modernbert-decision-90pct)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/Users/ameerhamza/HOBBY_CODING/LAYA/models/laya_finetuned_h200",
        help="Path to local fine-tuned model directory",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set repository visibility to private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (defaults to local cached token)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload fine-tuned LAYA ModernBERT 90.61% decision engine (PyTorch + MLX)",
        help="Git commit message for the upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all model files and authentication without uploading",
    )

    args = parser.parse_args()

    upload_model(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
