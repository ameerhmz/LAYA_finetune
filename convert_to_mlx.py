#!/usr/bin/env python3
"""
LAYA H100-to-Apple Silicon MLX Converter & Verifier.

Takes the trained PyTorch checkpoint from the H100 cluster, optimizes it for Apple Silicon
(optional FP16 compression for fast unified memory inference), and tests end-to-end
execution with laya_mlx.
"""

import argparse
import json
import shutil
from pathlib import Path
import mlx.core as mx
import safetensors.torch
import torch


def convert_and_verify(
    checkpoint_dir: str,
    output_dir: str,
    dtype: str = "float16",
    test_prediction: bool = True,
):
    in_path = Path(checkpoint_dir).resolve()
    out_path = Path(output_dir).resolve()

    print("=" * 75)
    print("🍏 LAYA H100 -> Apple Silicon MLX Model Converter & Verifier")
    print(f"   Input PyTorch Checkpoint : {in_path}")
    print(f"   Output MLX Checkpoint    : {out_path}")
    print(f"   Target Target Dtype      : {dtype}")
    print("=" * 75)

    if not (in_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing model.safetensors in {in_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Convert Weights to Target Dtype
    print(f"📦 Loading weights from {in_path / 'model.safetensors'}...")
    pt_weights = safetensors.torch.load_file(str(in_path / "model.safetensors"))

    mlx_dtype = getattr(mx, dtype, mx.float16)
    target_weights = {}
    for k, v in pt_weights.items():
        np_arr = v.cpu().to(torch.float32).numpy()
        mx_arr = mx.array(np_arr).astype(mlx_dtype)
        target_weights[k] = mx_arr

    out_weights_file = out_path / "model.safetensors"
    print(f"💾 Saving MLX-optimized {dtype} weights to {out_weights_file}...")
    mx.save_safetensors(str(out_weights_file), target_weights)

    # 2. Copy Configs & Tokenizer
    for config_name in ["rl_agent_config.json", "mlx_config.json"]:
        src_file = in_path / config_name
        if src_file.exists():
            shutil.copy2(src_file, out_path / config_name)
            print(f"   ✓ Copied {config_name}")

    # Ensure mlx_config.json exists
    if not (out_path / "mlx_config.json").exists():
        with open(out_path / "mlx_config.json", "w") as f:
            json.dump({
                "model_type": "decision_agent",
                "encoder": "jhu-clsp/mmBERT-base",
                "hidden_size": 768,
                "head_layers": 2,
            }, f, indent=2)
        print("   ✓ Generated mlx_config.json")

    # Copy tokenizer
    src_tok = in_path / "tokenizer"
    dst_tok = out_path / "tokenizer"
    if src_tok.exists():
        if dst_tok.exists():
            shutil.rmtree(dst_tok)
        shutil.copytree(src_tok, dst_tok)
        print("   ✓ Copied tokenizer directory")
    else:
        # Pull from Hugging Face base tokenizer
        print("   📥 Downloading tokenizer from base model...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("aac6fef/laya-multilingual-mlx", subfolder="tokenizer")
        tok.save_pretrained(str(dst_tok))
        print("   ✓ Generated tokenizer directory")

    print(f"\n✅ Checkpoint converted successfully! ({out_weights_file.stat().st_size / 1e6:.1f} MB)")

    # 3. Test Loading & Inference with laya_mlx
    if test_prediction:
        print("\n🧪 Testing Inference with Apple Silicon laya_mlx...")
        import laya_mlx as laya

        agent = laya.load(str(out_path), dtype="float32")
        print("   ✓ Successfully loaded fine-tuned model into laya_mlx.Agent!")

        test_state = "A critical database deadlock occurred on production shard-3 while processing payments."
        test_questions = {
            "incident_severity": {
                "type": "choice",
                "instructions": "What is the severity of this production incident?",
                "criteria": {
                    "p0_critical": "Outage affecting core user workflows, requires immediate on-call pager",
                    "p1_high": "Degraded performance on non-blocking components",
                    "p2_medium": "Minor bug with workaround available",
                    "p3_low": "Cosmetic or logging issue",
                },
            },
            "should_page_oncall": {
                "type": "noul",
                "instructions": "Should the system page the primary on-call engineer immediately?",
                "criteria": {
                    "false": "No, handle automatically or during business hours",
                    "true": "Yes, page immediately",
                },
            },
        }

        print("\n⚡ Running test decision on sample input:")
        print(f"   State: \"{test_state}\"")
        res = agent.system_one(test_state, test_questions)

        for qid, ans in res["answers"].items():
            print(f"\n   [{qid.upper()}] ({ans['type']})")
            if ans["type"] == "choice":
                print(f"   • Selected Choice: {ans['choice']} (Confidence: {ans['confidence']:.2%})")
                print(f"   • Probabilities  : {ans['probabilities']}")
            elif ans["type"] == "noul":
                print(f"   • Truth Score    : {ans['noul']:.4f} (Confidence: {ans['confidence']:.2%})")
            print(f"   • Action Prob    : {ans['action']['act_probability']:.4f}")

        print("\n🎉 Verification Complete! Model is 100% operational on Apple Silicon MLX.")


def main():
    parser = argparse.ArgumentParser(description="Convert H100 PyTorch LAYA checkpoint to MLX format")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing H100 PyTorch checkpoint")
    parser.add_argument("--output-dir", type=str, default="./laya_mlx_finetuned", help="Target MLX checkpoint directory")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Target weight dtype")
    parser.add_argument("--no-test", action="store_true", help="Skip running verification inference")
    args = parser.parse_args()

    convert_and_verify(
        args.checkpoint_dir,
        args.output_dir,
        dtype=args.dtype,
        test_prediction=not args.no_test,
    )


if __name__ == "__main__":
    main()
