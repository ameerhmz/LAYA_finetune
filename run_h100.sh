#!/bin/bash
set -e

echo "=========================================================================="
echo "⚡ Starting LAYA Fine-Tuning on NVIDIA H100 (Lightning.ai)"
echo "=========================================================================="

# 1. GPU Diagnostic
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Checking NVIDIA GPU configuration:"
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader
    NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
else
    echo "⚠️ nvidia-smi not found. Falling back to single process."
    NUM_GPUS=1
fi

echo "   Available GPUs: $NUM_GPUS"

# 2. Performance Environment Variables
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="9.0"  # Hopper H100 architecture
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 3. Execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
EPOCHS=10
BATCH_SIZE=64
LR=3.0e-5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --resume|-r)
            RESUME="$2"
            EXTRA_ARGS+=("--resume" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "$RESUME" ]; then
    echo "   Settings: Resuming from=$RESUME, Epochs=$EPOCHS, Batch Size=$BATCH_SIZE, Learning Rate=$LR"
else
    echo "   Settings: Epochs=$EPOCHS, Batch Size=$BATCH_SIZE, Learning Rate=$LR"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Launching Distributed Data Parallel with torchrun ($NUM_GPUS GPUs)..."
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train_h100.py \
        --train-data ./data/train_huge.json \
        --test-data ./data/test_huge.json \
        --checkpoint convaiinnovations/laya \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        "${EXTRA_ARGS[@]}"
else
    echo "🚀 Launching Single H100 Training..."
    python3 train_h100.py \
        --train-data ./data/train_huge.json \
        --test-data ./data/test_huge.json \
        --checkpoint convaiinnovations/laya \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        "${EXTRA_ARGS[@]}"
fi

echo "=========================================================================="
echo "✅ Training Completed Successfully!"
echo "   Output Checkpoint: ./laya_finetuned_h100/model.safetensors"
echo "=========================================================================="
