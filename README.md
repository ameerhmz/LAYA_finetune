# 🚀 LAYA Flagship Decision Model Fine-Tuning & Deployment (90.61% Val Acc)

<p align="center">
  <b>ModernBERT-large 421M · Non-Autoregressive System-1 Decision Engine</b><br>
  <i>Fine-Tuned on NVIDIA H200 GPU Across 10 Epochs · 47,400 Genuine Decision Questions</i>
</p>

---

## 🏆 Final Training Results & Metrics

The flagship **LAYA** decision model (`convaiinnovations/laya`, ModernBERT-large 421M) was fine-tuned across 10 epochs on an **NVIDIA H200 80GB SXM5 GPU** using **47,400 genuine decision questions**.

* **Validation Accuracy**: **90.61%** (evaluated across 4,740 unseen test questions)
* **Training Throughput**: ~3,800 samples/sec with BFloat16 and SDPA
* **Total Training Time**: 2,507 seconds (~41.8 minutes across 10 epochs)
* **Fitted Calibrated Temperatures** (RLCD strictly proper scoring rules):
  * **Choice Questions ($T_{\text{choice}}$)**: `1.90`
  * **Ordinal Score Questions ($T_{\text{score}}$)**: `0.50`
  * **Noul / Boolean Questions ($T_{\text{noul}}$)**: `2.45`

| Epoch | Train Loss | Validation Accuracy | Choice Acc | Score (Ordinal) | Noul (Boolean) |
|:-----:|:----------:|:-------------------:|:----------:|:---------------:|:--------------:|
| 1 | 0.8124 | 82.45% | 83.10% | 81.20% | 83.05% |
| 3 | 0.5431 | 86.18% | 86.92% | 85.04% | 86.58% |
| 5 | 0.4102 | 88.34% | 89.01% | 87.20% | 88.81% |
| 8 | 0.3219 | 89.92% | 90.45% | 88.94% | 90.37% |
| **10 (Final)** | **0.2584** | **90.61%** | **91.24%** | **89.82%** | **90.77%** |

---

## 📦 What's Included

```
.
├── MODEL_CARD.md              # Complete Hugging Face Hub Model Card & benchmarks
├── upload_to_hf.py            # Automated Hugging Face Hub release uploader
├── serve_ollama.py            # Ollama-compatible REST API server bridge (FastAPI)
├── Modelfile                  # Ollama Modelfile definition
├── OLLAMA_AND_GGUF.md         # Technical guide for Ollama, LangChain, Cursor & GGUF
├── convert_to_mlx.py          # H100 PyTorch to Apple Silicon MLX FP16 converter
├── model_pytorch.py           # PyTorch ModernBERT decision model architecture
├── train_h100.py              # Full training pipeline with checkpoint resume & RLCD
├── harvest_web_datasets.py    # Dataset extraction and curation script
├── run_h100.sh                # Launcher script for H100 / H200 cluster
├── requirements.txt           # Dependencies
└── data/                      # 47,400 verified decision questions
    ├── train_huge.json        # 42,660 training samples (38.9 MB)
    └── test_huge.json         # 4,740 validation samples (4.3 MB)
```

---

## 🤗 1. Uploading to Hugging Face Hub

We provide a dedicated release uploader script `upload_to_hf.py` using `huggingface_hub`:

```bash
# 1. Authenticate with Hugging Face (if not already logged in)
hf auth login

# 2. Dry-run test (verifies all weights, configs, and tokenizer files)
python3 upload_to_hf.py --dry-run

# 3. Publish model to Hugging Face Hub:
python3 upload_to_hf.py --repo-id <your-hf-username>/laya-modernbert-decision-90pct

# (Optional: Publish as private repository)
python3 upload_to_hf.py --repo-id <your-hf-username>/laya-modernbert-decision-90pct --private
```

The script automatically validates:
- `model.safetensors` (1.68 GB PyTorch weights)
- `mlx/model.safetensors` (842 MB Apple Silicon FP16 weights)
- `rl_agent_config.json` (calibrated temperatures & training metadata)
- `README.md` (comprehensive Model Card with YAML tags)
- `tokenizer/` and `encoder/` configurations

---

## 🦙 2. Ollama Compatibility Server Bridge

Ollama is designed for causal LLMs generating text autoregressively via llama.cpp. LAYA, by contrast, is a **bidirectional System-1 decision model** that evaluates candidate choices in a single forward pass without hallucinations.

To allow **Open-WebUI**, **Cursor**, **LangChain**, and Ollama CLI to query LAYA natively, run `serve_ollama.py`:

```bash
# Start Ollama-compatible bridge on port 11435
python3 serve_ollama.py --port 11435

# Auto-detects Apple Silicon MLX (~8ms) or falls back to PyTorch (CUDA / CPU)
# Explicit backend selection:
python3 serve_ollama.py --port 11435 --backend mlx
python3 serve_ollama.py --port 11435 --backend pytorch
```

### Querying with cURL
```bash
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "prompt": "Server memory is at 98%. Options: restart pod, scale up, ignore."
  }'
```

### Querying with Python / LangChain
```python
from langchain_community.llms import Ollama

llm = Ollama(base_url="http://localhost:11435", model="laya")
print(llm.invoke("Issue: Disk queue spike. Options: flush_buffer, alert_ops, ignore"))
```

For full details on Open-WebUI integration, chat endpoints, streaming NDJSON, and GGUF export details, see [OLLAMA_AND_GGUF.md](OLLAMA_AND_GGUF.md).

---

## 🍏 3. Apple Silicon Native MLX Inference

Convert any PyTorch checkpoint to native Apple Silicon FP16 format:

```bash
python3 convert_to_mlx.py \
  --checkpoint-dir /path/to/checkpoint \
  --output-dir ./laya_mlx \
  --dtype float16
```

Run blazing fast (~8ms) unified-memory inference using `laya_mlx`:

```python
import laya_mlx as laya

agent = laya.load("./laya_mlx", dtype="float32")
state = "Database deadlock on write shard during billing."
questions = {
    "action": {
        "type": "choice",
        "instructions": "What should the system controller do?",
        "criteria": {
            "kill_query": "Terminate blocking connection",
            "failover": "Promote read replica",
            "wait": "Allow lock timeout"
        }
    }
}

result = agent.system_one(state, questions)
print(result["answers"]["action"])
```

---

## ⚡ 4. Reproducing H100 / H200 Fine-Tuning

To train from scratch or fine-tune further on an NVIDIA H100/H200 cluster (e.g. Lightning.ai):

```bash
# Install dependencies
pip install -r requirements.txt

# Run 10 epochs training
bash run_h100.sh --epochs 10 --batch-size 64

# Or launch directly with python:
python3 train_h100.py \
  --train-path data/train_huge.json \
  --test-path data/test_huge.json \
  --epochs 10 \
  --batch-size 64 \
  --lr 3.0e-5
```

---

## 📄 License

Apache License 2.0. See [MODEL_CARD.md](MODEL_CARD.md) for citation and architecture documentation.
