---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags:
  - decision-engine
  - modernbert
  - calibration
  - rl
  - rlcd
  - classification
  - routing
  - system-one
  - mlx
  - apple-silicon
datasets:
  - SetFit/ag_news
  - dair-ai/emotion
  - google/boolq
  - LocalLLaMA/typed-decisions
metrics:
  - accuracy
base_model: convaiinnovations/laya
model_name: laya-modernbert-decision-90pct
---

# 🧠 LAYA Flagship Decision Model (ModernBERT-large 421M · 90.61% Val Acc)

<p align="center">
  <b>High-Precision Non-Autoregressive System-1 Decision Engine</b><br>
  <i>Fine-Tuned on NVIDIA H200 GPU Across 10 Epochs · 47,400 Genuine Decision Questions</i>
</p>

---

## 📌 Executive Summary

**LAYA** is a flagship **non-autoregressive System-1 decision model** based on `answerdotai/ModernBERT-large` (421M parameters). Given an arbitrary input **state** (natural language, system logs, code diffs, JSON documents, or emails) and **typed questions** with criteria, LAYA computes exact, mathematically calibrated probabilities across all options in a **single forward pass** (~10–30 ms) with **zero generation hallucination**.

This model checkpoint is the result of fine-tuning `convaiinnovations/laya` on an **NVIDIA H200 SXM5 GPU** across 10 epochs using a curated dataset of **47,400 genuine decision questions** (42,660 train, 4,740 test). It achieved **90.61% validation accuracy**, establishing state-of-the-art decision-routing performance with strictly calibrated confidence scores.

### Key Highlights
- **Architecture**: ModernBERT-large backbone (28 transformer layers, 16 attention heads, 1024 hidden dimension, 8192 token context window) + 2-layer Transformer decision scoring head.
- **Accuracy**: **90.61% validation accuracy** evaluated on 4,740 unseen multi-class decision, ordinal scoring, and Boolean validation questions.
- **Calibrated Temperatures**: Post-hoc temperature calibration via Strictly Proper Scoring Rules (**RLCD** - Reinforcement Learning with Calibrated Decisions):
  - **Choice questions**: $T_{\text{choice}} = 1.90$
  - **Score / Ordinal questions**: $T_{\text{score}} = 0.50$
  - **Noul (Boolean / Binary truth) questions**: $T_{\text{noul}} = 2.45$
- **Dual Format Included**:
  1. **PyTorch FP32/BF16**: `model.safetensors` (1.68 GB) for CUDA clusters, PyTorch, and Linux/cloud inference.
  2. **Native Apple Silicon MLX**: `mlx/model.safetensors` (842 MB FP16) for lightning-fast unified memory execution (~8ms latency) via `laya_mlx`.
- **Zero Hallucination**: Directly scores `[MASK]` tokens corresponding to candidate choices. It never produces autoregressive tokens, eliminating parsing failures, JSON syntax errors, and generation loops.

---

## 📊 Training & Benchmark Results

### Training Setup
- **Compute Cluster**: 1x NVIDIA H200 80GB SXM5 GPU
- **Dataset Size**: 47,400 total verified decision items
  - Training set: 42,660 samples
  - Validation & calibration set: 4,740 samples
- **Training Duration**: 2,507 seconds (~41.8 minutes across 10 epochs)
- **Optimizer**: AdamW with cosine learning rate schedule ($\eta = 3.0 \times 10^{-5}$)
- **Mixed Precision**: BFloat16 with Scaled Dot-Product Attention (SDPA)

### Validation Metrics across 10 Epochs

| Epoch | Train Loss | Validation Accuracy | Choice Acc | Score (Ordinal) | Noul (Boolean) |
|:-----:|:----------:|:-------------------:|:----------:|:---------------:|:--------------:|
| 1 | 0.8124 | 82.45% | 83.10% | 81.20% | 83.05% |
| 3 | 0.5431 | 86.18% | 86.92% | 85.04% | 86.58% |
| 5 | 0.4102 | 88.34% | 89.01% | 87.20% | 88.81% |
| 8 | 0.3219 | 89.92% | 90.45% | 88.94% | 90.37% |
| **10 (Final)** | **0.2584** | **90.61%** | **91.24%** | **89.82%** | **90.77%** |

---

## 🚀 Quickstart Guides

### 1. PyTorch / CUDA Quickstart (Bundled Upstream Engine)

The repository includes `rl_agent_api.py` and `rl_common.py`, allowing you to run predictions directly without external framework wrappers:

```python
import os
from huggingface_hub import snapshot_download
from rl_agent_api import RLAgent

# 1. Download checkpoint from Hugging Face (or point to local folder)
model_dir = snapshot_download(repo_id="ameerhmz5/laya-modernbert-decision-90pct")

# 2. Initialize Agent (automatically picks CUDA or CPU)
agent = RLAgent(model_dir=model_dir, device="cuda")

# 3. Define Context State & Questions
state = "High CPU alert: Pod 'payments-worker-9a8b' exceeded 98% memory usage for 5 minutes."

questions = {
    "recommended_action": {
        "type": "choice",
        "instructions": "What immediate remediation should be taken?",
        "criteria": {
            "restart_pod": "Gracefully restart the Kubernetes pod",
            "scale_replicas": "Increase HPA replica count",
            "page_oncall": "Trigger PagerDuty incident for on-call engineer",
            "suppress": "Known batch job spike, ignore alert"
        }
    },
    "is_p0_critical": {
        "type": "noul",
        "instructions": "Is this incident high severity (P0/P1) impacting user transactions?",
        "criteria": {
            "false": "Low or medium severity, non-blocking",
            "true": "Critical severity, user-facing impact"
        }
    }
}

# 4. Single-pass inference (<30ms)
result = agent.system_one(state, questions)

for qid, ans in result["answers"].items():
    print(f"[{qid}] Type: {ans['type']}")
    if ans["type"] == "choice":
        print(f"  Selected: {ans['choice']} (Confidence: {ans['confidence']:.2%})")
        print(f"  Probabilities: {ans['probabilities']}")
    elif ans["type"] == "noul":
        print(f"  Truth Probability: {ans['noul']:.4f}")
```

---

### 2. Apple Silicon Native MLX Quickstart (`laya_mlx`)

For Apple Silicon Macs (M1/M2/M3/M4), the `mlx/` subfolder provides pre-converted FP16 safetensors for unified memory inference at ~8ms latency:

```python
import laya_mlx as laya

# Load directly from the mlx subfolder
agent = laya.load("ameerhmz5/laya-modernbert-decision-90pct/mlx", dtype="float32")

state = "Database deadlock detected on master write shard during billing run."
questions = {
    "action": {
        "type": "choice",
        "instructions": "What should the system controller do?",
        "criteria": {
            "terminate_lock": "Terminate blocking transaction query",
            "failover": "Initiate read replica failover",
            "wait": "Allow automatic lock resolution timeout"
        }
    }
}

result = agent.system_one(state, questions)
print(result["answers"]["action"])
# Output: {'choice': 'terminate_lock', 'confidence': 0.892, 'probabilities': {...}}
```

---

### 3. Ollama REST API Compatibility Bridge

Ollama natively targets autoregressive text generation models. To allow Open-WebUI, Cursor, LangChain, and other Ollama clients to query LAYA transparently, run the bundled `serve_ollama.py` bridge:

```bash
# Start Ollama-compatible HTTP server on port 11435 (or 11434)
python3 serve_ollama.py --port 11435

# Query via standard Ollama REST API:
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "prompt": "Server memory is at 98%. Options: restart pod, scale up, ignore."
  }'
```

---

### 4. Transformers Backbone Loading

To inspect or use the underlying ModernBERT-large encoder directly in Hugging Face `transformers`:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

model_id = "ameerhmz5/laya-modernbert-decision-90pct"
tokenizer = AutoTokenizer.from_pretrained(f"{model_id}/tokenizer")
config = AutoConfig.from_pretrained(f"{model_id}/encoder")
encoder = AutoModel.from_pretrained(model_id, config=config)
```

---

## 📁 Repository Structure

```
.
├── README.md                 # Hugging Face Model Card & Documentation
├── .gitattributes            # Git LFS tracking configuration
├── config.json               # ModernBERT-large backbone architecture
├── model.safetensors         # 1.68 GB PyTorch FP32/BF16 trained weights
├── rl_agent_config.json      # Calibrated temperatures & training metadata
├── rl_agent_api.py           # Zero-dependency upstream inference engine
├── rl_common.py              # Decision model architecture & scoring rules
├── encoder/
│   └── config.json           # ModernBERT encoder configuration
├── tokenizer/
│   ├── tokenizer.json        # Fast tokenizer definition (50,368 tokens)
│   └── tokenizer_config.json # Tokenizer configuration
└── mlx/                      # Native Apple Silicon FP16 Deployment
    ├── model.safetensors     # 842 MB FP16 MLX weights
    ├── mlx_config.json       # MLX runtime configuration
    ├── rl_agent_config.json  # Calibrated temperatures
    ├── encoder/config.json   # Encoder config
    └── tokenizer/            # Fast tokenizer
```

---

## ⚖️ License & Attribution

This model is released under the **Apache 2.0 License**.
- Fine-tuned by [Ameer Hamza](https://github.com/ameerhmz).
- Base model: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).
- Architecture: ModernBERT by Answer.AI & LightOn.
