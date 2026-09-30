# 🦙 Ollama Integration & GGUF Technical Guide for LAYA

This guide explains how to run the fine-tuned **LAYA** ModernBERT decision model (90.61% validation accuracy) with **Ollama**, **Open-WebUI**, **Cursor**, **LangChain**, and details the architectural considerations for **GGUF** export.

---

## ⚡ Quickstart: Ollama-Compatible Local Server Bridge

Since Ollama is primarily designed for autoregressive generative models (causal LLMs running next-token prediction via llama.cpp), while LAYA is a **bidirectional System-1 decision engine** with a custom multi-task scoring head, this repository provides a dedicated **Ollama REST API server bridge** (`serve_ollama.py`).

The bridge exposes the standard Ollama endpoints:
* `GET  /api/tags` (lists `laya:latest`)
* `GET  /api/version` (reports Ollama version)
* `POST /api/show` (inspects modelfile and parameters)
* `POST /api/generate` (handles decision requests with streaming and JSON support)
* `POST /api/chat` (handles conversational decision-making)

### 1. Launching the Server

```bash
# Default: auto-detects Apple Silicon MLX (~8ms latency) or CUDA/CPU PyTorch
python3 serve_ollama.py --port 11435

# Or explicitly choose backend:
python3 serve_ollama.py --port 11435 --backend mlx     # Apple Silicon
python3 serve_ollama.py --port 11435 --backend pytorch # PyTorch (CUDA / CPU)
```

> **Note on Port 11434 vs 11435**: If you already have native Ollama running on port 11434, `serve_ollama.py` automatically binds to port `11435` so both can run concurrently without collision. You can direct your client to port 11435, or stop native Ollama (`pkill ollama` / quit Ollama app) and run with `--port 11434`.

---

## 🔌 Connecting to Ollama Clients

### 1. cURL (Command Line)
```bash
# Natural language query with options
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "prompt": "Server memory is at 98%. Options: restart pod, scale up, ignore."
  }'
```

```bash
# Chat endpoint
curl -X POST http://localhost:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "messages": [
      {"role": "system", "content": "Database transaction deadlock occurred on shard-2."},
      {"role": "user", "content": "Should we page on-call immediately? Options: page_now, wait_auto_resolve"}
    ]
  }'
```

### 2. LangChain / Ollama Python SDK
```python
from langchain_community.llms import Ollama

# Point directly to the LAYA bridge
llm = Ollama(base_url="http://localhost:11435", model="laya")
response = llm.invoke("Issue: Disk full on worker node. Options: clean_cache, expand_volume, alert")
print(response)
```

### 3. Open-WebUI
In your Open-WebUI settings:
1. Navigate to **Admin Settings** -> **Connections** -> **Ollama API**.
2. Add `http://localhost:11435` (or `http://host.docker.internal:11435` if running in Docker).
3. Select `laya:latest` from the model selector dropdown.
4. Query LAYA with any decision state and options!

---

## 📦 Modelfile

An Ollama `Modelfile` is provided in the root directory:

```dockerfile
# Ollama Modelfile for LAYA Flagship ModernBERT Decision Model
FROM convaiinnovations/laya

SYSTEM """You are LAYA, a high-precision non-autoregressive System-1 decision engine based on ModernBERT-large (421M parameters). Given a state context (logs, text, JSON) and questions with candidate options, you evaluate exact mathematical probabilities in a single forward pass with zero generation hallucination."""

PARAMETER temperature 0.0
PARAMETER top_k 1
PARAMETER top_p 1.0
PARAMETER stop "[SEP]"
PARAMETER stop "</s>"

TEMPLATE """{{ if .System }}Context: {{ .System }}
{{ end }}{{ if .Prompt }}Decision Request: {{ .Prompt }}
{{ end }}Assessment: """
```

---

## 🔬 GGUF Technical Deep Dive

### Why GGUF Causal Generation Differs from LAYA

1. **Architecture Difference**:
   - Standard GGUF models (`llama.cpp`) are **autoregressive decoder-only** transformers (predicting token $t_{n+1}$ given $t_1 \dots t_n$).
   - LAYA is a **bidirectional encoder** (`ModernBERT-large`) coupled with a **2-layer Transformer decision scoring head**.

2. **Scoring Mechanism**:
   - LAYA constructs a sequence inserting `[MASK]` tokens at each candidate option.
   - The decision head computes cross-attention over all options and produces logits directly for each `[MASK]` position.
   - Logits are then scaled by **calibrated temperatures** ($T_{\text{choice}} = 1.90, T_{\text{score}} = 0.50, T_{\text{noul}} = 2.45$) fitted via **Strictly Proper Scoring Rules (RLCD)**.

3. **Converting Base Encoder to GGUF**:
   You can convert the underlying ModernBERT encoder to GGUF using `llama.cpp`'s `convert_hf_to_gguf.py` for BERT/embedding tasks:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   python3 llama.cpp/convert_hf_to_gguf.py ./encoder --outtype f16 --outfile laya_modernbert.gguf
   ```
   *Note: Standard `llama.cpp` will execute the ModernBERT embeddings, but will not invoke the custom RLCD decision head. The included `serve_ollama.py` bridge provides the full, loss-free, calibrated decision engine with native Ollama API compatibility.*
