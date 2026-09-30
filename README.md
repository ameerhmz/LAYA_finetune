# 🚀 LAYA Flagship Model H100 Fine-Tuning (Lightning.ai)

This repository contains everything needed to fine-tune the flagship **LAYA Model** (`convaiinnovations/laya`, 421M ModernBERT-large) on an **NVIDIA H100 GPU** on **Lightning.ai**.

---

## 📊 Dataset Included (~47,400 Questions)
Located in `data/`:
* `data/train_huge.json`: **42,660 genuine decision questions** (38.9 MB)
* `data/test_huge.json`: **4,740 test & calibration questions** (4.3 MB)
* Sourced from genuine Hugging Face datasets (`SetFit/ag_news`, `dair-ai/emotion`, `google/boolq`, `LocalLLaMA/typed-decisions`, and desktop agent actions).

---

## ⚡ Quick Start on Lightning.ai Studio

### Step 1: Open Lightning.ai Terminal & Clone Repo
In your Lightning.ai H100 Studio terminal:
```bash
git clone https://github.com/ameerhmz/LAYA_finetune.git
cd LAYA_finetune
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Training with Custom Arguments
You can customize `--epochs` and `--batch-size` directly via CLI:
```bash
# Default: 10 epochs, batch size 64
bash run_h100.sh

# Or choose your own epochs and batch size:
bash run_h100.sh --epochs 10 --batch-size 64

# Or using short flags:
bash run_h100.sh -e 5 -b 128
```

You can also run `train_h100.py` directly with python:
```bash
python3 train_h100.py --epochs 10 --batch-size 64 --lr 3.0e-5
```

---

## 💾 Versioned Checkpoints Saved Every Epoch
After **every single epoch**, a complete, standalone model checkpoint is automatically saved in the root directory:
* `./checkpoint_epoch_1/`
* `./checkpoint_epoch_2/`
* ...
* `./checkpoint_epoch_10/`
* `./checkpoint_latest/` (always points to the most recent epoch)

Each checkpoint folder is completely self-contained and ready for deployment:
* `model.safetensors`: Fine-tuned PyTorch weights with calibrated temperature buffer
* `rl_agent_config.json`: Model configuration, fitted temperatures ($T_{\text{choice}}, T_{\text{score}}, T_{\text{noul}}$), epoch number, and validation accuracy
* `tokenizer/`: Vocabulary and tokenizer files

---

## ⏱️ Expected H100 Performance
* **GPU**: 1x NVIDIA H100 80GB SXM5
* **Throughput**: ~3,000–4,500 samples/sec with BFloat16 and FlashAttention-2 / SDPA
* **Training Time**: ~3–4 minutes per epoch (~30–40 minutes for 10 epochs over 42,660 samples)
* **VRAM**: ~14–18 GB (well within the 80GB VRAM headroom)
