#!/usr/bin/env python3
"""
High-Performance NVIDIA H100 Training & Fine-Tuning Engine for LAYA Flagship Model.

Target Model: convaiinnovations/laya (ModernBERT-large 421M Decision Engine)
Target Hardware: NVIDIA H100 80GB SXM5 (Hopper Architecture on Lightning.ai)

Features:
- Native PyTorch with CUDA & FlashAttention-2 / SDPA
- BFloat16 Mixed Precision & TensorFloat-32 (TF32) for H100 Tensor Cores
- Dynamic padding high-throughput DataLoader with pinned memory
- Combined Cross-Entropy + RLCD (Ranked Loss for Calibrated Decisions) proper scoring loss
- Post-training per-question-type temperature calibration for perfect confidence calibration
- Directly outputs standard Hugging Face / safetensors checkpoint
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


class DecisionModel(nn.Module):
    """
    Official Upstream LAYA DecisionModel architecture.
    Pretrained bidirectional encoder + 2-layer decision transformer head + calibrated scoring.
    """
    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Bidirectional Encoder
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2. Add Question Type Embedding
        h = h + self.type_emb(qtype)[:, None, :]

        # 3. Decision Transformer Head
        if self.head is not None:
            pad = ~attention_mask.bool()
            h = self.head(h, src_key_padding_mask=pad)

        # 4. Gather option markers
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)

        # 5. Score option markers
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        # 6. Action head (entropy, top-2 gap, marker count)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))

        return logits, act_logits


def render_criterion(c: Any) -> str:
    if isinstance(c, str):
        return c
    return json.dumps(c, ensure_ascii=False)


def render_options(q: Dict) -> List[str]:
    t, crit = q["type"], q.get("criteria", {})
    if t == "choice":
        return [
            k if v is None or v == "" else f"{k}: {render_criterion(v)}"
            for k, v in crit.items()
        ]
    if t == "score":
        if isinstance(crit, list):
            return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
        return [f"level {k}: {render_criterion(v)}" for k, v in crit.items()]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit else "no, statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit else "yes, statement holds"),
    ]


def build_sequence(
    tok: Any,
    state: Any,
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
) -> Tuple[List[int], List[int]]:
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q.get("instructions", "")).replace(mask_tok, " ")
    head_text = f"{q['type']} question: {ins}"
    head_ids = tok(head_text, add_special_tokens=False)["input_ids"]

    opt_ids = []
    for opt in opts:
        opt_text = " " + opt.replace(mask_tok, " ")
        encoded_opt = tok(opt_text, add_special_tokens=False)["input_ids"][:48]
        opt_ids.append([tok.mask_token_id] + encoded_opt)

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]

    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)

    # State truncation
    room = max(0, max_len - len(ids) - 1)
    state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    state_ids = tok(state_str.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = ids + state_ids + [tok.sep_token_id]

    return ids[:max_len], [m for m in markers if m < max_len]


class DecisionDataset(Dataset):
    def __init__(self, data_path: str, tok: Any, max_len: int = 512, head_max_len: int = 192):
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.tok = tok
        self.max_len = max_len
        self.head_max_len = head_max_len

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.raw_data[idx]
        state = item["state"]
        q = item["question"]
        target = item["target"]
        label = item.get("label", int(np.argmax(target)))
        qtype_str = q.get("type", "choice")
        qtype_id = QTYPES.get(qtype_str, 0)

        ids, markers = build_sequence(self.tok, state, q, self.max_len, self.head_max_len)
        return {
            "ids": ids,
            "markers": markers,
            "qtype": qtype_id,
            "target": target,
            "label": label,
        }


def collate_fn(batch: List[Dict[str, Any]], pad_id: int, pad_multiple: int = 8) -> Dict[str, torch.Tensor]:
    n = len(batch)
    max_len = max(len(b["ids"]) for b in batch)
    if pad_multiple > 1:
        max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple

    max_markers = max(2, max(len(b["markers"]) for b in batch))

    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.bool)
    marker_pos = torch.zeros((n, max_markers), dtype=torch.long)
    marker_mask = torch.zeros((n, max_markers), dtype=torch.bool)
    targets = torch.zeros((n, max_markers), dtype=torch.float32)
    labels = torch.zeros((n,), dtype=torch.long)
    qtypes = torch.zeros((n,), dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = len(b["ids"])
        n_m = len(b["markers"])
        input_ids[i, :seq_len] = torch.tensor(b["ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = True
        marker_pos[i, :n_m] = torch.tensor(b["markers"], dtype=torch.long)
        marker_mask[i, :n_m] = True

        t_vec = b["target"][:n_m]
        s_t = sum(t_vec)
        if s_t > 0:
            targets[i, :len(t_vec)] = torch.tensor([v / s_t for v in t_vec], dtype=torch.float32)
        else:
            targets[i, :len(t_vec)] = 1.0 / len(t_vec)

        labels[i] = min(b["label"], n_m - 1)
        qtypes[i] = b["qtype"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "targets": targets,
        "labels": labels,
        "qtypes": qtypes,
    }


def compute_calibrated_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    marker_mask: torch.Tensor,
    action_logits: torch.Tensor,
    w_sph: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combines Cross-Entropy with Spherical calibration loss for honest probability estimation."""
    masked_logits = torch.where(marker_mask, logits, torch.tensor(-1e4, device=logits.device))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = F.softmax(masked_logits, dim=-1)

    # Cross-Entropy / KL Divergence against target distribution
    ce_loss = -(targets * torch.where(marker_mask, log_probs, torch.zeros_like(log_probs))).sum(dim=-1).mean()

    # Spherical scoring rule loss (RLCD proper scoring)
    norm_p = torch.norm(probs, p=2, dim=-1).clamp(min=1e-6)
    dot_pt = (probs * targets).sum(dim=-1)
    sph_loss = (1.0 - (dot_pt / norm_p)).mean()

    decision_loss = ce_loss + w_sph * sph_loss

    # Action loss (autonomous confidence vs escalation)
    top_p = probs.max(dim=-1)[0]
    act_target = torch.where(top_p >= 0.70, 0, 1)
    act_loss = F.cross_entropy(action_logits, act_target)

    total_loss = decision_loss + 0.25 * act_loss
    return total_loss, decision_loss, act_loss


def calibrate_temperatures(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Fits post-training temperature scaling parameters on validation logits."""
    model.eval()
    qtype_logits = {0: [], 1: [], 2: []}
    qtype_labels = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            m_pos = batch["marker_pos"].to(device)
            m_mask = batch["marker_mask"].to(device)
            qt = batch["qtypes"].to(device)
            lbl = batch["labels"].to(device)

            logits, _ = model(ids, mask, m_pos, m_mask, qt)
            for i in range(len(qt)):
                k = int(m_mask[i].sum().item())
                q_idx = int(qt[i].item())
                qtype_logits[q_idx].append(logits[i, :k].cpu())
                qtype_labels[q_idx].append(lbl[i].cpu().item())

    temperatures = {}
    for q_idx, name in QTYPE_NAMES.items():
        if not qtype_logits[q_idx]:
            temperatures[name] = 1.0
            continue

        all_logits = qtype_logits[q_idx]
        all_labels = qtype_labels[q_idx]

        best_t, best_nll = 1.0, float("inf")
        for t_candidate in np.linspace(0.5, 3.0, 51):
            nll_sum = 0.0
            for z, y in zip(all_logits, all_labels):
                scaled = z / t_candidate
                lp = F.log_softmax(scaled, dim=-1)
                nll_sum -= lp[y].item()
            avg_nll = nll_sum / len(all_labels)
            if avg_nll < best_nll:
                best_nll = avg_nll
                best_t = float(t_candidate)
        temperatures[name] = round(best_t, 3)

    return temperatures


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LAYA Flagship Model on NVIDIA H100 (Lightning.ai)")
    parser.add_argument("--train-data", type=str, default="./data/train_huge.json", help="Path to training JSON")
    parser.add_argument("--test-data", type=str, default="./data/test_huge.json", help="Path to test JSON")
    parser.add_argument("--checkpoint", type=str, default="convaiinnovations/laya", help="Base model (convaiinnovations/laya)")
    parser.add_argument("--output-dir", type=str, default="./laya_finetuned_h100", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (64 or 128 for 80GB H100)")
    parser.add_argument("--lr", type=float, default=3.0e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    rank, world_size, local_rank, is_dist = setup_distributed()
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if is_main:
        print("=" * 80)
        print("🚀 LAYA Flagship Decision Model — NVIDIA H100 Training Engine")
        print(f"   Base Checkpoint    : {args.checkpoint} (Flagship ModernBERT-large 421M)")
        print(f"   Hardware           : {world_size}x GPU(s) ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"   Batch Size / GPU   : {args.batch_size} (Effective batch: {args.batch_size * world_size * args.grad_accum})")
        print(f"   Precision          : Native BFloat16 + TensorFloat-32 (TF32)")
        print(f"   Train Dataset      : {args.train_data}")
        print(f"   Test Dataset       : {args.test_data}")
        print(f"   Output Directory   : {args.output_dir}")
        print("=" * 80)

    # 1. Load Tokenizer & Config
    if is_main:
        print("📥 Loading tokenizer and base config from Hugging Face...")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, subfolder="tokenizer")

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if os.path.exists(os.path.join(args.checkpoint, "rl_agent_config.json")):
        with open(os.path.join(args.checkpoint, "rl_agent_config.json")) as f:
            cfg = json.load(f)
    else:
        cfg_path = hf_hub_download(args.checkpoint, "rl_agent_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

    # 2. Build Datasets & Loaders
    train_dataset = DecisionDataset(args.train_data, tok, max_len=args.max_len)
    test_dataset = DecisionDataset(args.test_data, tok, max_len=args.max_len)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_dist else None

    collate = lambda b: collate_fn(b, pad_id=tok.pad_token_id, pad_multiple=8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=8 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(torch.cuda.is_available()),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Build Model & Load Pretrained Weights
    if is_main:
        print(f"🧠 Building ModernBERT encoder ({cfg['encoder']}) with FlashAttention-2 / SDPA...")
    encoder = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    model = DecisionModel(
        encoder=encoder,
        head_layers=cfg.get("head_layers", 2),
        n_act=len(cfg.get("act_costs", {})) + 1,
    )

    if is_main:
        print(f"📦 Loading pretrained checkpoint weights from {args.checkpoint}...")
    if os.path.exists(os.path.join(args.checkpoint, "model.safetensors")):
        weight_file = os.path.join(args.checkpoint, "model.safetensors")
    else:
        weight_file = hf_hub_download(args.checkpoint, "model.safetensors")

    weights = load_file(weight_file)
    model.load_state_dict(weights, strict=False)
    model = model.to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if is_dist else model

    # 4. Optimizer & Schedule
    no_decay = ["bias", "LayerNorm.weight", "norm.weight", "final_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in raw_model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in raw_model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-6)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Training Loop
    if is_main:
        print(f"\n🏋️ Training {len(train_dataset):,} samples ({total_steps:,} steps across {args.epochs} epochs)")
        print("-" * 80)

    start_time = time.time()
    step_count = 0

    for epoch in range(1, args.epochs + 1):
        if is_dist:
            train_sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=95, disable=not is_main)
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            m_pos = batch["marker_pos"].to(device, non_blocking=True)
            m_mask = batch["marker_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            qtypes = batch["qtypes"].to(device, non_blocking=True)

            use_amp = torch.cuda.is_available()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                logits, act = model(ids, mask, m_pos, m_mask, qtypes)
                loss, dec_loss, act_loss = compute_calibrated_loss(
                    logits, targets, labels, m_mask, act
                )
                loss = loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            if is_main:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss.item() * args.grad_accum:.4f}",
                    "dec": f"{dec_loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                })

        # Epoch Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                m_pos = batch["marker_pos"].to(device)
                m_mask = batch["marker_mask"].to(device)
                labels = batch["labels"].to(device)
                qtypes = batch["qtypes"].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    logits, _ = model(ids, mask, m_pos, m_mask, qtypes)

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        if is_dist:
            metrics = torch.tensor([val_correct, val_total], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_correct, val_total = int(metrics[0].item()), int(metrics[1].item())

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        if is_main:
            print(f"📊 [Epoch {epoch}/{args.epochs}] Validation Accuracy: {val_acc:.2f}% ({val_correct:,}/{val_total:,})")

            # Temperature calibration for this epoch
            calib_temps = calibrate_temperatures(raw_model, test_loader, device)
            raw_model.temperature.data = torch.tensor(
                [calib_temps["choice"], calib_temps["score"], calib_temps["noul"]],
                dtype=torch.float32,
            )

            # Per-epoch checkpoint directory directly in root
            epoch_dir = Path(f"./checkpoint_epoch_{epoch}")
            latest_dir = Path("./checkpoint_latest")
            for save_dir in [epoch_dir, latest_dir]:
                save_dir.mkdir(parents=True, exist_ok=True)
                state_dict = raw_model.state_dict()
                safetensors.torch.save_file(state_dict, str(save_dir / "model.safetensors"))

                agent_cfg = {
                    "encoder": cfg["encoder"],
                    "head_layers": cfg.get("head_layers", 2),
                    "max_len": args.max_len,
                    "head_max_len": 192,
                    "max_prefixes": 6,
                    "act_costs": cfg.get("act_costs", {"escalate": 0.5}),
                    "cost_wrong_act": 3.0,
                    "amp_dtype": "bf16",
                    "model_name": f"laya-flagship-h100-epoch-{epoch}",
                    "temperature": [calib_temps["choice"], calib_temps["score"], calib_temps["noul"]],
                    "temperature_by_options": {},
                    "training": {
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "updates": step_count,
                        "duration_seconds": round(time.time() - start_time, 2),
                        "validation_acc": round(val_acc, 2),
                        "fine_tuned_from_checkpoint": args.checkpoint,
                    },
                }
                with open(save_dir / "rl_agent_config.json", "w") as f:
                    json.dump(agent_cfg, f, indent=2)

                tok.save_pretrained(str(save_dir / "tokenizer"))

            print(f"💾 [Epoch {epoch}/{args.epochs}] Checkpoint saved to: {epoch_dir.resolve()} and {latest_dir.resolve()}\n")

    if is_main:
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"🎉 All {args.epochs} Epochs Completed in {total_time / 60:.1f} minutes!")
        print(f"   All {args.epochs} checkpoints saved in: ./checkpoint_epoch_1 through ./checkpoint_epoch_{args.epochs}")
        print(f"   Latest checkpoint: ./checkpoint_latest/")
        print("=" * 80)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
