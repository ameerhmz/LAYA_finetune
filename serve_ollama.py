#!/usr/bin/env python3
"""
LAYA Ollama REST API Server Bridge.

Exposes an Ollama-compatible HTTP REST API (/api/generate, /api/chat, /api/tags,
/api/version, /api/show) backed by the fine-tuned LAYA ModernBERT-large 421M
decision engine (90.61% validation accuracy).

Enables Open-WebUI, LangChain, Cursor, Ollama CLI, and custom agents to query
LAYA as an Ollama model on localhost with zero hallucinations and calibrated
confidence probabilities.
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Pydantic Schemas for Ollama REST API
# -----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    model: str = "laya"
    prompt: str = ""
    system: Optional[str] = None
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    context: Optional[List[int]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "laya"
    messages: List[ChatMessage] = []
    stream: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class ShowRequest(BaseModel):
    name: Optional[str] = "laya"
    model: Optional[str] = "laya"


# -----------------------------------------------------------------------------
# Backend Model Wrapper (Apple Silicon MLX or PyTorch)
# -----------------------------------------------------------------------------
class LayaBackend:
    def __init__(self, model_dir: str, backend: str = "auto", device: Optional[str] = None):
        self.model_dir = Path(model_dir).resolve()
        self.backend_type = backend
        self.device = device
        self.agent = None
        self._init_backend()

    def _init_backend(self):
        mlx_dir = self.model_dir / "mlx"
        can_use_mlx = (
            sys.platform == "darwin"
            and (mlx_dir / "model.safetensors").exists()
            and self.backend_type in ("auto", "mlx")
        )

        if can_use_mlx:
            try:
                import laya_mlx as laya
                print(f"🍏 [MLX Backend] Loading native Apple Silicon checkpoint from {mlx_dir}...")
                self.agent = laya.load(str(mlx_dir), dtype="float32")
                self.backend_type = "mlx"
                print("   ✓ MLX backend initialized successfully! (~8ms latency)")
                return
            except Exception as e:
                print(f"⚠️ Failed to initialize MLX backend: {e}. Falling back to PyTorch...")

        # PyTorch fallback
        print(f"🔥 [PyTorch Backend] Loading checkpoint from {self.model_dir}...")
        sys.path.insert(0, str(self.model_dir))
        try:
            from rl_agent_api import RLAgent
            dev = self.device or ("cuda" if self._has_cuda() else "cpu")
            self.agent = RLAgent(model_dir=str(self.model_dir), device=dev)
            self.backend_type = f"pytorch ({dev})"
            print(f"   ✓ PyTorch backend initialized on {dev}!")
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            raise

    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Run decision inference through loaded agent."""
        t0 = time.perf_counter()
        res = self.agent.system_one(state, questions)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        res["inference_time_ms"] = elapsed_ms
        return res


# -----------------------------------------------------------------------------
# Prompt Parser & Natural Language Heuristics
# -----------------------------------------------------------------------------
def parse_prompt_to_decision(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Parses an incoming prompt into (state, questions) for LAYA.
    Handles:
      1. Native JSON format: {"state": ..., "questions": ...}
      2. Structured JSON: {"question": ..., "options": [...], "state": ...}
      3. Markdown / Plain Text with explicit options or questions.
    """
    prompt_str = prompt.strip()
    state = system_prompt or "General context"

    # Case 1: Native JSON
    if prompt_str.startswith("{") and prompt_str.endswith("}"):
        try:
            data = json.loads(prompt_str)
            if "questions" in data and isinstance(data["questions"], dict):
                st = data.get("state", state)
                return st, data["questions"]
            if "question" in data:
                q_text = data["question"]
                opts = data.get("options") or data.get("choices")
                st = data.get("state", state)
                if opts:
                    if isinstance(opts, list):
                        criteria = {str(opt).strip(): str(opt).strip() for opt in opts}
                    elif isinstance(opts, dict):
                        criteria = opts
                    else:
                        criteria = {"option_1": str(opts)}
                    return st, {
                        "decision": {
                            "type": "choice",
                            "instructions": q_text,
                            "criteria": criteria,
                        }
                    }
                else:
                    return st, {
                        "verification": {
                            "type": "noul",
                            "instructions": q_text,
                            "criteria": {"false": "No / False", "true": "Yes / True"},
                        }
                    }
        except Exception:
            pass

    # Case 2: Natural Language Extraction
    # Look for options formatted like:
    # 1. Option A / 1) Option A / - Option A / Options: A, B, C
    options_dict = {}

    # Extract state if marked: "State: ... \n Question: ..."
    state_match = re.search(r"(?:State|Context):\s*(.*?)(?=\n(?:Question|Prompt|Options):|\Z)", prompt_str, re.DOTALL | re.IGNORECASE)
    if state_match:
        state = state_match.group(1).strip()
        prompt_str = prompt_str[state_match.end():].strip()

    # Look for bullet points or numbered lists
    opt_lines = re.findall(r"^(?:[-*•]|\d+[\.\)]|[A-Za-z][\.\)])\s+(.+)$", prompt_str, re.MULTILINE)
    if len(opt_lines) >= 2:
        for i, opt in enumerate(opt_lines):
            key = re.sub(r"[^a-zA-Z0-9_]+", "_", opt.split(":")[0].strip().lower())
            if not key or key in options_dict:
                key = f"option_{i+1}"
            options_dict[key] = opt.strip()

    # Look for inline "Options: A, B, C" or "Choices: A, B, C"
    if not options_dict:
        inline_match = re.search(r"(?:Options|Choices|Select from):\s*([^\n\.]+)", prompt_str, re.IGNORECASE)
        if inline_match:
            raw_opts = inline_match.group(1).split(",")
            if len(raw_opts) >= 2:
                for i, opt in enumerate(raw_opts):
                    clean = opt.strip()
                    key = re.sub(r"[^a-zA-Z0-9_]+", "_", clean.lower())
                    if not key or key in options_dict:
                        key = f"option_{i+1}"
                    options_dict[key] = clean

    # Determine question instructions
    instr = prompt_str
    if options_dict:
        # Strip the options section from the question instruction
        instr_candidate = re.split(r"(?:Options|Choices|\n[-*•]|\n\d+[\.\)])", prompt_str, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if instr_candidate:
            instr = instr_candidate
        return state, {
            "decision": {
                "type": "choice",
                "instructions": instr or "Select the best option.",
                "criteria": options_dict,
            }
        }

    # If no options found, treat as Boolean / truth question (noul)
    return state, {
        "statement_truth": {
            "type": "noul",
            "instructions": instr or "Evaluate the truthfulness of the statement.",
            "criteria": {
                "false": "No, the statement is false or unsupported.",
                "true": "Yes, the statement is true and supported.",
            },
        }
    }


def format_laya_response(result: Dict[str, Any], requested_format: Optional[str] = None) -> str:
    """Format LAYA decision output into readable Markdown or JSON."""
    answers = result.get("answers", {})

    if requested_format == "json":
        clean_out = {}
        for qid, ans in answers.items():
            if ans["type"] == "choice":
                clean_out[qid] = {
                    "type": "choice",
                    "choice": ans["choice"],
                    "confidence": ans["confidence"],
                    "probabilities": ans["probabilities"],
                    "action_prob": ans.get("action", {}).get("act_probability") or ans.get("rl_agent", {}).get("act_probability"),
                }
            elif ans["type"] == "noul":
                clean_out[qid] = {
                    "type": "noul",
                    "truth_score": ans["noul"],
                    "confidence": ans["confidence"],
                    "action_prob": ans.get("action", {}).get("act_probability") or ans.get("rl_agent", {}).get("act_probability"),
                }
            elif ans["type"] == "score":
                clean_out[qid] = {
                    "type": "score",
                    "score": ans["score"],
                    "confidence": ans["confidence"],
                    "probabilities": ans.get("probabilities", {}),
                }
        return json.dumps(clean_out, indent=2)

    # Markdown format
    lines = ["### 🎯 LAYA Decision Assessment\n"]
    for qid, ans in answers.items():
        qtype = ans["type"]
        act_prob = ans.get("action", {}).get("act_probability") or ans.get("rl_agent", {}).get("act_probability")

        if qtype == "choice":
            conf = ans.get("confidence", 0.0)
            choice = ans.get("choice", "")
            lines.append(f"**Recommended Decision**: `{choice}` (Confidence: {conf:.1%})\n")
            lines.append("**Calibrated Probabilities**:")
            for opt, p in ans.get("probabilities", {}).items():
                marker = "👉" if opt == choice else "  •"
                lines.append(f"{marker} `{opt}`: {p:.2%}")
        elif qtype == "noul":
            score = ans.get("noul", 0.0)
            status = "TRUE" if score >= 0.5 else "FALSE"
            lines.append(f"**Truth Assessment**: `{status}` (Truth Score: {score:.4f}, Confidence: {ans.get('confidence', 0.0):.1%})\n")
        elif qtype == "score":
            lines.append(f"**Ordinal Score**: `{ans.get('score', 0.0)}` (Confidence: {ans.get('confidence', 0.0):.1%})\n")

        if act_prob is not None:
            escalate = "🚨 Action Required" if act_prob > 0.5 else "✅ Normal Policy (No Escalation)"
            lines.append(f"\n**RL Safety Action Risk**: `{act_prob:.4f}` — {escalate}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# FastAPI Ollama Server Factory
# -----------------------------------------------------------------------------
def create_app(model_dir: str, backend: str = "auto", device: Optional[str] = None) -> FastAPI:
    backend_engine = LayaBackend(model_dir=model_dir, backend=backend, device=device)

    app = FastAPI(
        title="LAYA Ollama Compatibility Bridge",
        description="Ollama REST API server for LAYA ModernBERT-large 421M decision model",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    @app.get("/api/version")
    async def get_version():
        return {"version": "0.3.14"}

    @app.get("/api/tags")
    @app.get("/tags")
    async def get_tags():
        return {
            "models": [
                {
                    "name": "laya:latest",
                    "model": "laya:latest",
                    "modified_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "size": 842609157,
                    "digest": "sha256:laya-modernbert-decision-90pct",
                    "details": {
                        "parent_model": "convaiinnovations/laya",
                        "format": "safetensors",
                        "family": "modernbert",
                        "families": ["modernbert", "laya"],
                        "parameter_size": "421M",
                        "quantization_level": "F16",
                    },
                }
            ]
        }

    @app.post("/api/show")
    async def show_model(req: ShowRequest):
        return {
            "modelfile": (
                "FROM convaiinnovations/laya\n"
                "PARAMETER temperature 0.0\n"
                "SYSTEM You are LAYA, a non-autoregressive calibrated System-1 decision engine.\n"
            ),
            "parameters": "temperature 0.0\nstop [SEP]",
            "template": "{{ .System }}\n{{ .Prompt }}",
            "details": {
                "parent_model": "convaiinnovations/laya",
                "format": "safetensors",
                "family": "modernbert",
                "families": ["modernbert", "laya"],
                "parameter_size": "421M",
                "quantization_level": "F16",
            },
            "model_info": {
                "general.architecture": "modernbert",
                "general.name": "laya-modernbert-decision-90pct",
                "general.parameter_count": 421000000,
                "validation_accuracy": 90.61,
            },
        }

    @app.post("/api/generate")
    async def generate(req: GenerateRequest):
        t0 = time.perf_counter()
        state, questions = parse_prompt_to_decision(req.prompt, system_prompt=req.system)
        result = backend_engine.predict(state, questions)
        text_response = format_laya_response(result, requested_format=req.format)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        n_tokens = result.get("usage", {}).get("input_tokens", 42)

        if not req.stream:
            return {
                "model": req.model,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "response": text_response,
                "done": True,
                "done_reason": "stop",
                "context": [50281, 101, 50282],
                "total_duration": elapsed_ns,
                "load_duration": 0,
                "prompt_eval_count": n_tokens,
                "prompt_eval_duration": elapsed_ns,
                "eval_count": 0,
                "eval_duration": 0,
            }

        # Streaming response (NDJSON)
        async def stream_generator():
            chunks = text_response.split("\n")
            for chunk in chunks:
                msg = {
                    "model": req.model,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "response": chunk + "\n",
                    "done": False,
                }
                yield json.dumps(msg) + "\n"
                await asyncio.sleep(0.005)
            # Final chunk
            final_msg = {
                "model": req.model,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "response": "",
                "done": True,
                "done_reason": "stop",
                "total_duration": elapsed_ns,
                "prompt_eval_count": n_tokens,
            }
            yield json.dumps(final_msg) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        t0 = time.perf_counter()
        system_content = ""
        user_content = ""
        for m in req.messages:
            if m.role == "system":
                system_content += m.content + "\n"
            elif m.role == "user":
                user_content = m.content

        state, questions = parse_prompt_to_decision(user_content, system_prompt=system_content.strip() or None)
        result = backend_engine.predict(state, questions)
        text_response = format_laya_response(result, requested_format=req.format)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        n_tokens = result.get("usage", {}).get("input_tokens", 42)

        if not req.stream:
            return {
                "model": req.model,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": text_response,
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": elapsed_ns,
                "prompt_eval_count": n_tokens,
                "eval_count": 0,
            }

        # Streaming chat
        async def stream_chat_generator():
            chunks = text_response.split("\n")
            for chunk in chunks:
                msg = {
                    "model": req.model,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": chunk + "\n",
                    },
                    "done": False,
                }
                yield json.dumps(msg) + "\n"
                await asyncio.sleep(0.005)
            final_msg = {
                "model": req.model,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": elapsed_ns,
                "prompt_eval_count": n_tokens,
            }
            yield json.dumps(final_msg) + "\n"

        return StreamingResponse(stream_chat_generator(), media_type="application/x-ndjson")

    return app


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def check_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Launch LAYA Ollama-compatible local REST server"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/Users/ameerhamza/HOBBY_CODING/LAYA/models/laya_finetuned_h200",
        help="Path to fine-tuned LAYA checkpoint directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Port to bind server to (default: 11435; port 11434 is standard Ollama)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "mlx", "pytorch"],
        default="auto",
        help="Inference engine backend (auto selects MLX on Apple Silicon)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device: cuda, cpu, or mps (ignored if using MLX)",
    )

    args = parser.parse_args()

    port = args.port
    if port == 11434 and check_port_in_use(args.host, 11434):
        print("⚠️ Warning: Port 11434 is already in use by another Ollama instance!")
        print("   Switching to port 11435. You can set OLLAMA_HOST=127.0.0.1:11435 for your clients.")
        port = 11435

    print("=" * 75)
    print("🦙 LAYA Ollama Compatibility REST Server")
    print(f"   Model Directory : {args.model_dir}")
    print(f"   Serving on      : http://{args.host}:{port}")
    print(f"   Backend         : {args.backend}")
    print("=" * 75)

    app = create_app(
        model_dir=args.model_dir,
        backend=args.backend,
        device=args.device,
    )

    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
