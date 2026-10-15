from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
RECORDS_DIR = DATA_DIR / "records"
RESULTS_ROOT = PROJECT_ROOT / "results"

SEED = 20261010

TASKS = ["esci_us", "esci_es", "esci_jp"]
LOCALE = {"esci_us": "us", "esci_es": "es", "esci_jp": "jp"}
N_QUERIES = {"esci_us": 1000, "esci_es": 200, "esci_jp": 200}
ROBUSTNESS_QUERIES = 50
PRODUCT_TEXT_CHARS = 1200
SUPERVISED_TRAIN_PAIRS = 50000

DESIGNS = ["choice", "score", "noul"]
ALL_CONDITIONS = ["choice", "score", "noul", "choice_reversed", "choice_shuffled", "repeat1", "repeat2"]

GAINS = {"exact": 1.0, "substitute": 0.1, "complement": 0.01, "irrelevant": 0.0}
ESCI_LABELS = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}

JEV = {
    "base_url": os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
    "model": os.environ.get("JEV_MODEL", "jev-1.13.0"),
    "api_key_env": "TYPESAFE_API_KEY",
    "usd_per_m_input": 0.042,
    "concurrency": int(os.environ.get("JEV_CONCURRENCY", "12")),
    "timeout_s": 60.0,
    "max_retries": 8,
}
PPLX_API = {
    "base_url": os.environ.get("PPLX_BASE_URL", "https://api.perplexity.ai/v1/decisions"),
    "model": os.environ.get("PPLX_MODEL", "pplx-decider-v1.1-27b"),
    "api_key_env": "PERPLEXITY_API_KEY",
    "usd_per_m_input": 0.02,
    "concurrency": int(os.environ.get("PPLX_CONCURRENCY", "8")),
    "timeout_s": 90.0,
    "max_retries": 8,
}
API_PROVIDERS = {"jev": JEV, "pplx_api": PPLX_API}

HF_MODELS = {
    "clef": "Cloudflare/clef",
    "clef_flash": "Cloudflare/clef-flash",
    "pplx": "perplexity-ai/pplx-decider-v1.1-27b",
    "qwen": "Qwen/Qwen3.8-27B-FP8",
    "gemma": "google/gemma-4-31B-it",
    "reranker": "Qwen/Qwen3-Reranker-4B",
    "modernbert": "answerdotai/ModernBERT-large",
    "mac_readout": "Qwen/Qwen3-0.6B",
    "mac_reranker": "Qwen/Qwen3-Reranker-0.6B",
}

READOUT_TOP_LOGPROBS = 64
READOUT_PRUNE = 1e-5
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = float(os.environ.get("VLLM_GPU_MEM", "0.90"))
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "reasoning_effort": "low"}

GPU_HOUR_USD = 2.50
TOST_MARGIN = 0.02
BOOTSTRAP_N = 10000


def results_dir(run_name: str = "main") -> Path:
    return RESULTS_ROOT / run_name


def raw_path(run_name: str, model: str, task: str, condition: str) -> Path:
    return results_dir(run_name) / "raw" / model / task / f"{condition}.jsonl"


def records_path(task: str, condition: str) -> Path:
    return RECORDS_DIR / task / f"{condition}.jsonl"


def conditions_for(task: str, model: str) -> list[str]:
    us = task == "esci_us"
    if model == "qwen_reranker":
        return ["noul"]
    if model == "modernbert":
        return ["choice"] if us else []
    if model in ("qwen_gen", "gemma_gen"):
        return ["choice"]
    if model == "mock":
        return ["choice", "score", "noul"]
    conds = ["choice", "score", "noul"] if us else ["choice"]
    if us and model != "pplx_api":
        conds += ["choice_reversed", "choice_shuffled"]
    if us and model in ("jev", "pplx_api"):
        conds += ["repeat1", "repeat2"]
    return conds
