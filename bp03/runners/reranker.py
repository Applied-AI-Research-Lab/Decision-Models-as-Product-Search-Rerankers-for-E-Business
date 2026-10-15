from __future__ import annotations

import math

from ..config import HF_MODELS, SEED, VLLM_GPU_MEM, VLLM_MAX_MODEL_LEN
from ..utils import Timer, append_jsonl, chunked, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("reranker")
MODEL_KEY = "qwen_reranker"
INSTRUCTION = "Given an e-commerce shopping query, judge whether the product satisfies the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompt(rec: dict) -> str:
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {rec['query']}\n<Document>: {rec['text']}{SUFFIX}"


def p_yes(lp_yes: float | None, lp_no: float | None) -> float:
    a = -1e9 if lp_yes is None else lp_yes
    b = -1e9 if lp_no is None else lp_no
    m = max(a, b)
    return math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))


def main():
    ap = base_parser("Run Qwen3-Reranker on records")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    args = ap.parse_args()
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        llm = LLM(model=HF_MODELS["reranker"], max_model_len=VLLM_MAX_MODEL_LEN, gpu_memory_utilization=VLLM_GPU_MEM,
                  enable_prefix_caching=True, max_logprobs=20, seed=SEED, logprobs_mode="processed_logprobs")
        tok = llm.get_tokenizer()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(HF_MODELS["mac_reranker"])
        model = AutoModelForCausalLM.from_pretrained(HF_MODELS["mac_reranker"]).to(device).eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    if args.backend == "vllm":
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, allowed_token_ids=[yes_id, no_id])
    for task, cond, recs, out in jobs(args, MODEL_KEY):
        for chunk in chunked(recs, 512 if args.backend == "vllm" else 8):
            ids = [tok.encode(prompt(r), add_special_tokens=False) for r in chunk]
            with Timer() as t:
                if args.backend == "vllm":
                    outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp, use_tqdm=False)
                    lps = [{k: v.logprob for k, v in o.outputs[0].logprobs[0].items()} for o in outs]
                else:
                    lps = []
                    for x in ids:
                        with torch.inference_mode():
                            lg = torch.log_softmax(model(torch.tensor([x], device=device)).logits[0, -1].float(), -1)
                        lps.append({yes_id: lg[yes_id].item(), no_id: lg[no_id].item()})
            per = t.seconds / len(chunk)
            rows = []
            for r, lp, x in zip(chunk, lps, ids):
                if yes_id not in lp or no_id not in lp:
                    raise RuntimeError(f"{r['id']}: yes/no logprobs missing: {lp}")
                p = p_yes(lp[yes_id], lp[no_id])
                rows.append(pred_row(r, MODEL_KEY, {"yes": p, "no": 1.0 - p}, per, len(x)))
            append_jsonl(out, rows)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
