from __future__ import annotations

import sys
from pathlib import Path

from ..config import HF_MODELS
from ..prompts import map_option_probs, pplx_row
from ..utils import Timer, append_jsonl, chunked, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("pplx")
MODEL_KEY = "pplx"


def load():
    from huggingface_hub import snapshot_download

    ckpt = Path(snapshot_download(HF_MODELS["pplx"]))
    sys.path.insert(0, str(ckpt / "source" / "src"))
    from autojev.model import DecisionModel, answer

    return DecisionModel(ckpt, device="cuda"), answer


def _to_list(p):
    try:
        return [float(x) for x in p.tolist()]
    except AttributeError:
        return [float(x) for x in p]


def to_probs(rec: dict, vec: list[float], lib) -> tuple[dict[str, float], dict]:
    meta = {"lib": str(lib)[:200]}
    if rec["qtype"] == "noul":
        if isinstance(lib, dict) and "noul" in lib:
            p_true = float(lib["noul"])
        elif isinstance(lib, (int, float)):
            p_true = float(lib)
        else:
            raise ValueError(f"noul: cannot read P(true) from answer() = {lib!r}")
        meta["vector"] = vec
        return {"yes": p_true, "no": 1.0 - p_true}, meta
    n = len(rec["options"])
    if len(vec) != n:
        raise ValueError(f"{len(vec)} probabilities for {n} options")
    if rec["qtype"] == "score":
        probs = map_option_probs(rec, [str(i) for i in range(n)], vec)
        if isinstance(lib, dict) and isinstance(lib.get("score"), (int, float)):
            meta["score_check"] = abs(sum(i * v for i, v in enumerate(vec)) - float(lib["score"]))
        return probs, meta
    probs = dict(zip(rec["options"], vec))
    lib_choice = lib.get("choice") if isinstance(lib, dict) else lib
    meta["lib_choice_mismatch"] = lib_choice is not None and str(lib_choice) != max(probs, key=probs.get)
    return probs, meta


def run_records(state, recs: list[dict], out, batch_size: int):
    model, answer_fn = state
    for chunk in chunked(recs, batch_size):
        rows_in = [pplx_row(r) for r in chunk]
        try:
            with Timer() as t:
                preds = model.predict(rows_in)
            per = t.seconds / len(chunk)
            rows = []
            for r, row_in, p in zip(chunk, rows_in, preds):
                probs, meta = to_probs(r, _to_list(p), answer_fn(row_in["question"], p))
                rows.append(pred_row(r, MODEL_KEY, probs, per, meta=meta))
        except Exception as e:
            if len(chunk) == 1:
                rows = [pred_row(chunk[0], MODEL_KEY, {}, 0.0, error=repr(e)[:300])]
            else:
                log.warning("batch failed (%s); retrying one by one", repr(e)[:160])
                run_records(state, chunk, out, 1)
                continue
        append_jsonl(out, rows)


def main():
    ap = base_parser("Run Perplexity Decider on records")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    state = load()
    for task, cond, recs, out in jobs(args, MODEL_KEY):
        run_records(state, recs, out, args.batch_size)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
