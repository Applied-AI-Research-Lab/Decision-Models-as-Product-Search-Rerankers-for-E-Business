from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .config import (ALL_CONDITIONS, API_PROVIDERS, BOOTSTRAP_N, GAINS, GPU_HOUR_USD, SEED, TASKS, TASKS_DIR,
                     records_path, results_dir)
from .records import SCORE_ORDER
from .stats import holm
from .utils import get_logger, iter_jsonl, latest_ok, read_jsonl

log = get_logger("analyze")
API_MODELS = set(API_PROVIDERS)
DECISION_MODELS = ["jev", "clef", "clef_flash", "pplx", "pplx_api"]
DESIGNS = ["choice", "score", "noul"]


def discover(run):
    raw = results_dir(run) / "raw"
    return sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.exists() else []


def load(run, model, task, cond):
    rp, pp = records_path(task, cond), results_dir(run) / "raw" / model / task / f"{cond}.jsonl"
    if not rp.exists() or not pp.exists():
        return None
    recs, preds = read_jsonl(rp), latest_ok(pp)
    n_err = sum(1 for r in iter_jsonl(pp) if r.get("error") and r["id"] not in preds)
    recs = [r for r in recs if r["id"] in preds]
    if not recs:
        return None
    opts = recs[0]["options"]
    P = np.array([[preds[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs], float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    y = np.array([opts.index(r["gold"]) for r in recs])
    return {"recs": recs, "preds": preds, "opts": opts, "P": P, "y": y, "n_err": n_err, "ids": [r["id"] for r in recs]}


def pair_scores(d, design) -> dict[str, float]:
    opts, P = d["opts"], d["P"]
    if design.startswith("choice") or design.startswith("repeat"):
        w = np.array([GAINS[o] for o in opts])
    elif design == "score":
        w = np.array([SCORE_ORDER.index(o) for o in opts], float)
    else:
        w = np.array([1.0 if o == "yes" else 0.0 for o in opts])
    return dict(zip(d["ids"], P @ w))


def ndcg_at(gains_in_rank_order: list[float], k: int = 10) -> float | None:
    ideal = sorted(gains_in_rank_order, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    if idcg == 0:
        return None
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains_in_rank_order[:k])) / idcg


def per_query(scores: dict[str, float], pairs: list[dict]) -> dict[int, dict]:
    by_q = defaultdict(list)
    for p in pairs:
        if p["id"] in scores:
            by_q[p["query_id"]].append(p)
    out = {}
    for q, items in by_q.items():
        ranked = sorted(items, key=lambda p: (-scores[p["id"]], p["id"]))
        gains = [GAINS[p["label"]] for p in ranked]
        nd = ndcg_at(gains)
        labels = [p["label"] for p in ranked]
        mrr = next((1.0 / (i + 1) for i, l in enumerate(labels) if l == "exact"), None)
        out[q] = {"ndcg": nd, "mrr": mrr, "hard": ("exact" in labels and "irrelevant" in labels)}
    return out


def bm25_scores(pairs: list[dict], k1=1.5, b=0.75) -> dict[str, float]:
    tokenize = lambda s: re.findall(r"\w+", s.lower())
    docs = {p["id"]: tokenize(p["product_text"]) for p in pairs}
    N = len(docs)
    avgdl = np.mean([len(d) for d in docs.values()]) or 1.0
    df = Counter(t for d in docs.values() for t in set(d))
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    out = {}
    for p in pairs:
        tf = Counter(docs[p["id"]])
        dl = len(docs[p["id"]])
        out[p["id"]] = sum(idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
                           for t in set(tokenize(p["query"])) if t in tf)
    return out


def summarize(pq: dict[int, dict]) -> dict:
    nd = [v["ndcg"] for v in pq.values() if v["ndcg"] is not None]
    mr = [v["mrr"] for v in pq.values() if v["mrr"] is not None]
    ndh = [v["ndcg"] for v in pq.values() if v["ndcg"] is not None and v["hard"]]
    return {"queries": len(nd), "ndcg10": float(np.mean(nd)) if nd else np.nan,
            "mrr_exact": float(np.mean(mr)) if mr else np.nan,
            "hard_queries": len(ndh), "ndcg10_hard": float(np.mean(ndh)) if ndh else np.nan}


def paired_query_bootstrap(a: dict, b: dict, n_boot: int, hard_only=False):
    qs = [q for q in a if q in b and a[q]["ndcg"] is not None and b[q]["ndcg"] is not None
          and (a[q]["hard"] or not hard_only)]
    if not qs:
        return np.nan, np.nan, np.nan, 0, np.nan
    d = np.array([a[q]["ndcg"] - b[q]["ndcg"] for q in qs])
    rng = np.random.default_rng(SEED)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = max(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)), 1.0 / n_boot)
    return float(d.mean()), float(lo), float(hi), len(qs), float(min(p, 1.0))


def add_holm(rows: list[dict], family: list[str]) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in family)].append(r)
    for g in groups.values():
        ok = [r for r in g if not np.isnan(r["p"])]
        for r, ph in zip(ok, holm([r["p"] for r in ok])):
            r["p_holm"] = ph


def per_pair_cost(model, preds) -> float:
    rows = list(preds.values())
    if model in API_MODELS:
        return float(np.mean([r["input_tokens"] or 0 for r in rows])) * API_PROVIDERS[model]["usd_per_m_input"] / 1e6
    return float(np.mean([r["latency_s"] or 0 for r in rows])) * GPU_HOUR_USD / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    args = ap.parse_args()
    run = args.run_name
    out_t, out_f = results_dir(run) / "tables", results_dir(run) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    models = discover(run)
    log.info("models found: %s", models)

    cache, cov = {}, []
    for m in models:
        for t in TASKS:
            for c in ALL_CONDITIONS:
                d = load(run, m, t, c)
                if d is not None:
                    cache[(m, t, c)] = d
                    cov.append({"model": m, "task": t, "condition": c, "records": len(read_jsonl(records_path(t, c))),
                                "answered": len(d["recs"]), "unresolved_errors": d["n_err"]})

    pairs = {t: read_jsonl(TASKS_DIR / t / "test.jsonl") for t in TASKS if (TASKS_DIR / t / "test.jsonl").exists()}
    pq_all, rank_rows = {}, []
    for t, prs in pairs.items():
        rng = np.random.default_rng(SEED)
        rand = [per_query({p["id"]: rng.random() for p in prs}, prs) for _ in range(20)]
        rand_pq = {q: {"ndcg": np.mean([r[q]["ndcg"] for r in rand]) if rand[0][q]["ndcg"] is not None else None,
                       "mrr": np.mean([r[q]["mrr"] for r in rand]) if rand[0][q]["mrr"] is not None else None,
                       "hard": rand[0][q]["hard"]} for q in rand[0]}
        pq_all[("random", t, "-")] = rand_pq
        pq_all[("bm25", t, "-")] = per_query(bm25_scores(prs), prs)
    for (m, t, c), d in cache.items():
        if c in DESIGNS and t in pairs:
            pq_all[(m, t, c)] = per_query(pair_scores(d, c), pairs[t])
    for (m, t, c), pq in pq_all.items():
        rank_rows.append({"model": m, "task": t, "design": c, **summarize(pq)})

    tests = []
    for (m, t, c), pq in pq_all.items():
        for ref in [("bm25", t, "-"), ("qwen_reranker", t, "noul"), ("jev", t, "choice")]:
            if ref == (m, t, c) or ref not in pq_all:
                continue
            for hard in (False, True):
                diff, lo, hi, n, p = paired_query_bootstrap(pq, pq_all[ref], args.n_boot, hard)
                tests.append({"model": m, "task": t, "design": c, "reference": f"{ref[0]}:{ref[2]}",
                              "subset": "hard" if hard else "all", "n_queries": n, "ndcg_diff": diff, "lo": lo, "hi": hi,
                              "p": p})
    add_holm(tests, ["task", "reference", "subset"])

    designs = []
    for m in DECISION_MODELS + ["qwen_readout", "gemma_readout"]:
        for t in TASKS:
            for a, b in [("choice", "score"), ("choice", "noul"), ("score", "noul")]:
                if (m, t, a) in pq_all and (m, t, b) in pq_all:
                    diff, lo, hi, n, p = paired_query_bootstrap(pq_all[(m, t, a)], pq_all[(m, t, b)], args.n_boot)
                    designs.append({"model": m, "task": t, "comparison": f"{a} - {b}", "n_queries": n,
                                    "ndcg_diff": diff, "lo": lo, "hi": hi, "p": p})
    add_holm(designs, ["task"])

    cls, conf_rows, eff = [], [], []
    for (m, t, c), d in cache.items():
        if c not in DESIGNS:
            continue
        cls.append({"model": m, "task": t, "design": c, **M.summary(d["P"], d["y"])})
        if c == "choice":
            pred = d["P"].argmax(1)
            for k, o in enumerate(d["opts"]):
                mask = d["y"] == k
                if mask.any():
                    conf_rows.append({"model": m, "task": t, "label": o, "n": int(mask.sum()),
                                      "recall": float(np.mean(pred[mask] == k))})
        n_pairs_per_q = np.mean(list(Counter(r["query_id"] for r in d["recs"]).values()))
        lat = np.array([p["latency_s"] or 0 for p in d["preds"].values()])
        eff.append({"model": m, "task": t, "design": c, "latency_per_pair_s": lat.mean(),
                    "latency_per_query_s_sequential": lat.mean() * n_pairs_per_q,
                    "input_tokens_per_pair": np.mean([p["input_tokens"] or 0 for p in d["preds"].values()]),
                    "usd_per_1k_queries": per_pair_cost(m, d["preds"]) * n_pairs_per_q * 1000})

    consistency = []
    from scipy.stats import spearmanr
    for m in models:
        for t in TASKS:
            a, b = cache.get((m, t, "choice")), cache.get((m, t, "noul"))
            if a is None or b is None:
                continue
            ids = sorted(set(a["ids"]) & set(b["ids"]))
            pa = [a["preds"][i]["probs"].get("exact", 0) for i in ids]
            pb = [b["preds"][i]["probs"].get("yes", 0) for i in ids]
            rho = spearmanr(pa, pb).correlation if len(ids) > 2 else np.nan
            consistency.append({"check": "corr P(exact|choice) vs P(yes|noul) (must be > 0)", "model": m, "task": t,
                                "n": len(ids), "value": rho})
    for t in TASKS:
        for c in DESIGNS:
            a, b = cache.get(("pplx_api", t, c)), cache.get(("pplx", t, c))
            if a is None or b is None:
                continue
            ids = sorted(set(a["ids"]) & set(b["ids"]))
            Pa = np.array([[a["preds"][i]["probs"].get(o, 0) for o in b["opts"]] for i in ids])
            Pb = np.array([[b["preds"][i]["probs"].get(o, 0) for o in b["opts"]] for i in ids])
            consistency.append({"check": f"hosted vs local Perplexity: decision agreement ({c})", "model": "pplx",
                                "task": t, "n": len(ids), "value": 1 - M.flip_rate(Pa.argmax(1), Pb.argmax(1))})

    rob = []
    for (m, t, c), d in cache.items():
        base = cache.get((m, t, "choice"))
        if c not in ("choice_reversed", "choice_shuffled", "repeat1", "repeat2") or base is None:
            continue
        ids = sorted(set(d["ids"]) & set(base["ids"]))
        Pa = np.array([[base["preds"][i]["probs"].get(o, 0) for o in base["opts"]] for i in ids])
        Pb = np.array([[d["preds"][i]["probs"].get(o, 0) for o in base["opts"]] for i in ids])
        rob.append({"model": m, "task": t, "condition": c, "n": len(ids),
                    "flip_rate": M.flip_rate(Pa.argmax(1), Pb.argmax(1)), "mean_tv": M.total_variation(Pa, Pb)})

    tables = {"coverage": cov, "ranking": rank_rows, "ranking_tests": tests, "designs": designs,
              "classification": cls, "confusion": conf_rows, "consistency": consistency, "robustness": rob,
              "efficiency": eff}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_t / f"{name}.csv", index=False)
    log.info("tables written to %s", out_t)
    make_figures(pd.DataFrame(rank_rows), pd.DataFrame(eff), out_f)


def make_figures(rank_df, eff_df, out_f: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rank_df.empty:
        return
    for t, g in rank_df.groupby("task"):
        g = g.sort_values("ndcg10")
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(g) + 1.5))
        labels = [f"{m} ({d})" if d != "-" else m for m, d in zip(g.model, g.design)]
        ax.barh(labels, g.ndcg10, label="all queries")
        ax.scatter(g.ndcg10_hard, labels, color="k", s=12, zorder=3, label="hard queries")
        ax.set_xlabel("nDCG@10 (ESCI gains)"); ax.set_title(f"Product-search ranking: {t}"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out_f / f"ndcg_{t}.png", dpi=200); plt.close(fig)
    if not eff_df.empty:
        merged = eff_df.merge(rank_df, on=["model", "task", "design"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for (m, d), g in merged.groupby(["model", "design"]):
            ax.scatter(g.usd_per_1k_queries.clip(lower=1e-6), g.ndcg10, label=f"{m} ({d})")
        ax.set_xscale("log"); ax.set_xlabel("$ per 1,000 queries (log)"); ax.set_ylabel("nDCG@10")
        ax.legend(fontsize=5); fig.tight_layout(); fig.savefig(out_f / "cost_vs_ndcg.png", dpi=200); plt.close(fig)
    log.info("figures written to %s", out_f)


if __name__ == "__main__":
    main()
