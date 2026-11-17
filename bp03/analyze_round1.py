from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .analyze import load, ndcg_at, pair_scores, paired_query_bootstrap, per_query, summarize
from .config import API_PROVIDERS, BOOTSTRAP_N, GAINS, GPU_HOUR_USD, SEED, TASKS_DIR, results_dir
from .lexical import LANG, analyser, bm25_pair_scores
from .stats import holm
from .utils import get_logger, read_jsonl

log = get_logger("analyze_round1")
ESCI = ["esci_us", "esci_es", "esci_jp"]
NEW = ["esci_us_trainprobe", "wands", "esci_us_retrieval"]
CONDS = ["choice", "score", "noul", "choice_reversed", "choice_shuffled", "repeat1", "repeat2"]
DESIGNS = ["choice", "score", "noul"]
DEC = ["jev", "clef", "clef_flash", "pplx", "pplx_api"]
RERANKERS = ["qwen_reranker"] + [f"qwen_reranker{s}_{i}" for s in ("4b", "8b") for i in ("default", "ecommerce", "esci")
                                 if (s, i) != ("4b", "ecommerce")]
GEN = ["qwen_gen", "gemma_gen"]
SUPERVISED = ["modernbert", "modernbert_nfull_s0"]
BASE = ["bm25_word", "bm25_language", "random"]


def ci_boot(values, n_boot, seed=SEED):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(v.mean()), *map(float, np.percentile(b, [2.5, 97.5]))


def load_all(run: str):
    cache = {}
    roots = [("main", results_dir("main") / "raw"), (run, results_dir(run) / "raw")]
    for name, raw in roots:
        if not raw.exists():
            continue
        for mdir in sorted(p for p in raw.iterdir() if p.is_dir()):
            for tdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
                for f in sorted(tdir.glob("*.jsonl")):
                    m, t, c = mdir.name, tdir.name, f.stem
                    key_c = "drift" if (name != "main" and c == "repeat1") else c
                    if (m, t, key_c) in cache:
                        continue
                    d = load(name, m, t, c)
                    if d is not None:
                        d["source"] = name
                        cache[(m, t, key_c)] = d
    return cache


def readout_names(cache):
    single = any(m.endswith("_readout1") for (m, _, _) in cache)
    return ["qwen_readout1", "gemma_readout1"] if single else ["qwen_readout", "gemma_readout"]


def bm25_variants(task: str, pairs: list[dict]) -> dict[str, dict[str, float]]:
    lang = LANG.get(task, "en")
    out = {"bm25_word": bm25_pair_scores(pairs, analyser("word", lang)),
           "bm25_language": bm25_pair_scores(pairs, analyser("language", lang))}
    if lang == "ja":
        for a in ("cjk_unigram", "cjk_bigram", "mecab"):
            try:
                out[f"bm25_{a}"] = bm25_pair_scores(pairs, analyser(a, lang))
            except ImportError as e:
                log.warning("skip %s: %s", a, e)
    return out


def random_pq(pairs, gains=None):
    rng = np.random.default_rng(SEED)
    rand = [per_query_g({p["id"]: rng.random() for p in pairs}, pairs, gains) for _ in range(20)]
    return {q: {"ndcg": np.mean([r[q]["ndcg"] for r in rand]) if rand[0][q]["ndcg"] is not None else None,
                "mrr": np.mean([r[q]["mrr"] for r in rand]) if rand[0][q]["mrr"] is not None else None,
                "hard": rand[0][q]["hard"]} for q in rand[0]}


def per_query_g(scores, pairs, gains=None):
    if gains is None:
        return per_query(scores, pairs)
    by_q = defaultdict(list)
    for p in pairs:
        if p["id"] in scores:
            by_q[p["query_id"]].append(p)
    out = {}
    for q, items in by_q.items():
        ranked = sorted(items, key=lambda p: (-scores[p["id"]], p["id"]))
        g = [gains(p) for p in ranked]
        labels = [p["label"] for p in ranked]
        out[q] = {"ndcg": ndcg_at(g), "mrr": next((1.0 / (i + 1) for i, l in enumerate(labels) if l == "exact"), None),
                  "hard": ("exact" in labels and "irrelevant" in labels)}
    return out


def retrieval_pq(scores, cand, judged_by_q, condensed=False):
    by_q, with_cand, scored = defaultdict(list), set(), set()
    for p in cand:
        with_cand.add(p["query_id"])
        if p["id"] in scores:
            scored.add(p["query_id"])
            if p["judged"] or not condensed:
                by_q[p["query_id"]].append(p)
    out = {}
    for q, judged in judged_by_q.items():
        if q in with_cand and q not in scored:
            continue
        ranked = sorted(by_q.get(q, []), key=lambda p: (-scores[p["id"]], p["id"]))
        g = [GAINS[p["label"]] for p in ranked][:10]
        ideal = sorted((GAINS[l] for l in judged), reverse=True)[:10]
        idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
        nd = None if idcg == 0 else sum(x / math.log2(i + 2) for i, x in enumerate(g)) / idcg
        labs = [p["label"] for p in ranked]
        mrr = next((1.0 / (i + 1) for i, l in enumerate(labs) if l == "exact"), 0.0 if "exact" in judged else None)
        out[q] = {"ndcg": nd, "mrr": mrr, "hard": "exact" in judged and "irrelevant" in judged,
                  "judged10": float(np.mean([p["judged"] for p in ranked[:10]])) if ranked else None}
    return out


def build_pq(cache, n_boot):
    pq, extra = {}, {}
    sens = {}
    for t in ESCI + NEW:
        f = TASKS_DIR / t / "test.jsonl"
        if not f.exists():
            continue
        pairs = read_jsonl(f)
        if t == "esci_us_retrieval":
            judged = defaultdict(list)
            for p in read_jsonl(TASKS_DIR / "esci_us" / "test.jsonl"):
                judged[p["query_id"]].append(p["label"])
            first = {p["id"]: -p["bm25_rank"] for p in pairs}
            pq[("bm25_language", t, "-")] = retrieval_pq(first, pairs, judged)
            pq[("bm25_language", t, "-", "condensed")] = retrieval_pq(first, pairs, judged, condensed=True)
            for (m, tt, c), d in cache.items():
                if tt == t and c in ("choice", "noul"):
                    s = pair_scores(d, c)
                    pq[(m, t, c)] = retrieval_pq(s, pairs, judged)
                    pq[(m, t, c, "condensed")] = retrieval_pq(s, pairs, judged, condensed=True)
            extra["retrieval_recall"] = _recall_at_k(pairs, judged)
            continue
        for name, sc in bm25_variants(t, pairs).items():
            pq[(name, t, "-")] = per_query(sc, pairs)
        pq[("random", t, "-")] = random_pq(pairs)
        for (m, tt, c), d in cache.items():
            if tt == t and c in DESIGNS:
                pq[(m, t, c)] = per_query(pair_scores(d, c), pairs)
        if t == "wands":
            g05 = lambda p: {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}[p["orig_label"]]
            for (m, tt, c), d in cache.items():
                if tt == t and c in ("choice", "noul"):
                    sens[(m, c)] = per_query_g(pair_scores(d, c), pairs, g05)
            for name, sc in bm25_variants(t, pairs).items():
                sens[(name, "-")] = per_query_g(sc, pairs, g05)
    extra["wands_sens"] = sens
    return pq, extra


def _recall_at_k(cand, judged):
    got = defaultdict(int)
    for p in cand:
        if p["label"] == "exact" and p["judged"]:
            got[p["query_id"]] += 1
    rec = [got[q] / Counter(l)["exact"] for q, l in judged.items() if "exact" in l]
    return float(np.mean(rec))


def reported(task, pq, readouts):
    syst = []
    for m in DEC:
        for c in DESIGNS:
            if (m, task, c) in pq:
                syst.append((m, task, c))
    for m in readouts:
        for c in DESIGNS:
            if (m, task, c) in pq:
                syst.append((m, task, c))
    syst += [(m, task, "choice") for m in GEN + SUPERVISED if (m, task, "choice") in pq]
    syst += [(m, task, "noul") for m in RERANKERS if (m, task, "noul") in pq]
    syst += [(m, task, "-") for m in BASE if (m, task, "-") in pq]
    return syst


def best_reranker(task, pq):
    cands = [(summarize(pq[(m, task, "noul")])["ndcg10"], m) for m in RERANKERS if (m, task, "noul") in pq]
    return max(cands)[1] if cands else None


def run_tests(pq, readouts, n_boot):
    rows = []
    for t in ESCI + NEW:
        syst = reported(t, pq, readouts)
        if not syst:
            continue
        refs = [("bm25_word", t, "-"), ("bm25_language", t, "-")]
        br = best_reranker(t, pq)
        if br:
            refs.append((br, t, "noul"))
        for ref in refs:
            if ref not in pq:
                continue
            for hard in (False, True):
                fam = []
                for s in syst:
                    if s == ref:
                        continue
                    diff, lo, hi, n, p = paired_query_bootstrap(pq[s], pq[ref], n_boot, hard)
                    fam.append({"model": s[0], "task": t, "design": s[2], "reference": f"{ref[0]}:{ref[2]}",
                                "reference_role": "best_reranker" if ref[0] == br else ref[0], "subset": "hard" if hard else "all",
                                "n_queries": n, "ndcg_diff": diff, "lo": lo, "hi": hi, "p": p})
                ok = [r for r in fam if not np.isnan(r["p"])]
                for r, ph in zip(ok, holm([r["p"] for r in ok])):
                    r["p_holm"] = ph
                rows += fam
    return pd.DataFrame(rows)


def run_designs(pq, readouts, n_boot):
    rows = []
    for t in ESCI:
        fam = []
        for m in DEC + readouts:
            for a, b in [("choice", "score"), ("choice", "noul"), ("score", "noul")]:
                if (m, t, a) in pq and (m, t, b) in pq:
                    diff, lo, hi, n, p = paired_query_bootstrap(pq[(m, t, a)], pq[(m, t, b)], n_boot)
                    fam.append({"model": m, "task": t, "comparison": f"{a} - {b}", "n_queries": n, "ndcg_diff": diff,
                                "lo": lo, "hi": hi, "p": p})
        for r, ph in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = ph
        rows += fam
    return pd.DataFrame(rows)


def classification(cache):
    cls, conf = [], []
    for (m, t, c), d in cache.items():
        if c not in DESIGNS:
            continue
        P, y = d["P"], d["y"]
        conf_gap = float(P.max(1).mean() - (P.argmax(1) == y).mean())
        cls.append({"model": m, "task": t, "design": c, **M.summary(P, y), "conf_minus_acc": conf_gap})
        if c == "choice":
            pred = P.argmax(1)
            for k, o in enumerate(d["opts"]):
                mask = y == k
                if mask.any():
                    conf.append({"model": m, "task": t, "label": o, "n": int(mask.sum()),
                                 "recall": float(np.mean(pred[mask] == k))})
    return pd.DataFrame(cls), pd.DataFrame(conf)


def _aligned(a, b):
    ids = sorted(set(a["ids"]) & set(b["ids"]))
    ia, ib = {i: k for k, i in enumerate(a["ids"])}, {i: k for k, i in enumerate(b["ids"])}
    Pa = np.array([[a["preds"][i]["probs"].get(o, 0) for o in a["opts"]] for i in ids])
    Pb = np.array([[b["preds"][i]["probs"].get(o, 0) for o in a["opts"]] for i in ids])
    qid = {r["id"]: r["query_id"] for r in a["recs"]}
    return ids, Pa, Pb, [qid[i] for i in ids]


def cluster_ci(flags_by_q: dict, n_boot):
    qs = sorted(flags_by_q)
    num = np.array([sum(flags_by_q[q]) for q in qs], float)
    den = np.array([len(flags_by_q[q]) for q in qs], float)
    rng = np.random.default_rng(SEED)
    b = []
    for _ in range(n_boot):
        k = rng.integers(0, len(qs), len(qs))
        b.append(num[k].sum() / den[k].sum())
    return float(num.sum() / den.sum()), *map(float, np.percentile(b, [2.5, 97.5]))


def robustness(cache, n_boot):
    rows, diffs = [], []
    for (m, t, c), d in cache.items():
        base = cache.get((m, t, "choice"))
        if c not in ("choice_reversed", "choice_shuffled", "repeat1", "repeat2", "drift") or base is None:
            continue
        ids, Pa, Pb, qs = _aligned(base, d)
        if not ids:
            continue
        flips = Pa.argmax(1) != Pb.argmax(1)
        by_q = defaultdict(list)
        for f, q in zip(flips, qs):
            by_q[q].append(float(f))
        est, lo, hi = cluster_ci(by_q, n_boot)
        rows.append({"model": m, "task": t, "condition": c, "n": len(ids), "queries": len(by_q), "flip_rate": est,
                     "lo": lo, "hi": hi, "mean_tv": M.total_variation(Pa, Pb)})
    for (m, t, c), d in cache.items():
        if c not in ("choice_reversed", "choice_shuffled"):
            continue
        for rep in ("repeat1", "repeat2"):
            r = cache.get((m, t, rep))
            base = cache.get((m, t, "choice"))
            if r is None or base is None:
                continue
            ids, Pa, Po, qs = _aligned(base, d)
            ids_r, _, Pr, _ = _aligned(base, r)
            if not ids or ids != ids_r:
                continue
            fo, fr = (Pa.argmax(1) != Po.argmax(1)).astype(float), (Pa.argmax(1) != Pr.argmax(1)).astype(float)
            qset = sorted(set(qs))
            qi = {q: k for k, q in enumerate(qset)}
            no, nr, den = np.zeros(len(qset)), np.zeros(len(qset)), np.zeros(len(qset))
            for a, b, q in zip(fo, fr, qs):
                no[qi[q]] += a; nr[qi[q]] += b; den[qi[q]] += 1
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(n_boot):
                k = rng.integers(0, len(qset), len(qset))
                bs.append((no[k].sum() - nr[k].sum()) / den[k].sum())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            diffs.append({"model": m, "task": t, "order": c, "resend": rep, "diff": float((no.sum() - nr.sum()) / den.sum()),
                          "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def hosted_local(cache, pq):
    rows = []
    for t in ESCI + NEW:
        for c in DESIGNS:
            a, b = cache.get(("pplx_api", t, c)), cache.get(("pplx", t, c))
            if a is None or b is None:
                continue
            ids, Pb, Pa, _ = _aligned(b, a)
            row = {"task": t, "design": c, "n": len(ids), "agreement": float(np.mean(Pa.argmax(1) == Pb.argmax(1)))}
            if ("pplx", t, c) in pq and ("pplx_api", t, c) in pq:
                row.update(ndcg_local=summarize(pq[("pplx", t, c)])["ndcg10"], ndcg_api=summarize(pq[("pplx_api", t, c)])["ndcg10"])
            rows.append(row)
    return pd.DataFrame(rows)


def efficiency(cache):
    rows = []
    for (m, t, c), d in cache.items():
        if c not in DESIGNS:
            continue
        npq = np.mean(list(Counter(r["query_id"] for r in d["recs"]).values()))
        lat = np.array([p["latency_s"] or 0 for p in d["preds"].values()])
        tok = np.array([p["input_tokens"] or 0 for p in d["preds"].values()])
        per_pair = (tok.mean() * API_PROVIDERS[m]["usd_per_m_input"] / 1e6) if m in API_PROVIDERS else lat.mean() * GPU_HOUR_USD / 3600
        rows.append({"model": m, "task": t, "design": c, "source": d["source"], "latency_per_pair_s": lat.mean(),
                     "input_tokens_per_pair": tok.mean(), "usd_per_1k_queries": per_pair * npq * 1000,
                     "pairs_per_query": npq})
    return pd.DataFrame(rows)


def probe(cache, pq, n_boot):
    rows = []
    for (m, t, c), d in cache.items():
        if t != "esci_us_trainprobe" or c not in ("choice", "noul") or (m, "esci_us", c) not in cache:
            continue
        a, b = pq.get((m, t, c)), pq.get((m, "esci_us", c))
        if a is None or b is None:
            continue
        na = [v["ndcg"] for v in a.values() if v["ndcg"] is not None]
        nb = [v["ndcg"] for v in b.values() if v["ndcg"] is not None]
        rng = np.random.default_rng(SEED)
        bs = [np.mean(rng.choice(na, len(na))) - np.mean(rng.choice(nb, len(nb))) for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dt, de = d, cache[(m, "esci_us", c)]
        rows.append({"model": m, "design": c, "ndcg_probe": float(np.mean(na)), "ndcg_test": float(np.mean(nb)),
                     "ndcg_diff": float(np.mean(na) - np.mean(nb)), "lo": float(lo), "hi": float(hi),
                     "acc_probe": M.accuracy(dt["P"], dt["y"]), "acc_test": M.accuracy(de["P"], de["y"]),
                     "queries_probe": len(na), "queries_test": len(nb)})
    return pd.DataFrame(rows)


def budget(cache, pq):
    rows = []
    for (m, t, c), d in cache.items():
        if not m.startswith("modernbert") or c != "choice" or t not in ("esci_us", "wands"):
            continue
        n, s = ("50000", "0") if m == "modernbert" else m[len("modernbert_n"):].split("_s")
        rows.append({"model": m, "task": t, "train_n": n, "seed_offset": int(s), "source": d["source"],
                     "ndcg10": summarize(pq[(m, t, c)])["ndcg10"], "accuracy": M.accuracy(d["P"], d["y"]),
                     "macro_f1": M.macro_f1(d["P"], d["y"])})
    return pd.DataFrame(rows)


def retrieval_table(pq, extra, n_boot):
    t = "esci_us_retrieval"
    rows = []
    for k, v in pq.items():
        if k[1] != t or len(k) != 3:
            continue
        cond = pq.get(k + ("condensed",))
        s = summarize(v)
        rows.append({"model": k[0], "design": k[2], "ndcg10": s["ndcg10"], "ndcg10_hard": s["ndcg10_hard"],
                     "mrr_exact": s["mrr_exact"], "ndcg10_condensed": summarize(cond)["ndcg10"] if cond else np.nan,
                     "judged_at10": float(np.mean([x["judged10"] for x in v.values() if x["judged10"] is not None])),
                     "queries": s["queries"],
                     "exact_recall_at_k": extra.get("retrieval_recall")})
    return pd.DataFrame(rows)


def readout_agreement(cache, pq):
    rows = []
    for (m, t, c), d in cache.items():
        if not m.endswith("_readout1"):
            continue
        old = cache.get((m[:-1], t, c))
        if old is None:
            continue
        ids, P1, P0, _ = _aligned(d, old)
        row = {"model": m, "task": t, "condition": c, "n": len(ids), "argmax_agreement": float(np.mean(P1.argmax(1) == P0.argmax(1))),
               "mean_tv": M.total_variation(P1, P0)}
        if (m, t, c) in pq and (m[:-1], t, c) in pq:
            a, b = pq[(m, t, c)], pq[(m[:-1], t, c)]
            qs = set(a) & set(b)
            row.update(ndcg_queries=len(qs), ndcg_single=summarize({q: a[q] for q in qs})["ndcg10"],
                       ndcg_trie=summarize({q: b[q] for q in qs})["ndcg10"])
        rows.append(row)
    return pd.DataFrame(rows)


def check_main(n_boot):
    from .analyze import bm25_scores
    frozen = pd.read_csv(results_dir("main") / "tables" / "ranking_tests.csv")
    for t in ESCI:
        pairs = read_jsonl(TASKS_DIR / t / "test.jsonl")
        a, b = bm25_scores(pairs), bm25_pair_scores(pairs, analyser("word", LANG[t]))
        assert all(abs(a[k] - b[k]) < 1e-12 for k in a), t
    log.info("word analyser reproduces the frozen BM25 exactly (3 markets)")
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="round1")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--check-main", action="store_true")
    args = ap.parse_args()
    if args.check_main:
        check_main(args.n_boot)
    out = results_dir(args.run_name) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    cache = load_all(args.run_name)
    readouts = readout_names(cache)
    log.info("%d files loaded; readouts in the paper: %s", len(cache), readouts)
    pq, extra = build_pq(cache, args.n_boot)

    rank = []
    for k, v in pq.items():
        if len(k) == 3:
            rank.append({"model": k[0], "task": k[1], "design": k[2], **summarize(v)})
    tables = {"ranking": pd.DataFrame(rank), "tests": run_tests(pq, readouts, args.n_boot),
              "designs": run_designs(pq, readouts, args.n_boot)}
    tables["classification"], tables["confusion"] = classification(cache)
    tables["robustness"], tables["robustness_paired"] = robustness(cache, args.n_boot)
    tables["hosted_local"] = hosted_local(cache, pq)
    tables["efficiency"] = efficiency(cache)
    tables["probe"] = probe(cache, pq, args.n_boot)
    tables["budget"] = budget(cache, pq)
    tables["retrieval"] = retrieval_table(pq, extra, args.n_boot)
    tables["readout_agreement"] = readout_agreement(cache, pq)
    sens = [{"model": m, "design": c, "ndcg10_partial05": summarize(v)["ndcg10"]} for (m, c), v in extra["wands_sens"].items()]
    tables["wands_sensitivity"] = pd.DataFrame(sens)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    log.info("tables written to %s: %s", out, ", ".join(f"{k} ({len(v)})" for k, v in tables.items()))


if __name__ == "__main__":
    main()
