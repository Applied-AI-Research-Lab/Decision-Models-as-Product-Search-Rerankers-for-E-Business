from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .config import PRODUCT_TEXT_CHARS, SEED, TASKS_DIR, records_path
from .data import _read, _rows, _sha
from .lexical import BM25Index, analyser
from .records import CHOICE_ORDER, DESIGN, make_record
from .utils import get_logger, read_jsonl, write_jsonl

log = get_logger("data_round1")
PROBE_QUERIES = 300
WANDS_PER_QUERY = 30
RETRIEVAL_K = 40
WANDS_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/"
WANDS_LABEL = {"Exact": "exact", "Partial": "substitute", "Irrelevant": "irrelevant"}
ROUND1_TASKS = ["esci_us_trainprobe", "wands", "esci_us_retrieval"]


def _write_task(task: str, rows: list[dict], extra: dict) -> dict:
    assert len({r["id"] for r in rows}) == len(rows), f"{task}: duplicate ids"
    d = TASKS_DIR / task
    write_jsonl(d / "test.jsonl", rows)
    (d / "labels.json").write_text(json.dumps(CHOICE_ORDER))
    labs = {k: sum(r["label"] == k for r in rows) for k in CHOICE_ORDER}
    info = {"task": task, "n_queries": len({r["query_id"] for r in rows}), "n_pairs": len(rows), "labels": labs,
            "sha256_test": _sha(d / "test.jsonl"), **extra}
    log.info("%s", info)
    return info


def _records(task: str, rows: list[dict]) -> None:
    for design in ("choice", "noul"):
        write_jsonl(records_path(task, design), [make_record(r, task, design, design, DESIGN[design]["options"])
                                                  for r in rows])


def build_train(limit_q: int | None) -> tuple[list[dict], dict]:
    train = _read("train", n_shards=1 if limit_q else None)
    train = train[train.product_locale == "us"]
    full = _rows(train.sort_values(["query_id", "example_id"]))
    test_q = {r["query_id"] for r in read_jsonl(TASKS_DIR / "esci_us" / "test.jsonl")}
    assert not ({r["query_id"] for r in full} & test_q), "train and test queries overlap"
    write_jsonl(TASKS_DIR / "esci_us" / "train_full.jsonl", full)
    log.info("esci_us train_full: %d pairs, %d queries", len(full), len({r['query_id'] for r in full}))

    sup_q = sorted({r["query_id"] for r in read_jsonl(TASKS_DIR / "esci_us" / "train.jsonl")})
    by_q = defaultdict(list)
    for r in full:
        by_q[r["query_id"]].append(r)
    pool = [q for q in sup_q if q in by_q and any(x["label"] == "exact" for x in by_q[q])]
    rng = random.Random(f"{SEED}-trainprobe")
    chosen = set(rng.sample(pool, min(limit_q or PROBE_QUERIES, len(pool))))
    rows = [r for r in full if r["query_id"] in chosen]
    return rows, {"source": "tasksource/esci (ESCI reduced TRAIN split, US)", "license": "Apache-2.0",
                  "n_train_full_pairs": len(full)}


def _wands_csv(name: str):
    import pandas as pd
    import urllib.request
    raw = urllib.request.urlopen(WANDS_BASE + name, timeout=120).read()
    sha = hashlib.sha256(raw).hexdigest()
    for sep in ("\t", ","):
        df = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)
        if len(df.columns) > 2:
            return df, sha
    raise ValueError(f"cannot parse {name}")


def _clean(v: str) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def wands_text(p) -> str:
    parts = [_clean(p["product_name"])]
    if _clean(p.get("product_class", "")):
        parts.append(f"Class: {_clean(p['product_class'])}")
    if _clean(p.get("category_hierarchy", "")):
        parts.append(f"Category: {_clean(p['category_hierarchy'])}")
    if _clean(p.get("product_description", "")):
        parts.append(f"Description: {_clean(p['product_description'])}")
    return " ".join(x for x in parts if x)[:PRODUCT_TEXT_CHARS]


def build_wands(limit_q: int | None) -> tuple[list[dict], dict]:
    prod, s1 = _wands_csv("product.csv")
    qry, s2 = _wands_csv("query.csv")
    lab, s3 = _wands_csv("label.csv")
    products = {r["product_id"]: r for r in prod.to_dict("records")}
    queries = {r["query_id"]: r["query"] for r in qry.to_dict("records")}
    by_q = defaultdict(list)
    for r in lab.to_dict("records"):
        if r["label"] in WANDS_LABEL and r["product_id"] in products:
            by_q[r["query_id"]].append(r)
    qids = sorted(by_q, key=int)
    if limit_q:
        qids = qids[:limit_q]
    rows = []
    for q in qids:
        items = sorted(by_q[q], key=lambda r: int(r["id"]))
        rng = random.Random(f"{SEED}-wands-{q}")
        keep = sorted(rng.sample(items, min(WANDS_PER_QUERY, len(items))), key=lambda r: int(r["id"]))
        for r in keep:
            rows.append({"id": f"wands-{r['id']}", "query_id": int(q), "query": _clean(queries[q]),
                         "product_text": wands_text(products[r["product_id"]]), "label": WANDS_LABEL[r["label"]],
                         "orig_label": r["label"]})
    return rows, {"source": "wayfair/WANDS (github, main)", "license": "MIT", "per_query_cap": WANDS_PER_QUERY,
                  "sha256_raw": {"product.csv": s1, "query.csv": s2, "label.csv": s3}}


def build_retrieval(limit_q: int | None) -> tuple[list[dict], dict]:
    test = read_jsonl(TASKS_DIR / "esci_us" / "test.jsonl")
    raw = _read("test", n_shards=1 if limit_q else None)
    pid = {f"esci-{int(e)}": p for e, p in zip(raw.example_id, raw.product_id)}
    missing = [r["id"] for r in test if r["id"] not in pid]
    if missing and not limit_q:
        raise RuntimeError(f"{len(missing)} test pairs without product_id")
    test = [r for r in test if r["id"] in pid]
    docs, judged = {}, {}
    for r in test:
        docs.setdefault(pid[r["id"]], r["product_text"])
        judged[(r["query_id"], pid[r["id"]])] = r["label"]
    idx = BM25Index(docs, analyser("language", "en"))
    queries = {}
    for r in test:
        queries.setdefault(r["query_id"], r["query"])
    qids = sorted(queries)[:limit_q] if limit_q else sorted(queries)
    rows, recall = [], []
    for q in qids:
        top = idx.top(queries[q], RETRIEVAL_K)
        exact = {p for (qq, p), l in judged.items() if qq == q and l == "exact"}
        got = {p for p, _ in top}
        recall.append(len(exact & got) / len(exact) if exact else float("nan"))
        for rank, (p, s) in enumerate(top):
            lab = judged.get((q, p))
            rows.append({"id": f"ret-{q}-{p}", "query_id": q, "query": queries[q], "product_text": docs[p],
                         "label": lab or "irrelevant", "judged": lab is not None, "product_id": p,
                         "bm25_rank": rank + 1, "bm25_score": s})
    import math
    rec = [x for x in recall if not math.isnan(x)]
    return rows, {"source": "ESCI test sample (US), BM25 language-aware analyser over the pooled products",
                  "pool_products": len(docs), "k": RETRIEVAL_K,
                  "mean_exact_recall_at_k": sum(rec) / len(rec) if rec else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--only", default="trainprobe,wands,retrieval")
    args = ap.parse_args()
    manifest_path = TASKS_DIR / "manifest_round1.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    only = args.only.split(",")
    if "trainprobe" in only:
        rows, extra = build_train(args.limit_queries)
        manifest["esci_us_trainprobe"] = _write_task("esci_us_trainprobe", rows, extra)
        _records("esci_us_trainprobe", rows)
    if "wands" in only:
        rows, extra = build_wands(args.limit_queries)
        manifest["wands"] = _write_task("wands", rows, extra)
        _records("wands", rows)
    if "retrieval" in only:
        rows, extra = build_retrieval(args.limit_queries)
        manifest["esci_us_retrieval"] = _write_task("esci_us_retrieval", rows, extra)
        _records("esci_us_retrieval", rows)
    manifest["limited"] = bool(args.limit_queries)
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
