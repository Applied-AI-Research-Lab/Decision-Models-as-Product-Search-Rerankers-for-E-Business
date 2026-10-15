from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .config import (ESCI_LABELS, LOCALE, N_QUERIES, PRODUCT_TEXT_CHARS, SEED, SUPERVISED_TRAIN_PAIRS, TASKS,
                     TASKS_DIR)
from .utils import get_logger, write_jsonl

log = get_logger("data")
REPO = "tasksource/esci"
COLS = ["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "small_version",
        "product_title", "product_bullet_point", "product_brand", "product_color"]


def _shards(split: str) -> list[str]:
    from huggingface_hub import list_repo_files
    return sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith(f"data/{split}-"))


def _read(split: str, n_shards: int | None = None):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    files = _shards(split)[:n_shards] if n_shards else _shards(split)
    parts = [pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset"), columns=COLS) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.small_version == 1]
    n = len(df)
    df = df.drop_duplicates(subset="example_id", keep="first")
    if len(df) < n:
        log.info("%s: dropped %d duplicate example_id rows", split, n - len(df))
    return df


def _field(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def product_text(row) -> str:
    parts = [_field(row.product_title)]
    if _field(row.product_brand):
        parts.append(f"Brand: {_field(row.product_brand)}")
    if _field(row.product_color):
        parts.append(f"Color: {_field(row.product_color)}")
    bullets = [b.strip() for b in re.split(r"\n+", _field(row.product_bullet_point)) if b.strip()][:3]
    if bullets:
        parts.append("Features: " + " | ".join(bullets))
    txt = re.sub(r"\s+", " ", " \n".join(p for p in parts if p)).strip()
    return txt[:PRODUCT_TEXT_CHARS]


def _rows(df) -> list[dict]:
    return [{"id": f"esci-{int(r.example_id)}", "query_id": int(r.query_id), "query": r.query,
             "product_text": product_text(r), "label": esci_label(r.esci_label)} for r in df.itertuples()]


def esci_label(v) -> str:
    return ESCI_LABELS[str(v).strip()[0].upper()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="queries per locale (smoke test; 1 shard)")
    args = ap.parse_args()
    test = _read("test", n_shards=1 if args.limit else None)
    log.info("ESCI reduced test pairs loaded: %d", len(test))
    manifest = []
    for task in TASKS:
        loc = test[test.product_locale == LOCALE[task]]
        qids = sorted(loc.query_id.unique())
        import random
        rng = random.Random(f"{SEED}-{task}")
        n = args.limit or N_QUERIES[task]
        chosen = set(rng.sample(qids, min(n, len(qids))))
        sub = loc[loc.query_id.isin(chosen)].sort_values(["query_id", "example_id"])
        rows = _rows(sub)
        assert len({r["id"] for r in rows}) == len(rows), "duplicate pair ids"
        d = TASKS_DIR / task
        write_jsonl(d / "test.jsonl", rows)
        (d / "labels.json").write_text(json.dumps(["exact", "substitute", "complement", "irrelevant"]))
        labs = {k: sum(r["label"] == k for r in rows) for k in ["exact", "substitute", "complement", "irrelevant"]}
        log.info("%s: %d queries, %d pairs, labels %s", task, len(chosen), len(rows), labs)
        manifest.append({"task": task, "source": f"{REPO} (ESCI reduced test)", "license": "Apache-2.0",
                         "n_queries": len(chosen), "n_pairs": len(rows), "labels": labs,
                         "sha256_test": _sha(d / "test.jsonl"), "limited": bool(args.limit)})
    train = _read("train", n_shards=1 if args.limit else 2)
    train = train[train.product_locale == "us"]
    n = min(len(train), 2000 if args.limit else SUPERVISED_TRAIN_PAIRS)
    train = train.sample(n=n, random_state=SEED)
    write_jsonl(TASKS_DIR / "esci_us" / "train.jsonl", _rows(train))
    log.info("esci_us train pairs: %d", n)
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
