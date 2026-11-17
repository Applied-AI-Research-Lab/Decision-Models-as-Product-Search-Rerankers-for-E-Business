import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
LBL = ["exact", "substitute", "complement", "irrelevant"]
SEED = 20261010
SYSTEMS = ["jev", "clef", "clef_flash", "pplx", "pplx_api", "qwen_readout1", "gemma_readout1"]
GRID = np.round(np.arange(-3.0, 3.01, 0.25), 2)


def rd(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def preds(m):
    for run in ("main", "round1"):
        p = ROOT / f"results/{run}/raw/{m}/esci_us/choice.jsonl"
        if p.exists():
            return {r["id"]: r for r in rd(p) if not r.get("error")}
    raise FileNotFoundError(m)


rec = rd(ROOT / "data/records/esci_us/choice.jsonl")
qs = sorted({r["query_id"] for r in rec})
random.Random(f"{SEED}-thresholds").shuffle(qs)
half = {q: i % 2 for i, q in enumerate(qs)}
fold = np.array([half[r["query_id"]] for r in rec])
y = np.array([LBL.index(r["gold"]) for r in rec])


def metrics(pred, yy):
    f1 = []
    for k in range(4):
        tp = np.sum((pred == k) & (yy == k)); fp = np.sum((pred == k) & (yy != k)); fn = np.sum((pred != k) & (yy == k))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(pred == yy)), float(np.mean(f1)), float(np.mean(pred[yy == 2] == 2))


def fit(L, yy):
    b = np.zeros(4)
    best = metrics((L + b).argmax(1), yy)[1]
    for _ in range(3):
        for k in range(4):
            for v in GRID:
                c = b.copy(); c[k] = v
                f = metrics((L + c).argmax(1), yy)[1]
                if f > best + 1e-12:
                    best, b = f, c
    return b


print(f"{'system':16s} acc    F1     Compl  ->  acc    F1     Compl")
rows = {}
for m in SYSTEMS:
    pr = preds(m)
    P = np.array([[pr[r["id"]]["probs"].get(o, 0.0) for o in LBL] for r in rec]); P /= P.sum(1, keepdims=True)
    L = np.log(np.clip(P, 1e-12, 1))
    adj = np.empty(len(rec), int)
    for k in (0, 1):
        b = fit(L[fold == k], y[fold == k])
        adj[fold != k] = (L[fold != k] + b).argmax(1)
    before, after = metrics(L.argmax(1), y), metrics(adj, y)
    rows[m] = (before, after)
    print(f"{m:16s} {before[0]:.3f}  {before[1]:.3f}  {before[2]:.3f}  ->  {after[0]:.3f}  {after[1]:.3f}  {after[2]:.3f}")
json.dump({m: {"before": b, "after": a} for m, (b, a) in rows.items()}, open(Path(__file__).with_name("class_thresholds.json"), "w"), indent=1)
