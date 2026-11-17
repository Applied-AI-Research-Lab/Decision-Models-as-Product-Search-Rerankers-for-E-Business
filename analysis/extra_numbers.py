import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(sys.argv[1])
LBL = ["exact", "substitute", "complement", "irrelevant"]
GAIN = dict(zip(LBL, [1.0, 0.1, 0.01, 0.0]))
DEC = ["jev", "clef", "clef_flash", "pplx", "pplx_api"]
CM = DEC + ["qwen_readout1", "gemma_readout1", "qwen_gen", "gemma_gen", "modernbert"]


def rd(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def preds(m, t, c):
    for run in ("main", "round1"):
        p = ROOT / f"results/{run}/raw/{m}/{t}/{c}.jsonl"
        if p.exists():
            return {r["id"]: r for r in rd(p) if not r.get("error")}
    raise FileNotFoundError((m, t, c))


rec = rd(ROOT / "data/records/esci_us/choice.jsonl")
gold = {r["id"]: r["gold"] for r in rec}; qid = {r["id"]: r["query_id"] for r in rec}
byq = defaultdict(list)
for r in rec:
    byq[r["query_id"]].append(r["id"])

pair, miss_sub, outr, ex_rec = {}, {}, {}, {}
for m in CM:
    pr = preds(m, "esci_us", "choice")
    assert set(pr) == set(gold), m
    P = {i: np.array([pr[i]["probs"].get(o, 0.0) for o in LBL]) for i in gold}
    pred = {i: LBL[int(P[i].argmax())] for i in gold}
    score = {i: float(P[i] @ np.array([GAIN[o] for o in LBL]) / P[i].sum()) for i in gold}
    cm = Counter((gold[i], pred[i]) for i in gold)
    err = sum(v for (a, b), v in cm.items() if a != b)
    pair[m] = (cm[("exact", "substitute")] + cm[("substitute", "exact")]) / err
    assert pair[m] == max((cm[(a, b)] + cm[(b, a)]) / err for a in LBL for b in LBL if a < b), m
    missed = {b: v for (a, b), v in cm.items() if a == "exact" and b != "exact"}
    assert max(missed, key=missed.get) == "substitute", m
    miss_sub[m] = missed["substitute"] / sum(missed.values())
    ex_rec[m] = cm[("exact", "exact")] / sum(v for (a, b), v in cm.items() if a == "exact")
    win = tot = 0
    for i in gold:
        if gold[i] == "exact" and pred[i] == "substitute":
            for j in byq[qid[i]]:
                if gold[j] in ("complement", "irrelevant"):
                    tot += 1; win += score[i] > score[j]
    outr[m] = win / tot

rho = {}
for m in DEC + ["qwen_readout1", "gemma_readout1"]:
    a, b = preds(m, "esci_us", "choice"), preds(m, "esci_us", "noul")
    ids = sorted(set(a) & set(b))
    rho[m] = spearmanr([a[i]["probs"].get("exact", 0) for i in ids], [b[i]["probs"].get("yes", 0) for i in ids]).correlation


def rng(d, pct=True, nd=0):
    lo, hi = min(d.values()), max(d.values())
    f = (lambda x: f"{100 * x:.{nd}f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(lo)}--{f(hi)}  (min {min(d, key=d.get)}, max {max(d, key=d.get)})"


print("Exact/Substitute share of all errors:", rng(pair))
print("missed Exact labelled Substitute:", rng(miss_sub))
print("Exact recall:", rng(ex_rec, nd=1))
print("mislabelled Exact still above C/I of its query:", rng(outr))
print("Spearman four-way P(Exact) vs yes/no P(yes):", rng(rho, pct=False))

probe = rd(ROOT / "data/tasks/esci_us_trainprobe/test.jsonl")
train50 = {r["id"] for r in rd(ROOT / "data/tasks/esci_us/train.jsonl")}
pq = Counter(r["query_id"] for r in probe)
inq = Counter(r["query_id"] for r in probe if r["id"] in train50)
print(f"probe pairs in the 50,000 training pairs: {sum(inq.values())} of {len(probe)} "
      f"({100 * sum(inq.values()) / len(probe):.1f}%); probe queries with >=1 such pair: {sum(1 for q in pq if inq[q])} of {len(pq)}")
