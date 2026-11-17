from __future__ import annotations

import random

from ..config import HF_MODELS, HF_REVISIONS, SEED, SUPERVISED_TRAIN_PAIRS, TASKS_DIR
from ..records import CHOICE_ORDER
from ..utils import Timer, append_jsonl, get_logger, read_jsonl
from .common import base_parser, jobs, pred_row

log = get_logger("baselines")


def run_mock(task, recs, out):
    rng = random.Random(SEED)
    rows = []
    for r in recs:
        w = [rng.random() for _ in r["options"]]
        rows.append(pred_row(r, "mock", {o: x / sum(w) for o, x in zip(r["options"], w)}, 0.001))
    append_jsonl(out, rows)


_STATE = {}


def training_pairs(train_n: str = str(SUPERVISED_TRAIN_PAIRS), seed_offset: int = 0) -> list[dict]:
    if train_n == "full":
        return read_jsonl(TASKS_DIR / "esci_us" / "train_full.jsonl")
    train = read_jsonl(TASKS_DIR / "esci_us" / "train.jsonl")
    n = int(train_n)
    if n >= len(train):
        return train
    rng = random.Random(f"{SEED}-budget-{seed_offset}")
    return [train[i] for i in sorted(rng.sample(range(len(train)), n))]


def model_key(train_n: str, seed_offset: int) -> str:
    if train_n == str(SUPERVISED_TRAIN_PAIRS) and seed_offset == 0:
        return "modernbert"
    return f"modernbert_n{train_n}_s{seed_offset}"


def _train_modernbert(epochs=2, lr=2e-5, bs=32, max_len=256, clip=1.0, train_n=str(SUPERVISED_TRAIN_PAIRS),
                      seed_offset=0):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed = SEED + seed_offset
    torch.manual_seed(seed)
    train = training_pairs(train_n, seed_offset)
    log.info("ModernBERT: %d training pairs (train_n=%s, seed=%d)", len(train), train_n, seed)
    lab2id = {l: i for i, l in enumerate(CHOICE_ORDER)}
    rev = HF_REVISIONS["modernbert"]
    tok = AutoTokenizer.from_pretrained(HF_MODELS["modernbert"], revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODELS["modernbert"], num_labels=4,
                                                               revision=rev).cuda()
    data = [(r["query"], r["product_text"], lab2id[r["label"]]) for r in train]

    def collate(b):
        enc = tok([q for q, _, _ in b], [p for _, p, _ in b], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, _, y in b])
        return enc

    dl = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * epochs), len(dl) * epochs)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for step, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if step % 200 == 0:
                log.info("epoch %d step %d/%d loss %.4f", ep, step, len(dl), tot / (step + 1))
    model.eval()
    return model, tok, lab2id, max_len


def run_modernbert(task, recs, out, key="modernbert", train_n=str(SUPERVISED_TRAIN_PAIRS), seed_offset=0):
    import torch

    if "mb" not in _STATE:
        _STATE["mb"] = _train_modernbert(train_n=train_n, seed_offset=seed_offset)
    model, tok, lab2id, max_len = _STATE["mb"]
    rows = []
    for i in range(0, len(recs), 128):
        chunk = recs[i:i + 128]
        enc = tok([r["query"] for r in chunk], [r["text"] for r in chunk], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to("cuda")
        with Timer() as t, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = model(**enc).logits.float().softmax(-1).cpu().tolist()
        for r, p in zip(chunk, P):
            rows.append(pred_row(r, key, {o: p[lab2id[o]] for o in r["options"]}, t.seconds / len(chunk)))
    append_jsonl(out, rows)


RUNNERS = {"modernbert": run_modernbert, "mock": run_mock}


def main():
    ap = base_parser("Supervised baseline / mock")
    ap.add_argument("--which", choices=list(RUNNERS), required=True)
    ap.add_argument("--train-n", default=str(SUPERVISED_TRAIN_PAIRS), help="5000 | 20000 | 50000 | full")
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()
    if args.which == "modernbert":
        key = model_key(args.train_n, args.seed_offset)
        for task, cond, recs, out in jobs(args, key):
            run_modernbert(task, recs, out, key, args.train_n, args.seed_offset)
            log.info("done %s", out)
        return
    for task, cond, recs, out in jobs(args, args.which):
        RUNNERS[args.which](task, recs, out)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
