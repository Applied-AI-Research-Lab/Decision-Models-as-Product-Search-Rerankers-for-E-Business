from __future__ import annotations

import random

from ..config import HF_MODELS, SEED, TASKS_DIR
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


def _train_modernbert(epochs=2, lr=2e-5, bs=32, max_len=256, clip=1.0):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    train = read_jsonl(TASKS_DIR / "esci_us" / "train.jsonl")
    lab2id = {l: i for i, l in enumerate(CHOICE_ORDER)}
    tok = AutoTokenizer.from_pretrained(HF_MODELS["modernbert"])
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODELS["modernbert"], num_labels=4).cuda()
    data = [(r["query"], r["product_text"], lab2id[r["label"]]) for r in train]

    def collate(b):
        enc = tok([q for q, _, _ in b], [p for _, p, _ in b], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, _, y in b])
        return enc

    dl = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
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


def run_modernbert(task, recs, out):
    import torch

    if "mb" not in _STATE:
        _STATE["mb"] = _train_modernbert()
    model, tok, lab2id, max_len = _STATE["mb"]
    rows = []
    for i in range(0, len(recs), 128):
        chunk = recs[i:i + 128]
        enc = tok([r["query"] for r in chunk], [r["text"] for r in chunk], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to("cuda")
        with Timer() as t, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = model(**enc).logits.float().softmax(-1).cpu().tolist()
        for r, p in zip(chunk, P):
            rows.append(pred_row(r, "modernbert", {o: p[lab2id[o]] for o in r["options"]}, t.seconds / len(chunk)))
    append_jsonl(out, rows)


RUNNERS = {"modernbert": run_modernbert, "mock": run_mock}


def main():
    ap = base_parser("Supervised baseline / mock")
    ap.add_argument("--which", choices=list(RUNNERS), required=True)
    args = ap.parse_args()
    for task, cond, recs, out in jobs(args, args.which):
        RUNNERS[args.which](task, recs, out)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
