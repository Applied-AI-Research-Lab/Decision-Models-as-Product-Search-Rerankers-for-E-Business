from __future__ import annotations

import random

from .config import ROBUSTNESS_QUERIES, SEED, TASKS, TASKS_DIR, records_path
from .utils import get_logger, read_jsonl, write_jsonl

log = get_logger("records")

ESCI_CRITERIA = {
    "exact": ("Exact: the product is relevant for the query and satisfies all of the query's specifications "
              "(e.g. product type, brand, size, color, quantity)."),
    "substitute": ("Substitute: the product is somewhat relevant; it fails some aspects of the query but could "
                   "be used as a functional substitute."),
    "complement": ("Complement: the product does not fulfil the query but could be used together with an exact "
                   "item (e.g. a case for a requested phone)."),
    "irrelevant": "Irrelevant: the product is irrelevant to the query or fails a central aspect of it.",
}
CHOICE_ORDER = ["exact", "substitute", "complement", "irrelevant"]
SCORE_ORDER = ["irrelevant", "complement", "substitute", "exact"]

DESIGN = {
    "choice": {"qtype": "choice", "options": CHOICE_ORDER,
               "instructions": "How does this product relate to the customer's shopping query?"},
    "score": {"qtype": "score", "options": SCORE_ORDER,
              "instructions": "Rate how well this product satisfies the customer's shopping query."},
    "noul": {"qtype": "noul", "options": ["yes", "no"],
             "instructions": ("Is this product an exact match for the customer's shopping query, satisfying all "
                              "of the query's specifications?")},
}
NOUL_CRITERIA = {"yes": "Yes: the product is an exact match.", "no": "No: the product is not an exact match."}


def gold_for(design: str, label: str) -> str:
    if design == "noul":
        return "yes" if label == "exact" else "no"
    return label


def make_record(item: dict, task: str, condition: str, design: str, options: list[str]) -> dict:
    d = DESIGN[design]
    crit = NOUL_CRITERIA if design == "noul" else ESCI_CRITERIA
    return {
        "id": item["id"], "task": task, "condition": condition, "qtype": d["qtype"],
        "query_id": item["query_id"], "query": item["query"], "text": item["product_text"],
        "state": f"Shopping query: {item['query']}\nProduct: {item['product_text']}",
        "gold": gold_for(design, item["label"]), "instructions": d["instructions"], "options": options,
        "criteria": {o: crit[o] for o in options}, "meta": {"esci_label": item["label"], "query_id": item["query_id"]},
    }


def robustness_subset(test: list[dict], task: str) -> list[dict]:
    qids = sorted({r["query_id"] for r in test})
    rng = random.Random(f"{SEED}-{task}-subset")
    keep = set(rng.sample(qids, min(ROBUSTNESS_QUERIES, len(qids))))
    return [r for r in test if r["query_id"] in keep]


def build_task(task: str) -> dict[str, int]:
    test = read_jsonl(TASKS_DIR / task / "test.jsonl")
    subset = robustness_subset(test, task)
    counts = {}

    def emit(cond, items, design, options):
        write_jsonl(records_path(task, cond), [make_record(it, task, cond, design, options) for it in items])
        counts[cond] = len(items)

    for design in ["choice", "score", "noul"]:
        emit(design, test, design, DESIGN[design]["options"])
    emit("choice_reversed", subset, "choice", list(reversed(CHOICE_ORDER)))
    shuffled = list(CHOICE_ORDER)
    random.Random(f"{SEED}-{task}-shuffle").shuffle(shuffled)
    if shuffled in (CHOICE_ORDER, list(reversed(CHOICE_ORDER))):
        shuffled = CHOICE_ORDER[1:] + CHOICE_ORDER[:1]
    emit("choice_shuffled", subset, "choice", shuffled)
    emit("repeat1", subset, "choice", CHOICE_ORDER)
    emit("repeat2", subset, "choice", CHOICE_ORDER)
    return counts


def main():
    for t in TASKS:
        log.info("%s: %s", t, build_task(t))


if __name__ == "__main__":
    main()
