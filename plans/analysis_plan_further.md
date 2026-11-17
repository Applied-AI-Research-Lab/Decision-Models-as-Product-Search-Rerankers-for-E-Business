# Analysis plan: further conditions

Committed before the run of the further conditions: commit `57aca027921254cac7426af5ade35ad15ecb5d43`, 12 November 2026, 11:39 (UTC+2). The code as run is tagged `round1-P03` (commit `80387d57797b5fb5d011b0b5dc726d51938bbcd4`; it adds only `max_num_seqs=512` for Qwen3.8, which changes batch scheduling, not scores). Reproduced verbatim.

Code: tag `round1-P03`. One launcher: `scripts/round1.sh prepare | pilot | full | status`; results in `results/round1`
(pilot: `results/round1_pilot`), new data in `data/tasks/{esci_us_trainprobe,wands,esci_us_retrieval}` and
`data/tasks/esci_us/train_full.jsonl`; `results/main` and the main test files are only read.

## Conditions
| Id | What | Systems | Data |
|---|---|---|---|
| H | Single-pass readout: next-token distribution over the K option digits, decoding restricted to those digits, processed logprobs | Qwen3.8 FP8, Gemma-4 (`*_readout1`) | every main condition (US 3 designs + order variants, ES/JP four-way), ES/JP ordinal + yes/no, new tasks |
| G | Qwen3-Reranker 4B and 8B x instruction {default (model card), ecommerce (main run), esci (ecommerce + ESCI exact-match definition)} | `qwen_reranker` (= 4B ecommerce), `qwen_reranker{4b,8b}_{default,ecommerce,esci}` | ESCI US/ES/JP, new tasks |
| I | ModernBERT label budget: N = 5k, 20k, 50k (nested subsets of the 50k pairs) x seed offsets 0-2, and the full US reduced train split (1 seed); same recipe | `modernbert` (50k, s0), `modernbert_n{N}_s{k}` | US test, probe, WANDS, retrieval |
| J | Ordinal and yes/no designs in Spanish and Japanese | Jev, Clef, Clef-flash, Perplexity local + API, readouts (single pass) | ES/JP |
| K1 | WANDS (Wayfair, MIT): 480 queries, <= 30 judged products per query (seeded); four-way ESCI question unchanged | every system (generative LLMs included) | `wands` |
| K2 | ESCI train-split exposure probe: 300 US train queries drawn from the queries of the 50k supervised training file, all their pairs | every system | `esci_us_trainprobe` |
| L | Retrieve-then-rerank: pool = all distinct products of the 1,000 US test queries; first stage BM25 (language-aware English analyser) top-40; unjudged = irrelevant | decision models, readouts, rerankers, supervised references | `esci_us_retrieval` |
| D | Hosted-API drift control: US robustness subset (1,049 pairs), four-way, sent again | Jev, Perplexity API | `esci_us/repeat1` |

## Analysis (fixed now)
- nDCG@10 with ESCI gains (E 1, S 0.1, C 0.01, I 0); WANDS: Exact 1, Partial 0.1, Irrelevant 0 (sensitivity: Partial 0.5).
- Query-level paired bootstrap (10,000 draws, seed 20261010), two-sided p floored at 1e-4, Holm across the systems of
  one table, per market x reference x subset (as in the main analysis).
- BM25: two analysers, `word` (pre-specified) and `language` (EN/ES: accent folding + Snowball; JA: CJK character
  bigrams). Japanese tokenisation ablation: word, character unigrams, character bigrams, MeCab (unidic-lite).
- Reranker reference for "vs reranker" tests: the best of the six reranker configurations in each market (by nDCG@10
  on all queries), i.e. the configuration least favourable to the other systems.
- Readouts: the single-pass readout replaces the digit-trie readout in all tables (quality and cost); agreement between
  the two is reported in the notes (argmax agreement, mean total variation).
- Label budget: nDCG@10 and accuracy per N (mean and range over seeds).
- Probe: per system, nDCG@10 and four-way accuracy on train-probe queries minus the same on the 1,000 test queries,
  with a two-sample query bootstrap CI. Positive control: supervised cross-encoders (trained on these queries).
- Retrieval: Exact recall@40 of the first stage; nDCG@10 after reranking with the ideal ranking over all judged
  products of the query (unjudged = irrelevant), condensed nDCG@10 (unjudged removed) as a sensitivity; judged@10.
  All 1,000 queries are scored: 7 queries share no token with any pooled product (misspellings, model numbers), the
  first stage returns nothing and they count as nDCG 0 (MRR 0) for every system.
- Order stability (main data): query-cluster bootstrap 95% CIs for flip rates; paired difference order variant minus
  identical resend (Jev).
