# Analysis plan: main run

Committed with the frozen code and the SHA-256 manifest of the test files before the main run: tag `prereg-P03`, commit `99cbd24c53be81055efea24d732613d96813cea7`, 12 October 2026, 10:03 (UTC+3). The sections on data, method, metrics and statistics are reproduced verbatim from the committed plan; project logistics (schedule, budget, local test steps) are omitted.

## 5. Data
Amazon Shopping Queries Dataset (ESCI, Apache-2.0; loaded from the HF mirror `tasksource/esci`, duplicate rows removed), reduced version (Task 1), test split, US English. A seeded sample of **1,000 queries (~20,000 query–product pairs)**. Product text = title + brand + color + first 3 bullet points (truncated to 300 tokens). Included: 200 Spanish + 200 Japanese queries (choice design), to test other e-commerce markets.

**"Hard" subset:** queries with at least one Exact and at least one Irrelevant product, so the random baseline is not inflated by all-Exact queries (a limitation noted in the Elastic blog).

## 6. Method
1. Each pair is one request: state = query + product text; questions = `choice` (E/S/C/I with definitions from the ESCI paper), `score` (4 levels), `noul` (exact match). Where supported, all three are asked in one request.
2. Ranking policies: expected gain Σ p(label)·gain(label); score expectation; P(exact).
3. Qwen3-Reranker scores pairs directly. LLM readouts use the same `choice` question.
4. Measure latency per query (batch of all candidates).

## 7. Metrics and statistics
nDCG@10 with ESCI gains (E = 1.0, S = 0.1, C = 0.01, I = 0), MRR of the first Exact, macro-F1 on E/S/C/I, ECE of the `choice` probabilities, latency, $ per 1,000 queries. Query-level paired bootstrap; Holm across systems.
