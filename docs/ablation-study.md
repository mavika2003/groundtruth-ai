# GroundTruth AI: Retrieval Ablation Study

**Status:** Measured run. Generated on 2026-11-14. Re-run `scripts/run_ablation.sh` to reproduce.

## Research question

Does BM25 fusion and cross-encoder reranking improve grounded-answer quality enough to justify their latency and cost versus dense-only retrieval, while preserving document-level access control?

## Controlled setup

- Corpus: `groundtruth.benchmark seed`, 30 policy facts plus conflicting and restricted companion documents.
- Query sets: **synthetic** (150 template-derived cases) and **human_reviewed** (41 natural-language cases in `groundtruth/human_eval.py`).
- Identity: employee principal, no access to `admin` documents.
- Modes: dense; BM25; hybrid reciprocal-rank fusion; hybrid plus cross-encoder reranking.

### Synthetic benchmark (150 cases)

| Metric | Dense | BM25 | Hybrid | Hybrid + rerank |
| --- | ---: | ---: | ---: | ---: |
| Recall@5 | 100.0% | 100.0% | 100.0% | 100.0% |
| MRR | 0.942 | 0.900 | 0.942 | 0.557 |
| Citation ID validity | 100.0% | 100.0% | 100.0% | 100.0% |
| Answer proxy accuracy | 82.0% | 91.3% | 82.7% | 82.7% |
| p95 latency (ms) | 5758 | 6005 | 47827 | 41849 |
| Cost / successful case (USD) | $0.00008 | $0.00007 | $0.00008 | $0.00007 |
| Permission leaks | 0 | 0 | 0 | 0 |

### Human-reviewed benchmark (41 cases)

| Metric | Dense | BM25 | Hybrid | Hybrid + rerank |
| --- | ---: | ---: | ---: | ---: |
| Recall@5 | 100.0% | 97.0% | 100.0% | 100.0% |
| MRR | 1.000 | 0.912 | 0.985 | 0.985 |
| Citation ID validity | 93.9% | 87.9% | 93.9% | 93.9% |
| Answer proxy accuracy | 90.2% | 87.8% | 90.2% | 90.2% |
| p95 latency (ms) | 5867 | 5734 | 41784 | 41871 |
| Cost / successful case (USD) | $0.00007 | $0.00007 | $0.00007 | $0.00007 |
| Permission leaks | 0 | 0 | 0 | 0 |

## How to read the comparison

- **Recall@5 / MRR** are retrieval-only signals on answerable cases.
- **Citation ID validity** checks cited chunk IDs were in the retrieved set, not semantic entailment.
- **Answer proxy accuracy** uses expected-string containment plus citations; treat as a lower bound.
- **Permission leaks** should remain **0** on employee runs.

## Failure analysis

Load `benchmark-results/*.json` in the Evaluation ablation explorer. Compare dense versus hybrid_rerank case by case.

## Interpretation limits

Synthetic cases stress reproducibility; human-reviewed cases stress natural phrasing. Neither substitutes for a licensed production corpus. Ragas metrics are optional via `groundtruth.ragas_eval`.
