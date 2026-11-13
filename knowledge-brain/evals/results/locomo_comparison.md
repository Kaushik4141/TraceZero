# Live LoCoMo Benchmark Comparison (TraceZero vs. OpenViking vs. Mem0)

- **Dataset**: Official `snap-research/locomo` (`locomo10.json`, sample `conv-26`)
- **Scope**: 19 multi-session dialogues, 199 QA items across Categories 1–5
- **Methodology**: 100% live local execution (no hardcoded/published baseline rows, no synthetic clamping).
- **Scoring**: Deterministic retrieval evaluation (`tiktoken cl100k_base` context tokens, Token Overlap Recall, Best-Sentence Token F1, Category-5 native abstention rate, p50 query latency, and raw on-disk byte scan after deletion of `LoCoMoVaultSecret#9921`).

## Overall Comparison

| Engine | Cat 1-4 Token Recall (%) | Cat 1-4 Best-Line F1 (%) | Cat 5 Abstention (%) | Mean Context Tokens | p50 Latency (ms) | Post-Delete Residual Bytes | Byte-Clean (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TraceZero | 45.17% | 20.28% | 65.96% | 379.4 | 105.74 | 0 | 100.0% |
| OpenViking (openviking) | 62.07% | 28.85% | 0.00% | 1894.0 | 8.36 | 3 | 0.0% |
| Mem0 (mem0ai) | 65.29% | 27.95% | 0.00% | 940.3 | 16.83 | 4 | 0.0% |

## Per-Category Breakdown

| Engine | Single-Hop Recall (%) | Temporal Recall (%) | Multi-Hop Recall (%) | Open-Domain Recall (%) | Adversarial Abstention (%) |
|---|---:|---:|---:|---:|---:|
| TraceZero | 40.18% | 52.43% | 11.54% | 49.86% | 65.96% |
| OpenViking (openviking) | 56.37% | 58.20% | 37.89% | 71.22% | 0.00% |
| Mem0 (mem0ai) | 56.92% | 71.08% | 32.37% | 72.16% | 0.00% |

## Deletion & Forensic Audit Details

- **TraceZero**: ingest=13.74s (419 items) | pre-delete secret byte matches=4 -> post-delete secret byte matches=0 | post-delete abstained=True
- **OpenViking (openviking)**: ingest=28.00s (19 items) | pre-delete secret byte matches=3 -> post-delete secret byte matches=3 | post-delete abstained=False
- **Mem0 (mem0ai)**: ingest=3.73s (419 items) | pre-delete secret byte matches=3 -> post-delete secret byte matches=4 | post-delete abstained=False
