# Live LoCoMo Benchmark Comparison (TraceZero vs. OpenViking vs. Mem0)

- **Dataset**: Official `snap-research/locomo` (`locomo10.json`, sample `conv-26`)
- **Scope**: 19 multi-session dialogues, 199 QA items across Categories 1–5
- **Methodology**: 100% live local execution (no hardcoded/published baseline rows, no synthetic clamping).
- **Scoring**: Deterministic retrieval evaluation (`tiktoken cl100k_base` context tokens, Token Overlap Recall, Best-Sentence Token F1, Category-5 native abstention rate, p50 query latency, and raw on-disk byte scan after deletion of `LoCoMoVaultSecret#9921`).

## Overall Comparison

| Engine | Cat 1-4 Token Recall (%) | Cat 1-4 Best-Line F1 (%) | Cat 5 Abstention (%) | All-Cat Score (1-5 %) | Mean Context Tokens | Recall / 1K Tok | p50 Latency (ms) | Post-Delete Residual Bytes | Byte-Clean (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TraceZero (Strict PS-06 Budget=800) | 61.08% | 25.78% | 59.57% | 60.72% | 621.5 | 98.3 | 192.38 | 0 | 100.0% |
| TraceZero (Mid Budget=2000, ~OpenViking) | 69.22% | 28.93% | 59.57% | 66.94% | 1553.0 | 44.6 | 423.20 | 0 | 100.0% |
| TraceZero (High Budget=7000, ~Mem0 Research) | 75.51% | 31.62% | 59.57% | 71.75% | 5486.1 | 13.8 | 864.56 | 0 | 100.0% |
| OpenViking (openviking) | 62.07% | 28.85% | 0.00% | 47.41% | 1894.0 | 32.8 | 8.44 | 3 | 0.0% |
| Mem0 (mem0ai) | 65.29% | 27.95% | 0.00% | 49.87% | 940.3 | 69.4 | 15.05 | 4 | 0.0% |

## Per-Category Breakdown

| Engine | Single-Hop Recall (%) | Temporal Recall (%) | Multi-Hop Recall (%) | Open-Domain Recall (%) | Adversarial Abstention (%) |
|---|---:|---:|---:|---:|---:|
| TraceZero (Strict PS-06 Budget=800) | 48.19% | 65.50% | 20.94% | 72.08% | 59.57% |
| TraceZero (Mid Budget=2000, ~OpenViking) | 67.15% | 68.74% | 30.72% | 77.58% | 59.57% |
| TraceZero (High Budget=7000, ~Mem0 Research) | 82.71% | 72.07% | 40.51% | 80.54% | 59.57% |
| OpenViking (openviking) | 56.37% | 58.20% | 37.89% | 71.22% | 0.00% |
| Mem0 (mem0ai) | 56.92% | 71.08% | 32.37% | 72.16% | 0.00% |

## Deletion & Forensic Audit Details

- **TraceZero (Strict PS-06 Budget=800)**: ingest=15.11s (419 items) | pre-delete secret byte matches=4 -> post-delete secret byte matches=0 | post-delete abstained=True
- **TraceZero (Mid Budget=2000, ~OpenViking)**: ingest=19.93s (419 items) | pre-delete secret byte matches=4 -> post-delete secret byte matches=0 | post-delete abstained=True
- **TraceZero (High Budget=7000, ~Mem0 Research)**: ingest=11.67s (419 items) | pre-delete secret byte matches=4 -> post-delete secret byte matches=0 | post-delete abstained=True
- **OpenViking (openviking)**: ingest=28.83s (19 items) | pre-delete secret byte matches=3 -> post-delete secret byte matches=3 | post-delete abstained=False
- **Mem0 (mem0ai)**: ingest=8.64s (419 items) | pre-delete secret byte matches=3 -> post-delete secret byte matches=4 | post-delete abstained=False
