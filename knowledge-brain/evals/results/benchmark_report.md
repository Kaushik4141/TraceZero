# TraceZero (Knowledge Brain): 5-Strategy Benchmark Report

- **Execution Mode**: `live`
- **Deterministic Seed**: `random.seed(42)`
- **Tokenizer**: `tiktoken` (`cl100k_base`)
- **Target Namespace**: `acme_corp/users/emp_042` (30 migration sessions: **17,539 full-history tokens**; 17,580 w/ profile)
- **Verified Deletion Cleanliness**: **100%** (`byte_scan_matches: 0` across `.sqlite`, `-wal`, `.okf.md`, `sources/*.txt`) vs. **0%** (`Naive RAG / Prompt-stuffed`)

## 5-Strategy x 5-Question-Type Evaluation Matrix

| Storage Strategy | Preference (%) | Multi-hop (%) | What Changed (%) | Count/Aggregate (%) | Task State (%) | Overall Acc (%) | Avg Tokens (`tiktoken`) | p50 (ms) | p95 (ms) | Byte-Clean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt-stuffed** | 67.5% | 60.0% | 60.0% | 62.5% | 60.0% | **62.0%** | 17,600 | 0.03 | 0.04 | 0% |
| **Pure Vector (sqlite-vec)** | 92.5% | 91.7% | 55.0% | 62.5% | 60.0% | **72.3%** | 732 | 40.27 | 41.67 | 0% |
| **Claims / Knowledge Graph** | 92.5% | 100.0% | 75.0% | 85.0% | 100.0% | **90.5%** | 360 | 1.92 | 2.50 | 0% |
| **OKF Files (.okf.md)** | 92.5% | 85.0% | 60.0% | 80.0% | 85.0% | **80.5%** | 1,195 | 0.10 | 0.12 | 0% |
| **SQL / Code (run_readonly_sql)** | 100.0% | 91.7% | 100.0% | 100.0% | 100.0% | **98.3%** | 150 | 0.48 | 0.58 | 0% |
| **Hybrid Scoped Engine** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** | 353 | 49.88 | 51.94 | 100% |

## Live TokenLedger Accounting Summary (10 Queries / Strategy)

| Strategy | Calls | Total Context Tokens | Avg Context Tokens | Total Prompt Tokens |
| :--- | :---: | :---: | :---: | :---: |
| **Prompt-stuffed** | 10 | 176,000 | 17,600.0 | 176,000 |
| **Pure Vector (sqlite-vec)** | 10 | 7,325 | 732.5 | 7,325 |
| **Claims / Knowledge Graph** | 10 | 3,596 | 359.6 | 3,596 |
| **OKF Files (.okf.md)** | 10 | 11,953 | 1,195.3 | 11,953 |
| **SQL / Code (run_readonly_sql)** | 10 | 1,504 | 150.4 | 1,504 |
| **Hybrid Scoped Engine** | 10 | 3,527 | 352.7 | 3,527 |

## Key Measured Findings

1. **Token Savings**: `Hybrid Scoped Engine` averages **353 tokens/query** (strictly `<= 800` budget) vs. **17,539 full-history tokens** (**97.99% reduction**).
2. **Low Tail Latency**: `Hybrid Scoped Engine` achieves **p50 = 49.88ms**, **p95 = 51.94ms**.
3. **Single-Strategy Gaps Exposed**: `OKF Files` misses historical updates (`What Changed`), `Pure Vector` leaks stale/disputed/redundant chunks without status filtering, `Prompt-stuffed` blows context budgets and leaks poisoned injections, while `Hybrid Scoped Engine` dynamically routes across SQL, Recursive CTE, OKF, and Vector.

## Full-History vs. Naive Top-K Vector RAG vs. TraceZero (Knowledge Brain)

| System / Baseline | Avg Tokens / Query | Token Savings vs 30-Session History | p50 / p95 Latency (ms) | Contradiction Acc (%) | Poison Resistance (%) | Abstention Precision (%) | Verified Byte-Clean Deletion (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full-History Prompt Stuffing (30 Sessions)** | 17,539 | 0.0% | 0.01 / 0.02 | 0.0% | 0.0% | 0.0% | 0.0% |
| **Plain Top-K Vector RAG / OpenMemory Baseline** | 273.7 | 98.4% | 49.13 / 49.30 | 0.0% | 0.0% | 0.0% | 0.0% |
| **TraceZero (Knowledge Brain)** | **244.2** | **98.61%** | **7.35 / 55.10** | **100.0%** | **100.0%** | **100.0%** | **100.0% (`bytes=0`)** |
