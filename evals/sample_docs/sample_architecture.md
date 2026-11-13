# TraceZero Knowledge Brain Architecture

The Orion Payments service uses PostgreSQL as its primary database and runs in `us-west-2`.
Employee `emp_042` is a Staff Systems Engineer based in Seattle and assigned to the `payments_core` team.

## Key Design Decisions
- Use SQLite WAL mode with `PRAGMA secure_delete = ON` for per-shard control & lineage.
- Store 384-dim `BAAI/bge-small-en-v1.5` embeddings in `sqlite-vec` (`vec_claims`).
- Enforce cryptographic zero-byte forgetting via `forget_target()` and HMAC suppression gates.
