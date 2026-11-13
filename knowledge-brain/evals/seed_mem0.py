#!/usr/bin/env python3
"""Companion Mem0 Seeder for Live Side-by-Side Antigravity IDE Demo.

Wipes ~/.mem0/ (or custom --storage-dir) and feeds
`knowledge-brain/evals/data/enterprise_shared_feed.jsonl` into Mem0
(`from mem0 import Memory` or `MemoryClient`) using `infer=False` bulk fast-mode
by default (<10s for all 100 employees + 30 sessions), with `--infer` support.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_FEED_PATH = Path(__file__).resolve().parent / "data" / "enterprise_shared_feed.jsonl"
DEFAULT_MEM0_DIR = Path.home() / ".mem0"


def wipe_mem0_storage(storage_dir: Path) -> None:
    """Cleanly remove existing ~/.mem0 and /tmp/qdrant state before seeding."""
    for target in (storage_dir, Path("/tmp/qdrant")):
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    storage_dir.mkdir(parents=True, exist_ok=True)


def init_mem0_instance(storage_dir: Path, use_infer: bool = False) -> Any:
    """Initialize Mem0 MemoryClient (if MEM0_API_KEY set) or local Mem0 Memory."""
    api_key = os.environ.get("MEM0_API_KEY")
    if api_key:
        from mem0 import MemoryClient

        return MemoryClient(api_key=api_key)

    from mem0 import Memory

    os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
    qdrant_path = str(storage_dir / "qdrant")
    history_db_path = str(storage_dir / "history.db")

    config: Dict[str, Any] = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_enterprise_demo",
                "embedding_model_dims": 384,
                "path": qdrant_path,
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {
                "model": "BAAI/bge-small-en-v1.5",
            },
        },
        "history_db_path": history_db_path,
    }
    if not use_infer and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-local-fast-mode-no-llm"

    return Memory.from_config(config)


def seed_mem0_from_jsonl(
    feed_path: Path = DEFAULT_FEED_PATH,
    storage_dir: Path = DEFAULT_MEM0_DIR,
    infer: bool = False,
    include_filler: bool = False,
    flat_user_id: str | None = None,
) -> Dict[str, Any]:
    """Wipe Mem0 state and bulk-ingest enterprise_shared_feed.jsonl."""
    if not feed_path.exists():
        from evals.seed_30_sessions import export_shared_feed, generate_dataset

        ds = generate_dataset()
        export_shared_feed(ds, feed_path)

    wipe_mem0_storage(storage_dir)
    mem = init_mem0_instance(storage_dir, use_infer=infer)

    records: List[Dict[str, Any]] = []
    with open(feed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not include_filler and rec.get("origin") == "tool" and "Terraform state verification" not in rec.get("text", ""):
                continue
            records.append(rec)

    t0 = time.perf_counter()
    ingested_count = 0
    for rec in records:
        uid = flat_user_id or rec["user_id"]
        meta = {
            "namespace": rec["namespace"],
            "employee_id": rec["user_id"],
            "project": rec.get("project", "orion_payments"),
            "session_id": rec.get("session_id", "session_00_profile"),
            "origin": rec.get("origin", "user"),
        }
        mem.add(
            rec["text"],
            user_id=uid,
            metadata=meta,
            infer=infer,
        )
        ingested_count += 1

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "feed_path": str(feed_path),
        "storage_dir": str(storage_dir),
        "infer": infer,
        "records_ingested": ingested_count,
        "elapsed_seconds": elapsed,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe ~/.mem0 and seed enterprise_shared_feed.jsonl into Mem0.")
    parser.add_argument(
        "--feed",
        type=Path,
        default=DEFAULT_FEED_PATH,
        help="Path to enterprise_shared_feed.jsonl",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_MEM0_DIR,
        help="Mem0 local storage directory to wipe and seed (default: ~/.mem0)",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        default=False,
        help="Enable LLM inference during Mem0 ingestion (default: False for <10s fast bulk seed)",
    )
    parser.add_argument(
        "--include-filler",
        action="store_true",
        default=False,
        help="Also ingest verbose tool telemetry filler paragraphs into Mem0",
    )
    parser.add_argument(
        "--flat-user-id",
        type=str,
        default=None,
        help="Optional single shared user_id to demonstrate flat cross-employee collision in Mem0",
    )
    args = parser.parse_args()

    summary = seed_mem0_from_jsonl(
        feed_path=args.feed,
        storage_dir=args.storage_dir,
        infer=args.infer,
        include_filler=args.include_filler,
        flat_user_id=args.flat_user_id,
    )
    print(
        f"[seed_mem0] Seeded {summary['records_ingested']} records from {summary['feed_path']}\n"
        f"  - Storage dir: {summary['storage_dir']} (wiped clean first)\n"
        f"  - Mode: infer={summary['infer']}\n"
        f"  - Elapsed: {summary['elapsed_seconds']}s"
    )


if __name__ == "__main__":
    main()
