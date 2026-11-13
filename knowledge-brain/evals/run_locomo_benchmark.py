#!/usr/bin/env python3
"""Live LoCoMo Benchmark Runner: TraceZero vs. OpenViking vs. Mem0.

Evaluates the official `snap-research/locomo` (`locomo10.json`) benchmark across
three real, on-disk memory engines:
  1. TraceZero (`knowledge-brain` `MemoryStore` + `ingest_message` + `recall_context` + `forget`)
  2. OpenViking (`openviking.service.core.OpenVikingService` with local FastEmbed `/v1/embeddings`)
  3. Mem0 (`mem0.Memory` backed by embedded Qdrant + FastEmbed + SQLite history)

No hardcoded or clamped scores: every accuracy, latency, token count, and
byte-level deletion figure in the output is directly measured from execution.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import socket
import statistics
import string
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure knowledge-brain root is importable regardless of cwd
KB_ROOT = Path(__file__).resolve().parent.parent
if str(KB_ROOT) not in sys.path:
    sys.path.insert(0, str(KB_ROOT))

# Disable Mem0 telemetry and prefer offline cached FastEmbed assets when available
os.environ.setdefault("MEM0_TELEMETRY", "false")
if Path("/tmp/fastembed_cache/models--Qdrant--bm25").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Force IPv4 DNS resolution in case HF Hub is contacted on networks with stalled IPv6
_ORIG_GETADDRINFO = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

from core.ingest import ingest_message  # noqa: E402
from core.retrieval import recall_context  # noqa: E402
from core.storage_api import MemoryStore  # noqa: E402
from core.verifier import scan_shard_bytes  # noqa: E402
from evals.token_utils import count_tokens  # noqa: E402

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CACHE_PATH = Path(__file__).resolve().parent / "data" / "locomo10.json"
RESULTS_MD_PATH = Path(__file__).resolve().parent / "results" / "locomo_comparison.md"
RESULTS_JSON_PATH = Path(__file__).resolve().parent / "results" / "locomo_comparison.json"

SECRET_MARKER = "LoCoMoVaultSecret#9921"
SECRET_TEXT = (
    f"Confidential break-glass vault credential marker is {SECRET_MARKER} for emergency audit."
)

CATEGORY_NAMES = {
    1: "Single-Hop",
    2: "Temporal",
    3: "Multi-Hop",
    4: "Open-Domain",
    5: "Adversarial",
}

_ARTICLES = {"a", "an", "the"}


def load_locomo_dataset() -> List[Dict[str, Any]]:
    """Load official locomo10.json, downloading to evals/data/ if missing."""
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOCOMO_URL, CACHE_PATH)
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def normalize_text(text: Any) -> str:
    """Standard SQuAD/LoCoMo normalization: lowercase, strip punctuation & articles."""
    s = str(text).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = [tok for tok in s.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def token_f1(pred_span: str, gold: str) -> float:
    """Compute token-level F1 between two short text strings."""
    pred_toks = normalize_text(pred_span).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return 1.0 if pred_toks == gold_toks else 0.0
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common == 0:
        return 0.0
    prec = common / len(pred_toks)
    rec = common / len(gold_toks)
    return (2.0 * prec * rec) / (prec + rec)


def score_retrieved_context(retrieved: str, gold_answer: Any) -> Dict[str, float]:
    """Score retrieved memory text against gold answer without external LLM calls.

    Computes:
      - token_recall: fraction of unique normalized gold tokens present in retrieved context.
      - best_line_f1: max token-F1 across lines/sentences in retrieved context vs gold answer.
      - substring_hit: 1.0 if full normalized gold answer appears as substring, else 0.0.
    """
    gold_str = str(gold_answer or "").strip()
    if not gold_str or not retrieved.strip():
        return {"token_recall": 0.0, "best_line_f1": 0.0, "substring_hit": 0.0}

    norm_ret = normalize_text(retrieved)
    norm_gold = normalize_text(gold_str)
    if not norm_gold:
        return {"token_recall": 0.0, "best_line_f1": 0.0, "substring_hit": 0.0}

    ret_toks = set(norm_ret.split())
    gold_toks = norm_gold.split()
    matched = sum(1 for tok in gold_toks if tok in ret_toks)
    tok_recall = matched / len(gold_toks)

    sub_hit = 1.0 if norm_gold in norm_ret else 0.0

    # Split retrieved context into candidate sentences/clauses for span-level F1
    spans = [
        seg.strip()
        for seg in re.split(r"[\n.!?;]+", retrieved)
        if seg.strip()
    ]
    best_f1 = max((token_f1(sp, gold_str) for sp in spans), default=0.0)
    return {
        "token_recall": tok_recall,
        "best_line_f1": best_f1,
        "substring_hit": sub_hit,
    }


def scan_dir_bytes(root_dir: Path, forbidden: str) -> int:
    """Scan every file under root_dir at raw byte level (case-insensitive)."""
    needle = forbidden.encode("utf-8").lower()
    matches = 0
    for path in root_dir.rglob("*"):
        if path.is_file():
            try:
                raw = path.read_bytes().lower()
                matches += raw.count(needle)
            except OSError:
                continue
    return matches


def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    """Extract ordered (session_key, date_time_str, turns) tuples from a LoCoMo conversation."""
    indices = []
    for k in conv_obj:
        m = re.fullmatch(r"session_(\d+)", k)
        if m:
            indices.append(int(m.group(1)))
    sessions = []
    for idx in sorted(indices):
        s_key = f"session_{idx}"
        dt_str = str(conv_obj.get(f"{s_key}_date_time", ""))
        turns = conv_obj.get(s_key, [])
        sessions.append((s_key, dt_str, turns))
    return sessions


def summarize_engine_records(
    name: str,
    records: List[Dict[str, Any]],
    ingest_secs: float,
    ingest_items: int,
    deletion_audit: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate per-QA records into category metrics, latency, tokens, and deletion audit."""
    cat_stats: Dict[int, Dict[str, Any]] = {}
    for cat_id in sorted(CATEGORY_NAMES):
        sub = [r for r in records if r["category"] == cat_id]
        if not sub:
            continue
        if cat_id == 5:
            abstain_rate = 100.0 * sum(1 for r in sub if r["abstained"]) / len(sub)
            adv_leak_rate = 100.0 * sum(1 for r in sub if r.get("adv_hit", 0.0) >= 0.5) / len(sub)
            cat_stats[cat_id] = {
                "name": CATEGORY_NAMES[cat_id],
                "count": len(sub),
                "abstention_rate_pct": round(abstain_rate, 2),
                "adversarial_leak_pct": round(adv_leak_rate, 2),
            }
        else:
            avg_recall = 100.0 * statistics.mean(r["token_recall"] for r in sub)
            avg_f1 = 100.0 * statistics.mean(r["best_line_f1"] for r in sub)
            avg_sub = 100.0 * statistics.mean(r["substring_hit"] for r in sub)
            cat_stats[cat_id] = {
                "name": CATEGORY_NAMES[cat_id],
                "count": len(sub),
                "token_recall_pct": round(avg_recall, 2),
                "best_line_f1_pct": round(avg_f1, 2),
                "substring_hit_pct": round(avg_sub, 2),
            }

    ans_records = [r for r in records if r["category"] in (1, 2, 3, 4)]
    latencies = [r["latency_ms"] for r in records]
    tok_counts = [r["context_tokens"] for r in records]

    return {
        "engine": name,
        "total_qas": len(records),
        "ingest_secs": round(ingest_secs, 2),
        "ingest_items": ingest_items,
        "overall_token_recall_pct": round(
            100.0 * statistics.mean(r["token_recall"] for r in ans_records), 2
        ) if ans_records else 0.0,
        "overall_best_line_f1_pct": round(
            100.0 * statistics.mean(r["best_line_f1"] for r in ans_records), 2
        ) if ans_records else 0.0,
        "overall_substring_hit_pct": round(
            100.0 * statistics.mean(r["substring_hit"] for r in ans_records), 2
        ) if ans_records else 0.0,
        "cat5_abstention_pct": cat_stats.get(5, {}).get("abstention_rate_pct", 0.0),
        "mean_context_tokens": round(statistics.mean(tok_counts), 1) if tok_counts else 0.0,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(
            sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 2
        ) if latencies else 0.0,
        "categories": {str(k): v for k, v in cat_stats.items()},
        "deletion_audit": deletion_audit,
    }


def eval_qa_item(
    qa: Dict[str, Any],
    retrieved_text: str,
    explicit_abstained: bool,
    latency_ms: float,
    tokens_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Score one LoCoMo QA item uniformly across engines."""
    cat = int(qa["category"])
    ctx_tokens = (
        tokens_override if tokens_override is not None else count_tokens(retrieved_text)
    )
    if cat == 5:
        adv_ans = qa.get("adversarial_answer", "")
        adv_scores = score_retrieved_context(retrieved_text, adv_ans)
        return {
            "category": cat,
            "question": qa["question"],
            "abstained": explicit_abstained,
            "adv_hit": adv_scores["token_recall"],
            "context_tokens": ctx_tokens,
            "latency_ms": latency_ms,
        }

    scores = score_retrieved_context(retrieved_text, qa.get("answer", ""))
    return {
        "category": cat,
        "question": qa["question"],
        "token_recall": scores["token_recall"],
        "best_line_f1": scores["best_line_f1"],
        "substring_hit": scores["substring_hit"],
        "abstained": explicit_abstained,
        "context_tokens": ctx_tokens,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# 1. TraceZero Engine
# ---------------------------------------------------------------------------
def run_tracezero_on_locomo(
    conv: Dict[str, Any], qa_list: List[Dict[str, Any]], token_budget: int = 500
) -> Dict[str, Any]:
    """Run real TraceZero MemoryStore + ingest_message + recall_context + forget."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="tz_locomo_"))
    ns = "locomo/conv0"
    try:
        store = MemoryStore(root_dir=str(tmp_dir))
        sessions = extract_sessions(conv)

        t_ing0 = time.perf_counter()
        turns_ingested = 0
        for s_key, dt_str, turns in sessions:
            for turn in turns:
                speaker = turn.get("speaker", "user")
                text = turn.get("text", "").strip()
                if not text:
                    continue
                payload = f"[{dt_str}] {speaker}: {text}" if dt_str else f"{speaker}: {text}"
                ingest_message(
                    store=store,
                    namespace=ns,
                    text=payload,
                    origin="user",
                    session_id=s_key,
                    subject=speaker,
                )
                turns_ingested += 1
        ingest_secs = time.perf_counter() - t_ing0

        records: List[Dict[str, Any]] = []
        for qa in qa_list:
            q_text = qa["question"]
            t0 = time.perf_counter()
            res = recall_context(store=store, namespace=ns, query=q_text, token_budget=token_budget)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            abstained = res.status == "abstained"
            retrieved_text = "" if abstained else res.packed_prompt
            records.append(
                eval_qa_item(
                    qa=qa,
                    retrieved_text=retrieved_text,
                    explicit_abstained=abstained,
                    latency_ms=lat_ms,
                    tokens_override=res.tokens_used,
                )
            )

        # Live Deletion & Byte-Level Audit
        ingest_message(
            store=store,
            namespace=ns,
            text=SECRET_TEXT,
            origin="user",
            session_id="audit_secret",
        )
        shard_dir = store.sm.get_shard_dir(ns)
        pre_matches = scan_shard_bytes(shard_dir, [SECRET_MARKER])
        receipt = store.forget(ns, SECRET_MARKER)
        post_matches = scan_shard_bytes(shard_dir, [SECRET_MARKER])
        post_recall = recall_context(store=store, namespace=ns, query=f"What is {SECRET_MARKER}?", token_budget=token_budget)

        deletion_audit = {
            "pre_delete_byte_matches": pre_matches,
            "post_delete_byte_matches": post_matches,
            "receipt_byte_scan_matches": receipt.byte_scan_matches,
            "post_delete_abstained": post_recall.status == "abstained",
            "byte_clean_pct": 100.0 if post_matches == 0 else 0.0,
        }
        return summarize_engine_records(
            name="TraceZero",
            records=records,
            ingest_secs=ingest_secs,
            ingest_items=turns_ingested,
            deletion_audit=deletion_audit,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Mem0 Engine (Real mem0ai + embedded Qdrant + FastEmbed + SQLite history)
# ---------------------------------------------------------------------------
def run_mem0_on_locomo(
    conv: Dict[str, Any], qa_list: List[Dict[str, Any]], top_k: int = 5
) -> Dict[str, Any]:
    """Run real mem0ai Memory with local FastEmbed + embedded Qdrant vector store."""
    from mem0 import Memory

    tmp_dir = Path(tempfile.mkdtemp(prefix="mem0_locomo_"))
    mem0_uid = "locomo_conv0"
    try:
        cfg = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "locomo_conv0",
                    "path": str(tmp_dir / "qdrant"),
                    "embedding_model_dims": 384,
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5",
                    "embedding_dims": 384,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini", "api_key": "sk-local"},
            },
            "history_db_path": str(tmp_dir / "history.db"),
        }
        mem = Memory.from_config(cfg)
        sessions = extract_sessions(conv)

        t_ing0 = time.perf_counter()
        turns_ingested = 0
        for s_key, dt_str, turns in sessions:
            for turn in turns:
                speaker = turn.get("speaker", "user")
                text = turn.get("text", "").strip()
                if not text:
                    continue
                payload = f"[{dt_str}] {speaker}: {text}" if dt_str else f"{speaker}: {text}"
                mem.add(payload, user_id=mem0_uid, metadata={"session": s_key}, infer=False)
                turns_ingested += 1
        ingest_secs = time.perf_counter() - t_ing0

        records: List[Dict[str, Any]] = []
        for qa in qa_list:
            q_text = qa["question"]
            t0 = time.perf_counter()
            search_res = mem.search(q_text, filters={"user_id": mem0_uid}, limit=top_k)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            hits = search_res.get("results", []) if isinstance(search_res, dict) else []
            memories = [str(h.get("memory", "")) for h in hits if h.get("memory")]
            abstained = len(memories) == 0
            retrieved_text = "\n".join(memories)
            records.append(
                eval_qa_item(
                    qa=qa,
                    retrieved_text=retrieved_text,
                    explicit_abstained=abstained,
                    latency_ms=lat_ms,
                )
            )

        # Live Deletion & Byte-Level Audit
        add_sec = mem.add(SECRET_TEXT, user_id=mem0_uid, infer=False)
        pre_matches = scan_dir_bytes(tmp_dir, SECRET_MARKER)
        sec_id = add_sec["results"][0]["id"]
        mem.delete(sec_id)
        post_matches = scan_dir_bytes(tmp_dir, SECRET_MARKER)
        post_search = mem.search(f"What is {SECRET_MARKER}?", filters={"user_id": mem0_uid}, limit=3)
        post_hits = post_search.get("results", []) if isinstance(post_search, dict) else []

        deletion_audit = {
            "pre_delete_byte_matches": pre_matches,
            "post_delete_byte_matches": post_matches,
            "post_delete_abstained": len(post_hits) == 0,
            "byte_clean_pct": 100.0 if post_matches == 0 else 0.0,
        }
        return summarize_engine_records(
            name="Mem0 (mem0ai)",
            records=records,
            ingest_secs=ingest_secs,
            ingest_items=turns_ingested,
            deletion_audit=deletion_audit,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. OpenViking Engine (Real OpenVikingService + local FastEmbed HTTP server)
# ---------------------------------------------------------------------------
class _LocalOpenAIServer:
    """In-process OpenAI-compatible HTTP endpoint for OpenViking embeddings & VLM."""

    def __init__(self):
        from fastembed import TextEmbedding

        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        model = self.embed_model

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path.endswith("/embeddings"):
                    inp = body.get("input", [])
                    if isinstance(inp, str):
                        inp = [inp]
                    vecs = [v.tolist() for v in model.embed(inp)]
                    resp = {
                        "object": "list",
                        "data": [
                            {"object": "embedding", "index": i, "embedding": v}
                            for i, v in enumerate(vecs)
                        ],
                        "model": body.get("model", "BAAI/bge-small-en-v1.5"),
                        "usage": {"prompt_tokens": len(inp), "total_tokens": len(inp)},
                    }
                else:
                    msgs = body.get("messages", [])
                    txt = msgs[-1].get("content", "") if msgs else ""
                    # Extract actual file/summary content after OpenViking template header if present
                    marker = "【File Content】"
                    if marker in txt:
                        after = txt.split(marker, 1)[1]
                        end_idx = after.find("Output requirements:")
                        content = after[:end_idx] if end_idx != -1 else after
                    else:
                        content = txt
                    summary = " ".join(content.strip().split())[:280] or "Session context summary."
                    resp = {
                        "id": "ov-local",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": summary},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                payload = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_LocalOpenAIServer":
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()


async def _run_openviking_async(
    conv: Dict[str, Any], qa_list: List[Dict[str, Any]], top_k: int = 5
) -> Dict[str, Any]:
    from openviking.server.identity import RequestContext, Role
    from openviking.service.core import OpenVikingService

    tmp_dir = Path(tempfile.mkdtemp(prefix="ov_locomo_"))
    ov_workspace = tmp_dir / "ov_data"
    src_dir = tmp_dir / "sessions"
    src_dir.mkdir(parents=True, exist_ok=True)
    prev_conf_env = os.environ.get("OPENVIKING_CONFIG_FILE")

    try:
        with _LocalOpenAIServer() as api:
            conf = {
                "embedding": {
                    "dense": {
                        "provider": "openai",
                        "model": "BAAI/bge-small-en-v1.5",
                        "api_key": "sk-local",
                        "api_base": f"http://127.0.0.1:{api.port}/v1",
                        "dimension": 384,
                    }
                },
                "vlm": {
                    "provider": "openai",
                    "model": "local-extractive",
                    "api_key": "sk-local",
                    "api_base": f"http://127.0.0.1:{api.port}/v1",
                },
                "storage": {
                    "workspace": str(ov_workspace),
                    "vectordb": {"backend": "local"},
                },
            }
            conf_path = tmp_dir / "ov.conf"
            conf_path.write_text(json.dumps(conf), encoding="utf-8")
            os.environ["OPENVIKING_CONFIG_FILE"] = str(conf_path)

            svc = OpenVikingService(path=str(ov_workspace))
            await svc.initialize()
            ctx = RequestContext(user=svc.user, role=Role.ROOT)
            try:
                sessions = extract_sessions(conv)
                t_ing0 = time.perf_counter()
                sessions_ingested = 0
                for s_key, dt_str, turns in sessions:
                    lines = [f"# {s_key} ({dt_str})"]
                    for turn in turns:
                        speaker = turn.get("speaker", "user")
                        text = turn.get("text", "").strip()
                        if text:
                            lines.append(f"- {speaker}: {text}")
                    s_file = src_dir / f"{s_key}.md"
                    s_file.write_text("\n".join(lines), encoding="utf-8")
                    await svc.resources.add_resource(
                        path=str(s_file), ctx=ctx, wait=True, summarize=False
                    )
                    sessions_ingested += 1
                ingest_secs = time.perf_counter() - t_ing0

                # Cache L2 file reads per URI so scoring stays fast while executing real fs.read
                uri_cache: Dict[str, str] = {}
                records: List[Dict[str, Any]] = []
                for qa in qa_list:
                    q_text = qa["question"]
                    t0 = time.perf_counter()
                    find_res = await svc.search.find(
                        query=q_text,
                        ctx=ctx,
                        target_uri="viking://resources",
                        limit=top_k,
                    )
                    chunks: List[str] = []
                    for hit in getattr(find_res, "resources", []) or []:
                        uri = getattr(hit, "uri", "")
                        lvl = getattr(hit, "level", 2)
                        if lvl == 2 and uri:
                            if uri not in uri_cache:
                                try:
                                    uri_cache[uri] = await svc.fs.read(uri=uri, ctx=ctx)
                                except Exception:
                                    uri_cache[uri] = getattr(hit, "abstract", "") or ""
                            if uri_cache[uri]:
                                chunks.append(uri_cache[uri])
                        elif getattr(hit, "abstract", None):
                            chunks.append(str(hit.abstract))
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    retrieved_text = "\n".join(chunks)
                    abstained = len(chunks) == 0
                    records.append(
                        eval_qa_item(
                            qa=qa,
                            retrieved_text=retrieved_text,
                            explicit_abstained=abstained,
                            latency_ms=lat_ms,
                        )
                    )

                # Live Deletion & Byte-Level Audit
                sec_file = src_dir / "audit_secret.md"
                sec_file.write_text(SECRET_TEXT, encoding="utf-8")
                add_sec = await svc.resources.add_resource(
                    path=str(sec_file), ctx=ctx, wait=True, summarize=False
                )
                pre_matches = scan_dir_bytes(ov_workspace, SECRET_MARKER)
                root_uri = add_sec.get("root_uri")
                if root_uri:
                    await svc.fs.rm(uri=root_uri, ctx=ctx, recursive=True, wait=True)
                post_find = await svc.search.find(
                    query=f"What is {SECRET_MARKER}?",
                    ctx=ctx,
                    target_uri="viking://resources",
                    limit=3,
                )
                await svc.close()
                post_matches = scan_dir_bytes(ov_workspace, SECRET_MARKER)

                post_hits = getattr(post_find, "resources", []) or []
                deletion_audit = {
                    "pre_delete_byte_matches": pre_matches,
                    "post_delete_byte_matches": post_matches,
                    "post_delete_abstained": len(post_hits) == 0,
                    "byte_clean_pct": 100.0 if post_matches == 0 else 0.0,
                }
                return summarize_engine_records(
                    name="OpenViking (openviking)",
                    records=records,
                    ingest_secs=ingest_secs,
                    ingest_items=sessions_ingested,
                    deletion_audit=deletion_audit,
                )
            except Exception:
                await svc.close()
                raise
    finally:
        if prev_conf_env is None:
            os.environ.pop("OPENVIKING_CONFIG_FILE", None)
        else:
            os.environ["OPENVIKING_CONFIG_FILE"] = prev_conf_env
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_openviking_on_locomo(
    conv: Dict[str, Any], qa_list: List[Dict[str, Any]], top_k: int = 5
) -> Dict[str, Any]:
    """Run real OpenVikingService benchmark synchronously."""
    return asyncio.run(_run_openviking_async(conv, qa_list, top_k=top_k))


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def render_markdown_report(
    sample_id: str,
    num_sessions: int,
    num_qas: int,
    results: List[Dict[str, Any]],
) -> str:
    """Format empirical results into a concise Markdown report."""
    lines = [
        "# Live LoCoMo Benchmark Comparison (TraceZero vs. OpenViking vs. Mem0)",
        "",
        f"- **Dataset**: Official `snap-research/locomo` (`locomo10.json`, sample `{sample_id}`)",
        f"- **Scope**: {num_sessions} multi-session dialogues, {num_qas} QA items across Categories 1–5",
        "- **Methodology**: 100% live local execution (no hardcoded/published baseline rows, no synthetic clamping).",
        "- **Scoring**: Deterministic retrieval evaluation (`tiktoken cl100k_base` context tokens, Token Overlap Recall, Best-Sentence Token F1, Category-5 native abstention rate, p50 query latency, and raw on-disk byte scan after deletion of `LoCoMoVaultSecret#9921`).",
        "",
        "## Overall Comparison",
        "",
        "| Engine | Cat 1-4 Token Recall (%) | Cat 1-4 Best-Line F1 (%) | Cat 5 Abstention (%) | Mean Context Tokens | p50 Latency (ms) | Post-Delete Residual Bytes | Byte-Clean (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        da = r["deletion_audit"]
        lines.append(
            f"| {r['engine']} | {r['overall_token_recall_pct']:.2f}% | "
            f"{r['overall_best_line_f1_pct']:.2f}% | {r['cat5_abstention_pct']:.2f}% | "
            f"{r['mean_context_tokens']:.1f} | {r['p50_latency_ms']:.2f} | "
            f"{da['post_delete_byte_matches']} | {da['byte_clean_pct']:.1f}% |"
        )

    lines.extend(["", "## Per-Category Breakdown", ""])
    lines.append(
        "| Engine | Single-Hop Recall (%) | Temporal Recall (%) | Multi-Hop Recall (%) | Open-Domain Recall (%) | Adversarial Abstention (%) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        cats = r["categories"]
        c1 = cats.get("1", {}).get("token_recall_pct", 0.0)
        c2 = cats.get("2", {}).get("token_recall_pct", 0.0)
        c3 = cats.get("3", {}).get("token_recall_pct", 0.0)
        c4 = cats.get("4", {}).get("token_recall_pct", 0.0)
        c5 = cats.get("5", {}).get("abstention_rate_pct", 0.0)
        lines.append(
            f"| {r['engine']} | {c1:.2f}% | {c2:.2f}% | {c3:.2f}% | {c4:.2f}% | {c5:.2f}% |"
        )

    lines.extend(["", "## Deletion & Forensic Audit Details", ""])
    for r in results:
        da = r["deletion_audit"]
        lines.append(
            f"- **{r['engine']}**: ingest={r['ingest_secs']:.2f}s ({r['ingest_items']} items) | "
            f"pre-delete secret byte matches={da['pre_delete_byte_matches']} -> "
            f"post-delete secret byte matches={da['post_delete_byte_matches']} | "
            f"post-delete abstained={da['post_delete_abstained']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live LoCoMo benchmark across TraceZero, OpenViking, and Mem0.")
    parser.add_argument("--conv-index", type=int, default=0, help="LoCoMo conversation index (0-9).")
    parser.add_argument("--max-qas", type=int, default=0, help="Optional cap on QA count (0 = all).")
    parser.add_argument("--token-budget", type=int, default=800, help="TraceZero recall token budget.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval limit for Mem0 and OpenViking.")
    args = parser.parse_args()

    dataset = load_locomo_dataset()
    sample = dataset[args.conv_index]
    conv = sample["conversation"]
    qa_list = sample["qa"] if args.max_qas <= 0 else sample["qa"][: args.max_qas]
    sessions = extract_sessions(conv)

    print(
        f"[LoCoMo] Sample={sample.get('sample_id', args.conv_index)} | "
        f"sessions={len(sessions)} | qas={len(qa_list)}",
        flush=True,
    )

    print("[1/3] Running TraceZero...", flush=True)
    tz_res = run_tracezero_on_locomo(conv, qa_list, token_budget=args.token_budget)
    print(f"  -> TraceZero done in {tz_res['ingest_secs']}s ingest | recall={tz_res['overall_token_recall_pct']}% | post-delete bytes={tz_res['deletion_audit']['post_delete_byte_matches']}", flush=True)

    print("[2/3] Running OpenViking (openviking)...", flush=True)
    ov_res = run_openviking_on_locomo(conv, qa_list, top_k=args.top_k)
    print(f"  -> OpenViking done in {ov_res['ingest_secs']}s ingest | recall={ov_res['overall_token_recall_pct']}% | post-delete bytes={ov_res['deletion_audit']['post_delete_byte_matches']}", flush=True)

    print("[3/3] Running Mem0 (mem0ai)...", flush=True)
    m0_res = run_mem0_on_locomo(conv, qa_list, top_k=args.top_k)
    print(f"  -> Mem0 done in {m0_res['ingest_secs']}s ingest | recall={m0_res['overall_token_recall_pct']}% | post-delete bytes={m0_res['deletion_audit']['post_delete_byte_matches']}", flush=True)

    results = [tz_res, ov_res, m0_res]
    payload = {
        "sample_id": sample.get("sample_id", str(args.conv_index)),
        "num_sessions": len(sessions),
        "num_qas": len(qa_list),
        "results": results,
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_text = render_markdown_report(
        sample_id=payload["sample_id"],
        num_sessions=len(sessions),
        num_qas=len(qa_list),
        results=results,
    )
    RESULTS_MD_PATH.write_text(md_text, encoding="utf-8")
    print("\n" + md_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
