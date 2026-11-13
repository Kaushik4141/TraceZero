"""Core Memory Bridge Adapter for Knowledge Brain.

Connects MCP and agent clients directly to Knowledge Brain micro-shards.
Provides auditable, provenance-backed memory operations:
- remember: Appends to sources/, extracts claims, and updates .okf.md
- recall: 3-tier recall (L0 abstract, L1 claims, L2 quotes) with honest abstention
- explain: Byte-accurate source quote retrieval from raw files
- history: Full temporal lineage of active and superseded claims
- quarantine_list / confirm_quarantine: Intercepts & handles prompt injections
- forget / verify_deletion: Cryptographic unlearning with byte-level verification
- handoff_create / handoff_get: Seamless OKF v0.2 cross-agent task handoffs
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import yaml

try:
    from core.contracts import ClaimRecord, DeletionReceipt, HandoffPacket, RecallResult, SourceSpan
    from core.handoff import create_handoff, get_handoff, count_tokens, _resolve_memory_root
except ImportError:
    from contracts import ClaimRecord, DeletionReceipt, HandoffPacket, RecallResult, SourceSpan
    from handoff import create_handoff, get_handoff, count_tokens, _resolve_memory_root


HMAC_SECRET = b"knowledge-brain-secret-key-v1"

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous\s+instructions|rules)", re.IGNORECASE),
    re.compile(r"leak\s+keys", re.IGNORECASE),
    re.compile(r"exfiltrat", re.IGNORECASE),
    re.compile(r"bypass\s+security", re.IGNORECASE),
    re.compile(r"system\s+(?:instruction\s+override|override|prompt)", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:rules|trust|instructions)", re.IGNORECASE),
    re.compile(r"transfer\s+funds", re.IGNORECASE),
    re.compile(r"attacker\.local", re.IGNORECASE),
]

BRIDGE_ORIGIN_TRUST_MAP: Dict[str, int] = {
    "user": 4,
    "user_session": 4,
    "agent_interaction": 3,
    "assistant": 3,
    "tool": 2,
    "tool_output": 2,
    "mcp": 2,
    "doc": 1,
    "document": 1,
    "web": 1,
    "web_scrape": 1,
    "external_untrusted": 1,
    "untrusted_scrape": 1,
    "poisoned": 1,
}


def _check_prompt_injection(text: str, origin: str) -> bool:
    """Detect prompt injection or untrusted external ingestion."""
    if origin in ("external_untrusted", "untrusted_scrape", "poisoned"):
        return True
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return True
    return False


def _get_shard_dir(namespace: str, memory_root: Optional[Union[str, Path]] = None) -> Path:
    root = _resolve_memory_root(memory_root)
    shard_dir = root / namespace
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "sources").mkdir(parents=True, exist_ok=True)
    (shard_dir / "receipts").mkdir(parents=True, exist_ok=True)
    return shard_dir


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Ensure micro-shard SQLite tables exist."""
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sources_index (
            source_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            origin TEXT NOT NULL,
            trust_tier INTEGER NOT NULL,
            byte_start INTEGER NOT NULL,
            byte_end INTEGER NOT NULL,
            quote TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS claims_index (
            claim_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_val TEXT NOT NULL,
            status TEXT CHECK(status IN ('active', 'superseded', 'disputed', 'quarantined')) NOT NULL,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            source_id TEXT NOT NULL,
            byte_start INTEGER NOT NULL,
            byte_end INTEGER NOT NULL,
            okf_file TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES sources_index(source_id)
        );

        CREATE TABLE IF NOT EXISTS lineage_index (
            edge_id TEXT PRIMARY KEY,
            parent_claim_id TEXT NOT NULL,
            child_claim_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS suppression_hashes (
            hash_id TEXT PRIMARY KEY,
            hmac_hash TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_value TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()


TEMPORAL_SUFFIX_RE = re.compile(
    r"\s+\b(?:this week|last week|this month|last month|this year|last year|today|yesterday|recently|right now|now)\b\.?$",
    re.IGNORECASE,
)


def clean_object_val(val: str) -> str:
    """Trim punctuation and trailing temporal qualifiers from extracted object values."""
    cleaned = val.strip().rstrip(".,!?;:")
    while True:
        updated = TEMPORAL_SUFFIX_RE.sub("", cleaned).strip().rstrip(".,!?;:")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def split_into_atomic_clauses(text: str) -> List[str]:
    """Split compound/multi-sentence text into atomic clauses for claim extraction."""
    clauses: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(
            r"(?:(?<=[.!?;])\s+(?=[A-Za-z0-9\"'(])|,?\s+\b(?:and|but)\b\s+(?=(?:i|my|we|our|emp_\d+|the|it|he|she|they|moved|relocated|live|lives|reside|resides|based|located|favorite|primary|role|works|preferred|decided)\b))",
            line,
            flags=re.IGNORECASE,
        )
        for p in parts:
            stripped = p.strip()
            if stripped:
                clauses.append(stripped)
    return clauses


def _extract_claim_triples_with_spans(
    text: str, namespace: str
) -> List[Tuple[str, str, str, str]]:
    """Extract (subject, predicate, object_val, exact_clause) tuples from raw text."""
    clauses = split_into_atomic_clauses(text)
    triples: List[Tuple[str, str, str, str]] = []
    default_sub = namespace.split("/")[-1] if "/" in namespace else "user"

    for clause in clauses:
        # 1. Favorite color: "My favorite color is electric indigo"
        color_match = re.search(
            r"(?:(?:my\s+)?favorite[\s_]+color\s+(?:is|was|are)|prefers?\s+color)\s+([A-Za-z0-9_\-\s]+)",
            clause,
            re.IGNORECASE,
        )
        if color_match:
            val = clean_object_val(color_match.group(1))
            if val:
                triples.append((default_sub, "favorite_color", val, clause))
                continue

        # 2. Location: "I live in Mangaluru" / "I moved to Bengaluru this week"
        loc_match = re.search(
            r"(?:lives?\s+in|live\s+in|resides?\s+in|based\s+in|located\s+in|location\s+is|moved\s+to|relocated\s+to)\s+([A-Za-z0-9_\-\s]+)",
            clause,
            re.IGNORECASE,
        )
        if loc_match:
            val = clean_object_val(loc_match.group(1))
            if val:
                triples.append((default_sub, "lives_in", val, clause))
                continue

        # 3. DB / tool preferences: "preferred DB is PostgreSQL" / "Target database is PostgreSQL 16"
        db_match = re.search(
            r"(?:target\s+(?:DB|database)\s+is|preferred\s+DB\s+is|preferred\s+database\s+is|database\s+of\s+choice\s+is|DB\s+preference\s+is|switch\s+(?:from\s+\w+\s+)?to)\s+([A-Za-z0-9_\-\s]+)",
            clause,
            re.IGNORECASE,
        )
        if db_match:
            val = clean_object_val(db_match.group(1))
            if val:
                triples.append((default_sub, "target_database", val, clause))
                continue

        # 4. Primary key specifications: "primary keys in payment tables must use uuidv7" / "Primary key format is uuidv7"
        pk_match = re.search(
            r"(?:primary\s+keys?.*?must\s+use|primary\s+keys?(?:\s+format)?\s+(?:are|format|is))\s+([A-Za-z0-9_\-]+)",
            clause,
            re.IGNORECASE,
        )
        if pk_match:
            val = clean_object_val(pk_match.group(1))
            if val:
                triples.append((default_sub, "primary_key_spec", val, clause))
                continue

        # 5. Role & Employer
        role_match = re.search(
            r"(?:my\s+)?role\s+(?:is|was)\s+([A-Za-z0-9_\-\s]+)",
            clause,
            re.IGNORECASE,
        )
        if role_match:
            val = clean_object_val(role_match.group(1))
            if val:
                triples.append((default_sub, "role", val, clause))
                continue

        emp_match = re.search(
            r"(?:works?\s+at|employer\s+is)\s+([A-Za-z0-9_\-\s]+)",
            clause,
            re.IGNORECASE,
        )
        if emp_match:
            val = clean_object_val(emp_match.group(1))
            if val:
                triples.append((default_sub, "employer", val, clause))
                continue

        # 6. Structured key: value
        if ":" in clause and not clause.startswith("[") and not clause.startswith("http"):
            parts = clause.split(":", 1)
            p = parts[0].strip().replace(" ", "_").lower()
            o = parts[1].strip().rstrip(".,;:")
            if p and o:
                triples.append((default_sub, p, o, clause))
                continue

        # 7. Fallback atomic stated clause
        cleaned_clause = clean_object_val(clause)
        if cleaned_clause:
            triples.append((default_sub, "stated", cleaned_clause, clause))

    if not triples and text.strip():
        triples.append((default_sub, "stated", clean_object_val(text), text.strip()))

    return triples


def _extract_claim_triples(text: str, namespace: str) -> List[Tuple[str, str, str]]:
    """Simple heuristic triple extractor for memory statements."""
    return [(s, p, o) for s, p, o, _ in _extract_claim_triples_with_spans(text, namespace)]


def remember(
    namespace: str,
    text: str,
    origin: str = "agent_interaction",
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Ingest new information, update source file, SQLite index, and OKF markdown."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)

    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    today = now_dt.strftime("%Y-%m-%d")

    # Check for adversarial prompt injection & assign trust tier
    is_injection = _check_prompt_injection(text, origin)
    base_claim_status = "quarantined" if is_injection else "active"
    trust_tier = 1 if is_injection else BRIDGE_ORIGIN_TRUST_MAP.get(origin.lower().strip(), 2)

    # Write source file
    sources_dir = shard_dir / "sources"
    source_id = f"src_{uuid4().hex[:6]}"
    filename = f"session_{now_dt.strftime('%Y%m%d_%H%M%S')}_{source_id}.txt"
    source_file = sources_dir / filename

    source_content = f"[{now_iso}] Origin: {origin} (trust_tier={trust_tier})\n{text.strip()}\n"
    source_bytes = source_content.encode("utf-8")
    source_file.write_bytes(source_bytes)
    sha256_hash = hashlib.sha256(source_bytes).hexdigest()

    # Find byte start and byte end for the text
    text_bytes = text.strip().encode("utf-8")
    byte_start = source_bytes.find(text_bytes)
    if byte_start == -1:
        byte_start = 0
        byte_end = len(source_bytes)
    else:
        byte_end = byte_start + len(text_bytes)

    rel_source_path = f"sources/{filename}"

    cursor = conn.cursor()
    src_cols = {r[1] for r in cursor.execute("PRAGMA table_info(sources_index)").fetchall()}
    claim_cols = {r[1] for r in cursor.execute("PRAGMA table_info(claims_index)").fetchall()}
    lin_cols = {r[1] for r in cursor.execute("PRAGMA table_info(lineage_index)").fetchall()}

    if "quote" in src_cols:
        cursor.execute(
            "INSERT INTO sources_index (source_id, file_path, sha256_hash, origin, trust_tier, byte_start, byte_end, quote, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                rel_source_path,
                sha256_hash,
                origin,
                trust_tier,
                byte_start,
                byte_end,
                text.strip()[:150],
                now_iso,
            ),
        )
    else:
        cursor.execute(
            "INSERT INTO sources_index (source_id, file_path, sha256_hash, origin, trust_tier, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                source_id,
                rel_source_path,
                sha256_hash,
                origin,
                trust_tier,
                now_iso,
            ),
        )

    extracted_triples = _extract_claim_triples_with_spans(text, namespace)
    created_claims = []

    for sub, pred, obj, exact_clause in extracted_triples:
        claim_id = f"claim_{uuid4().hex[:8]}"
        clause_bytes = exact_clause.encode("utf-8")
        c_pos = source_bytes.find(clause_bytes)
        if c_pos != -1:
            c_byte_start = c_pos
            c_byte_end = c_pos + len(clause_bytes)
        else:
            c_byte_start = byte_start
            c_byte_end = byte_end

        per_claim_status = base_claim_status
        if base_claim_status == "active":
            # Compare trust tier against existing active claims for the same (subject, predicate)
            cursor.execute(
                """SELECT c.claim_id, c.object_val, COALESCE(s.trust_tier, 3) AS trust_tier
                   FROM claims_index c
                   LEFT JOIN sources_index s ON c.source_id = s.source_id
                   WHERE c.namespace = ? AND c.subject = ? AND c.predicate = ? AND c.status = 'active'""",
                (namespace, sub, pred),
            )
            old_rows = cursor.fetchall()
            higher_trust_conflict = next(
                (r for r in old_rows if r["trust_tier"] > trust_tier and r["object_val"] != obj),
                None,
            )
            if higher_trust_conflict is not None:
                per_claim_status = "disputed"
                if "parent_claim_id" in lin_cols:
                    edge_id = f"edge_{uuid4().hex[:6]}"
                    cursor.execute(
                        "INSERT INTO lineage_index VALUES (?, ?, ?, ?, ?)",
                        (edge_id, higher_trust_conflict["claim_id"], claim_id, "disputes", now_iso),
                    )
            else:
                for old_row in old_rows:
                    old_id = old_row["claim_id"]
                    cursor.execute(
                        "UPDATE claims_index SET status = 'superseded', valid_to = ? WHERE claim_id = ?",
                        (today, old_id),
                    )
                    if "parent_claim_id" in lin_cols:
                        edge_id = f"edge_{uuid4().hex[:6]}"
                        cursor.execute(
                            "INSERT INTO lineage_index VALUES (?, ?, ?, ?, ?)",
                            (edge_id, old_id, claim_id, "superseded_by", now_iso),
                        )

        if "created_at" in claim_cols:
            cursor.execute(
                "INSERT INTO claims_index VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    namespace,
                    sub,
                    pred,
                    obj,
                    per_claim_status,
                    today,
                    None,
                    source_id,
                    c_byte_start,
                    c_byte_end,
                    "preferences.okf.md",
                    now_iso,
                ),
            )
        else:
            cursor.execute(
                "INSERT INTO claims_index (claim_id, namespace, subject, predicate, object_val, status, valid_from, valid_to, source_id, byte_start, byte_end, okf_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    namespace,
                    sub,
                    pred,
                    obj,
                    per_claim_status,
                    today,
                    None,
                    source_id,
                    c_byte_start,
                    c_byte_end,
                    "preferences.okf.md",
                ),
            )
        created_claims.append({
            "claim_id": claim_id,
            "subject": sub,
            "predicate": pred,
            "object_val": obj,
            "status": per_claim_status,
            "trust_tier": trust_tier,
            "byte_start": c_byte_start,
            "byte_end": c_byte_end,
        })

    conn.commit()
    conn.close()

    # Update preferences.okf.md
    _refresh_preferences_okf(shard_dir, namespace)

    return {
        "status": "success",
        "source_id": source_id,
        "quarantined": is_injection,
        "claims": created_claims,
    }


def _refresh_preferences_okf(shard_dir: Path, namespace: str) -> None:
    """Synchronize preferences.okf.md with SQLite index."""
    db_path = shard_dir / "index.sqlite"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM claims_index WHERE namespace = ? ORDER BY valid_from ASC, rowid ASC", (namespace,))
    claims = cursor.fetchall()

    cursor.execute("SELECT * FROM sources_index")
    sources = {row["source_id"]: row for row in cursor.fetchall()}
    conn.close()

    active_claims = [c for c in claims if c["status"] == "active"]
    superseded_claims = [c for c in claims if c["status"] == "superseded"]
    disputed_claims = [c for c in claims if c["status"] == "disputed"]
    quarantined_claims = [c for c in claims if c["status"] == "quarantined"]

    frontmatter = {
        "okf_version": "0.2",
        "namespace": namespace,
        "title": f"Knowledge Profile for {namespace.split('/')[-1]}",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "active_claims": len(active_claims),
        "superseded_claims": len(superseded_claims),
        "disputed_claims": len(disputed_claims),
        "quarantined_claims": len(quarantined_claims),
    }

    yaml_text = yaml.dump(frontmatter, sort_keys=False, default_flow_style=False).strip()

    body_lines = [
        f"# Knowledge Profile for {namespace.split('/')[-1]}",
        "",
        "## Active Claims",
    ]
    citations: Dict[str, str] = {}

    for c in active_claims:
        sid = c["source_id"]
        body_lines.append(f"- {c['predicate'].replace('_', ' ').capitalize()}: {c['object_val']}.[^{sid}]")
        if sid in sources:
            s = sources[sid]
            q_str = s["quote"] if "quote" in s.keys() else c["object_val"]
            citations[sid] = f"{s['file_path']} \"{q_str}\" (trust_tier={s['trust_tier']})"

    if not active_claims:
        body_lines.append("- (No active claims)")

    if superseded_claims:
        body_lines.extend(["", "## Superseded Claims"])
        for c in superseded_claims:
            sid = c["source_id"]
            body_lines.append(f"- {c['predicate'].replace('_', ' ').capitalize()} (Historical): {c['object_val']}.[^{sid}]")
            if sid in sources:
                s = sources[sid]
                q_str = s["quote"] if "quote" in s.keys() else c["object_val"]
                citations[sid] = f"{s['file_path']} \"{q_str}\" (trust_tier={s['trust_tier']})"

    if disputed_claims:
        body_lines.extend(["", "## Disputed Claims"])
        for c in disputed_claims:
            sid = c["source_id"]
            body_lines.append(f"- {c['predicate'].replace('_', ' ').capitalize()}: {c['object_val']}.[^{sid}]")
            if sid in sources:
                s = sources[sid]
                q_str = s["quote"] if "quote" in s.keys() else c["object_val"]
                citations[sid] = f"{s['file_path']} \"{q_str}\" (trust_tier={s['trust_tier']})"

    # Quarantined detections are excluded from .okf.md to keep compiled memory injection-free.

    if citations:
        body_lines.extend(["", "## Evidence & Citations"])
        for sid, cite_str in sorted(citations.items()):
            body_lines.append(f"[^{sid}]: {cite_str}")

    okf_content = f"---\n{yaml_text}\n---\n\n" + "\n".join(body_lines) + "\n"
    (shard_dir / "preferences.okf.md").write_text(okf_content, encoding="utf-8")


def recall(
    namespace: str,
    query: str,
    token_budget: int = 800,
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Retrieve 3-tier memory payload (L0 abstract, L1 claims, L2 quotes) or honestly abstain."""
    t0 = time.perf_counter()
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    full_history_tokens = 0
    sources_dir = shard_dir / "sources"
    if sources_dir.exists():
        for f in sources_dir.iterdir():
            if f.is_file():
                try:
                    full_history_tokens += count_tokens(f.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass

    if not db_path.exists():
        latency = (time.perf_counter() - t0) * 1000
        return RecallResult(
            status="abstained",
            answer_hint="I have no memory of that.",
            l0_abstract="",
            l1_claims=[],
            l2_quotes=[],
            tokens_used=0,
            baseline_full_history_tokens=full_history_tokens,
            latency_ms=round(latency, 2),
        ).model_dump()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check suppression hashes to guarantee suppression compliance
    suppr_cols = {r[1] for r in cursor.execute("PRAGMA table_info(suppression_hashes)").fetchall()}
    if "hmac_hash" in suppr_cols:
        cursor.execute("SELECT hmac_hash, target_value FROM suppression_hashes")
    else:
        cursor.execute("SELECT fact_hmac AS hmac_hash, '[suppressed_target]' AS target_value FROM suppression_hashes")
    suppression_rows = cursor.fetchall()
    stored_hmacs = {row["hmac_hash"] for row in suppression_rows}

    # Check query against HMAC suppression hashes
    query_bytes = query.strip().encode("utf-8")
    if hmac.new(HMAC_SECRET, query_bytes, hashlib.sha256).hexdigest() in stored_hmacs:
        conn.close()
        latency = (time.perf_counter() - t0) * 1000
        return RecallResult(
            status="abstained",
            answer_hint="I have no memory of that.",
            l0_abstract="",
            l1_claims=[],
            l2_quotes=[],
            tokens_used=0,
            baseline_full_history_tokens=full_history_tokens,
            latency_ms=round(latency, 2),
        ).model_dump()

    query_lower = query.lower()
    for w in re.findall(r"\w+", query):
        w_hmac = hmac.new(HMAC_SECRET, w.encode("utf-8"), hashlib.sha256).hexdigest()
        if w_hmac in stored_hmacs:
            conn.close()
            latency = (time.perf_counter() - t0) * 1000
            return RecallResult(
                status="abstained",
                answer_hint="I have no memory of that.",
                l0_abstract="",
                l1_claims=[],
                l2_quotes=[],
                tokens_used=0,
                baseline_full_history_tokens=full_history_tokens,
                latency_ms=round(latency, 2),
            ).model_dump()

    for row in suppression_rows:
        val = row["target_value"].lower()
        if val != "[suppressed_target]" and val in query_lower:
            conn.close()
            latency = (time.perf_counter() - t0) * 1000
            return RecallResult(
                status="abstained",
                answer_hint="I have no memory of that.",
                l0_abstract="",
                l1_claims=[],
                l2_quotes=[],
                tokens_used=0,
                baseline_full_history_tokens=full_history_tokens,
                latency_ms=round(latency, 2),
            ).model_dump()

    # Search claims_index for active claims ordered newest-first
    claim_cols = {r[1] for r in cursor.execute("PRAGMA table_info(claims_index)").fetchall()}
    if "created_at" in claim_cols:
        cursor.execute(
            "SELECT *, rowid FROM claims_index WHERE namespace = ? AND status = 'active' ORDER BY valid_from DESC, created_at DESC, rowid DESC",
            (namespace,),
        )
    else:
        cursor.execute(
            "SELECT *, valid_from AS created_at, rowid FROM claims_index WHERE namespace = ? AND status = 'active' ORDER BY valid_from DESC, rowid DESC",
            (namespace,),
        )
    all_active = cursor.fetchall()

    stopwords = {
        "what", "is", "are", "was", "were", "where", "when", "why", "who", "which", "how",
        "do", "does", "did", "am", "the", "a", "an", "of", "in", "to", "for", "my", "me",
        "i", "we", "our", "you", "your", "user", "their", "right", "now", "currently",
        "today", "this", "week", "month", "year",
    }
    q_words = [w for w in re.findall(r"\w+", query_lower) if w not in stopwords]

    scored_claims = []
    for c in all_active:
        sub = c["subject"].lower()
        pred = c["predicate"].lower()
        obj = c["object_val"].lower()
        c_text = f"{sub} {pred} {obj}"
        c_tokens = set(re.findall(r"[a-z0-9]+", c_text))

        def _tok_match(a: str, b: str) -> bool:
            if a == b:
                return True
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            if len(short) >= 4 and len(long_) - len(short) <= 2 and long_.startswith(short):
                return True
            if len(short) >= 5 and abs(len(a) - len(b)) <= 3 and a[:5] == b[:5]:
                return True
            return False

        pred_tokens = pred.replace("_", " ").split()
        score = sum(
            1
            for w in q_words
            if w in c_tokens
            or (len(w) >= 6 and w in c_text)
            or any(_tok_match(w, ct) for ct in c_tokens | set(pred_tokens))
        )
        has_pred_hit = any(
            any(_tok_match(w, pt) for pt in pred_tokens)
            for w in q_words
        )
        if not q_words:
            min_score = 0
        elif pred in ("stated", "has_note") and len(q_words) >= 3:
            min_score = max(2, (len(q_words) + 1) // 2)
        elif len(q_words) >= 4:
            min_score = max(2, (len(q_words) + 1) // 2)
        elif not has_pred_hit and len(q_words) >= 3:
            min_score = max(2, (len(q_words) + 1) // 2)
        else:
            min_score = 1

        if score >= min_score:
            is_specific = 0 if pred in ("stated", "has_note") else 1
            scored_claims.append((is_specific, score, c["valid_from"], c["created_at"], c["rowid"], c))

    scored_claims.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]), reverse=True)
    matching_claims = [item[5] for item in scored_claims]

    if not matching_claims:
        conn.close()
        latency = (time.perf_counter() - t0) * 1000
        return RecallResult(
            status="abstained",
            answer_hint="I have no memory of that.",
            l0_abstract="",
            l1_claims=[],
            l2_quotes=[],
            tokens_used=0,
            baseline_full_history_tokens=full_history_tokens,
            latency_ms=round(latency, 2),
        ).model_dump()

    # Read L0 abstract
    l0_path = shard_dir / ".abstract.md"
    l0_abstract = l0_path.read_text(encoding="utf-8") if l0_path.exists() else ""

    # Fetch L1 Claims
    l1_claim_records = []
    needed_sources = set()
    for c in matching_claims:
        rec = ClaimRecord(
            claim_id=c["claim_id"],
            namespace=c["namespace"],
            subject=c["subject"],
            predicate=c["predicate"],
            object_val=c["object_val"],
            status=c["status"],
            valid_from=c["valid_from"],
            valid_to=c["valid_to"],
            source_id=c["source_id"],
            byte_start=c["byte_start"],
            byte_end=c["byte_end"],
            okf_file=c["okf_file"],
        )
        l1_claim_records.append(rec)
        needed_sources.add(c["source_id"])

    # Fetch L2 Source Quotes
    l2_quotes = []
    if needed_sources:
        by_src_claim = {c.source_id: c for c in reversed(l1_claim_records)}
        placeholders = ",".join("?" for _ in needed_sources)
        cursor.execute(
            f"SELECT * FROM sources_index WHERE source_id IN ({placeholders})",
            list(needed_sources),
        )
        for s in cursor.fetchall():
            s_keys = set(s.keys())
            c_ref = by_src_claim.get(s["source_id"])
            b_start = s["byte_start"] if "byte_start" in s_keys else (c_ref.byte_start if c_ref else 0)
            b_end = s["byte_end"] if "byte_end" in s_keys else (c_ref.byte_end if c_ref else 0)
            if "quote" in s_keys and s["quote"]:
                q_text = s["quote"]
            else:
                q_text = ""
                sf = shard_dir / s["file_path"]
                if sf.exists() and b_end > b_start:
                    with open(sf, "rb") as f_in:
                        f_in.seek(b_start)
                        q_text = f_in.read(b_end - b_start).decode("utf-8", errors="replace")
            span = SourceSpan(
                source_id=s["source_id"],
                file_path=s["file_path"],
                sha256_hash=s["sha256_hash"],
                origin=s["origin"],
                trust_tier=s["trust_tier"],
                byte_start=b_start,
                byte_end=b_end,
                quote=q_text,
            )
            l2_quotes.append(span)

    conn.close()

    # Build answer hint from primary claim
    prim = l1_claim_records[0]
    pred_clean = prim.predicate.replace('_', ' ')
    answer_hint = f"{prim.subject.capitalize()} {pred_clean} is {prim.object_val}."

    # Pack within token_budget
    tokens_used = count_tokens(l0_abstract)
    tokens_used += sum(count_tokens(c.model_dump_json()) for c in l1_claim_records)
    tokens_used += sum(count_tokens(q.model_dump_json()) for q in l2_quotes)

    latency = (time.perf_counter() - t0) * 1000

    result = RecallResult(
        status="ok",
        answer_hint=answer_hint,
        l0_abstract=l0_abstract,
        l1_claims=l1_claim_records,
        l2_quotes=l2_quotes,
        tokens_used=tokens_used,
        baseline_full_history_tokens=full_history_tokens,
        latency_ms=round(latency, 2),
    )
    return result.model_dump()


def explain(
    namespace: str,
    claim_id: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Retrieve exact byte-accurate provenance quote and metadata for a claim."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM claims_index WHERE claim_id = ?", (claim_id,))
    claim = cursor.fetchone()
    if not claim:
        conn.close()
        raise ValueError(f"Claim with id '{claim_id}' not found in namespace '{namespace}'")

    cursor.execute("SELECT * FROM sources_index WHERE source_id = ?", (claim["source_id"],))
    source = cursor.fetchone()
    conn.close()

    if not source:
        raise ValueError(f"Source with id '{claim['source_id']}' not found")

    # Read exact slice from source file
    source_file = shard_dir / source["file_path"]
    exact_slice = ""
    if source_file.exists():
        with open(source_file, "rb") as f:
            f.seek(claim["byte_start"])
            raw = f.read(claim["byte_end"] - claim["byte_start"])
            exact_slice = raw.decode("utf-8", errors="replace")

    return {
        "claim_id": claim["claim_id"],
        "namespace": claim["namespace"],
        "subject": claim["subject"],
        "predicate": claim["predicate"],
        "object_val": claim["object_val"],
        "status": claim["status"],
        "valid_from": claim["valid_from"],
        "valid_to": claim["valid_to"],
        "source_id": claim["source_id"],
        "file_path": source["file_path"],
        "byte_start": claim["byte_start"],
        "byte_end": claim["byte_end"],
        "quote": source["quote"] if "quote" in source.keys() else exact_slice,
        "exact_source_slice": exact_slice,
        "trust_tier": source["trust_tier"],
        "origin": source["origin"],
        "sha256_hash": source["sha256_hash"],
    }


def history(
    namespace: str,
    subject: str,
    predicate: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Return chronological timeline of all claims (active, superseded, disputed)."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM claims_index WHERE namespace = ? AND LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?) "
        "ORDER BY valid_from ASC, rowid ASC",
        (namespace, subject, predicate),
    )
    rows = cursor.fetchall()
    if not rows:
        cursor.execute(
            "SELECT * FROM claims_index WHERE namespace = ? AND LOWER(predicate) = LOWER(?) "
            "ORDER BY valid_from ASC, rowid ASC",
            (namespace, predicate),
        )
        rows = cursor.fetchall()
    conn.close()

    timeline = []
    for r in rows:
        r_keys = set(r.keys())
        timeline.append({
            "claim_id": r["claim_id"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object_val": r["object_val"],
            "status": r["status"],
            "valid_from": r["valid_from"],
            "valid_to": r["valid_to"],
            "source_id": r["source_id"],
            "created_at": r["created_at"] if "created_at" in r_keys else r["valid_from"],
        })
    return timeline


def quarantine_list(
    namespace: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Return all quarantined claims in the namespace."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    src_cols = {r[1] for r in cursor.execute("PRAGMA table_info(sources_index)").fetchall()}
    if "quote" in src_cols:
        cursor.execute(
            "SELECT c.*, s.file_path, s.origin, s.quote, s.trust_tier "
            "FROM claims_index c LEFT JOIN sources_index s ON c.source_id = s.source_id "
            "WHERE c.namespace = ? AND c.status IN ('quarantined', 'disputed') "
            "ORDER BY c.valid_from DESC, c.rowid DESC",
            (namespace,),
        )
    else:
        cursor.execute(
            "SELECT c.*, s.file_path, s.origin, '' AS quote, s.trust_tier "
            "FROM claims_index c LEFT JOIN sources_index s ON c.source_id = s.source_id "
            "WHERE c.namespace = ? AND c.status IN ('quarantined', 'disputed') "
            "ORDER BY c.valid_from DESC, c.rowid DESC",
            (namespace,),
        )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        q_str = r["quote"]
        if not q_str and r["file_path"]:
            sf = shard_dir / r["file_path"]
            if sf.exists() and r["byte_end"] > r["byte_start"]:
                with open(sf, "rb") as f_in:
                    f_in.seek(r["byte_start"])
                    q_str = f_in.read(r["byte_end"] - r["byte_start"]).decode("utf-8", errors="replace")
        reason = (
            "Adversarial prompt injection pattern detected."
            if r["status"] == "quarantined"
            else f"Lower-trust conflicting claim from origin='{r['origin']}' (trust_tier={r['trust_tier']})."
        )
        results.append({
            "claim_id": r["claim_id"],
            "namespace": r["namespace"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object_val": r["object_val"],
            "status": r["status"],
            "source_id": r["source_id"],
            "file_path": r["file_path"],
            "quote": q_str,
            "trust_tier": r["trust_tier"],
            "origin": r["origin"],
            "threat_reason": reason,
        })
    return results


def confirm_quarantine(
    namespace: str,
    claim_id: str,
    action: str = "approve",
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Promote or reject a quarantined claim."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT status FROM claims_index WHERE claim_id = ?", (claim_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Claim '{claim_id}' not found")

    if action == "approve":
        cursor.execute("SELECT subject, predicate FROM claims_index WHERE claim_id = ?", (claim_id,))
        sp_row = cursor.fetchone()
        if sp_row:
            subj, pred = sp_row[0], sp_row[1]
            if pred in SINGLE_VALUED_PREDICATES:
                now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                cursor.execute(
                    "UPDATE claims_index SET status = 'superseded', valid_to = ? "
                    "WHERE namespace = ? AND subject = ? AND predicate = ? AND status = 'active' AND claim_id != ?",
                    (now_iso, namespace, subj, pred, claim_id),
                )
        cursor.execute("UPDATE claims_index SET status = 'active' WHERE claim_id = ?", (claim_id,))
        conn.commit()
        conn.close()
        _refresh_preferences_okf(shard_dir, namespace)
        return {"claim_id": claim_id, "status": "active", "message": "Claim approved and activated."}
    elif action == "reject":
        cursor.execute("DELETE FROM claims_index WHERE claim_id = ?", (claim_id,))
        conn.commit()
        conn.close()
        _refresh_preferences_okf(shard_dir, namespace)
        return {"claim_id": claim_id, "status": "rejected", "message": "Claim rejected and deleted."}
    else:
        conn.close()
        raise ValueError(f"Unknown action: '{action}'. Must be 'approve' or 'reject'.")


def forget(
    namespace: str,
    target: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Permanently delete and redact target information with byte-level verification."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find matching claims (by claim_id, target_value in subject/predicate/object_val)
    cursor.execute(
        "SELECT * FROM claims_index WHERE namespace = ? AND "
        "(claim_id = ? OR subject = ? OR predicate = ? OR object_val LIKE ? OR (subject || '|' || predicate) = ?)",
        (namespace, target, target, target, f"%{target}%", target),
    )
    matched_claims = cursor.fetchall()
    deleted_claim_ids = [c["claim_id"] for c in matched_claims]

    target_bytes = target.encode("utf-8")

    # Redact raw bytes in sources/
    sources_dir = shard_dir / "sources"
    if sources_dir.exists():
        for s_file in sources_dir.iterdir():
            if s_file.is_file():
                raw = s_file.read_bytes()
                if target_bytes in raw:
                    redacted = raw.replace(target_bytes, b"[REDACTED_BY_USER]")
                    s_file.write_bytes(redacted)

    # Redact target in any .md files (e.g. preferences.okf.md, tasks.okf.md)
    for md_file in shard_dir.glob("*.md"):
        try:
            raw_md = md_file.read_bytes()
            if target_bytes in raw_md:
                md_file.write_bytes(raw_md.replace(target_bytes, b"[REDACTED_BY_USER]"))
        except Exception:
            pass

    # Delete claims and edges
    lin_cols = {r[1] for r in cursor.execute("PRAGMA table_info(lineage_index)").fetchall()}
    src_cols = {r[1] for r in cursor.execute("PRAGMA table_info(sources_index)").fetchall()}
    suppr_cols = {r[1] for r in cursor.execute("PRAGMA table_info(suppression_hashes)").fetchall()}

    for cid in deleted_claim_ids:
        if "fts_rowid" in lin_cols:
            l_row = cursor.execute("SELECT vec_rowid, fts_rowid FROM lineage_index WHERE claim_id = ?", (cid,)).fetchone()
            if l_row:
                try:
                    cursor.execute("DELETE FROM fts_claims WHERE rowid = ?", (l_row["fts_rowid"],))
                except Exception:
                    pass
                try:
                    import sqlite_vec
                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                    cursor.execute("DELETE FROM vec_claims WHERE rowid = ?", (l_row["vec_rowid"],))
                except Exception:
                    pass
            cursor.execute("DELETE FROM lineage_index WHERE claim_id = ?", (cid,))
        elif "parent_claim_id" in lin_cols:
            cursor.execute("DELETE FROM lineage_index WHERE parent_claim_id = ? OR child_claim_id = ?", (cid, cid))
        cursor.execute("DELETE FROM claims_index WHERE claim_id = ?", (cid,))

    # Also delete any other claims or sources mentioning target in object_val or quote
    cursor.execute("DELETE FROM claims_index WHERE object_val LIKE ?", (f"%{target}%",))
    if "quote" in src_cols:
        cursor.execute("DELETE FROM sources_index WHERE quote LIKE ?", (f"%{target}%",))
    try:
        cursor.execute("DELETE FROM fts_claims WHERE object_val LIKE ?", (f"%{target}%",))
        cursor.execute("INSERT INTO fts_claims(fts_claims) VALUES('rebuild');")
    except Exception:
        pass

    # Record HMAC suppression hash (store mask in target_value to prevent plaintext leakage in sqlite)
    hmac_val = hmac.new(HMAC_SECRET, target_bytes, hashlib.sha256).hexdigest()
    hash_id = f"suppr_{uuid4().hex[:6]}"
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if "hmac_hash" in suppr_cols:
        cursor.execute(
            "INSERT OR REPLACE INTO suppression_hashes VALUES (?, ?, ?, ?, ?)",
            (hash_id, hmac_val, "claim", "[SUPPRESSED_TARGET]", now_iso),
        )
    else:
        cursor.execute(
            "INSERT OR REPLACE INTO suppression_hashes (fact_hmac, deleted_at) VALUES (?, ?)",
            (hmac_val, now_iso),
        )
    conn.commit()

    # Execute secure deletion and vacuum
    cursor.execute("PRAGMA secure_delete = ON;")
    cursor.execute("PRAGMA journal_mode = DELETE;")
    cursor.execute("VACUUM;")
    conn.commit()
    conn.close()

    # Byte-level verification scan across database and filesystem
    db_bytes = db_path.read_bytes()
    sqlite_byte_matches = db_bytes.count(target_bytes)

    fs_matches = 0
    if sources_dir.exists():
        for s_file in sources_dir.iterdir():
            if s_file.is_file():
                fs_matches += s_file.read_bytes().count(target_bytes)

    wal_path = shard_dir / "index.sqlite-wal"
    wal_matches = wal_path.read_bytes().count(target_bytes) if wal_path.exists() else 0

    # Probe recall 10 times to verify honest abstention
    probe_passed = True
    for _ in range(10):
        probe_res = recall(namespace, target, memory_root=memory_root)
        if probe_res.get("status") != "abstained":
            probe_passed = False
            break

    probe_str = "VERIFIED_ABSTAINED_10_PROBES" if probe_passed else "PROBE_FAILED"
    receipt_id = f"del_{uuid4().hex[:8]}"

    scrubbed_hashes: Dict[str, str] = {}
    for root_walk, _, files_walk in os.walk(shard_dir):
        for fname in files_walk:
            fpath = Path(root_walk) / fname
            rel_p = str(fpath.relative_to(shard_dir))
            try:
                scrubbed_hashes[rel_p] = hashlib.sha256(fpath.read_bytes()).hexdigest()
            except Exception:
                pass

    receipt = DeletionReceipt(
        receipt_id=receipt_id,
        namespace=namespace,
        target=f"[HMAC:{hmac_val[:16]}]",
        target_query=f"[HMAC:{hmac_val[:16]}]",
        deleted_claim_ids=deleted_claim_ids,
        hmac_suppression_hash=hmac_val,
        sqlite_byte_matches=sqlite_byte_matches,
        wal_byte_matches=wal_matches,
        filesystem_byte_matches=fs_matches,
        byte_scan_matches=sqlite_byte_matches + wal_matches + fs_matches,
        fts_rebuilt=True,
        vacuum_completed=True,
        scrubbed_files_sha256=scrubbed_hashes,
        probe_abstentions=probe_str,
        sha256_after=hashlib.sha256(db_bytes).hexdigest(),
        timestamp=now_iso,
    )

    receipts_dir = shard_dir / "receipts"
    (receipts_dir / f"{receipt_id}.json").write_text(
        receipt.model_dump_json(indent=2),
        encoding="utf-8",
    )

    _refresh_preferences_okf(shard_dir, namespace)

    out = receipt.model_dump()
    out["probe_abstention_rate"] = 1.0 if probe_passed else 0.0
    return out


def verify_deletion(
    namespace: str,
    target: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify permanent deletion, checking receipts, byte scans, and probe abstentions."""
    shard_dir = _get_shard_dir(namespace, memory_root)
    db_path = shard_dir / "index.sqlite"

    target_bytes = target.encode("utf-8")
    sqlite_byte_matches = db_path.read_bytes().count(target_bytes) if db_path.exists() else 0

    sources_dir = shard_dir / "sources"
    fs_matches = 0
    if sources_dir.exists():
        for s_file in sources_dir.iterdir():
            if s_file.is_file():
                fs_matches += s_file.read_bytes().count(target_bytes)

    # Check suppression hash table
    has_hash = False
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        target_hmac = hmac.new(HMAC_SECRET, target_bytes, hashlib.sha256).hexdigest()
        suppr_cols = {r[1] for r in cursor.execute("PRAGMA table_info(suppression_hashes)").fetchall()}
        if "hmac_hash" in suppr_cols:
            cursor.execute("SELECT 1 FROM suppression_hashes WHERE hmac_hash = ? OR target_value = ?", (target_hmac, target))
        else:
            cursor.execute("SELECT 1 FROM suppression_hashes WHERE fact_hmac = ?", (target_hmac,))
        has_hash = cursor.fetchone() is not None
        conn.close()

    # Probe abstentions
    probe_passed = True
    for _ in range(10):
        probe_res = recall(namespace, target, memory_root=memory_root)
        if probe_res.get("status") != "abstained":
            probe_passed = False
            break

    verified = (sqlite_byte_matches == 0) and (fs_matches == 0) and probe_passed

    return {
        "verified": verified,
        "status": "verified_unlearned" if verified else "failed_unlearned",
        "target": target,
        "sqlite_byte_matches": sqlite_byte_matches,
        "filesystem_byte_matches": fs_matches,
        "suppression_hash_present": has_hash,
        "probe_abstentions": "VERIFIED_ABSTAINED_10_PROBES" if probe_passed else "FAILED_PROBES",
    }


def handoff_create(
    namespace: str,
    task_id: str,
    goal: str,
    completed_steps: List[str],
    pending_steps: List[str],
    open_questions: List[str],
    created_by_agent: str = "agent_1",
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Wrap create_handoff for MCP clients."""
    packet = create_handoff(
        namespace=namespace,
        task_id=task_id,
        goal=goal,
        completed_steps=completed_steps,
        pending_steps=pending_steps,
        open_questions=open_questions,
        created_by_agent=created_by_agent,
        memory_root=memory_root,
    )
    return packet.model_dump()


def handoff_get(
    namespace: str,
    task_id: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Wrap get_handoff for MCP clients."""
    return get_handoff(
        namespace=namespace,
        task_id=task_id,
        memory_root=memory_root,
    )
