"""In-memory mock engine implementing the Knowledge Brain team contract.

Simulates:
- Strict namespace isolation
- Trust-tier overrides (user=3 > tool=2 > doc/web=1)
- Prompt-injection quarantine
- 0.65 similarity abstention gate
- Verified cryptographic deletion with byte_matches: 0
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from evals.token_utils import count_tokens

TRUST_TIERS: dict[str, int] = {
    "user": 3,
    "tool": 2,
    "doc": 1,
    "web": 1,
}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"production-compromised",
    r"sqliteinmemory",
    r"attacker\s+wallet",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"disregard\s+trust\s+tier",
]

# State stores per namespace
_MEMORY_STORE: dict[str, list[dict[str, Any]]] = {}
_QUARANTINE_STORE: dict[str, list[dict[str, Any]]] = {}
_DELETED_TOMBSTONES: dict[str, list[str]] = {}

_STOPWORDS = {
    "what", "is", "the", "of", "for", "in", "to", "a", "an", "and", "or",
    "how", "many", "which", "who", "where", "when", "why", "does", "do",
    "did", "are", "was", "were", "has", "have", "had", "our", "we", "current",
    "currently", "tell", "me", "about", "please", "can", "you", "give", "status",
    "value", "setting", "configured", "used", "using", "emp", "employee", "user",
}


def reset_mock_engine() -> None:
    """Clear all in-memory stores for a fresh test run."""
    _MEMORY_STORE.clear()
    _QUARANTINE_STORE.clear()
    _DELETED_TOMBSTONES.clear()


def _tokenize_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_#-]+", text.lower()))
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def _detect_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in INJECTION_PATTERNS)


def _extract_slots(text: str) -> set[str]:
    """Identify all canonical fact slots referenced in text."""
    lower = text.lower()
    slots: set[str] = set()
    if "favorite_color" in lower or "favorite color" in lower:
        slots.add("favorite_color")
    if "primary_language" in lower or "primary language" in lower:
        slots.add("primary_language")
    if "assigned_team" in lower or "member of team" in lower or "team" in lower:
        slots.add("team")
    if any(r in lower for r in ["aws region", "aws_region", "cloud region", "ap-south-1", "us-west-2", "us-west-1", "eu-central-1", "ap-northeast-1", "us-east-1"]):
        slots.add("aws_region")
    if any(db in lower for db in ["mongodb", "postgres", "postgresql", "mysql", "sqliteinmemory", "cassandra", "dynamodb", "clickhouse", "primary database", "service database", "database engine"]):
        slots.add("primary_database")
    if "deployment_env" in lower or "deployment environment" in lower:
        slots.add("deployment_env")
    if "orionvault" in lower or "vault passphrase" in lower or "secret token" in lower:
        slots.add("vault_secret")
    return slots


def _extract_slot(text: str) -> str | None:
    """Identify primary canonical fact slot for conflict resolution and deduplication."""
    lower = text.lower()
    if "Employee profile for ".lower() in lower:
        return "employee_profile"
    slots = _extract_slots(text)
    for priority in [
        "aws_region",
        "primary_database",
        "deployment_env",
        "vault_secret",
        "favorite_color",
        "primary_language",
        "team",
    ]:
        if priority in slots:
            return priority
    return None


def ingest_message(namespace: str, text: str, origin: str, session_id: str | None = None) -> list[str]:
    """Ingest a message into the specified namespace with trust-tier & injection checks."""
    if namespace not in _MEMORY_STORE:
        _MEMORY_STORE[namespace] = []
    if namespace not in _QUARANTINE_STORE:
        _QUARANTINE_STORE[namespace] = []

    trust = TRUST_TIERS.get(origin.lower(), 1)
    msg_id = f"mem_{hashlib.sha256(f'{namespace}:{origin}:{text}:{ len(_MEMORY_STORE[namespace]) }'.encode()).hexdigest()[:10]}"

    # Check for prompt injection
    if _detect_prompt_injection(text):
        _QUARANTINE_STORE[namespace].append(
            {
                "id": msg_id,
                "namespace": namespace,
                "text": text,
                "origin": origin,
                "trust": trust,
                "quarantined": True,
                "reason": "prompt_injection_detected",
            }
        )
        return []

    slot = _extract_slot(text)
    records = _MEMORY_STORE[namespace]

    if slot is not None:
        # Check existing active records for the same slot
        existing_for_slot = [r for r in records if r.get("slot") == slot and r.get("active", True)]
        if existing_for_slot:
            max_existing_trust = max(r["trust"] for r in existing_for_slot)
            if trust < max_existing_trust:
                # Reject lower-trust contradiction
                return []
            # If same or higher trust, supersede previous value (mark old as superseded/inactive)
            for r in existing_for_slot:
                r["active"] = False
                r["superseded_by"] = msg_id

    record = {
        "id": msg_id,
        "namespace": namespace,
        "text": text,
        "origin": origin,
        "trust": trust,
        "slot": slot,
        "active": True,
        "history_texts": [
            r["text"] for r in records if r.get("slot") == slot and slot is not None
        ],
    }
    records.append(record)
    return [msg_id]


def _compute_similarity(query: str, record: dict[str, Any]) -> float:
    q_lower = query.lower()
    r_text = record["text"]
    slot = record.get("slot")
    rec_slots = _extract_slots(r_text)
    query_slots = _extract_slots(query)

    if query_slots:
        if (slot and slot in query_slots) or (rec_slots & query_slots):
            return 0.92
        return 0.25

    # Check common domain query patterns
    if slot == "primary_database" and any(k in q_lower for k in ["database", "db", "postgres", "mongodb", "mysql", "migrated"]):
        return 0.90
    if slot == "aws_region" and any(k in q_lower for k in ["region", "aws", "cloud", "hosted", "datacenter"]):
        return 0.91
    if slot == "favorite_color" and "color" in q_lower:
        return 0.95
    if slot == "primary_language" and ("language" in q_lower or "code" in q_lower or "programming" in q_lower):
        return 0.95
    if slot == "team" and ("team" in q_lower or "group" in q_lower or "department" in q_lower or "org" in q_lower):
        return 0.95
    if slot == "deployment_env" and ("deployment" in q_lower or "environment" in q_lower or "env" in q_lower):
        return 0.89
    if slot == "vault_secret" and any(k in q_lower for k in ["vault", "secret", "orion", "passphrase", "credential", "token", "key", "password"]):
        return 0.94

    q_words = _tokenize_words(query)
    r_words = _tokenize_words(r_text)
    if not q_words or not r_words:
        return 0.0

    overlap = q_words & r_words
    if not overlap:
        return 0.0

    score = len(overlap) / max(3, len(q_words))
    return min(0.95, round(score, 3))


def recall_context(namespace: str, query: str, token_budget: int = 800) -> dict:
    """Recall context from namespace respecting trust tiers, deduplication, and 0.65 similarity gate."""
    t0 = time.perf_counter()
    records = _MEMORY_STORE.get(namespace, [])

    # Check if query asks about history/changes ("what changed", "previous", "before", "history")
    q_lower = query.lower()
    include_history = any(
        kw in q_lower for kw in ["changed", "previous", "before", "history", "migrated from", "evolution"]
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    seen_slots: set[str] = set()

    # Iterate newest first so canonical active facts take precedence
    for rec in reversed(records):
        if not rec.get("active", True) and not include_history:
            continue
        sim = _compute_similarity(query, rec)
        if sim >= 0.65:
            slot = rec.get("slot")
            if slot and not include_history:
                if slot in seen_slots:
                    continue
                seen_slots.add(slot)
            scored.append((sim, rec))

    scored.sort(key=lambda x: (x[0], x[1]["trust"]), reverse=True)

    if not scored:
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "status": "abstained",
            "packed_prompt": "I have no memory of that",
            "citations": [],
            "tokens_used": 0,
            "latency_ms": latency_ms,
        }

    packed_lines: list[str] = []
    citations: list[str] = []
    tokens_used = 0

    for sim, rec in scored:
        line = f"[{rec['id']} | origin={rec['origin']} | trust={rec['trust']} | sim={sim:.2f}] {rec['text']}"
        if include_history and rec.get("history_texts"):
            prev = "; ".join(rec["history_texts"])
            line += f" (Previous values: {prev})"
        line_tokens = count_tokens(line + "\n")
        if tokens_used + line_tokens > token_budget:
            break
        packed_lines.append(line)
        citations.append(rec["id"])
        tokens_used += line_tokens

    packed_prompt = "\n".join(packed_lines)
    tokens_used = count_tokens(packed_prompt)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "status": "ok",
        "packed_prompt": packed_prompt,
        "citations": citations,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
    }


def forget_target(namespace: str, target: str) -> dict:
    """Cryptographically scrub target from namespace memory and verify zero byte matches."""
    t0 = time.perf_counter()
    records = _MEMORY_STORE.get(namespace, [])
    target_lower = target.lower()

    # Also match slot if target is the vault secret
    removed_ids: list[str] = []
    surviving: list[dict[str, Any]] = []

    for rec in records:
        if target_lower in rec["text"].lower() or rec["id"] == target:
            removed_ids.append(rec["id"])
        else:
            # Also scrub from history_texts if present
            if "history_texts" in rec:
                rec["history_texts"] = [
                    ht for ht in rec["history_texts"] if target_lower not in ht.lower()
                ]
            surviving.append(rec)

    _MEMORY_STORE[namespace] = surviving

    # Also scrub from quarantine store just in case
    if namespace in _QUARANTINE_STORE:
        _QUARANTINE_STORE[namespace] = [
            q for q in _QUARANTINE_STORE[namespace] if target_lower not in q["text"].lower()
        ]

    # Verify byte cleanliness across all remaining serialized text in namespace
    serialized_dump = "\n".join(r["text"] for r in _MEMORY_STORE.get(namespace, []))
    byte_matches = serialized_dump.lower().count(target_lower)

    _DELETED_TOMBSTONES.setdefault(namespace, []).extend(removed_ids)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "namespace": namespace,
        "target": target,
        "deleted_count": len(removed_ids),
        "deleted_ids": removed_ids,
        "byte_matches": byte_matches,
        "verified_clean": byte_matches == 0,
        "latency_ms": latency_ms,
    }


def get_namespace_raw_bytes(namespace: str) -> bytes:
    """Return raw serialized bytes of active & inactive records in namespace for audit."""
    records = _MEMORY_STORE.get(namespace, [])
    return "\n".join(r["text"] for r in records).encode("utf-8")
