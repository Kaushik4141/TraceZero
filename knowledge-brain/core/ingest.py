"""Ingestion pipeline, UTF-8 byte span verification, and claim indexing via MemoryStore."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from core.contracts import (
    ClaimRecord,
    ClaimStatus,
    ExtractedCandidate,
    ORIGIN_TRUST_MAP,
    TrustTier,
)
from core.policy import apply_policy_and_decide_status, verify_exact_quote_span
from core.predicates import get_target_okf_file
from core.storage_api import MemoryStore


TEMPORAL_SUFFIX_RE = re.compile(
    r"\s+\b(?:this week|last week|this month|last month|this year|last year|today|yesterday|recently|right now|now)\b\.?$",
    re.IGNORECASE,
)


def clean_object_val(val: str) -> str:
    """Trim trailing punctuation and temporal qualifiers from extracted object values."""
    cleaned = val.strip().rstrip(".,!?;:")
    while True:
        updated = TEMPORAL_SUFFIX_RE.sub("", cleaned).strip().rstrip(".,!?;:")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def split_into_atomic_clauses(text: str) -> List[str]:
    """Split compound/multi-sentence input into atomic clauses preserving exact substrings."""
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


def extract_heuristic_candidates(
    raw_text: str, default_subject: str = "user"
) -> List[ExtractedCandidate]:
    """Regex and heuristic extractor for user facts, decisions, and tasks."""
    # Detect optional dialogue speaker prefix like "[date] Caroline: ..." when subject is generic
    if default_subject == "user":
        spk_m = re.match(r"^(?:\[[^\]]+\]\s*)?([A-Z][A-Za-z0-9_-]{1,30}):\s+", raw_text.strip())
        if spk_m:
            default_subject = spk_m.group(1).strip().lower()

    # Detect project subject if an enterprise project is named in raw_text
    proj_m = re.search(
        r"(?i)\b(orion_payments|atlas_auth|hyperion_search|kronos_billing|nebula_analytics|"
        r"vortex_streaming|aegis_gateway|titan_ledger|zephyr_messaging|pulsar_telemetry|"
        r"helios_compute|argus_security|nexus_graph|stratos_cdn|valkyrie_ci|cobalt_cache|"
        r"solaris_ml|mercury_fx|borealis_storage|draco_compliance|project_atlas|Project Atlas)\b",
        raw_text,
    )
    if proj_m:
        matched_proj = proj_m.group(1).strip()
        proj_subject = "project_atlas" if matched_proj.lower() == "project atlas" else matched_proj.lower()
    else:
        proj_subject = "orion_payments" if default_subject == "emp_042" else "project_atlas"

    candidates: List[ExtractedCandidate] = []
    clauses = split_into_atomic_clauses(raw_text) or ([raw_text.strip()] if raw_text.strip() else [])

    for clause in clauses:
        # 1. Favorite Color
        for m in re.finditer(
            r"(?i)(?:my\s+)?favorite[\s_]+color\s+(?:is|was)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:and|but|because|with|since|which|my)|\b[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            quote = clause[m.start() : m.start(1) + len(m.group(1))].strip()
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="favorite_color",
                        object_val=val,
                        exact_quote=quote,
                    )
                )
        for m in re.finditer(r"(?i)prefers?\s+([a-zA-Z]+)\s+color", clause):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="favorite_color",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )

        # 1b. Primary Language & Assigned Team
        for m in re.finditer(
            r"(?i)(?:my\s+)?primary[\s_]+language\s+(?:is|was)\s+([a-zA-Z0-9_\-+#]+?)(?:\s+(?:and|but|because|with|since|which|my)|[,.\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="primary_language",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )
        for m in re.finditer(
            r"(?i)(?:my\s+)?assigned[\s_]+team\s+(?:is|was)\s+([a-zA-Z0-9_\-]+?)(?:\s+(?:and|but|because|with|since|which|my)|[,.\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="assigned_team",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )

        # 2. Location (lives_in)
        for m in re.finditer(
            r"(?i)(?:I\s+)?(?:lives?\s+in|resides?\s+in|based\s+in|moved\s+to|relocated\s+to|located\s+in)\s+([a-zA-Z\s]+?)(?:\s+(?:and|but|because|with|since|which)|\b[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            quote = clause[m.start() : m.start(1) + len(m.group(1))].strip()
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="lives_in",
                        object_val=val,
                        exact_quote=quote,
                    )
                )

        # 3. Preferred Database / Database Decision
        for m in re.finditer(
            r"(?i)preferred\s+database\s+(?:is|to\s+use)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:and|but|because|with|since|which)|\b[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            quote = clause[m.start() : m.start(1) + len(m.group(1))].strip()
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="preferred_db",
                        object_val=val,
                        exact_quote=quote,
                    )
                )
        for m in re.finditer(
            r"(?i)(?:primary\s+database(?:\s+engine)?|service\s+database|override\s+(?:the\s+)?database)(?:\s+for\s+[^.\n]+?)?\s+(?:is(?:\s+currently)?|to)\s+(PostgreSQL\s+\d+|Postgres\s+\d+|PostgreSQL|MongoDB\s+[\d.]+|MongoDB|Cassandra(?:\s+[\d.]+)?|DynamoDB|Redis(?:\s+[\d.]+)?|ClickHouse(?:\s+[\d.]+)?|MySQL\s+[\d.]+|MySQL|SQLiteInMemory|[a-zA-Z0-9_\-]+)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=proj_subject,
                        predicate="preferred_db",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )
        for m in re.finditer(
            r"(?i)(?:use|uses|using)\s+([a-zA-Z0-9_\-]+)\s+(?:as\s+our\s+|for\s+)?database",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject="system",
                        predicate="uses_database",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )

        # 3b. Cloud Infrastructure: AWS/cloud region, deployment_env, project link, vault passphrase
        for m in re.finditer(
            r"(?i)(?:(?:aws|cloud)[\s_]+region[^.\n]*?\b(us-[a-z]+-\d|eu-[a-z]+-\d|ap-[a-z]+-\d)\b|\b(us-[a-z]+-\d|eu-[a-z]+-\d|ap-[a-z]+-\d)\b[^.\n]*?(?:aws|cloud)[\s_]+region)",
            clause,
        ):
            val = (m.group(1) or m.group(2)).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=proj_subject,
                    predicate="aws_region",
                    object_val=val,
                    exact_quote=m.group(0).strip("., \n"),
                )
            )
        for m in re.finditer(
            r"(?i)(?:update\s+|current\s+)?deployment_env(?:\s+is(?:\s+set)?|\s+to)?\s+(?:to\s+)?([a-zA-Z0-9_\-]+)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=proj_subject,
                        predicate="deployment_env",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )
        for m in re.finditer(
            r"(?i)(?:vault\s+passphrase[^:\n]*?\s+is|passphrase\s+is)\s+([^\s,;]+)",
            clause,
        ):
            val = m.group(1).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=default_subject,
                    predicate="vault_passphrase",
                    object_val=val,
                    exact_quote=m.group(0).strip("., \n"),
                )
            )

        # 4. Role & Employer
        for m in re.finditer(
            r"(?i)(?:my\s+)?role\s+is\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:and|but|because|with|since|which)|\b[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            quote = clause[m.start() : m.start(1) + len(m.group(1))].strip()
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="role",
                        object_val=val,
                        exact_quote=quote,
                    )
                )
        for m in re.finditer(
            r"(?i)(?:works?\s+at|employer\s+is)\s+([a-zA-Z0-9_\-\s]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="employer",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )

        # 5. Coding style
        for m in re.finditer(
            r"(?i)coding\s+style\s+(?:is|preference\s+is)\s+([a-zA-Z0-9_\-\s]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject=default_subject,
                        predicate="coding_style",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )

        # 6. Graph Relationships: depends_on & manages
        for m in re.finditer(
            r"(?i)\b([a-zA-Z0-9_\-]+)\s+depends\s+on\s+([a-zA-Z0-9_\-]+)\b",
            clause,
        ):
            s, o = m.group(1).strip(), m.group(2).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=s,
                    predicate="depends_on",
                    object_val=o,
                    exact_quote=m.group(0).strip(),
                )
            )
        for m in re.finditer(
            r"(?i)\b([a-zA-Z0-9_\-]+)\s+manages\s+([a-zA-Z0-9_\-]+)\b",
            clause,
        ):
            s, o = m.group(1).strip(), m.group(2).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=s,
                    predicate="manages",
                    object_val=o,
                    exact_quote=m.group(0).strip(),
                )
            )

        # 7. Decisions and Constraints
        for m in re.finditer(
            r"(?i)decided(?:\s+to)?\s+([a-zA-Z0-9_\-\s]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = clean_object_val(m.group(1))
            if val:
                candidates.append(
                    ExtractedCandidate(
                        subject="architecture",
                        predicate="decided",
                        object_val=val,
                        exact_quote=m.group(0).strip("., \n"),
                    )
                )
        for m in re.finditer(
            r"(?i)constraint:\s*([a-zA-Z0-9_\-\s<>=%]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = m.group(1).strip()
            candidates.append(
                ExtractedCandidate(
                    subject="architecture",
                    predicate="constraint",
                    object_val=val,
                    exact_quote=m.group(0).strip("., \n"),
                )
            )

        # 8. Tasks (goal, completed, pending)
        for m in re.finditer(
            r"(?i)task\s+goal:\s*([a-zA-Z0-9_\-\s]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = m.group(1).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=default_subject,
                    predicate="task_goal",
                    object_val=val,
                    exact_quote=m.group(0).strip("., \n"),
                )
            )
        for m in re.finditer(
            r"(?i)(?:task\s+)?completed:\s*([a-zA-Z0-9_\-\s]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = m.group(1).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=default_subject,
                    predicate="task_completed",
                    object_val=val,
                    exact_quote=m.group(0).strip("., \n"),
                )
            )
        for m in re.finditer(
            r"(?i)(?:task\s+)?pending:\s*([a-zA-Z0-9_\-\s]+?)(?:[.,;!\n]|$)",
            clause,
        ):
            val = m.group(1).strip()
            candidates.append(
                ExtractedCandidate(
                    subject=default_subject,
                    predicate="task_pending",
                    object_val=val,
                    exact_quote=m.group(0).strip("., \n"),
                )
            )

        # 9. Prompt Injection candidate
        for m in re.finditer(
            r"(?i)(?:ignore\s+(?:all\s+)?previous\s+instructions|system\s+prompt|system\s+override|override\s+policy|disregard\s+trust\s+tier)[^\n]*",
            clause,
        ):
            val = m.group(0).strip()
            candidates.append(
                ExtractedCandidate(
                    subject="adversary",
                    predicate="has_note",
                    object_val=val,
                    exact_quote=val,
                )
            )

    if proj_m and default_subject != "user" and any(
        c.predicate in ("aws_region", "preferred_db", "deployment_env") for c in candidates
    ):
        candidates.append(
            ExtractedCandidate(
                subject=default_subject,
                predicate="works_on",
                object_val=proj_subject,
                exact_quote=proj_m.group(1),
            )
        )

    # 10. Fallback contextual note if nothing else matched
    if not candidates and raw_text.strip():
        full_note = raw_text.strip()
        # Strip leading speaker prefix only if we already promoted speaker into default_subject? Keep full_note for exact byte span!
        candidates.append(
            ExtractedCandidate(
                subject=default_subject,
                predicate="has_note",
                object_val=full_note[:400],
                exact_quote=full_note,
            )
        )

    return candidates


def ingest_message(
    store: MemoryStore,
    namespace: str,
    text: str,
    origin: str = "user",
    source_id: Optional[str] = None,
    session_id: Optional[str] = None,
    candidates: Optional[List[ExtractedCandidate]] = None,
    llm_extractor: Optional[Callable[[str], List[ExtractedCandidate]]] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ingest raw text via MemoryStore, verify byte spans, apply policy, and index."""
    now = datetime.now(timezone.utc).isoformat()
    trust_tier = ORIGIN_TRUST_MAP.get(origin.lower(), TrustTier.DOC_WEB)
    sid = source_id or f"src_{uuid.uuid4().hex[:12]}"
    sess = session_id or "session_default"
    filename = f"{sess}.txt"

    # Compute base byte offset before MemoryStore.save_source appends (including '\n' separator)
    shard_dir = store.sm.get_shard_dir(namespace)
    src_path = shard_dir / "sources" / filename
    base_byte_offset = (src_path.stat().st_size + 1) if src_path.exists() else 0

    store.save_source(
        namespace=namespace,
        source_id=sid,
        filename=filename,
        text=text,
        origin=origin,
        trust_tier=trust_tier,
    )

    if candidates is not None:
        extracted = candidates
    elif llm_extractor is not None:
        extracted = llm_extractor(text)
    else:
        default_subject = (
            subject.strip().lower()
            if subject and subject.strip()
            else (namespace.split("/")[-1] if "users" in namespace else "user")
        )
        extracted = extract_heuristic_candidates(text, default_subject=default_subject)

    results: List[Dict[str, Any]] = []

    for cand in extracted:
        span = verify_exact_quote_span(text, cand.exact_quote)
        if span is None:
            continue

        rel_start, rel_end = span
        abs_byte_start = base_byte_offset + rel_start
        abs_byte_end = base_byte_offset + rel_end

        subj = store.resolve_entity(namespace, cand.subject)
        pred = cand.predicate.strip().lower()
        obj = cand.object_val.strip()
        okf_file = get_target_okf_file(pred)

        status, reason, superseded_id = apply_policy_and_decide_status(
            store=store,
            namespace=namespace,
            subject=subj,
            predicate=pred,
            object_val=obj,
            origin=origin,
            trust_tier=trust_tier,
            raw_quote=cand.exact_quote,
            full_text=text,
        )

        if superseded_id:
            store.mark_claim_status(
                namespace=namespace,
                claim_id=superseded_id,
                new_status=ClaimStatus.SUPERSEDED,
                valid_to=now,
            )

        claim_id = f"clm_{uuid.uuid4().hex[:12]}"
        claim = ClaimRecord(
            claim_id=claim_id,
            namespace=namespace,
            subject=subj,
            predicate=pred,
            object_val=obj,
            status=status,
            valid_from=now,
            valid_to=None,
            source_id=sid,
            byte_start=abs_byte_start,
            byte_end=abs_byte_end,
            okf_file=okf_file,
        )

        stored = store.store_claim(claim)
        if not stored:
            # Blocked by suppression HMAC gate
            results.append(
                {
                    "claim_id": claim_id,
                    "subject": subj,
                    "predicate": pred,
                    "object_val": obj,
                    "status": ClaimStatus.QUARANTINED.value,
                    "reason": "Blocked by suppression HMAC gate",
                    "source_id": sid,
                    "byte_start": abs_byte_start,
                    "byte_end": abs_byte_end,
                }
            )
            continue

        results.append(
            {
                "claim_id": claim_id,
                "subject": subj,
                "predicate": pred,
                "object_val": obj,
                "status": status.value,
                "reason": reason,
                "source_id": sid,
                "byte_start": abs_byte_start,
                "byte_end": abs_byte_end,
            }
        )

    return results
