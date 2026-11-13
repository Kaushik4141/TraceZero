"""Deterministic query routing, tiered context packing, exact byte spans, and abstention gate."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set

import tiktoken

from core.contracts import Citation, ClaimRecord, ClaimStatus, RetrievalResult
from core.storage_api import MemoryStore

_TIKTOKEN_ENC = None


def get_tokenizer():
    """Return cached tiktoken encoding."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base."""
    return len(get_tokenizer().encode(text))


PREDICATE_KEYWORDS: Dict[str, Set[str]] = {
    "employer": {"employer", "company", "work", "works", "working", "job", "office", "organization", "firm", "employ", "employed"},
    "lives_in": {"lives", "live", "living", "reside", "resides", "residing", "location", "city", "place", "based", "home", "country", "moved", "relocated"},
    "favorite_color": {"color", "colour", "shade", "tint", "hue", "favorite_color"},
    "primary_language": {"primary_language", "programming", "language", "rust", "python"},
    "assigned_team": {"team", "assigned_team", "squad", "group"},
    "preferred_db": {"db", "database", "engine", "storage", "datastore", "postgres", "postgresql", "mysql", "mongodb", "preferred_db"},
    "aws_region": {"aws", "region", "regions", "aws_region", "oregon"},
    "deployment_env": {"deployment_env", "deployment", "environment", "canary"},
    "vault_passphrase": {"vault", "passphrase", "credential", "password", "break-glass", "orionvault"},
    "role": {"role", "title", "position", "designation", "job", "engineer", "architect", "lead"},
    "coding_style": {"coding", "style", "format", "conventions", "standards", "pep8", "linting"},
    "programming_language": {"language", "python", "typescript", "java", "golang", "rust", "c++"},
    "timezone": {"timezone", "zone", "tz", "utc", "gmt", "ist", "est", "pst"},
    "depends_on": {"depends", "dependency", "dependencies", "depend", "depends_on"},
    "manages": {"manages", "manager", "reports", "lead", "subordinates"},
    "decided": {"decision", "decided", "chose", "chosen", "architecture", "strategy"},
    "constraint": {"constraint", "limit", "restriction", "requirement", "latency", "sla"},
    "task_goal": {"goal", "objective", "task", "milestone", "mission", "target"},
    "task_completed": {"completed", "done", "finished", "resolved"},
    "task_pending": {"pending", "remaining", "todo", "upcoming"},
    "task_status": {"status", "phase", "state", "progress"},
    "has_note": {"note", "notes", "observation", "context", "memo"},
}


def classify_query_route(query: str) -> str:
    """Deterministic 0-LLM-token query router."""
    q = query.lower()

    if any(k in q for k in [
        "what changed", "history", "previously", "former", "timeline",
        "past", "used to", "before", "prior", "superseded", "historical",
    ]):
        return "TEMPORAL_HISTORY"

    if any(k in q for k in [
        "depends on", "dependency", "dependencies", "manages", "managed by",
        "hierarchy", "lineage", "chain", "path", "subordinates", "reports to",
    ]):
        return "GRAPH_CTE"

    if any(k in q for k in [
        "how many", "count", "number of", "total", "sum", "average",
        "aggregate", "list all tasks", "pending tasks count", "tally",
    ]):
        return "SQL_AGGREGATE"

    return "HYBRID_TIERED"


def _row_to_claim_record(r: Dict[str, Any], namespace: str) -> ClaimRecord:
    return ClaimRecord(
        claim_id=r["claim_id"],
        namespace=r.get("namespace", namespace),
        subject=r["subject"],
        predicate=r["predicate"],
        object_val=r["object_val"],
        status=ClaimStatus(r.get("status", "active")),
        valid_from=r["valid_from"],
        valid_to=r.get("valid_to"),
        source_id=r["source_id"],
        byte_start=r["byte_start"],
        byte_end=r["byte_end"],
        okf_file=r.get("okf_file", "preferences.okf.md"),
    )


def recall_context(
    store: MemoryStore,
    namespace: str,
    query: str,
    token_budget: int = 800,
) -> RetrievalResult:
    """Recall scoped memory via MemoryStore adhering to token budget and abstention gate."""
    t0 = time.perf_counter()
    route = classify_query_route(query)
    enc = get_tokenizer()

    # Step 1: Hybrid Semantic + FTS5 search via MemoryStore
    vec_hits = store.search_hybrid(
        namespace=namespace,
        query=query,
        top_k=18,
        include_superseded=(route == "TEMPORAL_HISTORY"),
    )
    active_vec_hits = [
        h for h in vec_hits
        if (route == "TEMPORAL_HISTORY" or h.get("status") == "active")
    ]

    # Step 2: Check active predicates & subjects in namespace for deterministic matching
    with store.sm.get_connection(namespace) as conn:
        active_pred_rows = conn.execute(
            "SELECT DISTINCT predicate FROM claims_index WHERE namespace = ? AND status = 'active'",
            (namespace,),
        ).fetchall()
        active_subj_rows = conn.execute(
            "SELECT DISTINCT subject FROM claims_index WHERE namespace = ? AND status = 'active'",
            (namespace,),
        ).fetchall()
    active_predicates = [r["predicate"] for r in active_pred_rows]
    active_subjects = {
        r["subject"].lower() for r in active_subj_rows if r["subject"] and r["subject"].lower() != "user"
    }

    q_words = set(re.findall(r"\w+", query.lower())) | set(
        re.findall(r"[a-z0-9_#-]+", query.lower())
    )
    matching_active_predicates: List[str] = []
    for pred in active_predicates:
        pred_lower = pred.lower()
        kw_set = PREDICATE_KEYWORDS.get(pred_lower, set())
        pred_parts = pred_lower.split("_")
        if pred_lower in q_words or pred_parts[-1] in q_words or bool(q_words & kw_set):
            matching_active_predicates.append(pred)

    # Check if the query asks about a predicate slot that was explicitly deleted (tombstone check)
    from core.forget import compute_fact_hmac
    with store.sm.get_connection(namespace) as conn:
        suppr_rows = conn.execute("SELECT fact_hmac FROM suppression_hashes").fetchall()
    suppr_hmacs = {r[0] for r in suppr_rows}
    for cand_pred, kw_set in PREDICATE_KEYWORDS.items():
        if cand_pred not in active_predicates and (cand_pred in q_words or bool(q_words & kw_set)):
            if compute_fact_hmac("__deleted_predicate__", cand_pred, "__any__") in suppr_hmacs:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return RetrievalResult(
                    status="abstained",
                    route=route,
                    packed_prompt="I have no memory of that.",
                    claims=[],
                    citations=[],
                    tokens_used=0,
                    latency_ms=latency_ms,
                )

    has_exact_predicate_match = len(matching_active_predicates) > 0

    stopwords = {
        "what", "when", "where", "who", "whom", "whose", "why", "how", "did", "does",
        "is", "are", "was", "were", "the", "a", "an", "to", "in", "on", "of", "for",
        "and", "or", "has", "have", "had", "go", "went", "do", "her", "his", "she",
        "he", "they", "their", "them", "this", "that", "from", "with", "about",
        "would", "could", "should", "can", "will", "after", "before", "during",
        "likely", "realize", "mean", "feel",
    }
    raw_content_terms = {
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 2 and t not in stopwords
    }
    non_subject_terms = raw_content_terms - active_subjects
    query_content_terms = non_subject_terms if non_subject_terms else raw_content_terms
    mentioned_subjects = raw_content_terms & active_subjects
    other_subjects = active_subjects - mentioned_subjects

    sensitive_terms = {
        "vault", "secret", "secrets", "password", "passcode", "passphrase",
        "key", "keys", "token", "tokens", "credential", "credentials",
        "pin", "ssn", "passport", "account", "routing", "cipher",
        "recovery", "break",
    }
    q_low = query.lower()
    is_sensitive_query = bool((q_words | raw_content_terms) & sensitive_terms) or any(
        s in q_low for s in ("vault", "secret", "passphrase", "password", "passcode", "credential")
    )

    def _stem(tok: str) -> str:
        for suf in ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "ly", "er", "s"):
            if len(tok) > len(suf) + 2 and tok.endswith(suf):
                base = tok[: -len(suf)]
                return base + ("y" if suf in ("ied", "ies") else "")
        return tok

    def _term_in_vocab(qt: str, vocab: Set[str]) -> bool:
        if qt in vocab:
            return True
        qs = _stem(qt)
        if qs in vocab or any(_stem(vt) == qs for vt in vocab):
            return True
        if len(qs) < 4:
            return False
        pref = qs[:5] if len(qs) >= 5 else qs
        return any(
            (
                "_" not in vt
                and len(vt) >= 4
                and abs(len(qs) - len(vt)) <= 3
                and (vt.startswith(pref) or (len(vt) >= 5 and qs.startswith(vt[:5])))
            )
            for vt in vocab
        )

    first_person_own_re = re.compile(r"\b(?:my|mine|our|ours|we|we're|we've)\b", re.I)
    first_person_pron_re = re.compile(r"\b(?:i|i'm|i've|i'd|me)\b", re.I)
    reactive_first_person_re = re.compile(
        r"\b(?:i\s+(?:love|like|think|bet|hear|heard|know|see|saw|remember|wonder|hope|wish)|"
        r"i(?:'m|\s+am)\s+(?:so\s+|really\s+|very\s+)?(?:glad|happy|excited|proud|sorry|sure|curious)|"
        r"happy\s+for\s+you|proud\s+of\s+you|good\s+for\s+you)\b",
        re.I,
    )
    second_person_re = re.compile(r"\b(?:you|you're|you've|you'd|your|yours|yourself)\b", re.I)

    def _is_second_person_attribution(text: str) -> bool:
        body = re.sub(r"^(?:\[[^\]]+\]\s*)?[A-Za-z0-9_]+:\s*", "", text).strip()
        clauses = [c.strip() for c in re.split(r"(?<=[.!?;])\s+", body) if c.strip()]
        flagged_2p = False
        for clause in clauses:
            clause_terms = set(re.findall(r"[a-z0-9]+", clause.lower()))
            has_topic = any(_term_in_vocab(qt, clause_terms) for qt in query_content_terms)
            if not has_topic:
                continue

            has_2p = bool(second_person_re.search(clause))
            has_1p_own = bool(first_person_own_re.search(clause))
            has_1p_pron = bool(first_person_pron_re.search(clause))
            is_reactive_1p = bool(reactive_first_person_re.search(clause))

            if has_1p_own or (has_1p_pron and not is_reactive_1p and not has_2p and not clause.endswith("?")):
                return False

            if clause.endswith("?"):
                flagged_2p = True
                continue

            if has_2p and not has_1p_own and (not has_1p_pron or is_reactive_1p):
                flagged_2p = True
                continue

            if not has_1p_own and not has_1p_pron:
                for other in other_subjects:
                    nick = other[:3] if len(other) >= 4 else other
                    if other in clause_terms or (len(nick) >= 3 and nick in clause_terms):
                        flagged_2p = True
                        break
        return flagged_2p

    # Filter semantic hits: separate strict abstention gate from secondary evidence expansion
    strict_vec_hits: List[Dict[str, Any]] = []
    support_vec_hits: List[Dict[str, Any]] = []
    for h in active_vec_hits:
        dist = float(h.get("distance", 999.0))
        pred = h.get("predicate", "")
        if pred not in ("has_note", "stated"):
            if is_sensitive_query and pred not in matching_active_predicates and not (set(pred.split("_")) & sensitive_terms):
                continue
            max_uncond_dist = 0.72 if is_sensitive_query else 0.76
            max_lex_dist = 0.80 if is_sensitive_query else 0.84
            if dist <= max_uncond_dist:
                if is_sensitive_query and query_content_terms:
                    struct_text = f"{h.get('subject', '')} {pred.replace('_', ' ')} {h.get('object_val', '')}".lower()
                    struct_terms = set(re.findall(r"[a-z0-9]+", struct_text))
                    if not (query_content_terms & struct_terms):
                        continue
                strict_vec_hits.append(h)
                support_vec_hits.append(h)
            elif dist <= max_lex_dist:
                if not query_content_terms:
                    strict_vec_hits.append(h)
                    support_vec_hits.append(h)
                else:
                    struct_text = f"{h.get('subject', '')} {pred.replace('_', ' ')} {h.get('object_val', '')}".lower()
                    struct_terms = set(re.findall(r"[a-z0-9]+", struct_text))
                    if query_content_terms & struct_terms:
                        strict_vec_hits.append(h)
                        support_vec_hits.append(h)
            continue

        if not query_content_terms:
            if dist <= 0.68:
                strict_vec_hits.append(h)
                support_vec_hits.append(h)
            continue

        obj_text = str(h.get("object_val", "")).lower()
        obj_terms = set(re.findall(r"[a-z0-9_]+", obj_text))
        matched_q = sum(1 for qt in query_content_terms if _term_in_vocab(qt, obj_terms))
        term_overlap = matched_q / len(query_content_terms)

        hit_subj = str(h.get("subject", "")).lower()
        if mentioned_subjects and hit_subj not in mentioned_subjects:
            if not is_sensitive_query and dist <= 0.64 and term_overlap >= 0.50:
                strict_vec_hits.append(h)
                support_vec_hits.append(h)
            continue

        if is_sensitive_query:
            strict_sens_terms = query_content_terms - {"session", "earlier", "memory", "stored", "recorded", "namespace", "temporary", "emergency", "migration", "cutover"}
            sens_matched = sum(1 for qt in strict_sens_terms if _term_in_vocab(qt, obj_terms))
            sens_overlap = sens_matched / max(len(strict_sens_terms), 1)
            passes_strict = (
                (dist <= 0.68 and sens_matched >= 1 and sens_overlap >= 0.25)
                or (dist <= 0.80 and sens_matched >= 2 and sens_overlap >= 0.40)
            )
            passes_support = passes_strict
        elif mentioned_subjects and _is_second_person_attribution(str(h.get("object_val", ""))):
            passes_strict = False
            passes_support = False
        elif mentioned_subjects:
            passes_strict = (
                dist <= 0.68
                or (dist <= 0.76 and matched_q >= 1)
                or (dist <= 0.88 and matched_q >= 2 and term_overlap >= 0.25)
                or (dist <= 0.92 and term_overlap >= 0.50)
            )
            passes_support = passes_strict
        else:
            passes_strict = (
                dist <= 0.72
                or (dist <= 0.84 and matched_q >= 1)
                or (dist <= 0.90 and term_overlap >= 0.25)
                or (dist <= 0.95 and term_overlap >= 0.34)
            )
            passes_support = passes_strict

        h["_strict"] = bool(passes_strict)
        h["_overlap"] = float(term_overlap)
        if passes_strict:
            strict_vec_hits.append(h)
        if passes_support:
            support_vec_hits.append(h)

    qualifying_vec_hits = support_vec_hits if strict_vec_hits else []

    # Step 3: Strict Abstention Gate
    if not has_exact_predicate_match and not qualifying_vec_hits:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            status="abstained",
            route=route,
            packed_prompt="I have no memory of that.",
            claims=[],
            citations=[],
            tokens_used=0,
            latency_ms=latency_ms,
        )

    # Step 4: Route-specific claim gathering
    matched_rows: List[Dict[str, Any]] = []
    seen_claim_ids: Set[str] = set()

    def _add_row(row_dict: Dict[str, Any]):
        cid = row_dict["claim_id"]
        if cid not in seen_claim_ids:
            seen_claim_ids.add(cid)
            matched_rows.append(row_dict)

    if route == "TEMPORAL_HISTORY":
        with store.sm.get_connection(namespace) as conn:
            if matching_active_predicates:
                placeholders = ",".join("?" for _ in matching_active_predicates)
                rows = conn.execute(
                    f"""SELECT c.*, s.file_path, s.origin, s.trust_tier
                        FROM claims_index c
                        JOIN sources_index s ON c.source_id = s.source_id
                        WHERE c.namespace = ? AND c.status IN ('active', 'superseded')
                          AND c.predicate IN ({placeholders})
                        ORDER BY c.valid_from ASC""",
                    [namespace, *matching_active_predicates],
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT c.*, s.file_path, s.origin, s.trust_tier
                       FROM claims_index c
                       JOIN sources_index s ON c.source_id = s.source_id
                       WHERE c.namespace = ? AND c.status IN ('active', 'superseded')
                         AND c.predicate != 'has_note'
                       ORDER BY c.valid_from ASC""",
                    (namespace,),
                ).fetchall()
        for r in rows:
            _add_row(dict(r))

    elif route == "GRAPH_CTE":
        # Find candidate root subjects from query words or existing active subjects
        with store.sm.get_connection(namespace) as conn:
            subj_rows = conn.execute(
                "SELECT DISTINCT subject FROM claims_index WHERE namespace = ? AND status = 'active'",
                (namespace,),
            ).fetchall()
        candidate_roots = [
            r["subject"] for r in subj_rows if r["subject"].lower() in q_words
        ]
        if not candidate_roots and subj_rows:
            candidate_roots = [r["subject"] for r in subj_rows[:3]]

        for root in candidate_roots:
            hops = store.query_graph_multihop(namespace, root, max_hops=5)
            if hops:
                hop_ids = [h["claim_id"] for h in hops]
                placeholders = ",".join("?" for _ in hop_ids)
                with store.sm.get_connection(namespace) as conn:
                    full_rows = conn.execute(
                        f"""SELECT c.*, s.file_path, s.origin, s.trust_tier
                            FROM claims_index c
                            JOIN sources_index s ON c.source_id = s.source_id
                            WHERE c.claim_id IN ({placeholders})""",
                        hop_ids,
                    ).fetchall()
                for r in full_rows:
                    _add_row(dict(r))

    elif route == "SQL_AGGREGATE":
        with store.sm.get_connection(namespace) as conn:
            if matching_active_predicates:
                placeholders = ",".join("?" for _ in matching_active_predicates)
                rows = conn.execute(
                    f"""SELECT c.*, s.file_path, s.origin, s.trust_tier
                        FROM claims_index c
                        JOIN sources_index s ON c.source_id = s.source_id
                        WHERE c.namespace = ? AND c.status = 'active'
                          AND c.predicate IN ({placeholders})
                        ORDER BY c.predicate, c.valid_from DESC""",
                    [namespace, *matching_active_predicates],
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT c.*, s.file_path, s.origin, s.trust_tier
                       FROM claims_index c
                       JOIN sources_index s ON c.source_id = s.source_id
                       WHERE c.namespace = ? AND c.status = 'active'
                         AND c.predicate != 'has_note'
                       ORDER BY c.predicate, c.valid_from DESC""",
                    (namespace,),
                ).fetchall()
        for r in rows:
            _add_row(dict(r))

    else:
        # HYBRID_TIERED: prioritize specific active predicates over stated/has_note fallback, newest first
        if matching_active_predicates:
            placeholders = ",".join("?" for _ in matching_active_predicates)
            with store.sm.get_connection(namespace) as conn:
                rows = conn.execute(
                    f"""SELECT c.*, s.file_path, s.origin, s.trust_tier
                        FROM claims_index c
                        JOIN sources_index s ON c.source_id = s.source_id
                        WHERE c.namespace = ? AND c.status = 'active'
                          AND c.predicate IN ({placeholders})
                        ORDER BY CASE WHEN c.predicate IN ('stated', 'has_note') THEN 1 ELSE 0 END ASC,
                                 c.valid_from DESC,
                                 c.rowid DESC""",
                    [namespace, *matching_active_predicates],
                ).fetchall()
            for r in rows:
                _add_row(dict(r))

        if not matched_rows:
            sorted_vec_hits = sorted(
                qualifying_vec_hits,
                key=lambda h: (
                    1 if h.get("predicate") in ("stated", "has_note") else 0,
                    0 if h.get("_strict", True) else 1,
                    float(h.get("distance", 999.0)),
                ),
            )
            note_count = 0
            for h in sorted_vec_hits:
                if h.get("predicate") in ("stated", "has_note"):
                    if note_count >= 10:
                        continue
                    note_count += 1
                _add_row(h)

            if not is_sensitive_query and note_count < 10:
                anchor_notes = [
                    r for r in matched_rows
                    if r.get("predicate") in ("stated", "has_note") and r.get("file_path")
                ][:2]
                if anchor_notes:
                    with store.sm.get_connection(namespace) as conn:
                        for anchor in anchor_notes:
                            if note_count >= 10:
                                break
                            rid_row = conn.execute(
                                "SELECT rowid FROM claims_index WHERE claim_id = ?",
                                (anchor["claim_id"],),
                            ).fetchone()
                            if not rid_row:
                                continue
                            rid = rid_row["rowid"]
                            adj_rows = conn.execute(
                                """SELECT c.*, s.file_path, s.origin, s.trust_tier
                                   FROM claims_index c
                                   JOIN sources_index s ON c.source_id = s.source_id
                                   WHERE c.namespace = ?
                                     AND c.status = 'active'
                                     AND c.predicate IN ('has_note', 'stated')
                                     AND s.file_path = ?
                                     AND c.rowid IN (?, ?)
                                   ORDER BY c.rowid ASC""",
                                (namespace, anchor["file_path"], rid - 1, rid + 1),
                            ).fetchall()
                            for ar in adj_rows:
                                if note_count >= 10:
                                    break
                                ar_dict = dict(ar)
                                if ar_dict["claim_id"] not in seen_claim_ids:
                                    _add_row(ar_dict)
                                    note_count += 1

    if not matched_rows:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            status="abstained",
            route=route,
            packed_prompt="I have no memory of that.",
            claims=[],
            citations=[],
            tokens_used=0,
            latency_ms=latency_ms,
        )

    # Step 5: Tiered Context Packing (L0 + L1 + L2) within token_budget
    shard_dir = store.sm.get_shard_dir(namespace)
    abstract_file = shard_dir / ".abstract.md"
    l0_content = abstract_file.read_text(encoding="utf-8").strip() if abstract_file.exists() else ""
    if l0_content and len(enc.encode(l0_content)) > 120:
        filtered_l0 = [
            ln for ln in l0_content.splitlines()
            if not ln.lstrip().startswith("- has_note:")
        ]
        l0_content = "\n".join(filtered_l0).strip()

    header_sections: List[str] = [f"# Retrieved Knowledge Context (Route: {route})"]
    if l0_content:
        header_sections.append(f"## OpenViking Tier L0 (Core Identity)\n{l0_content}")

    soft_pack_budget = min(token_budget, 660) if token_budget >= 750 else token_budget
    l1_lines: List[str] = ["## Tier L1 Active Triples"]
    l2_lines: List[str] = ["## Tier L2 Exact Quotes & Provenance"]
    citations: List[str] = []
    citation_details: List[Citation] = []
    seen_sources: Set[str] = set()
    packed_rows: List[Dict[str, Any]] = []

    for c in matched_rows:
        status_val = c.get("status", "active")
        status_suffix = f" [status={status_val}]" if status_val != "active" else ""
        obj_disp = str(c["object_val"])
        if c.get("predicate") in ("has_note", "stated"):
            if len(obj_disp) > 80:
                obj_disp = obj_disp[:77] + "..."
            cand_l1 = f"- **{c['subject']}**: {obj_disp}{status_suffix} [^{c['source_id']}]"
        else:
            cand_l1 = (
                f"- **{c['subject']}** -> **{c['predicate']}**: {obj_disp} "
                f"(since {c['valid_from']}){status_suffix} [^{c['source_id']}]"
            )

        span_text = store.read_verified_span(
            namespace, c["file_path"], c["byte_start"], c["byte_end"]
        )
        new_source = bool(span_text and "[REDACTED" not in span_text and c["source_id"] not in seen_sources)
        cand_l2 = (
            f"[^{c['source_id']}]: \"{span_text}\" ({c['file_path']} bytes {c['byte_start']}-{c['byte_end']})"
            if new_source
            else None
        )

        trial_l1 = "\n".join([*l1_lines, cand_l1])
        trial_l2 = "\n".join([*l2_lines, cand_l2]) if cand_l2 else ("\n".join(l2_lines) if len(l2_lines) > 1 else "")
        trial_sections = [*header_sections, trial_l1] + ([trial_l2] if trial_l2 else [])
        trial_tokens = len(enc.encode("\n\n".join(trial_sections)))
        cap = token_budget if len(packed_rows) < 2 else soft_pack_budget
        if packed_rows and trial_tokens > cap:
            continue

        packed_rows.append(c)
        l1_lines.append(cand_l1)
        if span_text and "[REDACTED" not in span_text:
            cite_tag = f"[^{c['source_id']}]"
            if cite_tag not in citations:
                citations.append(cite_tag)
            if cand_l2 is not None:
                seen_sources.add(c["source_id"])
                l2_lines.append(cand_l2)
                citation_details.append(
                    Citation(
                        source_id=c["source_id"],
                        file_path=c["file_path"],
                        byte_start=c["byte_start"],
                        byte_end=c["byte_end"],
                        exact_quote=span_text,
                    )
                )

    context_sections = [*header_sections, "\n".join(l1_lines)]
    if len(l2_lines) > 1:
        context_sections.append("\n".join(l2_lines))

    raw_packed = "\n\n".join(context_sections)

    # Step 6: Hard Token Budget Ceiling Enforcement
    raw_tokens = enc.encode(raw_packed)
    if len(raw_tokens) > token_budget:
        packed_prompt = enc.decode(raw_tokens[:token_budget])
    else:
        packed_prompt = raw_packed

    tokens_used = len(enc.encode(packed_prompt))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    claim_records = [_row_to_claim_record(r, namespace) for r in packed_rows]

    return RetrievalResult(
        status="ok",
        route=route,
        packed_prompt=packed_prompt,
        claims=claim_records,
        citations=citations,
        citation_details=citation_details,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )
