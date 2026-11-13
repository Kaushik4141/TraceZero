import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastembed import TextEmbedding
from core.contracts import ClaimRecord, ClaimStatus, DeletionReceipt, SourceRecord, TrustTier
from core.forget import compute_fact_hmac, forget_target
from core.okf_compiler import compile_namespace_okf
from core.predicates import is_single_valued
from core.schema import ShardManager, serialize_f32

# Singleton embedding model so it only loads once per process
_EMBED_MODEL: Optional[TextEmbedding] = None


def get_embedder() -> TextEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL


def embed_text(text: str) -> List[float]:
    model = get_embedder()
    vec = next(model.embed([text]))
    return vec.tolist()


class MemoryStore:
    """Unified Storage API that Role 2 (Ingest/Policy/Retrieval) and Role 3 (MCP) call."""

    def __init__(self, root_dir: str = "memory_root"):
        self.sm = ShardManager(root_dir=root_dir)

    # --- 1. ENTITY ALIAS RESOLUTION ---
    def register_alias(self, namespace: str, alias: str, canonical_subject: str):
        with self.sm.get_connection(namespace) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias, canonical_subject) VALUES (?, ?)",
                (alias.lower().strip(), canonical_subject.lower().strip()),
            )
            conn.commit()

    def resolve_entity(self, namespace: str, subject: str) -> str:
        clean = subject.lower().strip()
        with self.sm.get_connection(namespace) as conn:
            row = conn.execute(
                "SELECT canonical_subject FROM entity_aliases WHERE alias = ?",
                (clean,),
            ).fetchone()
            return row["canonical_subject"] if row else clean

    # --- 2. SUPPRESSION CHECK (BLOCKS RE-LEARNING DELETED FACTS) ---
    def is_suppressed(self, namespace: str, subject: str, predicate: str, object_val: str) -> bool:
        fhmac = compute_fact_hmac(subject, predicate, object_val)
        with self.sm.get_connection(namespace) as conn:
            row = conn.execute(
                "SELECT 1 FROM suppression_hashes WHERE fact_hmac = ?",
                (fhmac,),
            ).fetchone()
            return row is not None

    # --- 3. WRITE PATH FOR ROLE 2 (POLICY ENGINE) ---
    def save_source(
        self, namespace: str, source_id: str, filename: str, text: str, origin: str, trust_tier: TrustTier
    ) -> SourceRecord:
        return self.sm.append_raw_source(namespace, source_id, filename, text, origin, trust_tier)

    def get_active_claims_for_slot(self, namespace: str, subject: str, predicate: str) -> List[Dict[str, Any]]:
        canon_subj = self.resolve_entity(namespace, subject)
        with self.sm.get_connection(namespace) as conn:
            rows = conn.execute(
                """SELECT c.*, s.trust_tier, s.origin
                   FROM claims_index c
                   JOIN sources_index s ON c.source_id = s.source_id
                   WHERE c.subject = ? AND c.predicate = ? AND c.status = 'active'""",
                (canon_subj, predicate.lower().strip()),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_claim_status(self, namespace: str, claim_id: str, new_status: ClaimStatus, valid_to: Optional[str] = None):
        with self.sm.get_connection(namespace) as conn:
            conn.execute(
                "UPDATE claims_index SET status = ?, valid_to = COALESCE(?, valid_to) WHERE claim_id = ?",
                (str(new_status.value), valid_to, claim_id),
            )
            conn.commit()
        compile_namespace_okf(self.sm, namespace)

    def store_claim(self, claim: ClaimRecord) -> bool:
        """Embeds and stores a claim + updates .okf.md files. Returns False if suppressed."""
        claim.subject = self.resolve_entity(claim.namespace, claim.subject)
        if self.is_suppressed(claim.namespace, claim.subject, claim.predicate, claim.object_val):
            return False

        if claim.predicate in ("has_note", "stated"):
            clean_val = re.sub(
                r"^(?:\[[^\]]+\]\s*)?[A-Za-z0-9_-]{1,30}:\s*",
                "",
                claim.object_val,
            ).strip()
            semantic_text = f"{claim.subject}: {clean_val or claim.object_val}"
        else:
            semantic_text = f"{claim.subject} {claim.predicate} {claim.object_val}"
        vec = embed_text(semantic_text)
        self.sm.insert_claim_with_lineage(claim, vec)
        if claim.status == ClaimStatus.ACTIVE:
            compile_namespace_okf(self.sm, claim.namespace)
        return True

    # --- 4. READ PATHS FOR ROLE 2 (HYBRID, GRAPH CTE, TEMPORAL, L2 QUOTE) ---
    def search_hybrid(
        self, namespace: str, query: str, top_k: int = 8, include_superseded: bool = False
    ) -> List[Dict[str, Any]]:
        q_vec = embed_text(query)
        q_vec_bytes = serialize_f32(q_vec)
        allowed_statuses = ("active", "disputed", "superseded") if include_superseded else ("active", "disputed")
        placeholders = ",".join("?" for _ in allowed_statuses)
        vec_k = max(top_k * 2, 64)

        with self.sm.get_connection(namespace) as conn:
            vec_sql = f"""
                SELECT c.*, s.file_path, s.origin, s.trust_tier, v.distance
                FROM vec_claims v
                JOIN lineage_index l ON l.vec_rowid = v.rowid
                JOIN claims_index c ON c.claim_id = l.claim_id
                JOIN sources_index s ON s.source_id = c.source_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND c.status IN ({placeholders})
                ORDER BY v.distance ASC
            """
            rows = [dict(r) for r in conn.execute(vec_sql, (q_vec_bytes, vec_k, *allowed_statuses)).fetchall()]
            rows_by_id: Dict[str, Dict[str, Any]] = {}
            vec_rank_by_id: Dict[str, int] = {}
            for idx, r in enumerate(rows):
                cid = r["claim_id"]
                rows_by_id[cid] = r
                vec_rank_by_id[cid] = idx

            subj_rows = conn.execute(
                "SELECT DISTINCT LOWER(subject) AS subj FROM claims_index WHERE status IN ({})".format(placeholders),
                allowed_statuses,
            ).fetchall()
            shard_subjects = {r["subj"] for r in subj_rows if r["subj"] and r["subj"] != "user"}

            # Complement vector KNN with FTS5 BM25 rare-keyword matches + RRF fusion
            _fts_stop = {
                "what", "when", "where", "who", "whom", "whose", "why", "how", "which",
                "did", "does", "do", "are", "is", "was", "were", "be", "been", "being",
                "the", "and", "for", "with", "from", "into", "onto", "this", "that",
                "have", "has", "had", "not", "near", "would", "could", "should", "will",
                "can", "may", "might", "about", "there", "their", "them", "they", "she",
                "her", "hers", "him", "his", "you", "your", "our", "ours", "still",
                "just", "also", "even", "very", "really", "actually", "recently",
                "get", "got", "going", "went", "make", "made", "take", "took", "give",
                "gave", "come", "came", "see", "saw", "say", "said", "tell", "told",
                "know", "knew", "think", "thought", "want", "wanted", "need", "needed",
                "use", "used", "like", "likely", "thing", "things", "way", "lot", "lots",
                "much", "many", "more", "most", "some", "any", "time", "ago", "year",
                "years", "month", "months", "week", "weeks", "day", "days", "long",
                "during", "before", "after", "since", "between", "both", "each", "other",
            }
            terms = [
                t for t in re.findall(r"[a-z0-9]+", query.lower())
                if len(t) >= 3 and t not in _fts_stop
            ]
            rare_terms = [t for t in terms if t not in shard_subjects] or terms
            fts_rank_by_id: Dict[str, int] = {}
            fts_limit = max(top_k * 2, 48)
            fts_base_sql = f"""
                SELECT c.*, s.file_path, s.origin, s.trust_tier,
                       vec_distance_L2(v.embedding, ?) AS distance,
                       bm25(fts_claims) AS bm25_score
                FROM fts_claims
                JOIN lineage_index l ON l.fts_rowid = fts_claims.rowid
                JOIN claims_index c ON c.claim_id = l.claim_id
                JOIN sources_index s ON s.source_id = c.source_id
                JOIN vec_claims v ON v.rowid = l.vec_rowid
                WHERE fts_claims MATCH ?
                  AND c.status IN ({placeholders})
                ORDER BY bm25_score ASC, distance ASC
                LIMIT ?
            """

            def _run_fts(fts_q: str, lim: int) -> int:
                try:
                    fts_rows = conn.execute(
                        fts_base_sql, (q_vec_bytes, fts_q, *allowed_statuses, lim)
                    ).fetchall()
                    for r in fts_rows:
                        d = dict(r)
                        cid = d["claim_id"]
                        if cid not in fts_rank_by_id:
                            fts_rank_by_id[cid] = len(fts_rank_by_id)
                        if cid not in rows_by_id:
                            rows_by_id[cid] = d
                    return len(fts_rows)
                except sqlite3.OperationalError:
                    return 0

            # Query each rare content term individually first so common terms never crowd out rare ones
            for rt in list(dict.fromkeys(rare_terms))[:6]:
                tok = f'"{rt[:6]}"*' if len(rt) >= 6 else f'"{rt}"*'
                _run_fts(tok, 12)

            for term_set in ([rare_terms] if rare_terms == terms else [rare_terms, terms]):
                if not term_set:
                    continue
                fts_tokens = [f'"{t[:6]}"*' if len(t) >= 6 else f'"{t}"*' for t in dict.fromkeys(term_set[:12])]
                _run_fts(" OR ".join(fts_tokens), fts_limit)
                if len(fts_rank_by_id) >= top_k:
                    break

            rare_set = [t for t in dict.fromkeys(rare_terms) if t not in shard_subjects]
            scored_rows = []
            for cid, d in rows_by_id.items():
                v_rank = vec_rank_by_id.get(cid, vec_k + 25)
                f_rank = fts_rank_by_id.get(cid, fts_limit + 25)
                rrf = (1.0 / (45.0 + v_rank)) + (1.0 / (45.0 + f_rank))
                lex_hits = 0
                if rare_set:
                    obj_low = str(d.get("object_val", "")).lower()
                    lex_hits = sum(1 for t in rare_set if t in obj_low or (len(t) >= 5 and t[:5] in obj_low))
                    rrf += 0.008 * min(lex_hits, 4)
                d["_rrf_score"] = rrf
                d["_lex_hits"] = lex_hits
                scored_rows.append(d)

            scored_rows.sort(
                key=lambda r: (
                    float(r.get("distance", 999.0))
                    - min(0.12, 0.02 * int(r.get("_lex_hits", 0)) + 1.8 * float(r.get("_rrf_score", 0.0))),
                    float(r.get("distance", 999.0)),
                )
            )
            return scored_rows[:top_k]

    def query_graph_multihop(self, namespace: str, start_subject: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Recursive CTE graph traversal over claims_index triples."""
        canon = self.resolve_entity(namespace, start_subject)
        with self.sm.get_connection(namespace) as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE graph_walk(claim_id, subject, predicate, object_val, source_id, hop) AS (
                    SELECT claim_id, subject, predicate, object_val, source_id, 1
                    FROM claims_index
                    WHERE subject = ? AND status = 'active'
                    UNION ALL
                    SELECT c.claim_id, c.subject, c.predicate, c.object_val, c.source_id, g.hop + 1
                    FROM claims_index c
                    JOIN graph_walk g ON c.subject = LOWER(g.object_val)
                    WHERE c.status = 'active' AND g.hop < ?
                )
                SELECT DISTINCT claim_id, subject, predicate, object_val, source_id, hop
                FROM graph_walk
                ORDER BY hop ASC;
                """,
                (canon, max_hops),
            ).fetchall()
            return [dict(r) for r in rows]

    def read_verified_span(self, namespace: str, file_path: str, byte_start: int, byte_end: int) -> str:
        """L2 On-Demand Reader: seeks directly to exact byte offsets on disk."""
        target = self.sm.get_shard_dir(namespace) / file_path
        if not target.exists():
            return ""
        with open(target, "rb") as f:
            f.seek(byte_start)
            return f.read(byte_end - byte_start).decode("utf-8", errors="replace")

    # --- 5. DELETION PATH ---
    def forget(self, namespace: str, target_query: str) -> DeletionReceipt:
        q_vec = embed_text(target_query)
        return forget_target(self.sm, namespace, target_query, query_embedding=q_vec)

    # --- 6. MCP / DASHBOARD / EVAL EXTENSIONS ---
    def get_claim_history(
        self, namespace: str, subject: str, predicate: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all claims (active, superseded, disputed, quarantined) ordered by valid_from ASC."""
        canon_subj = self.resolve_entity(namespace, subject)
        with self.sm.get_connection(namespace) as conn:
            if predicate is not None:
                rows = conn.execute(
                    """SELECT c.*, s.file_path, s.origin, s.trust_tier
                       FROM claims_index c
                       JOIN sources_index s ON c.source_id = s.source_id
                       WHERE c.subject = ? AND c.predicate = ?
                         AND c.status IN ('active', 'superseded', 'disputed', 'quarantined')
                       ORDER BY c.valid_from ASC""",
                    (canon_subj, predicate.lower().strip()),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT c.*, s.file_path, s.origin, s.trust_tier
                       FROM claims_index c
                       JOIN sources_index s ON c.source_id = s.source_id
                       WHERE c.subject = ?
                         AND c.status IN ('active', 'superseded', 'disputed', 'quarantined')
                       ORDER BY c.valid_from ASC""",
                    (canon_subj,),
                ).fetchall()
            return [dict(r) for r in rows]

    def explain_claim(self, namespace: str, claim_id: str) -> Dict[str, Any]:
        """Return full claim provenance, lineage, source metadata, and exact L2 byte-span quote."""
        with self.sm.get_connection(namespace) as conn:
            row = conn.execute(
                """SELECT c.*, s.file_path, s.sha256_hash, s.origin, s.trust_tier,
                          s.created_at AS source_created_at,
                          l.vec_rowid, l.fts_rowid
                   FROM claims_index c
                   JOIN sources_index s ON c.source_id = s.source_id
                   LEFT JOIN lineage_index l ON c.claim_id = l.claim_id
                   WHERE c.claim_id = ?""",
                (claim_id,),
            ).fetchone()
        if not row:
            return {}
        info = dict(row)
        info["exact_quote"] = self.read_verified_span(
            namespace, info["file_path"], info["byte_start"], info["byte_end"]
        )
        return info

    def list_quarantined(self, namespace: str) -> List[Dict[str, Any]]:
        """Return all quarantined or disputed claims with source metadata and exact quote spans."""
        with self.sm.get_connection(namespace) as conn:
            rows = conn.execute(
                """SELECT c.*, s.file_path, s.sha256_hash, s.origin, s.trust_tier
                   FROM claims_index c
                   JOIN sources_index s ON c.source_id = s.source_id
                   WHERE c.status IN ('quarantined', 'disputed')
                   ORDER BY c.valid_from DESC"""
            ).fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["exact_quote"] = self.read_verified_span(
                namespace, item["file_path"], item["byte_start"], item["byte_end"]
            )
            results.append(item)
        return results

    def confirm_quarantined_claim(self, namespace: str, claim_id: str) -> bool:
        """Promote a quarantined or disputed claim to ACTIVE, superseding prior single-valued claims."""
        now = datetime.now(timezone.utc).isoformat()
        with self.sm.get_connection(namespace) as conn:
            target = conn.execute(
                "SELECT * FROM claims_index WHERE claim_id = ? AND status IN ('quarantined', 'disputed')",
                (claim_id,),
            ).fetchone()
            if not target:
                return False

            subj = target["subject"]
            pred = target["predicate"]

            if is_single_valued(pred):
                conn.execute(
                    """UPDATE claims_index
                       SET status = ?, valid_to = ?
                       WHERE subject = ? AND predicate = ? AND status = 'active' AND claim_id != ?""",
                    (ClaimStatus.SUPERSEDED.value, now, subj, pred, claim_id),
                )

            conn.execute(
                "UPDATE claims_index SET status = ?, valid_to = NULL WHERE claim_id = ?",
                (ClaimStatus.ACTIVE.value, claim_id),
            )
            conn.commit()

        compile_namespace_okf(self.sm, namespace)
        return True

    def run_readonly_sql(
        self, namespace: str, sql_query: str, params: tuple = ()
    ) -> List[Dict[str, Any]]:
        """Execute a validated read-only SELECT query against the namespace SQLite shard."""
        stripped = sql_query.strip()
        if not stripped.upper().startswith("SELECT"):
            raise ValueError("Only read-only SELECT queries are permitted.")

        # Ensure shard & tables exist before opening read-only URI connection
        init_conn = self.sm.get_connection(namespace)
        init_conn.close()

        db_path = (self.sm.get_shard_dir(namespace) / "index.sqlite").resolve()
        ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        ro_conn.row_factory = sqlite3.Row
        try:
            ro_conn.execute("PRAGMA query_only = ON;")
            rows = ro_conn.execute(stripped, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            ro_conn.close()

    def list_namespaces(self) -> List[str]:
        """Scan root_dir for all directories containing index.sqlite and return sorted namespace strings."""
        if not self.sm.root_dir.exists():
            return []
        namespaces: List[str] = []
        for db_file in sorted(self.sm.root_dir.rglob("index.sqlite")):
            rel_dir = db_file.parent.relative_to(self.sm.root_dir).as_posix()
            if rel_dir and rel_dir != ".":
                namespaces.append(rel_dir)
        return namespaces


def get_claim_history(
    store: MemoryStore, namespace: str, subject: str, predicate: Optional[str] = None
) -> List[Dict[str, Any]]:
    return store.get_claim_history(namespace, subject, predicate)


def explain_claim(store: MemoryStore, namespace: str, claim_id: str) -> Dict[str, Any]:
    return store.explain_claim(namespace, claim_id)


def list_quarantined(store: MemoryStore, namespace: str) -> List[Dict[str, Any]]:
    return store.list_quarantined(namespace)


def confirm_quarantined_claim(store: MemoryStore, namespace: str, claim_id: str) -> bool:
    return store.confirm_quarantined_claim(namespace, claim_id)


def run_readonly_sql(
    store: MemoryStore, namespace: str, sql_query: str, params: tuple = ()
) -> List[Dict[str, Any]]:
    return store.run_readonly_sql(namespace, sql_query, params)


def list_namespaces(store: MemoryStore) -> List[str]:
    return store.list_namespaces()

