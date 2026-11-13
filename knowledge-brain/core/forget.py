import hashlib
import hmac
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from core.contracts import DeletionReceipt
from core.okf_compiler import compile_namespace_okf
from core.schema import ShardManager, serialize_f32
from core.verifier import compute_shard_hashes, scan_shard_bytes

HMAC_SECRET = os.getenv("HMAC_SECRET", "sahyadri-hackathon-secret-key").encode("utf-8")


def compute_fact_hmac(subject: str, predicate: str, object_val: str) -> str:
    payload = (
        f"{subject.lower().strip()}|{predicate.lower().strip()}|{object_val.lower().strip()}"
    ).encode("utf-8")
    return hmac.new(HMAC_SECRET, payload, hashlib.sha256).hexdigest()


def forget_target(
    shard_manager: ShardManager,
    namespace: str,
    target_predicate_or_value: str,
    query_embedding: Optional[List[float]] = None,
) -> DeletionReceipt:
    shard_dir = shard_manager.get_shard_dir(namespace)
    now = datetime.now(timezone.utc).isoformat()

    conn = shard_manager.get_connection(namespace)
    target_lower = target_predicate_or_value.lower().strip()

    try:
        # 1. Find matching claims by predicate or object substring
        matched_rows = conn.execute(
            """SELECT c.*, l.vec_rowid, l.fts_rowid, s.file_path
               FROM claims_index c
               JOIN lineage_index l ON c.claim_id = l.claim_id
               JOIN sources_index s ON c.source_id = s.source_id
               WHERE LOWER(c.predicate) = ? OR LOWER(c.object_val) LIKE ?""",
            (target_lower, f"%{target_lower}%"),
        ).fetchall()

        # 2. Optional semantic neighbor catch for paraphrased free-text notes
        if query_embedding is not None:
            vec_rows = conn.execute(
                """SELECT c.*, l.vec_rowid, l.fts_rowid, s.file_path, v.distance
                   FROM vec_claims v
                   JOIN lineage_index l ON l.vec_rowid = v.rowid
                   JOIN claims_index c ON c.claim_id = l.claim_id
                   JOIN sources_index s ON c.source_id = s.source_id
                   WHERE v.embedding MATCH ? AND k = 5""",
                (serialize_f32(query_embedding),),
            ).fetchall()
            existing_ids = {r["claim_id"] for r in matched_rows}
            for vr in vec_rows:
                if vr["distance"] < 0.65 and vr["claim_id"] not in existing_ids:
                    matched_rows.append(vr)

        deleted_claim_ids: List[str] = []
        redacted_spans: List[str] = []
        suppression_hmacs: List[str] = []
        forbidden_strings: List[str] = [target_predicate_or_value]

        for row in matched_rows:
            cid = row["claim_id"]
            deleted_claim_ids.append(cid)
            forbidden_strings.append(row["object_val"])

            # Record suppression HMAC for both exact fact and predicate tombstones
            fhmac = compute_fact_hmac(row["subject"], row["predicate"], row["object_val"])
            pred_tombstone = compute_fact_hmac("__deleted_predicate__", row["predicate"], "__any__")
            suppression_hmacs.append(fhmac)
            conn.execute(
                "INSERT OR REPLACE INTO suppression_hashes (fact_hmac, deleted_at) VALUES (?, ?)",
                (fhmac, now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO suppression_hashes (fact_hmac, deleted_at) VALUES (?, ?)",
                (pred_tombstone, now),
            )

            # Redact exact byte span in raw L2 source file on disk
            src_file = shard_dir / row["file_path"]
            b_start, b_end = row["byte_start"], row["byte_end"]
            if src_file.exists() and b_end > b_start:
                with open(src_file, "r+b") as f:
                    f.seek(b_start)
                    span_len = b_end - b_start
                    replacement = (b"[REDACTED]" * ((span_len // 10) + 1))[:span_len]
                    f.write(replacement)
                    f.flush()
                    os.fsync(f.fileno())
                redacted_spans.append(f"{row['file_path']}:{b_start}:{b_end}")
                # Update source file sha256 after redaction
                new_sha = hashlib.sha256(src_file.read_bytes()).hexdigest()
                conn.execute(
                    "UPDATE sources_index SET sha256_hash = ? WHERE source_id = ?",
                    (new_sha, row["source_id"]),
                )

            # Cascade delete from vector, FTS, lineage, and claims tables
            conn.execute("DELETE FROM vec_claims WHERE rowid = ?", (row["vec_rowid"],))
            conn.execute("DELETE FROM fts_claims WHERE rowid = ?", (row["fts_rowid"],))
            conn.execute("DELETE FROM lineage_index WHERE claim_id = ?", (cid,))
            conn.execute("DELETE FROM claims_index WHERE claim_id = ?", (cid,))

        # 3. Rebuild FTS5 shadow tables, truncate WAL, and VACUUM to wipe freed pages
        conn.execute("INSERT INTO fts_claims(fts_claims) VALUES('rebuild');")
        conn.commit()
        conn.isolation_level = None
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.execute("PRAGMA secure_delete = ON;")
        conn.execute("VACUUM;")
    finally:
        conn.close()

    # 4. Regenerate L1 (.okf.md) and L0 (.abstract.md) files without the deleted claims
    compile_namespace_okf(shard_manager, namespace)

    # 5. Byte-level scan verification across entire shard directory
    byte_matches = scan_shard_bytes(shard_dir, forbidden_strings)
    file_hashes = compute_shard_hashes(shard_dir)

    return DeletionReceipt(
        receipt_id=f"del_{uuid.uuid4().hex[:8]}",
        namespace=namespace,
        target_query=target_predicate_or_value,
        deleted_claim_ids=deleted_claim_ids,
        redacted_source_spans=redacted_spans,
        suppression_hmacs=suppression_hmacs,
        fts_rebuilt=True,
        vacuum_completed=True,
        byte_scan_matches=byte_matches,
        scrubbed_files_sha256=file_hashes,
        timestamp=now,
    )
