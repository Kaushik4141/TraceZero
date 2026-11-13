from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import upload_cli
from core import MemoryStore, ingest_message, recall_context
from core.ingest import extract_heuristic_candidates
from core.policy import verify_exact_quote_span


def test_json_and_yaml_kv_extraction_preserves_exact_spans():
    cfg_path = REPO_ROOT / "evals" / "sample_docs" / "sample_config.json"
    raw_json = cfg_path.read_text(encoding="utf-8")

    candidates = extract_heuristic_candidates(raw_json, default_subject="emp_042")
    pred_map = {(c.subject, c.predicate): c for c in candidates}

    assert pred_map[("orion_payments", "preferred_db")].object_val == "postgresql"
    assert pred_map[("orion_payments", "aws_region")].object_val == "us-west-2"
    assert pred_map[("orion_payments", "primary_language")].object_val == "python"
    assert pred_map[("orion_payments", "owner")].object_val == "emp_042"
    assert ("emp_042", "works_on") in pred_map

    for cand in candidates:
        assert verify_exact_quote_span(raw_json, cand.exact_quote) is not None

    yaml_text = "service: billing_api\npreferred_database: CockroachDB\ncloud_region: eu-west-1\n"
    yaml_cands = extract_heuristic_candidates(yaml_text, default_subject="emp_042")
    yaml_map = {(c.subject, c.predicate): c for c in yaml_cands}
    assert yaml_map[("billing_api", "preferred_db")].object_val == "CockroachDB"
    assert yaml_map[("billing_api", "aws_region")].object_val == "eu-west-1"
    for cand in yaml_cands:
        assert verify_exact_quote_span(yaml_text, cand.exact_quote) is not None


def test_chunk_document_text_splits_sections_and_caps_size():
    arch_path = REPO_ROOT / "evals" / "sample_docs" / "sample_architecture.md"
    arch_text = arch_path.read_text(encoding="utf-8")

    chunks = upload_cli.chunk_document_text(arch_text, doc_type="MARKDOWN", max_chars=400)
    assert len(chunks) == 2
    assert "Orion Payments" in chunks[0]
    assert "Key Design Decisions" in chunks[1]
    assert all(len(c) <= 400 for c in chunks)

    long_prose = "\n\n".join(
        f"## Section {i}\n" + ("SQLite WAL mode ensures deterministic crash recovery. " * 12)
        for i in range(4)
    )
    long_chunks = upload_cli.chunk_document_text(long_prose, doc_type="MARKDOWN", max_chars=400)
    assert len(long_chunks) >= 8
    assert all(len(c) <= 400 for c in long_chunks)

    small_json = json.dumps({"service": "orion_payments", "preferred_db": "postgresql"}, indent=2)
    assert upload_cli.chunk_document_text(small_json, doc_type="JSON", max_chars=400) == [small_json]


def test_upload_sample_docs_cli_and_end_to_end_recall(tmp_path):
    sample_dir = REPO_ROOT / "evals" / "sample_docs"
    assert upload_cli.main([str(sample_dir), "--dry-run"]) == 0

    store = MemoryStore(root_dir=str(tmp_path))
    ns = "acme_corp/users/emp_042"
    files, is_bulk = upload_cli.collect_target_files(sample_dir)
    assert is_bulk and len(files) == 3

    for fp in files:
        doc_type, text = upload_cli.extract_any_document(fp)
        for chunk in upload_cli.chunk_document_text(text, doc_type=doc_type):
            claims = ingest_message(
                store=store,
                namespace=ns,
                text=chunk,
                origin="user",
                session_id=fp.stem,
            )
            assert claims

    db_recall = recall_context(store, ns, "What is the preferred database of orion_payments?", token_budget=800)
    assert db_recall.status == "ok"
    assert "postgresql" in db_recall.packed_prompt.lower()

    design_recall = recall_context(store, ns, "How are embeddings stored in Key Design Decisions?", token_budget=800)
    assert design_recall.status == "ok"
    assert "sqlite-vec" in design_recall.packed_prompt

    metrics_recall = recall_context(store, ns, "What is recall_latency_p50?", token_budget=800)
    assert metrics_recall.status == "ok"
    assert "14.2" in metrics_recall.packed_prompt
