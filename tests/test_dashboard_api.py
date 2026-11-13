"""Automated tests for Knowledge Brain Audit & Provenance Dashboard API."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from dashboard.server import app

client = TestClient(app)
NAMESPACE = "acme_corp/users/emp_042"


def test_serve_index():
    """Verify root / serves the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Knowledge Brain" in response.text
    assert "OpenViking" in response.text


def test_get_namespaces():
    """Verify /api/namespaces returns discovered shards."""
    response = client.get("/api/namespaces")
    assert response.status_code == 200
    data = response.json()
    assert "namespaces" in data
    ns_list = [item["namespace"] for item in data["namespaces"]]
    assert "acme_corp/shared" in ns_list
    assert NAMESPACE in ns_list


def test_get_namespace_details():
    """Verify /api/namespace/emp_042 returns L0, L1, and L2 tiers."""
    response = client.get(f"/api/namespace/{NAMESPACE}")
    assert response.status_code == 200
    data = response.json()
    assert data["namespace"] == NAMESPACE
    assert "l0" in data
    assert "content" in data["l0"]
    assert data["l0"]["tokens"] > 0
    assert "l1_files" in data
    assert len(data["l1_files"]) > 0
    assert "l2_sources" in data
    assert len(data["l2_sources"]) > 0


def test_get_claims():
    """Verify /api/claims returns rows from claims_index."""
    response = client.get(f"/api/claims/{NAMESPACE}")
    assert response.status_code == 200
    data = response.json()
    assert "claims" in data
    claims = data["claims"]
    assert len(claims) > 0
    statuses = {c["status"] for c in claims}
    assert "active" in statuses


def test_live_recall_known_and_abstained():
    """Verify /api/recall returns live answers and honest abstentions."""
    # Known query
    res_known = client.post(
        "/api/recall",
        json={"namespace": NAMESPACE, "query": "Where do I live?", "token_budget": 800},
    )
    assert res_known.status_code == 200
    known_data = res_known.json()
    assert known_data["status"] == "ok"
    assert "Bengaluru" in known_data["answer_hint"]
    assert known_data["tokens_used"] <= 800

    # Unknown query (Honest Abstention)
    res_unknown = client.post(
        "/api/recall",
        json={"namespace": NAMESPACE, "query": "What is my top secret unrecorded flight seat?", "token_budget": 800},
    )
    assert res_unknown.status_code == 200
    unknown_data = res_unknown.json()
    assert unknown_data["status"] == "abstained"
    assert unknown_data["answer_hint"] == "I have no memory of that."
    assert unknown_data["tokens_used"] == 0


def test_get_quarantine():
    """Verify /api/quarantine returns quarantined prompt injections."""
    response = client.get(f"/api/quarantine/{NAMESPACE}")
    assert response.status_code == 200
    data = response.json()
    assert "quarantined" in data


def test_live_byte_scan():
    """Verify /api/byte_scan performs binary inspection across SQLite, WAL, and sources."""
    response = client.post(
        "/api/byte_scan",
        json={"namespace": NAMESPACE, "query": "nonexistent_cryptographic_canary_value_9999"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sqlite_matches"] == 0
    assert data["wal_matches"] == 0
    assert data["source_matches"] == 0
    assert data["total_matches"] == 0
    assert "hmac_hash" in data
    assert "10/10 Probes ABSTAINED" in data["probe_result"]
    assert data["verdict"] == "VERIFIED_ZERO_LEAKAGE"


def test_get_benchmarks():
    """Verify /api/benchmarks returns 5-strategy comparison matrix and token stats."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert len(data["strategies"]) == 5
    assert "session_token_comparison" in data
    assert data["session_token_comparison"]["reduction_percentage"] > 90.0


def test_dashboard_interactive_handoff(tmp_path, monkeypatch):
    """Verify /api/handoff/create and /api/handoff/{namespace} create and restore tasks.okf.md."""
    monkeypatch.setenv("MEMORY_ROOT", str(tmp_path))
    create_res = client.post(
        "/api/handoff/create",
        json={
            "namespace": NAMESPACE,
            "task_id": "dash_handoff_01",
            "goal": "Cut over payment service to PostgreSQL 16.",
            "decisions": ["Target database is PostgreSQL 16."],
            "completed_steps": ["Step 1: Provisioned RDS instance"],
            "pending_steps": ["Step 2: Run dual-write worker"],
        },
    )
    assert create_res.status_code == 200
    created = create_res.json()
    assert created["exists"] is True
    assert created["task_id"] == "dash_handoff_01"
    assert len(created["recorded_claims"]) >= 1

    get_res = client.get(f"/api/handoff/{NAMESPACE}?task_id=dash_handoff_01")
    assert get_res.status_code == 200
    restored = get_res.json()
    assert restored["exists"] is True
    assert restored["goal"] == "Cut over payment service to PostgreSQL 16."
    assert len(restored["active_decisions"]) >= 1
