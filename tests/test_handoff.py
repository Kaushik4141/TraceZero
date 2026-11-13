"""Tests for OKF v0.2 cross-agent task handoff functionality."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import pytest

from core.contracts import HandoffPacket
from core.handoff import create_handoff, get_handoff, count_tokens


@pytest.fixture
def mock_memory_root(tmp_path: Path) -> Path:
    """Creates a temporary memory root copying emp_042 shard."""
    repo_root = Path(__file__).resolve().parent.parent
    src_emp_042 = repo_root / "memory_root" / "acme_corp" / "users" / "emp_042"
    if not src_emp_042.exists():
        src_emp_042 = repo_root / "knowledge-brain" / "memory_root" / "acme_corp" / "users" / "emp_042"

    dest_emp_042 = tmp_path / "acme_corp" / "users" / "emp_042"
    dest_emp_042.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_emp_042, dest_emp_042)

    return tmp_path


def test_create_handoff_success(mock_memory_root: Path):
    namespace = "acme_corp/users/emp_042"
    task_id = "task_relocation_verification_01"
    goal = "Verify employee relocation to Bengaluru and update distributed cache."
    completed_steps = [
        "Ingested session_12 transcript.",
        "Identified new location claim for Bengaluru.",
    ]
    pending_steps = [
        "Update geo-routing cache.",
        "Notify facilities team of workstation transfer.",
    ]
    open_questions = [
        "Does employee require parking spot allocation in Bengaluru?",
    ]

    packet = create_handoff(
        namespace=namespace,
        task_id=task_id,
        goal=goal,
        completed_steps=completed_steps,
        pending_steps=pending_steps,
        open_questions=open_questions,
        created_by_agent="agent_onboarding",
        memory_root=mock_memory_root,
    )

    # Validate HandoffPacket
    assert isinstance(packet, HandoffPacket)
    assert packet.okf_version == "0.2"
    assert packet.namespace == namespace
    assert packet.task_id == task_id
    assert packet.status == "in_progress"
    assert packet.created_by_agent == "agent_onboarding"
    assert packet.goal == goal
    assert packet.completed_steps == completed_steps
    assert packet.pending_steps == pending_steps
    assert packet.open_questions == open_questions
    assert "claim_042_loc_active" in packet.relied_on_claims
    # Superseded claim must not be in relied_on_claims
    assert "claim_042_loc_superseded" not in packet.relied_on_claims

    # Verify active decisions structure
    active_claims = [d for d in packet.active_decisions if d["claim_id"] == "claim_042_loc_active"]
    assert len(active_claims) == 1
    assert active_claims[0]["predicate"] == "lives_in"
    assert active_claims[0]["object_val"] == "Bengaluru"
    assert active_claims[0]["citation"] == "[^src_12]"

    # Check tasks.okf.md file exists on disk
    okf_file = mock_memory_root / namespace / "tasks.okf.md"
    assert okf_file.exists()
    content = okf_file.read_text(encoding="utf-8")

    # Verify frontmatter and section contents
    assert 'okf_version: "0.2"' in content or "okf_version: '0.2'" in content
    assert f"task_id: {task_id}" in content
    assert "# Task Handoff: task_relocation_verification_01" in content
    assert "## Goal" in content
    assert goal in content
    assert "## Completed Steps" in content
    assert "## Pending Steps" in content
    assert "## Active Decisions & Provenance" in content
    assert "## Open Questions" in content
    assert "## Evidence & Citations" in content
    assert "[^src_12]:" in content

    # Check token statistics
    assert packet.handoff_tokens > 0
    assert packet.full_history_tokens > 0
    assert packet.full_history_tokens >= 147


def test_token_compression_with_large_history(mock_memory_root: Path):
    namespace = "acme_corp/users/emp_042"
    task_id = "task_compression_demo"

    # Add a large session log to sources to simulate realistic conversational history
    sources_dir = mock_memory_root / namespace / "sources"
    large_log = "\n".join([f"[2026-11-12T10:{i:02d}:00Z] Chat history message {i} with extensive discussion..." for i in range(100)])
    (sources_dir / "session_history_bulk.txt").write_text(large_log, encoding="utf-8")

    packet = create_handoff(
        namespace=namespace,
        task_id=task_id,
        goal="Short goal",
        completed_steps=["Step 1"],
        pending_steps=["Step 2"],
        open_questions=[],
        created_by_agent="agent_1",
        memory_root=mock_memory_root,
    )

    # Handoff packet should be significantly more compact than total raw source history
    assert packet.handoff_tokens < packet.full_history_tokens


def test_get_handoff_success(mock_memory_root: Path):
    namespace = "acme_corp/users/emp_042"
    task_id = "task_roundtrip_02"
    create_handoff(
        namespace=namespace,
        task_id=task_id,
        goal="Ensure clean roundtrip serialization and retrieval.",
        completed_steps=["Step Alpha"],
        pending_steps=["Step Beta"],
        open_questions=["Question Gamma?"],
        created_by_agent="agent_1",
        memory_root=mock_memory_root,
    )

    result = get_handoff(namespace=namespace, task_id=task_id, memory_root=mock_memory_root)

    assert isinstance(result, dict)
    assert result["namespace"] == namespace
    assert result["task_id"] == task_id
    assert result["goal"] == "Ensure clean roundtrip serialization and retrieval."
    assert result["completed_steps"] == ["Step Alpha"]
    assert result["pending_steps"] == ["Step Beta"]
    assert result["open_questions"] == ["Question Gamma?"]
    assert result["stale_warnings"] == []
    assert "raw_okf_markdown" in result
    assert result["raw_okf_markdown"].startswith("---")
    assert "l0_abstract" in result
    assert "Employee 042" in result["l0_abstract"]


def test_get_handoff_flags_superseded_claim(mock_memory_root: Path):
    namespace = "acme_corp/users/emp_042"
    task_id = "task_stale_superseded_03"
    create_handoff(
        namespace=namespace,
        task_id=task_id,
        goal="Staleness test when claim gets superseded.",
        completed_steps=["Step 1"],
        pending_steps=["Step 2"],
        open_questions=[],
        created_by_agent="agent_1",
        memory_root=mock_memory_root,
    )

    # Now simulate an external event superseding 'claim_042_loc_active'
    db_path = mock_memory_root / namespace / "index.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE claims_index SET status = 'superseded' WHERE claim_id = 'claim_042_loc_active'"
    )
    conn.commit()
    conn.close()

    # get_handoff should now detect stale claim
    result = get_handoff(namespace=namespace, task_id=task_id, memory_root=mock_memory_root)

    assert len(result["stale_warnings"]) > 0
    assert any("claim_042_loc_active" in w and "superseded" in w for w in result["stale_warnings"])


def test_get_handoff_flags_deleted_claim(mock_memory_root: Path):
    namespace = "acme_corp/users/emp_042"
    task_id = "task_stale_deleted_04"
    create_handoff(
        namespace=namespace,
        task_id=task_id,
        goal="Staleness test when claim is purged/deleted.",
        completed_steps=["Step 1"],
        pending_steps=["Step 2"],
        open_questions=[],
        created_by_agent="agent_1",
        memory_root=mock_memory_root,
    )

    # Now simulate a deletion / unlearning event deleting 'claim_042_loc_active'
    db_path = mock_memory_root / namespace / "index.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM claims_index WHERE claim_id = 'claim_042_loc_active'"
    )
    conn.commit()
    conn.close()

    # get_handoff should flag missing claim in stale_warnings
    result = get_handoff(namespace=namespace, task_id=task_id, memory_root=mock_memory_root)

    assert len(result["stale_warnings"]) > 0
    assert any("claim_042_loc_active" in w and "deleted" in w for w in result["stale_warnings"])


def test_handoff_file_not_found(mock_memory_root: Path):
    namespace = "acme_corp/users/emp_042"
    # Ensure tasks.okf.md doesn't exist
    okf_file = mock_memory_root / namespace / "tasks.okf.md"
    if okf_file.exists():
        okf_file.unlink()

    with pytest.raises(FileNotFoundError):
        get_handoff(namespace=namespace, task_id="nonexistent_task", memory_root=mock_memory_root)


def test_realtime_handoff_console(mock_memory_root: Path):
    from agents.realtime_handoff import run_realtime_handoff

    restored = run_realtime_handoff(
        namespace="acme_corp/users/emp_042",
        task_id="test_realtime_task",
        mode="2",
        wait_for_enter=False,
        memory_root=mock_memory_root,
    )
    assert restored["task_id"] == "test_realtime_task"
    assert restored["status"] == "in_progress"
    assert len(restored["active_decisions"]) >= 2
    assert (mock_memory_root / "acme_corp" / "users" / "emp_042" / "tasks.okf.md").exists()
