"""Minimum Viable System (MVS) End-to-End Rubric & Subprocess Restart Verifier (Role 4).

Executes:
1. 30-session deterministic shard seed + 100-employee namespace isolation test
2. Agent A -> Agent B cross-agent OKF v0.2 handoff (handoff_create -> fresh subprocess restart -> handoff_get + recall_context without history paste)
3. Verified cryptographic deletion (stored secret + never-stored secret) with 0 residual bytes
   across .sqlite, -wal, .okf.md, .abstract.md, and sources/*.txt + 10 paraphrased probes abstaining.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.forget import forget_target
from core.retrieval import recall_context
from core.verifier import scan_shard_bytes
from evals.run_benchmarks import _update_summary_json, run_benchmark_suite
from evals.seed_30_sessions import seed_live_shards
from server.handoff import handoff_create, handoff_get


def run_mvs_eval(
    memory_root: Optional[str | Path] = None,
    subprocess_restart: bool = True,
) -> Dict[str, Any]:
    """Run full PS-06 MVS rubric + process restart + handoff + byte-verified deletion."""
    tmp_ctx = None
    if memory_root is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="tracezero_mvs_")
        active_root = Path(tmp_ctx.name)
    else:
        active_root = Path(memory_root)
        active_root.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Seed 30 sessions + emp_042 + isolated shards cleanly
        store, dataset = seed_live_shards(
            active_root, seed_all_employees=False, ingest_filler=True
        )
        ns = dataset["metadata"]["target_namespace"]
        probes = dataset["rubric_probes"]
        shard_dir = store.sm.get_shard_dir(ns)

        # Verify all 30 session files exist on disk
        source_files = sorted(p.name for p in (shard_dir / "sources").glob("session_*.txt"))
        assert len(source_files) >= 30, f"Expected >=30 session files, found {len(source_files)}"

        # 2. Agent A creates mid-task OKF v0.2 handoff packet
        db_exp = probes["db_update_expected"]
        reg_exp = probes["aws_region_expected"]
        pkt = handoff_create(
            store=store,
            namespace=ns,
            task_id="mig_wave_30",
            goal=f"Complete orion_payments {db_exp} cutover in {reg_exp}",
            completed_steps=[f"Validate {db_exp} schema parity", "Quarantine untrusted web override"],
            pending_steps=["Switch production DNS records to ap-south-1_az3", "Decommission PostgreSQL 14"],
            open_questions=["Confirm maintenance window for DNS cutover"],
            relied_on_claim_ids=[],
            agent_id="agent_a_claude",
        )
        assert pkt.okf_version == "0.2"
        assert (shard_dir / "tasks.okf.md").exists()

        # Drop in-memory store instance completely
        del store

        # 3. Verify persistence across a FRESH OS PROCESS RESTART (zero in-memory state)
        restart_verified = False
        agent_b_tokens = 0
        if subprocess_restart:
            child_code = (
                "import json, sys; "
                "from core.storage_api import MemoryStore; "
                "from core.retrieval import recall_context; "
                "from server.handoff import handoff_get; "
                f"s = MemoryStore(root_dir={str(active_root)!r}); "
                f"h = handoff_get(s, {ns!r}, 'mig_wave_30'); "
                f"r = recall_context(s, {ns!r}, 'What is our primary database engine?', token_budget=800); "
                "g = s.query_graph_multihop(" + repr(ns) + ", 'emp_042', max_hops=4); "
                "g_arch = s.query_graph_multihop(" + repr(ns) + ", 'orion_payments', max_hops=3); "
                "assert h['status'] == 'ok'; "
                "assert h['tokens_used'] < 350; "
                f"assert {db_exp!r} in r.packed_prompt; "
                f"assert any(x['object_val'] == {db_exp!r} for x in g); "
                "assert any(x['object_val'] == 'ap-south-1_az3' for x in g_arch); "
                "print(json.dumps({'handoff_tokens': h['tokens_used'], 'recall_tokens': r.tokens_used, 'hops': len(g)}))"
            )
            proc = subprocess.run(
                [sys.executable, "-c", child_code],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=True,
            )
            child_out = json.loads(proc.stdout.strip().splitlines()[-1])
            agent_b_tokens = child_out["handoff_tokens"]
            restart_verified = True
        else:
            from core.storage_api import MemoryStore

            fresh_store = MemoryStore(root_dir=str(active_root))
            h = handoff_get(fresh_store, ns, "mig_wave_30")
            r = recall_context(fresh_store, ns, "What is our primary database engine?", token_budget=800)
            assert h["status"] == "ok" and h["tokens_used"] < 350
            assert db_exp in r.packed_prompt
            agent_b_tokens = h["tokens_used"]
            restart_verified = True

        # 4. Run end-to-end deletion + byte scan + 10 probes
        from core.storage_api import MemoryStore

        store2 = MemoryStore(root_dir=str(active_root))
        secret = probes["deletion_secret"]
        rec_stored = forget_target(store2.sm, ns, secret)
        rec_never = forget_target(store2.sm, ns, probes["never_stored_secret"])

        remaining_bytes = scan_shard_bytes(shard_dir, [secret])
        assert remaining_bytes == 0, f"Residual bytes detected: {remaining_bytes}"
        assert rec_stored.byte_scan_matches == 0
        assert rec_never.byte_scan_matches == 0

        probe_abstained = 0
        for pq in probes["deletion_probes"]:
            pr = recall_context(store2, ns, pq, token_budget=800)
            if pr.status == "abstained" and "no memory of that" in pr.packed_prompt.lower():
                probe_abstained += 1

        assert probe_abstained == 10, f"Only {probe_abstained}/10 deletion probes abstained"

        mvs_summary = {
            "sessions_seeded": len(source_files),
            "process_restart_verified": restart_verified,
            "handoff_tokens": agent_b_tokens,
            "stored_deletion_receipt": rec_stored.model_dump(),
            "never_stored_deletion_receipt": rec_never.model_dump(),
            "post_deletion_probes_abstained": probe_abstained,
            "residual_shard_bytes": remaining_bytes,
        }
        _update_summary_json("mvs_verification", mvs_summary)
        return mvs_summary
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main() -> None:
    bench = run_benchmark_suite(mode="live")
    mvs = run_mvs_eval(subprocess_restart=True)
    print(
        f"[run_mvs_eval] MVS Verified: sessions={mvs['sessions_seeded']}, "
        f"restart={mvs['process_restart_verified']}, handoff_tokens={mvs['handoff_tokens']}, "
        f"deletion_bytes={mvs['residual_shard_bytes']}, probes={mvs['post_deletion_probes_abstained']}/10, "
        f"isolation={bench['namespace_collision_test']['isolation_rate_pct']:.1f}%"
    )


__all__ = ["run_mvs_eval", "run_benchmark_suite", "main"]

if __name__ == "__main__":
    main()
