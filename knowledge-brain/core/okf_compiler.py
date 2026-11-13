from collections import defaultdict
from datetime import datetime, timezone
import yaml
from core.schema import ShardManager


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compile_namespace_okf(shard_manager: ShardManager, namespace: str):
    shard_dir = shard_manager.get_shard_dir(namespace)
    now = datetime.now(timezone.utc).isoformat()

    with shard_manager.get_connection(namespace) as conn:
        rows = conn.execute("""
            SELECT c.*, s.file_path, s.origin, s.trust_tier
            FROM claims_index c
            JOIN sources_index s ON c.source_id = s.source_id
            WHERE c.status = 'active'
            ORDER BY c.predicate ASC
        """).fetchall()

    by_file = defaultdict(list)
    for r in rows:
        by_file[r["okf_file"]].append(r)

    # Always ensure preferences.okf.md and any existing .okf.md files are rewritten
    # (even if all claims in a file were deleted!)
    existing_okf = {
        p.name
        for p in shard_dir.glob("*.okf.md")
        if not (p.name == "tasks.okf.md" and "created_by_agent:" in p.read_text(encoding="utf-8", errors="ignore"))
    }
    okf_targets = set(by_file.keys()) | existing_okf | {"preferences.okf.md"}

    for okf_file in okf_targets:
        file_claims = by_file.get(okf_file, [])
        sources_meta = {}
        for c in file_claims:
            sid = c["source_id"]
            sources_meta[sid] = {
                "id": sid,
                "path": c["file_path"],
                "span": f"{c['byte_start']}:{c['byte_end']}",
                "origin": c["origin"],
                "trust_tier": c["trust_tier"],
            }

        frontmatter = {
            "okf_version": "0.2",
            "namespace": f"viking://{namespace.strip('/')}/{okf_file}",
            "status": "stable",
            "updated_at": now,
            "sources": list(sources_meta.values()),
        }

        lines = ["---", yaml.safe_dump(frontmatter, sort_keys=False).strip(), "---", ""]
        lines.append(f"# Active Knowledge ({namespace})")
        if not file_claims:
            lines.append("_No active claims recorded._")
        else:
            for c in file_claims:
                lines.append(
                    f"- **{c['subject']}.{c['predicate']}**: {c['object_val']} [^{c['source_id']}]"
                )

        (shard_dir / okf_file).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compile OpenViking L0 .abstract.md (~50-100 tokens)
    emp_label = namespace.split("/")[-1]
    if emp_label.startswith("emp_"):
        emp_title = f"Employee {emp_label.replace('emp_', '')} ({namespace})"
    else:
        emp_title = namespace
    abstract_lines = [f"# L0 Abstract: {emp_title}"]
    for r in rows[:8]:
        abstract_lines.append(f"- {r['predicate']}: {r['object_val']}")
    if not rows:
        abstract_lines.append("- Empty namespace")
    (shard_dir / ".abstract.md").write_text("\n".join(abstract_lines) + "\n", encoding="utf-8")
