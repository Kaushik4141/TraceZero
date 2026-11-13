"""Core module for OKF v0.2 Cross-Agent Task Handoffs.

Implements seamless handoffs using Google Open Knowledge Format (OKF v0.2)
Markdown files with YAML frontmatter, active decision provenance links,
token compression accounting, and SQLite staleness verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

try:
    from core.contracts import HandoffPacket
except ImportError:
    from contracts import HandoffPacket


def _resolve_memory_root(memory_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the memory_root directory with fallback discovery."""
    if memory_root is not None:
        p = Path(memory_root).resolve()
        return p

    env_root = os.environ.get("MEMORY_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Try relative to current working directory
    cwd_candidate = Path.cwd() / "memory_root"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    # Try knowledge-brain/memory_root relative to cwd
    cwd_kb_candidate = Path.cwd() / "knowledge-brain" / "memory_root"
    if cwd_kb_candidate.exists():
        return cwd_kb_candidate.resolve()

    # Try relative to this file
    current_file = Path(__file__).resolve()
    # File is at <repo>/knowledge-brain/core/handoff.py or <repo>/core/handoff.py
    for parent in [current_file.parent.parent, current_file.parent.parent.parent]:
        candidate = parent / "memory_root"
        if candidate.exists():
            return candidate.resolve()
        candidate_kb = parent / "knowledge-brain" / "memory_root"
        if candidate_kb.exists():
            return candidate_kb.resolve()

    # Fallback to Path.cwd() / "memory_root"
    return (Path.cwd() / "memory_root").resolve()


def count_tokens(text: str) -> int:
    """Count or approximate token count for a text string.
    
    Uses tiktoken if installed; otherwise approximates as round(len(text.split()) * 1.3).
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        words = len(text.split())
        return int(round(words * 1.3))


def _calculate_sources_tokens(shard_dir: Path) -> int:
    """Calculate total token count across all files in sources/."""
    sources_dir = shard_dir / "sources"
    if not sources_dir.exists() or not sources_dir.is_dir():
        return 0

    total_tokens = 0
    for file_path in sorted(sources_dir.iterdir()):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                total_tokens += count_tokens(content)
            except Exception:
                continue
    return total_tokens


def _parse_sections_from_markdown(md_text: str) -> Dict[str, Any]:
    """Parse structured markdown sections from OKF markdown body."""
    sections: Dict[str, Any] = {
        "goal": "",
        "completed_steps": [],
        "pending_steps": [],
        "active_decisions": [],
        "open_questions": [],
        "citations": {},
    }

    # Split into sections based on ## headers
    header_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(md_text))

    if not matches:
        return sections

    for i, match in enumerate(matches):
        section_name = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        chunk = md_text[start:end].strip()

        if "goal" in section_name:
            sections["goal"] = chunk
        elif "completed step" in section_name:
            items = []
            for line in chunk.splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "•")):
                    items.append(line.lstrip("-*• ").strip())
                elif line and not line.startswith("#"):
                    items.append(line)
            sections["completed_steps"] = items
        elif "pending step" in section_name:
            items = []
            for line in chunk.splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "•")):
                    items.append(line.lstrip("-*• ").strip())
                elif line and not line.startswith("#"):
                    items.append(line)
            sections["pending_steps"] = items
        elif "open question" in section_name:
            items = []
            for line in chunk.splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "•")):
                    items.append(line.lstrip("-*• ").strip())
                elif line and not line.startswith("#"):
                    items.append(line)
            sections["open_questions"] = items
        elif "active decision" in section_name:
            decisions = []
            for line in chunk.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                clean_line = line.lstrip("-*• ").strip()
                cite_match = re.search(r"\[\^([a-zA-Z0-9_\-]+)\]", clean_line)
                claim_match = re.search(r"claim:\s*`?([a-zA-Z0-9_\-]+)`?", clean_line)
                citation = f"[^{cite_match.group(1)}]" if cite_match else ""
                claim_id = claim_match.group(1) if claim_match else ""

                parts = clean_line.split(":", 1)
                if len(parts) == 2:
                    pred = parts[0].strip()
                    rest = parts[1].strip()
                    rest = re.sub(r"\[\^[a-zA-Z0-9_\-]+\]", "", rest)
                    rest = re.sub(r"\(claim:\s*`?[a-zA-Z0-9_\-]+`?\)", "", rest).strip()
                    decisions.append({
                        "claim_id": claim_id,
                        "predicate": pred,
                        "object_val": rest,
                        "citation": citation,
                    })
                else:
                    decisions.append({
                        "claim_id": claim_id,
                        "predicate": "decision",
                        "object_val": clean_line,
                        "citation": citation,
                    })
            sections["active_decisions"] = decisions
        elif "evidence" in section_name or "citation" in section_name:
            citations = {}
            for line in chunk.splitlines():
                line = line.strip()
                m = re.match(r"^\[\^([a-zA-Z0-9_\-]+)\]:\s*(.+)$", line)
                if m:
                    citations[m.group(1)] = m.group(2).strip()
            sections["citations"] = citations

    return sections


def create_handoff(
    namespace: str,
    task_id: str,
    goal: str,
    completed_steps: List[str],
    pending_steps: List[str],
    open_questions: List[str],
    created_by_agent: str = "agent_1",
    memory_root: Optional[Union[str, Path]] = None,
) -> HandoffPacket:
    """Create a structured OKF v0.2 task handoff packet and write tasks.okf.md.
    
    Args:
        namespace: Memory shard namespace (e.g., 'acme_corp/users/emp_042').
        task_id: Unique task identifier.
        goal: Overarching goal of the task.
        completed_steps: List of actions/milestones finished.
        pending_steps: List of remaining actions.
        open_questions: List of unresolved queries/decisions.
        created_by_agent: Agent identity performing the handoff.
        memory_root: Optional custom base directory for memory_root.

    Returns:
        Validated HandoffPacket model.
    """
    root = _resolve_memory_root(memory_root)
    shard_dir = root / namespace
    if not shard_dir.exists():
        shard_dir.mkdir(parents=True, exist_ok=True)

    db_path = shard_dir / "index.sqlite"
    active_decisions: List[Dict[str, Any]] = []
    relied_on_claims: List[str] = []
    sources_meta: List[Dict[str, Any]] = []
    citations_dict: Dict[str, str] = {}

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query active structured claims (excluding raw bulk has_note / stated notes so handoff remains compact)
        cursor.execute(
            "SELECT claim_id, subject, predicate, object_val, source_id, byte_start, byte_end "
            "FROM claims_index WHERE status = 'active' AND predicate NOT IN ('has_note', 'stated')"
        )
        active_claims = cursor.fetchall()

        # Collect source IDs
        needed_source_ids = {row["source_id"] for row in active_claims if row["source_id"]}
        sources_by_id = {}
        if needed_source_ids:
            placeholders = ",".join("?" for _ in needed_source_ids)
            cursor.execute(
                f"SELECT * FROM sources_index WHERE source_id IN ({placeholders})",
                list(needed_source_ids),
            )
            for s_row in cursor.fetchall():
                sources_by_id[s_row["source_id"]] = dict(s_row)

        conn.close()

        for c_row in active_claims:
            cid = c_row["claim_id"]
            pred = c_row["predicate"]
            obj = c_row["object_val"]
            src_id = c_row["source_id"]
            relied_on_claims.append(cid)
            citation_marker = f"[^{src_id}]" if src_id else ""
            active_decisions.append({
                "claim_id": cid,
                "predicate": pred,
                "object_val": obj,
                "citation": citation_marker,
            })

            if src_id and src_id in sources_by_id:
                s_info = sources_by_id[src_id]
                quote = s_info.get("quote", "")
                fpath = s_info.get("file_path", "")
                if not quote and fpath:
                    sf = shard_dir / fpath
                    if sf.exists() and c_row["byte_end"] > c_row["byte_start"]:
                        with open(sf, "rb") as f_in:
                            f_in.seek(c_row["byte_start"])
                            quote = f_in.read(c_row["byte_end"] - c_row["byte_start"]).decode("utf-8", errors="replace").strip()
                tier = s_info.get("trust_tier", 1)
                citations_dict[src_id] = f'{fpath} "{quote}" (trust_tier={tier})'

        for s_id, s_info in sources_by_id.items():
            sources_meta.append({
                "source_id": s_id,
                "file_path": s_info.get("file_path", ""),
                "trust_tier": s_info.get("trust_tier", 1),
                "origin": s_info.get("origin", ""),
            })

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build Frontmatter
    frontmatter_dict = {
        "okf_version": "0.2",
        "namespace": namespace,
        "task_id": task_id,
        "status": "in_progress",
        "created_by_agent": created_by_agent,
        "updated_at": updated_at,
        "relied_on_claims": relied_on_claims,
        "sources": sources_meta,
        "active_decisions": active_decisions,
    }

    yaml_frontmatter = yaml.dump(frontmatter_dict, sort_keys=False, default_flow_style=False).strip()

    # Build Markdown Body
    body_lines = [
        f"# Task Handoff: {task_id}",
        "",
        "## Goal",
        goal.strip(),
        "",
        "## Completed Steps",
    ]
    if completed_steps:
        for step in completed_steps:
            body_lines.append(f"- {step}")
    else:
        body_lines.append("- (None)")

    body_lines.extend(["", "## Pending Steps"])
    if pending_steps:
        for step in pending_steps:
            body_lines.append(f"- {step}")
    else:
        body_lines.append("- (None)")

    body_lines.extend(["", "## Active Decisions & Provenance"])
    if active_decisions:
        for dec in active_decisions:
            cite = f" {dec['citation']}" if dec.get("citation") else ""
            body_lines.append(f"- {dec['predicate']}: {dec['object_val']}{cite} (claim: `{dec['claim_id']}`)")
    else:
        body_lines.append("- No active decisions recorded.")

    body_lines.extend(["", "## Open Questions"])
    if open_questions:
        for q in open_questions:
            body_lines.append(f"- {q}")
    else:
        body_lines.append("- (None)")

    if citations_dict:
        body_lines.extend(["", "## Evidence & Citations"])
        for src_id, cite_val in sorted(citations_dict.items()):
            body_lines.append(f"[^{src_id}]: {cite_val}")

    body_text = "\n".join(body_lines) + "\n"
    full_markdown = f"---\n{yaml_frontmatter}\n---\n\n{body_text}"

    # Write tasks.okf.md
    okf_file_path = shard_dir / "tasks.okf.md"
    okf_file_path.write_text(full_markdown, encoding="utf-8")

    # Token calculations
    handoff_tokens = count_tokens(full_markdown)
    full_history_tokens = _calculate_sources_tokens(shard_dir)

    return HandoffPacket(
        okf_version="0.2",
        namespace=namespace,
        task_id=task_id,
        status="in_progress",
        created_by_agent=created_by_agent,
        updated_at=updated_at,
        goal=goal,
        completed_steps=completed_steps,
        pending_steps=pending_steps,
        active_decisions=active_decisions,
        open_questions=open_questions,
        relied_on_claims=relied_on_claims,
        okf_file_path=str(okf_file_path),
        handoff_tokens=handoff_tokens,
        full_history_tokens=full_history_tokens,
    )


def get_handoff(
    namespace: str,
    task_id: str,
    memory_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Read, parse, and verify an OKF v0.2 task handoff packet.

    Args:
        namespace: Memory shard namespace.
        task_id: Task identifier.
        memory_root: Optional custom base directory for memory_root.

    Returns:
        Structured dictionary matching HandoffPacket plus 'raw_okf_markdown',
        'stale_warnings', and 'l0_abstract'.
    """
    root = _resolve_memory_root(memory_root)
    shard_dir = root / namespace
    okf_file_path = shard_dir / "tasks.okf.md"

    if not okf_file_path.exists():
        raise FileNotFoundError(f"Handoff file not found at {okf_file_path}")

    raw_okf_markdown = okf_file_path.read_text(encoding="utf-8")

    # Read .abstract.md (L0) if present
    l0_path = shard_dir / ".abstract.md"
    l0_abstract = l0_path.read_text(encoding="utf-8") if l0_path.exists() else ""

    # Split YAML frontmatter and Markdown body
    frontmatter: Dict[str, Any] = {}
    md_body = raw_okf_markdown

    if raw_okf_markdown.startswith("---"):
        parts = raw_okf_markdown.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            md_body = parts[2]
            try:
                frontmatter = yaml.safe_load(fm_text) or {}
            except Exception:
                frontmatter = {}

    # Parse sections from Markdown body
    sections = _parse_sections_from_markdown(md_body)

    # Reconcile values between frontmatter and markdown sections
    parsed_goal = sections["goal"] or frontmatter.get("goal", "")
    parsed_completed_steps = sections["completed_steps"] or frontmatter.get("completed_steps", [])
    parsed_completed_steps = [s for s in parsed_completed_steps if s != "(None)"]

    parsed_pending_steps = sections["pending_steps"] or frontmatter.get("pending_steps", [])
    parsed_pending_steps = [s for s in parsed_pending_steps if s != "(None)"]

    parsed_open_questions = sections["open_questions"] or frontmatter.get("open_questions", [])
    parsed_open_questions = [q for q in parsed_open_questions if q != "(None)"]

    # Active decisions: prefer frontmatter if present and rich, otherwise markdown parsed
    active_decisions = frontmatter.get("active_decisions")
    if not active_decisions:
        active_decisions = sections.get("active_decisions", [])

    relied_on_claims = frontmatter.get("relied_on_claims", [])
    if not relied_on_claims:
        relied_on_claims = [d.get("claim_id") for d in active_decisions if d.get("claim_id")]

    # Verification against index.sqlite for staleness
    stale_warnings: List[str] = []
    db_path = shard_dir / "index.sqlite"

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        for cid in relied_on_claims:
            cursor.execute("SELECT status FROM claims_index WHERE claim_id = ?", (cid,))
            row = cursor.fetchone()
            if not row:
                stale_warnings.append(f"Claim '{cid}' was deleted from claims_index.")
            elif row[0] != "active":
                stale_warnings.append(f"Claim '{cid}' is no longer active (current status: '{row[0]}').")

        conn.close()
    else:
        if relied_on_claims:
            stale_warnings.append(f"Database {db_path.name} not found; unable to verify claims.")

    handoff_tokens = count_tokens(raw_okf_markdown)
    full_history_tokens = _calculate_sources_tokens(shard_dir)

    packet = HandoffPacket(
        okf_version=str(frontmatter.get("okf_version", "0.2")),
        namespace=namespace,
        task_id=task_id,
        status=str(frontmatter.get("status", "in_progress")),
        created_by_agent=str(frontmatter.get("created_by_agent", "unknown")),
        updated_at=str(frontmatter.get("updated_at", "")),
        goal=parsed_goal,
        completed_steps=parsed_completed_steps,
        pending_steps=parsed_pending_steps,
        active_decisions=active_decisions,
        open_questions=parsed_open_questions,
        relied_on_claims=relied_on_claims,
        okf_file_path=str(okf_file_path),
        handoff_tokens=handoff_tokens,
        full_history_tokens=full_history_tokens,
    )

    result = packet.model_dump()
    result["raw_okf_markdown"] = raw_okf_markdown
    result["stale_warnings"] = stale_warnings
    result["l0_abstract"] = l0_abstract

    return result


from server.handoff import handoff_create, handoff_get

