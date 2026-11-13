"""Universal zero-friction CLI for uploading any document into Knowledge Brain."""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure knowledge-brain is importable regardless of working directory
REPO_ROOT = Path(__file__).resolve().parent
KB_DIR = REPO_ROOT / "knowledge-brain"
if str(KB_DIR) not in sys.path:
    sys.path.insert(0, str(KB_DIR))

import tiktoken

from core.ingest import ingest_message
from core.storage_api import MemoryStore

# Folders automatically ignored during recursive directory scans
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "memory_root",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".antigravity",
}

# Known binary extensions to skip immediately without decoding
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".svgz",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".lib",
    ".sqlite", ".sqlite3", ".db", ".db-wal", ".db-shm",
    ".pyc", ".pyo", ".pyd", ".class", ".jar", ".war",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".avi", ".mov", ".mkv", ".webm",
    ".iso", ".img", ".dmg", ".parquet", ".feather", ".arrow", ".pkl", ".pickle",
}

TEXT_TYPE_MAP = {
    ".md": "MARKDOWN",
    ".markdown": "MARKDOWN",
    ".txt": "TEXT",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".jsonl": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".xml": "XML",
    ".log": "LOG",
    ".ini": "CONFIG",
    ".cfg": "CONFIG",
    ".toml": "TOML",
    ".env": "CONFIG",
}

CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".cmd", ".bat",
    ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".scala", ".r", ".lua", ".css", ".scss",
}

_TIKTOKEN_ENC = None


def _get_encoder() -> tiktoken.Encoding:
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


def count_tokens(text: str) -> int:
    """Count cl100k_base tokens for extracted document text."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def _extract_pdf(file_path: Path) -> Tuple[str, str]:
    try:
        import pypdf
    except ImportError:
        print(f"⚠️  Warning: 'pypdf' is not installed; cannot parse {file_path.name}")
        return ("PDF", "")

    reader = pypdf.PdfReader(str(file_path))
    if reader.is_encrypted:
        try:
            res = reader.decrypt("")
            if res == 0:
                print(f"⚠️  Warning: PDF '{file_path.name}' is encrypted and cannot be read.")
                return ("PDF", "")
        except Exception:
            print(f"⚠️  Warning: PDF '{file_path.name}' is encrypted and cannot be read.")
            return ("PDF", "")

    pages_text: List[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages_text.append(f"[Page {idx}]\n{txt}")

    if not pages_text:
        print(f"⚠️  Warning: PDF '{file_path.name}' has no selectable text.")
        return ("PDF", "")

    return ("PDF", "\n\n".join(pages_text))


def _extract_docx(file_path: Path) -> Tuple[str, str]:
    try:
        import docx
    except ImportError:
        print(f"⚠️  Warning: 'python-docx' is not installed; cannot parse {file_path.name}")
        return ("DOCX", "")

    doc = docx.Document(str(file_path))
    parts: List[str] = []

    for para in doc.paragraphs:
        txt = para.text.strip()
        if txt:
            parts.append(txt)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return ("DOCX", "\n".join(parts))


def _extract_pptx(file_path: Path) -> Tuple[str, str]:
    try:
        from pptx import Presentation
    except ImportError:
        print(f"⚠️  Warning: 'python-pptx' is not installed; cannot parse {file_path.name}")
        return ("PPTX", "")

    prs = Presentation(str(file_path))
    slides_out: List[str] = []

    for idx, slide in enumerate(prs.slides, start=1):
        slide_lines: List[str] = [f"[Slide {idx}]"]
        title_shape = slide.shapes.title
        title_text = ""
        if title_shape is not None and title_shape.has_text_frame:
            title_text = title_shape.text_frame.text.strip()
            if title_text:
                slide_lines.append(f"Title: {title_text}")

        for shape in slide.shapes:
            if shape == title_shape:
                continue
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = para.text.strip()
                    if line:
                        slide_lines.append(f"- {line}")

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_lines.append(f"[Notes] {notes}")

        if len(slide_lines) > 1:
            slides_out.append("\n".join(slide_lines))

    return ("PPTX", "\n\n".join(slides_out))


def _extract_excel(file_path: Path) -> Tuple[str, str]:
    try:
        import openpyxl
    except ImportError:
        print(f"⚠️  Warning: 'openpyxl' is not installed; cannot parse {file_path.name}")
        return ("XLSX", "")

    wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
    try:
        sheets_out: List[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_out: List[str] = [f"## Sheet: {sheet_name}"]
            header_emitted = False
            for row in ws.iter_rows(values_only=True):
                if row is None or all(cell is None or str(cell).strip() == "" for cell in row):
                    continue
                vals = [str(cell).strip() if cell is not None else "" for cell in row]
                rows_out.append("| " + " | ".join(vals) + " |")
                if not header_emitted:
                    rows_out.append("| " + " | ".join("---" for _ in vals) + " |")
                    header_emitted = True
            if len(rows_out) > 1:
                sheets_out.append("\n".join(rows_out))
        return ("XLSX", "\n\n".join(sheets_out))
    finally:
        wb.close()


def _extract_html(file_path: Path) -> Tuple[str, str]:
    raw_html = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        cleaned = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", raw_html)
        cleaned = re.sub(r"(?s)<[^>]+>", "\n", cleaned)
        text = html.unescape(cleaned)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ("HTML", "\n".join(lines))


def extract_any_document(file_path: Path) -> Tuple[str, str]:
    """Extract readable text and detected type from any supported document.

    Returns (doc_type, text). Binary files return ("BINARY", "").
    """
    name_lower = file_path.name.lower()
    ext = file_path.suffix.lower()

    if name_lower.endswith(".okf.md"):
        return ("OKF", file_path.read_text(encoding="utf-8", errors="replace"))

    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext == ".docx":
        return _extract_docx(file_path)
    if ext == ".pptx":
        return _extract_pptx(file_path)
    if ext in {".xlsx", ".xls", ".xlsm"}:
        return _extract_excel(file_path)
    if ext in {".html", ".htm"}:
        return _extract_html(file_path)

    if ext in TEXT_TYPE_MAP:
        return (TEXT_TYPE_MAP[ext], file_path.read_text(encoding="utf-8", errors="replace"))
    if ext in CODE_EXTENSIONS:
        return ("CODE", file_path.read_text(encoding="utf-8", errors="replace"))

    if ext in BINARY_EXTENSIONS:
        return ("BINARY", "")

    # Universal fallback: read raw bytes and reject binary content with null bytes
    raw_bytes = file_path.read_bytes()
    if b"\x00" in raw_bytes:
        return ("BINARY", "")

    doc_type = ext.lstrip(".").upper() if ext else "TEXT"
    return (doc_type, raw_bytes.decode("utf-8", errors="replace"))


def _is_ignored_path(file_path: Path, root_dir: Path) -> bool:
    """Return True if file_path resides inside any ignored or hidden folder."""
    try:
        rel_parts = file_path.relative_to(root_dir).parts[:-1]
    except ValueError:
        rel_parts = file_path.parts[:-1]

    for part in rel_parts:
        if part in IGNORED_DIRS or part.startswith("."):
            return True
    return False


def collect_target_files(target: Path) -> Tuple[List[Path], bool]:
    """Collect single file or recursively scan a directory. Returns (files, is_bulk)."""
    if target.is_file():
        return ([target], False)
    if target.is_dir():
        files = sorted(
            p for p in target.rglob("*")
            if p.is_file() and not _is_ignored_path(p, target)
        )
        return (files, True)
    raise FileNotFoundError(f"Path not found: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload any document or folder of documents into Knowledge Brain."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single file or a directory to scan recursively.",
    )
    parser.add_argument(
        "-n",
        "--namespace",
        default="acme_corp/users/emp_042",
        help="Target memory namespace (default: acme_corp/users/emp_042).",
    )
    parser.add_argument(
        "-o",
        "--origin",
        default="doc",
        choices=["user", "tool", "doc"],
        help="Trust tier origin passed to ingest_message() (default: doc).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and preview file type, chars, and tokens without writing to SQLite.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # Ensure UTF-8 console output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    target = args.path.resolve()
    try:
        files, is_bulk = collect_target_files(target)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not files:
        print(f"No eligible files found in: {target}")
        return 0

    store: Optional[MemoryStore] = None if args.dry_run else MemoryStore(root_dir="memory_root")

    mode_label = "DRY-RUN" if args.dry_run else f"INGEST -> {args.namespace} (origin={args.origin})"
    print(f"TraceZero Knowledge Brain Uploader | {mode_label}")
    print("-" * 80)

    total_processed = 0
    total_skipped = 0
    total_claims = 0
    total_tokens = 0
    t_all_start = time.perf_counter()

    for file_path in files:
        display_name = str(file_path.relative_to(target)) if is_bulk else file_path.name
        try:
            doc_type, text = extract_any_document(file_path)
            if doc_type == "BINARY":
                print(f"⏭️ Skipped binary file: {display_name}")
                total_skipped += 1
                continue

            if not text.strip():
                print(f"⚠️  Skipped empty/unreadable file [{doc_type}]: {display_name}")
                total_skipped += 1
                continue

            char_count = len(text)
            token_count = count_tokens(text)

            if args.dry_run:
                print(
                    f"[{doc_type:<8}] {display_name:<40} | "
                    f"{char_count:>7,} chars | {token_count:>6,} tokens | dry-run"
                )
                total_processed += 1
                total_tokens += token_count
                continue

            t0 = time.perf_counter()
            assert store is not None
            claims = ingest_message(
                store=store,
                namespace=args.namespace,
                text=text,
                origin=args.origin,
                session_id=file_path.stem,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            claim_ids = [c["claim_id"] for c in claims if "claim_id" in c]

            print(
                f"[{doc_type:<8}] {display_name:<40} | "
                f"{char_count:>7,} chars | {token_count:>6,} tokens | "
                f"{len(claim_ids):>3} claim(s) | {elapsed_ms:>7.1f} ms"
            )
            total_processed += 1
            total_claims += len(claim_ids)
            total_tokens += token_count

        except Exception as exc:
            print(f"⚠️  Warning: Failed to process '{display_name}': {exc}")
            total_skipped += 1

    total_elapsed_s = time.perf_counter() - t_all_start

    if is_bulk:
        print("=" * 80)
        print("BULK UPLOAD SUMMARY")
        print(f"  Files Processed : {total_processed}")
        print(f"  Files Skipped   : {total_skipped}")
        print(f"  Claims Extracted: {total_claims}")
        print(f"  Tokens Ingested : {total_tokens:,}")
        print(f"  Total Elapsed   : {total_elapsed_s:.2f} s")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
