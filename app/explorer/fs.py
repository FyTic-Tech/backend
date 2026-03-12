"""
Filesystem utilities for the FsExplorer agent.

PDF parsing uses a 3-layer fallback strategy:
  Layer 1: Docling (no OCR) — best quality, handles DOCX/PPTX/XLSX too
  Layer 2: pdfplumber      — reliable text extraction for most PDFs
  Layer 3: pypdf           — last resort, handles encrypted/malformed PDFs

This means even PDFs that Docling rejects will parse successfully.
"""

import os
import re
import glob as glob_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from app.utils.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"}
)

NON_PDF_EXTENSIONS: frozenset[str] = frozenset(
    {".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"}
)

DEFAULT_PREVIEW_CHARS = 3000
DEFAULT_SCAN_PREVIEW_CHARS = 1500
MAX_PREVIEW_LINES = 30
DEFAULT_MAX_WORKERS = 4
PREVIEW_MAX_PAGES = 10
RECOMMENDED_SECTION_SIZE = 50


# =============================================================================
# PDF Fallback Parsers
# =============================================================================


def _parse_pdf_pdfplumber(file_path: str, max_pages: int | None = None) -> str:
    """
    Layer 2 fallback: pdfplumber.
    Handles most PDFs including those with complex layouts.
    """
    try:
        import pdfplumber

        pages_text = []
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for page in pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        result = "\n\n".join(pages_text)
        if result.strip():
            log.info(
                f"pdfplumber extracted {len(result):,} chars from {os.path.basename(file_path)}"
            )
            return result
        return ""
    except Exception as e:
        log.warning(f"pdfplumber failed on {os.path.basename(file_path)}: {e}")
        return ""


def _parse_pdf_pypdf(file_path: str, max_pages: int | None = None) -> str:
    """
    Layer 3 fallback: pypdf.
    Last resort — handles encrypted and malformed PDFs.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)

        # Handle encrypted PDFs (try empty password)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return "[PDF is encrypted and could not be decrypted]"

        pages_text = []
        pages = reader.pages[:max_pages] if max_pages else reader.pages
        for page in pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        result = "\n\n".join(pages_text)
        if result.strip():
            log.info(
                f"pypdf extracted {len(result):,} chars from {os.path.basename(file_path)}"
            )
            return result
        return "[PDF appears to be image-based — no text layer found. OCR would be needed.]"
    except Exception as e:
        log.warning(f"pypdf failed on {os.path.basename(file_path)}: {e}")
        return f"[All parsers failed on this PDF: {e}]"


# =============================================================================
# Docling Converter Factory
# =============================================================================


def _make_converter() -> DocumentConverter:
    """
    Docling converter with OCR disabled.
    Fast and memory-safe for text-based PDFs.
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _parse_with_docling(file_path: str, max_pages: int | None = None) -> str | None:
    """
    Attempt Docling parsing. Returns None if Docling rejects the file.
    """
    try:
        converter = _make_converter()
        if max_pages:
            result = converter.convert(file_path, max_num_pages=max_pages)
        else:
            result = converter.convert(file_path)
        content = result.document.export_to_markdown()
        if content.strip():
            return content
        return None
    except Exception as e:
        log.warning(f"Docling failed on {os.path.basename(file_path)}: {e}")
        return None


def _parse_non_pdf(file_path: str, max_pages: int | None = None) -> str:
    """Parse non-PDF formats — always goes through Docling."""
    try:
        converter = DocumentConverter()
        if max_pages:
            result = converter.convert(file_path, max_num_pages=max_pages)
        else:
            result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        raise RuntimeError(f"Failed to parse {os.path.basename(file_path)}: {e}") from e


# =============================================================================
# Core Parse Function (with fallback chain)
# =============================================================================


def _parse_document(file_path: str, max_pages: int | None = None) -> str:
    """
    Parse any supported document with automatic fallback.

    PDF fallback chain:
      1. Docling (no OCR) — best structure preservation
      2. pdfplumber       — reliable text extraction
      3. pypdf            — handles encrypted/malformed files

    Non-PDF: always Docling.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext != ".pdf":
        return _parse_non_pdf(file_path, max_pages)

    # Layer 1 — Docling
    log.info(f"Trying Docling: {os.path.basename(file_path)}")
    content = _parse_with_docling(file_path, max_pages)
    if content:
        return content

    # Layer 2 — pdfplumber
    log.info(f"Docling failed, trying pdfplumber: {os.path.basename(file_path)}")
    content = _parse_pdf_pdfplumber(file_path, max_pages)
    if content and "[" not in content[:5]:  # not an error message
        return content

    # Layer 3 — pypdf
    log.info(f"pdfplumber failed, trying pypdf: {os.path.basename(file_path)}")
    return _parse_pdf_pypdf(file_path, max_pages)


# =============================================================================
# Document Caches
# =============================================================================

_PREVIEW_CACHE: dict[str, str] = {}
_PARSE_CACHE: dict[str, str] = {}


def clear_document_cache() -> None:
    _PREVIEW_CACHE.clear()
    _PARSE_CACHE.clear()
    log.info("Document cache cleared")


def _cache_key(file_path: str, suffix: str = "") -> str:
    abs_path = os.path.abspath(file_path)
    mtime = os.path.getmtime(abs_path)
    return f"{abs_path}:{mtime}{suffix}"


def _get_preview(file_path: str) -> str:
    key = _cache_key(file_path, ":preview")
    if key not in _PREVIEW_CACHE:
        log.info(
            f"Parsing preview: {os.path.basename(file_path)} (first {PREVIEW_MAX_PAGES} pages)"
        )
        _PREVIEW_CACHE[key] = _parse_document(file_path, max_pages=PREVIEW_MAX_PAGES)
    return _PREVIEW_CACHE[key]


def _get_ranged(file_path: str, page_start: int, page_end: int | None) -> str:
    range_suffix = f":p{page_start}-{page_end or 'end'}"
    key = _cache_key(file_path, range_suffix)
    if key not in _PARSE_CACHE:
        end = page_end if page_end is not None else 99999
        log.info(f"Parsing {os.path.basename(file_path)} (pages 1-{end})")
        # Docling parses from page 1 up to max_num_pages.
        # True arbitrary start-page support comes in MVP 3 with section chunking.
        _PARSE_CACHE[key] = _parse_document(file_path, max_pages=end)
    return _PARSE_CACHE[key]


def _get_full(file_path: str) -> str:
    key = _cache_key(file_path, ":full")
    if key not in _PARSE_CACHE:
        log.info(f"Parsing full document: {os.path.basename(file_path)}")
        _PARSE_CACHE[key] = _parse_document(file_path)
    return _PARSE_CACHE[key]


# =============================================================================
# Directory Operations
# =============================================================================


def describe_dir_content(directory: str) -> str:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"
    children = os.listdir(directory)
    if not children:
        return f"Directory {directory} is empty"

    files, dirs = [], []
    for child in children:
        fullpath = os.path.join(directory, child)
        (files if os.path.isfile(fullpath) else dirs).append(fullpath)

    desc = f"Content of {directory}\nFILES:\n- " + "\n- ".join(files)
    desc += (
        "\nThis folder does not have any sub-folders"
        if not dirs
        else "\nSUBFOLDERS:\n- " + "\n- ".join(dirs)
    )
    return desc


# =============================================================================
# Basic File Operations
# =============================================================================


def read_file(file_path: str) -> str:
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def grep_file_content(file_path: str, pattern: str) -> str:
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()

    # For binary formats, parse to text first then grep
    if ext in SUPPORTED_EXTENSIONS and ext not in {".md", ".html"}:
        try:
            content = _get_full(file_path)
        except Exception:
            return f"Could not extract text from {file_path} for grep"
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    matches = re.compile(pattern, flags=re.MULTILINE | re.IGNORECASE).findall(content)
    if matches:
        str_matches = [str(m) for m in matches[:50]]  # cap at 50 matches
        return (
            f"MATCHES for '{pattern}' in {os.path.basename(file_path)}:\n\n- "
            + "\n- ".join(str_matches)
        )
    return f"No matches found for '{pattern}' in {os.path.basename(file_path)}"


def glob_paths(directory: str, pattern: str) -> str:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"
    matches = glob_module.glob(str(Path(directory) / pattern))
    if matches:
        return f"MATCHES for '{pattern}' in {directory}:\n\n- " + "\n- ".join(matches)
    return "No matches found"


# =============================================================================
# Document Parsing (agent-facing tools)
# =============================================================================


def preview_file(file_path: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Quick preview — first 10 pages. Use for relevance assessment."""
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Unsupported extension '{ext}'"
    try:
        content = _get_preview(file_path)
        preview = content[:max_chars]
        if len(content) > max_chars:
            preview += (
                f"\n\n[TRUNCATED — use parse_file() with page ranges for full content]"
            )
        return f"=== PREVIEW: {os.path.basename(file_path)} (first {PREVIEW_MAX_PAGES} pages) ===\n\n{preview}"
    except Exception as e:
        log.error(f"Error previewing {file_path}: {e}")
        return f"Error previewing {file_path}: {e}"


def parse_file(
    file_path: str,
    page_start: int = 1,
    page_end: int | None = None,
) -> str:
    """
    Parse document content, optionally by page range.

    For documents over 50 pages, always use page ranges:
      parse_file(path, page_end=50)           → pages 1-50
      parse_file(path, page_start=51, page_end=100) → pages 51-100

    Stop reading as soon as you have enough to answer.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Unsupported extension '{ext}'"
    if page_start < 1:
        page_start = 1
    if page_end is not None and page_end < page_start:
        return (
            f"Invalid range: page_end ({page_end}) must be >= page_start ({page_start})"
        )

    try:
        is_full = page_start == 1 and page_end is None
        if is_full:
            content = _get_full(file_path)
            header = f"=== FULL CONTENT: {os.path.basename(file_path)} ==="
        else:
            content = _get_ranged(file_path, page_start, page_end)
            range_label = f"pages {page_start}–{page_end if page_end else 'end'}"
            next_start = (page_end or 0) + 1
            header = (
                f"=== CONTENT: {os.path.basename(file_path)} ({range_label}) ===\n"
                f"To continue: parse_file(path, page_start={next_start}, page_end={next_start + RECOMMENDED_SECTION_SIZE - 1})"
            )
        return f"{header}\n\n{content}\n\n[{len(content):,} characters]"
    except Exception as e:
        log.error(f"Error parsing {file_path}: {e}")
        return f"Error parsing {file_path}: {e}"


# =============================================================================
# Parallel Folder Scan
# =============================================================================


def _preview_single_file(file_path: str, preview_chars: int) -> dict:
    filename = os.path.basename(file_path)
    try:
        content = _get_preview(file_path)
        return {
            "file": file_path,
            "filename": filename,
            "preview": content[:preview_chars],
            "total_chars": len(content),
            "status": f"ok (first {PREVIEW_MAX_PAGES} pages)",
        }
    except Exception as e:
        log.error(f"Error scanning {filename}: {e}")
        return {
            "file": file_path,
            "filename": filename,
            "preview": "",
            "total_chars": 0,
            "status": f"error: {e}",
        }


def scan_folder(
    directory: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    preview_chars: int = DEFAULT_SCAN_PREVIEW_CHARS,
) -> str:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"

    doc_files = [
        os.path.join(directory, item)
        for item in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, item))
        and os.path.splitext(item)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not doc_files:
        return f"No supported documents in {directory}"

    log.info(f"Scanning {len(doc_files)} file(s) in {directory}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_preview_single_file, f, preview_chars): f
            for f in doc_files
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["filename"])

    out = [
        "═══════════════════════════════════════════════════════════════",
        f"  FOLDER SCAN: {directory}",
        f"  {len(results)} document(s) — first {PREVIEW_MAX_PAGES} pages each",
        "═══════════════════════════════════════════════════════════════",
        "",
    ]
    for i, r in enumerate(results, 1):
        out += [
            "┌─────────────────────────────────────────────────────────────",
            f"│ [{i}/{len(results)}] {r['filename']}",
            f"│ Path: {r['file']}",
            f"│ Status: {r['status']} | Preview: {r['total_chars']:,} chars",
            "├─────────────────────────────────────────────────────────────",
        ]
        if r["status"].startswith("ok") and r["preview"]:
            for line in r["preview"].split("\n")[:MAX_PREVIEW_LINES]:
                out.append(f"│ {line}")
        else:
            out.append(f"│ {r['status']}")
        out += ["└─────────────────────────────────────────────────────────────", ""]

    out += [
        "═══════════════════════════════════════════════════════════════",
        f"  FOR LARGE DOCS: parse_file(path, page_end={RECOMMENDED_SECTION_SIZE}),",
        f"  then parse_file(path, page_start=N, page_end=M) for next sections.",
        "═══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(out)
