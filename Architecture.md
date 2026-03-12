# Architecture — Agentic File Explorer

## Overview

A FastAPI backend for document Q&A. An LLM agent explores files in DATA/
dynamically to answer questions using filesystem tools rather than a pre-built index.

---

## Current State: MVP 1 — Pure Agentic Filesystem Search

```
POST /api/query
    |
    v
ExplorerService (creates fresh per-request)
    |
    v
FsExplorerAgent (Gemini 3 Flash)
    |
    +-- Tool: scan_folder     -> parallel preview of all files (first 10 pages)
    +-- Tool: preview_file    -> single-file preview (first 10 pages)
    +-- Tool: parse_file      -> page-ranged parse (pdfplumber or Docling)
    +-- Tool: grep            -> regex search across parsed text
    +-- Tool: glob            -> filename pattern matching
    |
    v
FsExplorerWorkflow (LlamaIndex event loop)
    StartEvent -> ToolCallEvent loop -> StopEvent
    |
    v
Response: { answer, error, usage }
```

### PDF Parsing — 3-layer Fallback

```
Layer 1: Docling (no OCR)     -> best structure, fails on image-based PDFs
Layer 2: pdfplumber           -> reliable text extraction, handles most PDFs
Layer 3: pypdf                -> last resort, handles encrypted/malformed files
```

OCR is disabled on Docling to prevent memory crashes (std::bad_alloc) when
loading 400+ page documents as bitmaps. Text-based PDFs do not need OCR.

### Page Range Strategy

parse_file(path, page_start, page_end) — agent reads in 50-page sections.
Prevents context overflow on large documents.

Current limitation: Docling's public API supports max_num_pages but not an
arbitrary start page offset. Currently parses from page 1 up to page_end.
True arbitrary start-offset support will be added in MVP 3 when section-based
chunking replaces page numbers as the primary navigation unit.

### Capabilities at MVP 1
- Any document size (memory-safe via page ranges)
- Any PDF format (3-layer fallback never fails silently)
- Cross-reference following via grep on cached parsed text
- Cited answers with file + section attribution
- Per-request token usage and cost tracking (~$0.003-0.015 per query)
- In-session parse caching (subsequent queries on same file are fast)

### Limits at MVP 1
- No map of document structure — agent navigates by sequential reading
- No persistent index — each new session re-parses from scratch
- No semantic similarity — agent finds only what it explicitly reads
- Single DATA/ folder

---

## MVP 2 — Flat Semantic Search Layer

Architecture: Agentic Search + Vector Retrieval

Adds ChromaDB and a flat embedding index as a tool the agent can optionally use.

```
New tools:
  semantic_search(query, n_results)  -> top-N relevant chunks from ChromaDB
  get_section(file, chunk_id)        -> retrieve a specific cached chunk

New infrastructure:
  app/retrieval/vector_store.py      -> ChromaDB wrapper (single flat collection)
  POST /api/index                    -> triggers indexing of DATA/

Indexing:
  file -> pdfplumber/Docling -> fixed-size chunks (512 tokens)
       -> multilingual-e5-small embeddings -> ChromaDB
```

Agent can now jump to relevant content without sequential reading.
Fixed-size chunks still lose cross-chunk context (resolved in MVP 3).

---

## MVP 3 — Hierarchical Index + Late Chunking

Architecture: Structure-Aware Agentic Retrieval with Contextual Embeddings

```
ChromaDB collections:
  level_1_summaries   -> one entry per document
  level_2_summaries   -> one entry per section/chapter
  level_3_chunks      -> late-chunked content with full-section context

New tools:
  hier_search(query, level)        -> navigate L1 -> L2 -> L3
  expand_context(chunk_id, n)      -> retrieve surrounding chunks

Indexing pipeline:
  file
    -> Docling -> structured markdown
    -> Structure parser: extract headings as L1/L2 labels (free, no LLM)
    -> LLM summary only for sections missing informative headings
    -> Late chunking per section: embed full section -> split into chunks
    -> Store in L1/L2/L3 ChromaDB collections
```

Hierarchy cost estimate: ~$0.005 per 200-article law, one-time, cached by file hash.

---

## MVP 4 — Relationship Graph + Cross-Reference Traversal

Architecture: Contextual Agentic Retrieval with Explicit and Implicit Knowledge Graph

```
Two edge types:
  EXPLICIT: parsed cross-references ("see Article 89", "per Schedule B")
            -> regex extraction during indexing
  IMPLICIT: cosine similarity above threshold (~0.85) between chunk vectors
            -> computed during indexing

New tools:
  follow_references(chunk_id)              -> traverse explicit edges
  find_related(chunk_id, cross_document)   -> traverse implicit edges

Storage: JSON adjacency structure alongside ChromaDB index
```

---

## MVP 5 — Production Hardening

```
Additions:
  Hash-based incremental indexing (re-index only changed files)
  GET /api/index/status    -> indexed files, coverage, last-updated
  Query observability      -> tool call traces, latency per step
  Chunk quality scoring    -> flag low-information chunks at index time
  Language auto-detection  -> select embedding model per document language
  Index health endpoint    -> detect stale/corrupted entries
```

---

## Final Architecture Diagram

```
Query
  |
  v
FsExplorerAgent
  |
  +-- hier_search(L1) -----------> ChromaDB L1 (document summaries)
  |        |
  |        +-- hier_search(L2) --> ChromaDB L2 (section summaries)
  |                 |
  |                 +-- hier_search(L3) -> ChromaDB L3 (late-chunked)
  |                          |
  |                          +-- expand_context() -> neighboring chunks
  |                          |
  |                          +-- follow_references() -> Graph (explicit)
  |                                    |
  |                                    +-- find_related() -> Graph (implicit)
  |
  +-- parse_file(page_range) ----> pdfplumber/Docling (exact fallback)
  +-- grep() --------------------> raw text search (unindexed fallback)
```

---

## Dependency Stack

```
Runtime (MVP 1 — active):
  FastAPI / Uvicorn         -> HTTP server
  google-genai              -> Gemini 3 Flash (agent reasoning)
  llama-index-core          -> workflow event loop
  Docling                   -> structured PDF/DOCX parsing
  pdfplumber / pypdf        -> PDF fallback parsers

MVP 2+:
  chromadb                  -> vector store
  sentence-transformers     -> multilingual-e5-small / bge-m3 embeddings

MVP 3+:
  jina-embeddings-v3        -> late chunking support (8K context length)

MVP 4+:
  No new libraries — graph stored as JSON
```