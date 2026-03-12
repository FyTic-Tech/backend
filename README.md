# Agentic File Explorer — Backend

An AI research agent that explores documents to answer questions —
the way a human researcher would. It scans, reasons, reads sections,
follows cross-references, and returns cited answers.

**Current state: MVP 1** — Pure agentic search with large-document support.
No vector index yet. The agent navigates files directly using page ranges.

---

## What MVP 1 Can Do

- Accept a question via API and search through all files in `DATA/`
- Handle documents of any size by reading them in page-range sections
  (no more memory crashes on 400+ page PDFs)
- Follow cross-references between sections using targeted reads and grep
- Return a cited answer with source file and section references
- Report token usage and estimated cost per query

## What MVP 1 Cannot Do Yet

- Vector/semantic search (comes in MVP 2)
- Hierarchical navigation — L1/L2/L3 (comes in MVP 3)
- Cross-reference graph traversal (comes in MVP 4)
- Index management endpoints (comes in MVP 3)

For now: one endpoint, one DATA folder, one question at a time.

---

## How It Works (MVP 1)

```
Your question
     │
     ▼
FsExplorer Agent (Gemini 3 Flash)
     │
     ├── scan_folder(DATA/)
     │     previews first 10 pages of every file → decides what's relevant
     │
     ├── preview_file(relevant_file)
     │     reads first 10 pages → confirms relevance
     │
     ├── parse_file(file, page_end=50)
     │     reads pages 1-50 → looks for the answer
     │
     ├── parse_file(file, page_start=51, page_end=100)
     │     continues reading if needed
     │
     ├── grep(file, "cross-referenced term")
     │     follows references found in earlier sections
     │
     └── STOP → returns answer with citations
```

The agent decides at each step what to do next. It stops as soon as it
has enough information — it does not read the entire document unless necessary.

---

## Setup

### 1. Fix folder structure (if set up manually)

```bash
# From backend/app/
mv retrieval/vectore_store.py retrieval/vector_store.py   # fix typo
mv indexing/indexing_service.py indexing/pipeline.py      # fix wrong name

touch api/__init__.py config/__init__.py services/__init__.py
touch utils/__init__.py graph/__init__.py indexing/__init__.py retrieval/__init__.py
touch graph/extractor.py indexing/embedder.py
touch retrieval/hier_search.py services/index_service.py
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create `.env` in `backend/`:

```
GOOGLE_API_KEY=your_key_here
```

Get your key at: https://aistudio.google.com/apikey
A paid key is strongly recommended — free tier quota exhausts after 1-2 large-doc queries.

### 4. Add your documents

```
backend/DATA/
  your_document.pdf
  another_file.docx
```

Supported: **PDF, DOCX, DOC, PPTX, XLSX, HTML, Markdown**

### 5. Run

```bash
# From backend/
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API

### POST `/api/query`

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the obligations for foreign entities?"}'
```

**Response:**
```json
{
  "answer": "Foreign entities must comply with... [Source: law.pdf, pages 1-50, Article 89]",
  "error": null,
  "usage": {
    "api_calls": 6,
    "prompt_tokens": 28000,
    "completion_tokens": 1200,
    "total_tokens": 29200,
    "documents_scanned": 1,
    "documents_parsed": 3,
    "tool_result_chars": 120000,
    "estimated_cost_usd": 0.00256
  }
}
```

### GET `/health`

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

## Performance Expectations (MVP 1)

| Document size | First query | Repeat queries |
|---------------|-------------|----------------|
| < 50 pages | 15–45 sec | 10–30 sec (cached) |
| 50–200 pages | 45–120 sec | 20–60 sec (cached) |
| 200–400 pages | 2–5 min | 30–90 sec (cached) |

First query is slow because Docling parses the PDF into text.
That result is cached for the rest of the session.

---

## Project Structure

```
backend/
├── .env                          # GOOGLE_API_KEY — never commit
├── requirements.txt
├── README.md
├── Architecture.md               # Full system design and MVP roadmap
│
├── app/
│   ├── main.py                   # FastAPI app + CORS
│   ├── api/
│   │   └── routers.py            # POST /api/query
│   ├── config/
│   │   └── settings.py           # GOOGLE_API_KEY, DATA_DIR
│   ├── explorer/                 # MVP 1 — ACTIVE
│   │   ├── agent.py              # Gemini client, tool registry, token tracking
│   │   ├── fs.py                 # scan, preview, parse (with page ranges)
│   │   ├── models.py             # Pydantic schemas for agent actions
│   │   └── workflow.py           # LlamaIndex event loop
│   ├── indexing/                 # MVP 3 — placeholder
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── pipeline.py
│   ├── retrieval/                # MVP 2/3 — placeholder
│   │   ├── vector_store.py
│   │   └── hier_search.py
│   ├── graph/                    # MVP 4 — placeholder
│   │   ├── extractor.py
│   │   └── store.py
│   ├── services/
│   │   ├── explorer_service.py   # Wires agent + workflow per request
│   │   └── index_service.py      # MVP 3 — placeholder
│   └── utils/
│       └── logging.py
│
└── DATA/                         # Your documents go here
```

---

## MVP Roadmap

| MVP | What it adds | Status |
|-----|-------------|--------|
| **1 — Large Document Support** | Page-range reads, any file size | ✅ Current |
| **2 — Semantic Search** | ChromaDB + embeddings, fast lookup | 🔲 Next |
| **3 — Hierarchical Index** | L1/L2/L3 navigation, late chunking | 🔲 Planned |
| **4 — Relationship Graph** | Cross-reference traversal | 🔲 Planned |
| **5 — Production Hardening** | Incremental indexing, observability | 🔲 Planned |

---

## Troubleshooting

**`std::bad_alloc` during parsing**
OCR is disabled. If you still see this, the document has dense image pages.
Reduce `PREVIEW_MAX_PAGES` in `fs.py` or add RAM.

**`WorkflowTimeoutError`**
Query took over 10 minutes. Ask a more specific question so the agent
stops reading earlier.

**`429 RESOURCE_EXHAUSTED`**
Gemini quota hit. Wait for reset or enable billing at aistudio.google.com.

**Vague or wrong answers**
Be specific. Instead of "tell me about X", ask "what does Article 89 say about X".
The agent's reading path is driven by your question — more specific = more targeted.