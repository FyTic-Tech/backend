# Technical Analysis — Agentic File Explorer
## What This Is, Why It Matters, What It's Missing, and When It Becomes Real

---

## 1. What This Project Is, Honestly

This is a document Q&A backend where an LLM agent navigates files using
tools — scan, read, grep — the way a human researcher would, rather than
embedding documents into a vector database and querying it statistically.

That distinction matters more than it sounds.

In standard RAG (Retrieval-Augmented Generation), the flow is mechanical:
embed everything upfront, find the closest vectors to the query, send
those chunks to the LLM, get an answer. The LLM's only job is synthesis.
All navigation decisions are made by cosine similarity.

In this system, the LLM makes all navigation decisions. It decides what to
read, what to skip, what to follow up, whether a cross-reference is
relevant, and when it has enough information to stop. The vector index
(when it arrives in MVP 2) is just another tool the agent can choose to
use — not the entire pipeline.

The practical difference: when a user asks "what are the exceptions to
Article 89 and how do they interact with the sanctions defined in Chapter
VI?", standard RAG returns the three chunks most similar to that query
string. This system reads Article 89, notices the cross-reference to
Chapter VI, reads Chapter VI, notices that one sanction applies only under
conditions defined in Article 134, reads that too, and then answers with
the full chain of reasoning and proper citations.

That chain-following behavior is not something you can bolt onto standard
RAG. It requires an agent that reasons about what to read next.

---

## 2. Current Architecture (MVP 1): The Honest Assessment

**Title: A Working Agent That Can Read But Not Navigate**

The system works. The CPEUM.pdf test (403 pages, image-readable, rejected
by Docling) completed successfully once the 3-layer PDF fallback was added.
pdfplumber extracted 180,365 characters in under 30 seconds. The agent
read pages 1-50, found sufficient information, and stopped. Total cost:
$0.0039 for 3 API calls.

What this architecture actually does well right now:

1. It is memory-safe on large documents. The page-range strategy plus
   OCR-disabled Docling means a 403-page PDF will not crash the server.
   This was a hard blocker and it is now solved.

2. It degrades gracefully. The 3-layer fallback (Docling -> pdfplumber
   -> pypdf) means the agent never gets a silent failure on a PDF. It
   either extracts text or returns an informative error message.

3. It produces cited answers. Every stop action includes references to
   the source file and sections. Standard RAG often loses this.

What this architecture genuinely cannot do yet:

1. It has no structural map of any document. The agent navigates by reading
   sequentially from page 1 forward, guided only by what the preview
   revealed. For a 403-page law where the answer is on page 280, the agent
   must read through pages 1-50, 51-100, etc., until it arrives. This is
   expensive and slow.

2. It has no persistence between sessions. Parse results are cached in
   memory within a session, but each new server start re-parses from
   scratch. A 180k-character parse takes 21 seconds (pdfplumber on CPEUM).
   That cost is paid on every cold start.

3. It cannot find semantically similar content it hasn't explicitly read.
   If the user asks about "foreign entity obligations" but the law uses
   the phrase "entities of foreign origin", the agent must happen to read
   the right section to discover the match. Embedding-based search would
   catch this immediately.

---

## 3. The Existing Research This Project Builds On

Before explaining what is innovative here, it is important to be honest
about what is not novel — and how that existing research is being applied.

### Standard RAG (2020-present)

The baseline. Chunk documents, embed chunks, store in a vector database,
retrieve by cosine similarity, synthesize with an LLM. This is what most
production document Q&A systems use. It is fast, scalable, and works well
for factual lookups from well-isolated information.

Its fundamental limitation for this project's use case (legal documents,
technical manuals, regulatory text): cross-references. Legal documents are
not collections of isolated facts. They are dense networks of dependencies.
"Article 89 applies except as provided in Article 134, subject to the
definitions in Title II." RAG retrieves chunks based on query similarity.
It cannot follow that chain.

**How this applies here:** Standard RAG is the baseline that MVP 2 brings
in as a supplemental tool. It is not the architecture — it is one tool
the agent can use when the query is a simple fact lookup.

### RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
*Sarthi et al., Stanford / ICLR 2024 — arxiv.org/abs/2401.18059*

RAPTOR addressed RAG's inability to handle questions that require
understanding across a whole document rather than a single chunk. It builds
a tree of summaries: raw chunks at the leaves, cluster summaries in the
middle, full-document summary at the root. At query time, the system can
retrieve from any level of the tree, matching the abstraction level of the
question to the abstraction level of the content.

RAPTOR showed significant improvements on multi-step reasoning benchmarks,
including a 20% absolute accuracy improvement on the QuALITY benchmark when
combined with GPT-4. The key insight is that some questions need a
high-level summary and some need a specific chunk — and retrieval should be
able to pick the right level.

**Where this project diverges from RAPTOR:** RAPTOR builds its tree from
the bottom up using recursive clustering and LLM-generated summaries at
every level. This is expensive (LLM calls for every cluster) and destroys
document structure (clusters are formed by similarity, not by the document's
own headings and sections). For legal documents, the document's own
structure is the correct hierarchy. Article 89 belongs to Chapter IV. That
is not a statistical similarity — it is the author's intent.

**How RAPTOR's insight is used here:** The hierarchical index in MVP 3 uses
the same three-level idea (L1: document, L2: section, L3: chunk), but the
hierarchy is extracted from the document's own structure rather than
constructed by recursive clustering. LLM summaries are only generated where
the document itself provides no informative heading.

### Late Chunking — Contextual Chunk Embeddings
*Günther et al., Jina AI — arxiv.org/abs/2409.04701*

Published September 2024. The core problem: when you split a document into
chunks before embedding them, each chunk's embedding captures only the
context of that chunk. A sentence that says "it increased by 3%" has a
meaningless embedding without knowing what "it" refers to — which was
established three sentences earlier in a different chunk.

Jina AI's paper introduced late chunking: embed the entire document (or the
entire section) first, using a long-context model, then split the resulting
token embeddings into chunks and apply mean pooling. This way each chunk's
vector carries the full document context, not just the isolated chunk context.

The paper demonstrated that with naive chunking, a sentence like "it became
the capital" has near-zero cosine similarity to "Berlin" if "Berlin" is
only mentioned in a previous chunk. With late chunking, the same sentence
scores high similarity because the embedding carries the full document context
where "it" and "Berlin" co-occurred.

This is directly applicable to a 403-page law where articles refer to
previously defined terms that were established in Title I. Standard chunking
loses those definitions. Late chunking preserves them.

**How this is used here:** Late chunking is the basis for the L3 chunk
embeddings in MVP 3. The unit of embedding is the section (an article, a
chapter subdivision), not the full 403-page document. Each section is
embedded in full using jina-embeddings-v3 (8K context length), then split
into 256-512 token chunks for storage. This preserves the intra-section
contextual dependencies while keeping chunk sizes reasonable for the agent's
context window.

**What this does NOT solve:** Late chunking preserves context within a
section. It does not preserve context across sections — a chunk from
Article 89 will not carry context from Article 134 just because they
reference each other. That is what the relationship graph (MVP 4) handles.

---

## 4. What Is Actually Novel in This Project

To be direct: no single component here is a new research contribution.
RAPTOR is published. Late chunking is published. Graph RAG is published
(Microsoft Research, 2024). Agentic search is a known pattern.

The novelty — if it is real — is in the specific combination and the
specific design choices for a specific class of documents.

### What does not exist as a packaged solution:

**An agent that uses a structure-extracted (not cluster-derived) hierarchy
to navigate, with late-chunked section embeddings, an explicit cross-reference
graph, and direct filesystem fallback tools in a single decision loop.**

Breaking that down:

**Structure-extracted hierarchy instead of clustered hierarchy.** RAPTOR
builds its tree by clustering similar content. This works for prose
documents. It is wrong for legal documents. "Article 89 is in Chapter IV"
is a structural fact, not a similarity finding. Building the hierarchy by
extracting the document's own structure preserves the author's organizational
intent. No published system does this automatically with the combination of
free structural extraction plus targeted LLM fill-in for uninformative headings.

**Structural edges in the relationship graph.** GraphRAG (Microsoft, 2024)
builds knowledge graphs from documents, but its edges are entity-relationship
edges: "Company X acquired Company Y." This works for narrative documents.
For legal text, the critical relationships are structural: "Article 89
defers to Article 134." These are not entity relationships — they are
normative dependencies. Parsing them explicitly (regex on "see Article N",
"pursuant to Section X") and storing them as graph edges is a different
approach that has not been productized for legal document navigation.

**`expand_context(chunk_id, window)` — agent-controlled narrative restoration.**
When the agent retrieves a relevant chunk, it can request the surrounding
chunks to read the full passage in context. This is not the same as
increasing chunk size. The agent decides how much context to expand based
on what it finds — one call might expand two chunks, another might expand
ten. The agent is the one controlling granularity, not a fixed parameter.
No equivalent exists in RAPTOR or any standard RAG framework.

**The filesystem layer as permanent fallback.** Every system that builds a
vector index eventually needs to handle documents that are not in the index,
documents where the index is stale, or queries that the index cannot answer.
This system's response is: the agent always has direct filesystem tools
available. If `hier_search` returns nothing useful, the agent reads the file
directly. The index makes things fast; the filesystem makes them correct.
Most systems treat these as separate pipelines. Here they are tools in the
same agent decision loop.

---

## 5. What Is Missing for This to Be Genuinely Innovative

Being direct about the gaps:

**The structural edge extraction is currently unbuilt.** In MVP 4, a regex
parser will extract cross-references. This sounds simple. It is not. Legal
cross-references in Spanish ("conforme al artículo 89, en relación con lo
dispuesto en el Título II") are not standardized. A regex that catches
"artículo 89" misses "la fracción III del artículo anterior" (the third
fraction of the preceding article). Building a robust reference parser for
the CPEUM requires a targeted NLP approach, not just regex. This is the
single component that requires real engineering investment to work well.

**Late chunking at section level has not been validated for 400-page
documents.** The Jina AI paper demonstrates late chunking on documents
within the 8K token limit of jina-embeddings-v3. A 403-page law is
approximately 300,000 tokens. Sections vary from one paragraph (Article 3)
to dozens of paragraphs (Article 123). Sections that exceed 8K tokens
cannot be fully embedded at once — they must be truncated or split further,
which reintroduces the context loss problem. The system needs a strategy
for handling large sections, which the current design does not yet address.

**The implicit similarity graph threshold is domain-dependent.** Computing
implicit edges between chunks at cosine similarity > 0.85 is a parameter,
not a law. The right threshold for a constitutional law is different from
the right threshold for a medical protocol or a financial contract. The
system needs a calibration step or a per-domain configuration. This is not
hard to build, but it must be built deliberately.

**No evaluation framework.** A system like this is hard to evaluate because
"correct answer" is subjective for legal questions. Is "Article 89 prohibits
foreign entities from..." a correct answer if it omits the exception in
Article 134? The system needs a test corpus with known questions and known
correct answers, plus an automated evaluation loop. Without this, it is
impossible to know whether MVP 3 is better than MVP 2, or whether a
different embedding model would improve retrieval quality.

---

## 6. The Stages — What Each One Actually Changes

### MVP 1: Agent That Can Read (Current)
**Technical complexity: An LLM calling filesystem tools in a loop**

The agent navigates by brute force — reading from page 1 until it finds
what it needs. Quality depends entirely on the Gemini reasoning about when
to stop and what to follow. No structural knowledge exists. The system is
correct but slow and expensive on large documents.

Appropriate for: Small corpora, documents under 50 pages, prototyping,
demonstrating that the agentic approach works at all.

### MVP 2: Agent That Can Search (Next)
**Technical complexity: Agent + flat vector retrieval as one tool**

The agent can now ask "find sections about foreign entity obligations" and
get the top-N closest chunks without reading anything. First query still
indexes. Subsequent queries are fast. The agent uses semantic search for
initial orientation and filesystem tools for verification.

Appropriate for: Medium corpora (10-50 documents), mixed document types,
queries where the question closely matches the document's own phrasing.
Fails on: queries where terminology differs from document language, or
questions requiring multi-section chains.

### MVP 3: Agent That Can Navigate (Structure Matters)
**Technical complexity: Hierarchical retrieval + late chunking + structure extraction**

The agent knows where it is in a document. "I need Article 89. It is in
Chapter IV. Here are the 4 articles in Chapter IV that are semantically
relevant. Here are the 12 chunks within those articles." The agent reads
only what is relevant, not what is physically nearby. Late chunking means
each chunk's embedding carries its full-section context, so "it" and "the
preceding article" are resolved at embedding time.

This is where the architecture becomes meaningfully better than standard
RAG for the specific use case of structured legal/technical documents.

Appropriate for: Any structured document corpus (laws, regulations,
contracts, technical manuals). Most queries will complete with 2-3 agent
steps instead of 6-10.

### MVP 4: Agent That Can Follow Dependencies (Full Architecture)
**Technical complexity: Hierarchical retrieval + late chunking + explicit/implicit knowledge graph**

The agent can now traverse cross-references. "Article 89 refers to Article
134 — follow that reference. Article 134 refers to the definitions in
Title II — follow that too." This is the capability that separates this
system from everything else for legal and regulatory document Q&A.
A question like "under what conditions is the penalty in Article 200
applicable, considering all the modifications introduced by subsequent
articles?" requires traversing a dependency chain that standard RAG simply
cannot handle.

This is the architecture that has a genuine claim to being non-trivially
better than existing published approaches for structured document Q&A.

### MVP 5: Production System
**Technical complexity: All of the above plus operational reliability**

Incremental indexing, health monitoring, observability, multilingual
support. The difference between a research prototype and a system someone
can actually run in production.

---

## 7. The Honest Summary

Right now this is a well-engineered prototype with a sound architectural
roadmap. It works, it handles the main edge cases (large PDFs, fallback
parsing, page-range navigation), and the code is structured to absorb
the planned additions without restructuring.

The gap between MVP 1 and genuinely innovative is three specific things:
the structural hierarchy extraction in MVP 3, the reference parser in MVP 4,
and an evaluation framework that can measure whether improvements are real.
Everything else — the late chunking, the hierarchical index, the
agent-controlled expand_context — is applied engineering using published
research. Good engineering, with clear design justifications, but not novel.

The combination of structure-extracted hierarchy + late chunking +
structural cross-reference graph + agent-controlled navigation in a single
decision loop does not exist as a packaged solution. Whether building it
constitutes innovation depends on how well it actually works on real
documents — which requires the evaluation framework that is currently absent.

Build that, and the claim is real.

---

## Sources

- Günther et al., "Late Chunking: Contextual Chunk Embeddings Using Long-Context
  Embedding Models," Jina AI, arXiv:2409.04701, September 2024.
  https://arxiv.org/abs/2409.04701
  Directly informs: L3 chunk embedding strategy in MVP 3, choice of
  jina-embeddings-v3 as the embedding model.

- Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized
  Retrieval," Stanford University, ICLR 2024, arXiv:2401.18059.
  https://arxiv.org/abs/2401.18059
  Directly informs: The three-level hierarchy concept (L1/L2/L3) and the
  principle that retrieval should match the abstraction level of the question.
  This project diverges from RAPTOR in using structure extraction instead of
  recursive clustering.

- Jina AI, "Late Chunking in Long-Context Embedding Models," October 2024.
  https://jina.ai/news/late-chunking-in-long-context-embedding-models/
  Directly informs: The specific mechanism of embedding sections before splitting,
  and the nDCG@10 benchmark results demonstrating retrieval improvement.

- Elastic / Jina AI, "Late Chunking in Elasticsearch with Jina Embeddings v2,"
  September 2025. https://www.elastic.co/search-labs/blog/late-chunking-elasticsearch-jina-embeddings
  Directly informs: Practical implementation details and the documented
  trade-off between context length and embedding compression/dilution.