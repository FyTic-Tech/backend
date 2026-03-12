"""
MVP 2 — Flat ChromaDB vector store.
MVP 3 — Upgraded to three-collection hierarchical store (L1/L2/L3).

Collections:
- level_1_summaries: one document summary per file
- level_2_summaries: one summary per section/chapter
- level_3_chunks: late-chunked content with full-section contextual embeddings

Metadata per chunk:
- source_file, level, section_id, page_start, page_end, parent_id, children_ids
"""

# Not implemented yet — placeholder for MVP 2
