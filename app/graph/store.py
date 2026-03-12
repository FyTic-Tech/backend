"""
MVP 4 — Relationship graph store.

Stores and traverses explicit + implicit edges between chunks.

Tools powered by this module:
- follow_references(chunk_id): traverse explicit edges → referenced chunks
- find_related(chunk_id, cross_document): traverse implicit similarity edges

Storage: JSON adjacency structure alongside ChromaDB index.
Lightweight by design — graph traversal is O(edges), not O(corpus).
"""

# Not implemented yet — placeholder for MVP 4
