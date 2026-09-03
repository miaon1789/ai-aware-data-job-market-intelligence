"""Hybrid retrieval over the private job-advertisement corpus.

The corpus is small by design (roughly 700 advertisements and a few thousand
chunks), so this package deliberately avoids a vector database: dense search is
brute-force cosine similarity in numpy, and lexical search uses the DuckDB FTS
extension that the analytics layer already depends on. See ``dense.py`` for the
reasoning and the point at which that judgement would change.
"""

from __future__ import annotations
