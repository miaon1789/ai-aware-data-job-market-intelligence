"""BM25 lexical retrieval on the DuckDB full-text-search extension.

DuckDB is already the analytics store for this project, so its FTS extension
keeps the lexical branch inside an existing dependency rather than adding a
second search engine. The index lives in its own database file so that
rebuilding it never touches the analysis tables.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pandas as pd

TABLE_NAME = "chunks"
# DuckDB's FTS tokeniser splits on non-alphanumerics, so a query typed as
# "Power BI" or "scikit-learn" arrives as several terms. Quoting is not
# supported by match_bm25, hence the conservative sanitiser below.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+#.]+")

FILTERABLE_COLUMNS = ("city", "analysis_role", "seniority", "source", "section")


def tokenise_query(query: str) -> str:
    """Reduce a free-text query to the terms DuckDB FTS can match."""

    return " ".join(_TOKEN_RE.findall(str(query).lower()))


class LexicalIndex:
    """A persisted BM25 index over chunk text."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    @classmethod
    def build(
        cls,
        chunks: pd.DataFrame,
        database_path: str | Path,
        *,
        stemmer: str = "porter",
        stopwords: str = "english",
    ) -> LexicalIndex:
        """Create (or replace) the BM25 index for ``chunks``."""

        required = {"chunk_id", "text"}
        missing = sorted(required - set(chunks.columns))
        if missing:
            raise ValueError(f"chunks are missing columns: {', '.join(missing)}")

        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        columns = ["chunk_id", "job_id", "text", *FILTERABLE_COLUMNS]
        frame = chunks.reindex(columns=[c for c in columns if c in chunks.columns]).astype(str)
        with duckdb.connect(str(path)) as connection:
            connection.execute("INSTALL fts; LOAD fts;")
            connection.register("chunk_frame", frame)
            connection.execute(
                f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM chunk_frame"
            )
            connection.execute(
                f"PRAGMA create_fts_index('{TABLE_NAME}', 'chunk_id', 'text', "
                f"stemmer='{stemmer}', stopwords='{stopwords}', overwrite=1)"
            )
        return cls(path)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, bm25_score)`` for the best lexical matches."""

        terms = tokenise_query(query)
        if not terms or top_k <= 0:
            return []

        conditions = []
        parameters: list[object] = [terms]
        for column, value in (filters or {}).items():
            if column not in FILTERABLE_COLUMNS:
                raise ValueError(f"cannot filter on {column!r}")
            if value:
                conditions.append(f"{column} = ?")
                parameters.append(str(value))
        where = f" AND {' AND '.join(conditions)}" if conditions else ""
        parameters.append(int(top_k))

        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            connection.execute("LOAD fts;")
            rows = connection.execute(
                f"""
                SELECT chunk_id, score FROM (
                    SELECT chunk_id, fts_main_{TABLE_NAME}.match_bm25(chunk_id, ?) AS score,
                           {', '.join(FILTERABLE_COLUMNS)}
                    FROM {TABLE_NAME}
                ) WHERE score IS NOT NULL{where}
                ORDER BY score DESC, chunk_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [(str(chunk_id), float(score)) for chunk_id, score in rows]
