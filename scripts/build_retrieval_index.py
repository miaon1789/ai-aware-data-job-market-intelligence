#!/usr/bin/env python3
"""Build the dense and lexical indexes for each chunk configuration.

Embeddings and the BM25 database are derived from licensed advertisement text
and stay under `data/private/`. Encoding is cached by content hash, so
re-running after a chunking change only encodes what actually changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.retrieval.config import EMBEDDING_MODELS  # noqa: E402
from job_market_intelligence.retrieval.embedding import ChunkEncoder  # noqa: E402
from job_market_intelligence.retrieval.lexical import LexicalIndex  # noqa: E402

PRIVATE_ROOT = (ROOT / "data/private").resolve()


def inside_private(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == PRIVATE_ROOT or PRIVATE_ROOT in resolved.parents


def model_slug(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir", type=Path, default=ROOT / "data/private/retrieval/chunks"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/private/retrieval/index"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data/private/retrieval/embedding_cache"
    )
    parser.add_argument(
        "--embedding-model",
        action="append",
        dest="models",
        help=f"Repeatable; defaults to {EMBEDDING_MODELS}",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-dense", action="store_true", help="Build only the BM25 indexes"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()

    for path in (args.output_dir, args.cache_dir):
        if not inside_private(path):
            parser.error("index and cache output must stay under data/private")
    chunk_files = sorted(args.chunk_dir.glob("chunk*.parquet"))
    if not chunk_files:
        parser.error(f"no chunk files found in {args.chunk_dir}; run build_chunks.py first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    models = args.models or list(EMBEDDING_MODELS)

    summary: dict[str, object] = {"indexes": []}
    for chunk_file in chunk_files:
        label = chunk_file.stem
        chunks = pd.read_parquet(chunk_file)
        started = time.perf_counter()
        lexical_path = args.output_dir / f"{label}.lexical.duckdb"
        LexicalIndex.build(chunks, lexical_path)
        lexical_seconds = time.perf_counter() - started

        entry: dict[str, object] = {
            "chunk_label": label,
            "chunks": int(len(chunks)),
            "lexical_index": str(lexical_path.relative_to(ROOT)),
            "lexical_build_seconds": round(lexical_seconds, 2),
            "dense_indexes": {},
        }

        if not args.skip_dense:
            for model in models:
                encoder = ChunkEncoder(model, cache_dir=args.cache_dir)
                started = time.perf_counter()
                matrix = encoder.encode(
                    chunks["text"].tolist(), batch_size=args.batch_size, show_progress=True
                )
                encoder.save_cache()
                vector_path = args.output_dir / f"{label}.{model_slug(model)}.npy"
                np.save(vector_path, matrix)
                vector_path.chmod(0o600)
                entry["dense_indexes"][model] = {
                    "path": str(vector_path.relative_to(ROOT)),
                    "shape": list(matrix.shape),
                    "encode_seconds": round(time.perf_counter() - started, 2),
                    "megabytes": round(matrix.nbytes / 1_048_576, 2),
                }
        summary["indexes"].append(entry)
        print(json.dumps(entry, indent=2), file=sys.stderr)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
