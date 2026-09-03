"""Sentence-embedding encoder with an on-disk cache.

Ablations re-encode the same chunk text many times -- once per retrieval mode,
once per reranking setting -- so encoding is cached by ``(model, text)`` hash.
The cache turns a repeated ablation sweep from minutes of GPU-less encoding
into a file read, which is what makes the evaluation matrix cheap to rerun
after a chunking change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
    from collections.abc import Sequence

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
ALTERNATE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Models on the Hub are mutable: the author can push new weights to the same
# name. An ablation whose embedding model can change underneath it has no
# control condition, and the nDCG figures in the README would stop being
# reproducible without anything appearing to break. Pinning the commit is what
# makes "we measured this" a claim that survives the next six months.
MODEL_REVISIONS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "sentence-transformers/all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
}

# BGE retrieval models are trained with an asymmetric instruction prefix on the
# query side only. Omitting it measurably degrades recall, so it is part of the
# model's contract rather than a tuning knob.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()
    return digest[:32]


@dataclass
class EmbeddingCache:
    """A flat vector store keyed by content hash, persisted as npy + json."""

    directory: Path
    model_name: str
    _keys: dict[str, int]
    _matrix: np.ndarray

    @classmethod
    def load(cls, directory: str | Path, model_name: str) -> EmbeddingCache:
        path = Path(directory)
        slug = model_name.replace("/", "__")
        keys_path = path / f"{slug}.keys.json"
        matrix_path = path / f"{slug}.vectors.npy"
        if keys_path.exists() and matrix_path.exists():
            keys = json.loads(keys_path.read_text(encoding="utf-8"))
            matrix = np.load(matrix_path)
        else:
            keys, matrix = {}, np.zeros((0, 0), dtype=np.float32)
        return cls(directory=path, model_name=model_name, _keys=keys, _matrix=matrix)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        slug = self.model_name.replace("/", "__")
        (self.directory / f"{slug}.keys.json").write_text(
            json.dumps(self._keys), encoding="utf-8"
        )
        np.save(self.directory / f"{slug}.vectors.npy", self._matrix)

    def get(self, text: str) -> np.ndarray | None:
        index = self._keys.get(cache_key(self.model_name, text))
        return None if index is None else self._matrix[index]

    def put(self, texts: Sequence[str], vectors: np.ndarray) -> None:
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must be the same length")
        if not len(texts):
            return
        if self._matrix.size == 0:
            self._matrix = np.zeros((0, vectors.shape[1]), dtype=np.float32)
        start = len(self._matrix)
        self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])
        for offset, text in enumerate(texts):
            self._keys[cache_key(self.model_name, text)] = start + offset


def normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so cosine similarity is a dot product."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class ChunkEncoder:
    """Encodes chunk and query text, reusing cached vectors where possible."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._cache = (
            EmbeddingCache.load(cache_dir, model_name) if cache_dir is not None else None
        )

    @property
    def model(self):
        """Load the transformer lazily so cache hits never import torch."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                revision=MODEL_REVISIONS.get(self.model_name),
            )
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return L2-normalised embeddings, one row per input text."""

        prepared = [
            (QUERY_PREFIXES.get(self.model_name, "") if is_query else "") + str(text)
            for text in texts
        ]
        if not prepared:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[np.ndarray | None] = [None] * len(prepared)
        missing: list[int] = []
        if self._cache is not None:
            for index, text in enumerate(prepared):
                cached = self._cache.get(text)
                if cached is None:
                    missing.append(index)
                else:
                    vectors[index] = cached
        else:
            missing = list(range(len(prepared)))

        if missing:
            fresh = self.model.encode(
                [prepared[index] for index in missing],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)
            for offset, index in enumerate(missing):
                vectors[index] = fresh[offset]
            if self._cache is not None:
                self._cache.put([prepared[index] for index in missing], fresh)

        return normalise(np.vstack([vector for vector in vectors if vector is not None]))

    def save_cache(self) -> None:
        if self._cache is not None:
            self._cache.save()
