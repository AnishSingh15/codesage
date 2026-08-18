"""Brute-force cosine-similarity retrieval over embedded chunks.

O(n) per query where n = number of chunks. Fine at the scale of one
repo (hundreds-low thousands of chunks). At millions of vectors you'd
reach for an ANN index (e.g. HNSW, a navigable small-world graph) —
noted here, not built, per the spec's explicit v1 scope.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from codesage.ingest import Chunk

INDEX_FILENAME = ".codesage_index.json"


class RetrievalIndex:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._matrix = np.array([c.vector for c in chunks]) if chunks else np.empty((0, 0))

    def search(self, query_vector: list[float], k: int = 5) -> list[Chunk]:
        if not self._chunks:
            return []
        query = np.array(query_vector)
        chunk_norms = np.linalg.norm(self._matrix, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = (self._matrix @ query) / (chunk_norms * query_norm + 1e-10)
        top_k_idx = np.argsort(-similarities)[:k]
        return [self._chunks[i] for i in top_k_idx]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in chunks]))


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text())
    return [Chunk(**d) for d in data]
