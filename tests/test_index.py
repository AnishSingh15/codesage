import json
from pathlib import Path

from codesage.ingest import Chunk
from codesage.index import RetrievalIndex, save_chunks, load_chunks


def _chunk(name: str, vector: list[float]) -> Chunk:
    return Chunk(text=f"code for {name}", file_path=f"{name}.py", line_start=1, line_end=10, vector=vector)


def test_search_ranks_by_cosine_similarity():
    chunks = [
        _chunk("auth", [1.0, 0.0, 0.0]),
        _chunk("db", [0.0, 1.0, 0.0]),
        _chunk("http", [0.9, 0.1, 0.0]),  # close to "auth"
    ]
    index = RetrievalIndex(chunks)

    results = index.search(query_vector=[1.0, 0.0, 0.0], k=2)

    assert [c.file_path for c in results] == ["auth.py", "http.py"]


def test_search_on_empty_index_returns_empty_list():
    index = RetrievalIndex([])
    assert index.search([1.0, 0.0], k=5) == []


def test_save_and_load_chunks_round_trip(tmp_path: Path):
    chunks = [_chunk("auth", [1.0, 0.0, 0.0])]
    path = tmp_path / "index.json"

    save_chunks(chunks, path)
    loaded = load_chunks(path)

    assert loaded == chunks
    assert json.loads(path.read_text())[0]["file_path"] == "auth.py"
