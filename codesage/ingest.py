"""Turn a repo's files into embedded, addressable chunks.

Chunking is the DSA-adjacent decision here: fixed-size line windows are
simple and predictable (O(file length) to produce), at the cost of
sometimes splitting a function across two chunks. Good enough for v1;
the tradeoff is called out in the README.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TEXT_EXTENSIONS = {".py", ".md", ".txt", ".rst", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}


@dataclass
class Chunk:
    text: str
    file_path: str
    line_start: int
    line_end: int
    vector: list[float] | None = None
    name: str | None = None
    docstring: str | None = None


def iter_source_files(repo_path: Path) -> Iterator[Path]:
    for path in repo_path.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in TEXT_EXTENSIONS:
            continue
        yield path


def chunk_file(path: Path, lines_per_chunk: int = 40) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    chunks: list[Chunk] = []
    for start in range(0, len(lines), lines_per_chunk):
        window = lines[start : start + lines_per_chunk]
        if not any(line.strip() for line in window):
            continue
        chunks.append(
            Chunk(
                text="\n".join(window),
                file_path=str(path),
                line_start=start + 1,
                line_end=min(start + lines_per_chunk, len(lines)),
            )
        )
    return chunks


def ingest_repo(repo_path: Path, llm, delay_seconds: float = 0.5) -> list[Chunk]:
    # The embeddings API has a free-tier per-minute rate limit; firing every
    # chunk's embed call back-to-back 429s partway through any repo big
    # enough to matter. A fixed delay is a blunt fix (no backoff-aware
    # pacing), but it's enough to keep a single-repo ingest under the limit.
    chunks: list[Chunk] = []
    for file_path in iter_source_files(repo_path):
        for chunk in chunk_file(file_path):
            # Store paths relative to repo_path, not the absolute/prefixed path
            # chunk_file produced — read_file/list_files also resolve relative to
            # repo_path, so citations and tool-call paths must speak the same
            # coordinate system or the agent will misread its own search results.
            chunk.file_path = str(Path(chunk.file_path).relative_to(repo_path))
            chunk.vector = llm.embed(chunk.text)
            chunks.append(chunk)
            if delay_seconds:
                time.sleep(delay_seconds)
    return chunks
