"""AST-based "vectorless" retrieval: navigate a repo's structure via LLM
reasoning instead of embedding-based nearest-neighbor search.

build_hierarchy_chunks extracts structure with Python's ast module —
module docstrings, top-level function/class names, docstrings, and exact
line spans — with zero LLM calls. HierarchicalIndex.search then makes
exactly one LLM call per query: show the model a table of contents, ask
which entries are relevant. Same cost shape as the vector strategy's one
embed() call per query, which is what makes `eval --compare` a fair
head-to-head.
"""

import ast
from pathlib import Path

from google.genai import types

from codesage.ingest import Chunk, chunk_file, iter_source_files


def build_hierarchy_chunks(repo_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file_path in iter_source_files(repo_path):
        relative_path = str(file_path.relative_to(repo_path))

        if file_path.suffix != ".py":
            for i, chunk in enumerate(chunk_file(file_path)):
                chunk.file_path = relative_path
                chunk.name = f"chunk_{i}"
                chunks.append(chunk)
            continue

        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        source_lines = source.splitlines()

        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.append(
                Chunk(
                    text=module_docstring,
                    file_path=relative_path,
                    line_start=1,
                    line_end=1,
                    name="module",
                )
            )

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = node.end_lineno or node.lineno
                text = "\n".join(source_lines[start - 1 : end])
                chunks.append(
                    Chunk(
                        text=text,
                        file_path=relative_path,
                        line_start=start,
                        line_end=end,
                        name=node.name,
                        docstring=ast.get_docstring(node),
                    )
                )
    return chunks


class HierarchicalIndex:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._by_key = {f"{c.file_path}:{c.name}": c for c in chunks}

    def _table_of_contents(self) -> str:
        lines = []
        for chunk in self._chunks:
            key = f"{chunk.file_path}:{chunk.name}"
            first_line = chunk.text.strip().splitlines()[0] if chunk.text.strip() else ""
            description = chunk.docstring or first_line
            lines.append(f"{key} — {description}")
        return "\n".join(lines)

    def search(self, query: str, llm, k: int = 5) -> list[Chunk]:
        if not self._chunks:
            return []
        prompt = (
            "Here is a table of contents for a codebase, one entry per line "
            "in the format 'file_path:name — short description':\n\n"
            f"{self._table_of_contents()}\n\n"
            f"Question: {query}\n\n"
            f"Reply with ONLY the {k} most relevant entry keys (the "
            "'file_path:name' part before the — ), one per line, nothing else."
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = llm.generate(contents)

        matched = []
        for line in (response.text or "").splitlines():
            key = line.strip()
            if key in self._by_key:
                matched.append(self._by_key[key])

        if not matched:
            return self._chunks[:k]
        return matched[:k]
