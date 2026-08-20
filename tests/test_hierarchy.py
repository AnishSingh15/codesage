from pathlib import Path
from types import SimpleNamespace

from codesage.hierarchy import HierarchicalIndex, build_hierarchy_chunks
from codesage.ingest import Chunk


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, contents, tools=None):
        self.calls.append({"contents": contents, "tools": tools})
        return self._responses.pop(0)


def _chunk(file_path: str, name: str, text: str, docstring: str | None = None) -> Chunk:
    return Chunk(text=text, file_path=file_path, line_start=1, line_end=1, name=name, docstring=docstring)


def test_build_hierarchy_chunks_extracts_module_functions_and_classes(tmp_path: Path):
    (tmp_path / "mod.py").write_text(
        '"""Module docstring."""\n'
        "\n"
        "def foo():\n"
        '    """Does foo things."""\n'
        "    return 1\n"
        "\n"
        "class Bar:\n"
        '    """A bar class."""\n'
        "    pass\n"
    )

    chunks = build_hierarchy_chunks(tmp_path)

    names = {c.name for c in chunks}
    assert names == {"module", "foo", "Bar"}

    foo_chunk = next(c for c in chunks if c.name == "foo")
    assert foo_chunk.docstring == "Does foo things."
    assert "def foo():" in foo_chunk.text
    assert foo_chunk.file_path == "mod.py"

    bar_chunk = next(c for c in chunks if c.name == "Bar")
    assert bar_chunk.docstring == "A bar class."


def test_build_hierarchy_chunks_skips_files_with_syntax_errors(tmp_path: Path):
    (tmp_path / "broken.py").write_text("def foo(:\n    pass")
    (tmp_path / "good.py").write_text("def bar():\n    pass\n")

    chunks = build_hierarchy_chunks(tmp_path)

    names = {c.name for c in chunks}
    assert "bar" in names
    assert not any(c.file_path == "broken.py" for c in chunks)


def test_build_hierarchy_chunks_falls_back_to_line_windows_for_non_python(tmp_path: Path):
    (tmp_path / "notes.md").write_text("\n".join(f"line{i}" for i in range(1, 50)))

    chunks = build_hierarchy_chunks(tmp_path)

    assert len(chunks) == 2  # 50 lines / 40-line window -> 2 chunks
    assert {c.name for c in chunks} == {"chunk_0", "chunk_1"}
    assert all(c.file_path == "notes.md" for c in chunks)


def test_search_sends_table_of_contents_and_query_to_llm():
    chunks = [
        _chunk("auth.py", "login", "def login(): ...", docstring="Logs a user in."),
        _chunk("db.py", "connect", "def connect(): ...", docstring="Opens a DB connection."),
    ]
    index = HierarchicalIndex(chunks)
    fake_llm = FakeLLM([SimpleNamespace(text="auth.py:login", function_calls=None)])

    results = index.search("how does login work?", fake_llm, k=1)

    assert results == [chunks[0]]
    prompt_text = fake_llm.calls[0]["contents"][0].parts[0].text
    assert "auth.py:login — Logs a user in." in prompt_text
    assert "how does login work?" in prompt_text


def test_search_falls_back_to_first_k_when_response_doesnt_match():
    chunks = [
        _chunk("a.py", "one", "..."),
        _chunk("b.py", "two", "..."),
        _chunk("c.py", "three", "..."),
    ]
    index = HierarchicalIndex(chunks)
    fake_llm = FakeLLM([SimpleNamespace(text="nonsense response", function_calls=None)])

    results = index.search("anything", fake_llm, k=2)

    assert results == chunks[:2]


def test_search_on_empty_index_returns_empty_list():
    index = HierarchicalIndex([])
    assert index.search("anything", FakeLLM([]), k=5) == []
