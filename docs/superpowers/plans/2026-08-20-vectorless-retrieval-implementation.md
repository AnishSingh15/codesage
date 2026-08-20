# Vectorless Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, AST-based "vectorless" retrieval strategy that coexists with the existing brute-force vector one, directly comparable via a new `codesage eval --compare` command.

**Architecture:** `codesage/hierarchy.py` builds `Chunk`s from Python's `ast` module (zero LLM calls) and a `HierarchicalIndex` that navigates them via one `generate()` call per query. A small adapter (`VectorRetrievalStrategy`) gives the existing `RetrievalIndex` the same `search(query, llm, k)` shape so `eval.py` can score both strategies uniformly — this is a real, necessary interface correction discovered while planning (see Task 2), not present in the original spec's assumption that no `eval.py` changes were needed.

**Tech Stack:** Same as the existing project — Python 3.13's `ast` module (stdlib, no new dependency), `google-genai`, `pytest`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-20-vectorless-retrieval-design.md` — every task implements one piece of it.
- Both retrieval strategies coexist; the vector strategy's existing behavior/tests must not regress.
- `build_hierarchy_chunks` makes zero LLM calls. `HierarchicalIndex.search` makes exactly one `generate()` call per query.
- All new logic is unit-tested against the existing `FakeLLM` double pattern, not real API calls — except one `@pytest.mark.integration` test.
- Decisions log at `docs/superpowers/decisions/2026-08-20-vectorless-retrieval-decisions.md` (gitignored). **Every task ends with an entry appended to it. Never `git add` this file — verify with `git status` before each commit.**

---

### Task 1: `Chunk` gets `name` and `docstring` fields

**Why:** `HierarchicalIndex` needs a stable lookup key (`"{file_path}:{name}"`) and a short description per chunk for its table-of-contents view. Neither exists on `Chunk` today. Both fields default to `None`, so this is fully backward compatible — existing persisted `.codesage_index.json` files (with no `name`/`docstring` keys) still load fine via `Chunk(**d)`.

**Files:**
- Modify: `codesage/ingest.py:17-23` (the `Chunk` dataclass)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Produces: `Chunk(text, file_path, line_start, line_end, vector=None, name=None, docstring=None)`. Every existing caller that constructs `Chunk` without these two new kwargs is unaffected (they're optional, defaulting to `None`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingest.py`:

```python
def test_chunk_name_and_docstring_default_to_none():
    chunk = Chunk(text="x = 1", file_path="a.py", line_start=1, line_end=1)

    assert chunk.name is None
    assert chunk.docstring is None


def test_chunk_accepts_name_and_docstring():
    chunk = Chunk(text="x = 1", file_path="a.py", line_start=1, line_end=1, name="foo", docstring="Does foo.")

    assert chunk.name == "foo"
    assert chunk.docstring == "Does foo."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `TypeError: Chunk.__init__() got an unexpected keyword argument 'name'`

- [ ] **Step 3: Update the `Chunk` dataclass**

In `codesage/ingest.py`, change:

```python
@dataclass
class Chunk:
    text: str
    file_path: str
    line_start: int
    line_end: int
    vector: list[float] | None = None
```

to:

```python
@dataclass
class Chunk:
    text: str
    file_path: str
    line_start: int
    line_end: int
    vector: list[float] | None = None
    name: str | None = None
    docstring: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass — new fields are optional, nothing existing constructs `Chunk` with conflicting positional args (every existing call site uses keyword args for these fields already, per the current codebase's style).

- [ ] **Step 6: Append a decisions log entry**

Append to `docs/superpowers/decisions/2026-08-20-vectorless-retrieval-decisions.md` (create the file if it doesn't exist yet):

```markdown
# Vectorless Retrieval — Decisions Log

Local-only running journal, one entry per task, written as I go. Not committed to git.

## Task 1: Chunk gets name/docstring, both optional

Decision: add `name: str | None = None` and `docstring: str | None = None`
to Chunk instead of creating a separate `HierarchyChunk` subclass or
dataclass.

Why: the vector strategy's chunks never need these fields (they stay
`None`), and one shared `Chunk` type means `save_chunks`/`load_chunks`,
the `search_code` tool's citation formatting, and every place that
already handles `Chunk` keeps working unchanged for both strategies —
no parallel type to keep in sync.
```

- [ ] **Step 7: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/ingest.py tests/test_ingest.py
git commit -m "feat: add optional name/docstring fields to Chunk

Backward compatible (both default to None) — needed by the upcoming
hierarchical retrieval strategy for its lookup keys and descriptions."
```

---

### Task 2: Unify the retrieval interface

**Why this task exists:** The design spec assumed `eval.py` needed no changes — wrong. `score_retrieval` currently hardcodes the vector strategy's shape (`llm.embed(question)` then `index.search(vector, k)`). `HierarchicalIndex.search` (Task 3) will take `(query: str, llm, k)` — a genuinely different signature, not a drop-in. To score both strategies through the same `run_eval`, both need a common `.search(query: str, llm, k: int) -> list[Chunk]` shape. Rather than changing `RetrievalIndex` itself (which `make_search_code_tool` depends on directly, tested and working), this task adds a thin adapter and updates only `eval.py` and its callers to go through the common shape.

**Files:**
- Modify: `codesage/index.py` (add `VectorRetrievalStrategy`)
- Modify: `codesage/eval.py:1-33` (docstring + `score_retrieval`)
- Modify: `tests/test_eval.py` (wrap `RetrievalIndex` in the adapter)
- Modify: `codesage/cli.py:88-93` (the plain `eval` command's index construction)

**Interfaces:**
- Consumes: `RetrievalIndex` from `codesage.index` (unchanged).
- Produces: `VectorRetrievalStrategy(index: RetrievalIndex)` with `.search(query: str, llm, k: int = 5) -> list[Chunk]`. Task 3's `HierarchicalIndex` will independently implement the same shape natively (no adapter needed there).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_index.py`:

```python
def test_vector_retrieval_strategy_embeds_query_then_searches():
    chunks = [_chunk("auth", [1.0, 0.0, 0.0])]
    index = RetrievalIndex(chunks)
    strategy = VectorRetrievalStrategy(index)

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            assert text == "how does auth work?"
            return [1.0, 0.0, 0.0]

    results = strategy.search("how does auth work?", FakeLLM(), k=1)

    assert results == [chunks[0]]
```

Add `VectorRetrievalStrategy` to the existing import line at the top of `tests/test_index.py`:
```python
from codesage.index import RetrievalIndex, VectorRetrievalStrategy, save_chunks, load_chunks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL — `ImportError: cannot import name 'VectorRetrievalStrategy'`

- [ ] **Step 3: Add `VectorRetrievalStrategy` to `codesage/index.py`**

Append to the end of `codesage/index.py`:

```python
class VectorRetrievalStrategy:
    """Adapts RetrievalIndex's (query_vector, k) shape to the (query, llm, k)
    shape both retrieval strategies expose uniformly to eval.py. RetrievalIndex
    itself stays untouched — make_search_code_tool still calls
    RetrievalIndex.search(vector, k) directly; this adapter exists only for
    code paths (eval.py, the eval --compare CLI flag) that need to treat both
    strategies the same way."""

    def __init__(self, index: RetrievalIndex):
        self._index = index

    def search(self, query: str, llm, k: int = 5) -> list[Chunk]:
        query_vector = llm.embed(query)
        return self._index.search(query_vector, k=k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_index.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Update `codesage/eval.py`**

Replace the module docstring and `score_retrieval`:

```python
"""A small, repeatable check that CodeSage's answers are actually grounded.

Two proxies, both cheap and deterministic:
- retrieval hit-rate: did we retrieve a chunk from the file we expected?
- answer score: fraction of expected keywords present in the final answer.

score_retrieval calls `index.search(question, llm, k) -> list[Chunk]` — a
shape both VectorRetrievalStrategy (index.py) and HierarchicalIndex
(hierarchy.py) implement, so this file needs no changes to score either
retrieval strategy.
"""
```

```python
def score_retrieval(index, llm, case: EvalCase) -> bool:
    results = index.search(case.question, llm, k=5)
    return any(case.expected_file_substring in c.file_path for c in results)
```

- [ ] **Step 6: Update `tests/test_eval.py`**

Change the import line:
```python
from codesage.index import RetrievalIndex
```
to:
```python
from codesage.index import RetrievalIndex, VectorRetrievalStrategy
```

In `test_score_retrieval_true_when_expected_file_is_returned`, change:
```python
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = RetrievalIndex([chunk])
    case = EvalCase(question="q", expected_file_substring="auth.py", expected_answer_keywords=[])

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [1.0, 0.0]

    assert score_retrieval(index, FakeLLM(), case) is True
```
to:
```python
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = VectorRetrievalStrategy(RetrievalIndex([chunk]))
    case = EvalCase(question="q", expected_file_substring="auth.py", expected_answer_keywords=[])

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [1.0, 0.0]

    assert score_retrieval(index, FakeLLM(), case) is True
```

In `test_run_eval_aggregates_scores`, change:
```python
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = RetrievalIndex([chunk])
```
to:
```python
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = VectorRetrievalStrategy(RetrievalIndex([chunk]))
```

In `test_run_eval_builds_a_fresh_agent_per_case`, change:
```python
    index = RetrievalIndex([])
```
to:
```python
    index = VectorRetrievalStrategy(RetrievalIndex([]))
```

- [ ] **Step 7: Update `codesage/cli.py`'s plain `eval` command**

In the `elif args.command == "eval":` block, change:
```python
        chunks = load_chunks(index_path)
        index = RetrievalIndex(chunks)
        registry = build_base_registry(repo_path)
        registry.register(make_search_code_tool(index, llm))

        results = run_eval(lambda: Agent(llm, tools=registry), index, llm, load_cases(cases_path))
```
to:
```python
        from codesage.index import VectorRetrievalStrategy

        chunks = load_chunks(index_path)
        index = RetrievalIndex(chunks)
        registry = build_base_registry(repo_path)
        registry.register(make_search_code_tool(index, llm))

        results = run_eval(
            lambda: Agent(llm, tools=registry), VectorRetrievalStrategy(index), llm, load_cases(cases_path)
        )
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass (40 + the 2 new tests from Task 1 + 1 new test from this task = 43 passed, 2 deselected)

- [ ] **Step 9: Append a decisions log entry**

```markdown
## Task 2: unified retrieval interface — VectorRetrievalStrategy adapter

Decision: don't change RetrievalIndex.search's signature. Add a separate
VectorRetrievalStrategy wrapper exposing `.search(query, llm, k)`, and
change eval.py's score_retrieval to call that shape instead of hardcoding
`llm.embed()` + `index.search(vector, k)` itself.

Why: RetrievalIndex.search(query_vector, k) is used directly by
make_search_code_tool and is well-tested; changing its signature would
mean updating that tool and re-verifying it for no real benefit. An
adapter gets both retrieval strategies onto one interface (what eval.py
and the upcoming `eval --compare` actually need) without touching code
that already works. This is the corrected version of the spec's original
(wrong) assumption that no eval.py changes would be needed — the
Strategy-pattern seam existed, but its assumed shape didn't match what a
truly vectorless strategy can provide (there's no vector to accept).
```

- [ ] **Step 10: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/index.py codesage/eval.py tests/test_index.py tests/test_eval.py codesage/cli.py
git commit -m "refactor: unify retrieval interface via VectorRetrievalStrategy

score_retrieval hardcoded the vector strategy's embed-then-search shape.
Added an adapter so eval.py can score any retrieval strategy through one
common search(query, llm, k) interface — required before a genuinely
vectorless strategy (no query vector to accept) can be compared."
```

---

### Task 3: `codesage/hierarchy.py` — AST-based chunks + navigation

**Files:**
- Create: `codesage/hierarchy.py`
- Test: `tests/test_hierarchy.py`

**Interfaces:**
- Consumes: `Chunk`, `iter_source_files`, `chunk_file` from `codesage.ingest`.
- Produces: `build_hierarchy_chunks(repo_path: Path) -> list[Chunk]`; `HierarchicalIndex(chunks: list[Chunk])` with `.search(query: str, llm, k: int = 5) -> list[Chunk]`. Task 4 wires both into a CLI-facing tool.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hierarchy.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hierarchy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'codesage.hierarchy'`

- [ ] **Step 3: Write `codesage/hierarchy.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hierarchy.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions

- [ ] **Step 6: Append a decisions log entry**

```markdown
## Task 3: hierarchy.py — top-level-only AST extraction, fallback-to-first-k

Decision: only walk `tree.body` (top-level definitions), not nested
functions/classes. And when the LLM's response doesn't name any known
key, fall back to returning the first k chunks rather than raising.

Why: top-level-only keeps the table of contents proportional to a file's
public surface, not every helper closure — matches how a human skims a
file first. The fallback matters because this strategy makes a real
network call whose output format isn't enforced by a schema (unlike the
vector strategy's tool-calling, which is API-structured) — a malformed
or off-format response is a real possibility, and "return something
plausible" beats "crash the whole ask() call."
```

- [ ] **Step 7: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/hierarchy.py tests/test_hierarchy.py
git commit -m "feat: add hierarchy.py — AST-based vectorless retrieval

build_hierarchy_chunks extracts structure via Python's ast module, zero
LLM calls. HierarchicalIndex.search makes one generate() call per query
to navigate that structure by reasoning instead of vector similarity."
```

---

### Task 4: Persistence filename + `search_code` tool for the hierarchical strategy

**Files:**
- Modify: `codesage/index.py` (add `HIERARCHY_INDEX_FILENAME`)
- Modify: `codesage/tools.py` (add `make_hierarchical_search_tool`)
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `HierarchicalIndex` from `codesage.hierarchy`.
- Produces: `HIERARCHY_INDEX_FILENAME = ".codesage_hierarchy_index.json"` (index.py); `make_hierarchical_search_tool(index: HierarchicalIndex, llm) -> Tool`, registered under the name `"search_code"` — same name the vector version uses, since a given `ToolRegistry` only ever holds one retrieval strategy's tool at a time (Task 5 decides which at `ask`/`eval` time).

- [ ] **Step 1: Add the filename constant**

In `codesage/index.py`, change:
```python
INDEX_FILENAME = ".codesage_index.json"
```
to:
```python
INDEX_FILENAME = ".codesage_index.json"
HIERARCHY_INDEX_FILENAME = ".codesage_hierarchy_index.json"
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_tools.py`:

```python
from codesage.hierarchy import HierarchicalIndex
from codesage.ingest import Chunk


def test_make_hierarchical_search_tool_returns_citation_formatted_results():
    chunk = Chunk(text="def login(): ...", file_path="auth.py", line_start=3, line_end=5, name="login")
    index = HierarchicalIndex([chunk])

    class FakeLLM:
        def generate(self, contents, tools=None):
            from types import SimpleNamespace
            return SimpleNamespace(text="auth.py:login", function_calls=None)

    tool = make_hierarchical_search_tool(index, FakeLLM())
    result = tool.handler(query="how does login work?")

    assert "auth.py:3-5" in result
    assert "def login(): ..." in result


def test_make_hierarchical_search_tool_reports_no_match():
    index = HierarchicalIndex([])
    tool = make_hierarchical_search_tool(index, llm=None)

    result = tool.handler(query="anything")

    assert result == "No matching code found."
```

Change the existing `codesage.tools` import line at the top of `tests/test_tools.py` from:
```python
from codesage.tools import Tool, ToolRegistry, list_files_handler, read_file_handler, repo_tree_handler
```
to:
```python
from codesage.tools import (
    Tool,
    ToolRegistry,
    list_files_handler,
    read_file_handler,
    repo_tree_handler,
    make_hierarchical_search_tool,
)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_hierarchical_search_tool'`

- [ ] **Step 4: Add `make_hierarchical_search_tool` to `codesage/tools.py`**

Add this import near the top of `codesage/tools.py`, alongside the existing `from codesage.index import RetrievalIndex`:
```python
from codesage.hierarchy import HierarchicalIndex
```

Append this function to the end of `codesage/tools.py`:

```python
def make_hierarchical_search_tool(index: HierarchicalIndex, llm) -> Tool:
    def handler(query: str) -> str:
        results = index.search(query, llm, k=5)
        if not results:
            return "No matching code found."
        return "\n\n".join(f"{c.file_path}:{c.line_start}-{c.line_end}\n{c.text}" for c in results)

    return Tool(
        name="search_code",
        description=(
            "Structure-based search over the ingested codebase (navigates a "
            "table of contents via reasoning, not embeddings). Returns "
            "relevant code chunks with file/line citations."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
        handler=handler,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass. Also sanity-check there's no import cycle: `tools.py` now imports from `hierarchy.py`, and `hierarchy.py` imports from `ingest.py` only — no path back to `tools.py`, so this is safe.

- [ ] **Step 7: Append a decisions log entry**

```markdown
## Task 4: search_code tool name reused for both strategies

Decision: make_hierarchical_search_tool registers under the same tool
name, "search_code", that make_search_code_tool (vector) uses.

Why: the agent loop and its prompts never need to know which retrieval
strategy is active — from the model's perspective there's just "a
search_code tool." Only one strategy's tool is ever registered in a
given ToolRegistry at a time (the CLI's --strategy flag decides which),
so there's no name collision in practice, and the agent's behavior is
identical either way — exactly the point of both strategies sharing an
interface.
```

- [ ] **Step 8: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/index.py codesage/tools.py tests/test_tools.py
git commit -m "feat: add hierarchical search_code tool + persistence filename

Same tool name as the vector strategy's — the agent never needs to know
which retrieval strategy is active, only the CLI's --strategy flag does."
```

---

### Task 5: CLI — `ingest-hierarchy` and `ask --strategy`

**Files:**
- Modify: `codesage/cli.py`

**Interfaces:**
- Consumes: `build_hierarchy_chunks` from `codesage.hierarchy`; `HIERARCHY_INDEX_FILENAME` from `codesage.index`; `make_hierarchical_search_tool` from `codesage.tools`.

- [ ] **Step 1: Add the `ingest-hierarchy` subparser**

In `codesage/cli.py`, after the `ingest_parser` block, add:

```python
    ingest_hierarchy_parser = subparsers.add_parser("ingest-hierarchy")
    ingest_hierarchy_parser.add_argument("repo_path")
```

- [ ] **Step 2: Add the `--strategy` flag to the `ask` subparser**

Change:
```python
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--repo", default=".", help="Target repo path")
```
to:
```python
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--repo", default=".", help="Target repo path")
    ask_parser.add_argument(
        "--strategy", choices=["vector", "hierarchy"], default="vector", help="Retrieval strategy"
    )
```

- [ ] **Step 3: Update the import line for `HIERARCHY_INDEX_FILENAME`**

Change:
```python
from codesage.index import INDEX_FILENAME, RetrievalIndex, load_chunks, save_chunks
```
to:
```python
from codesage.index import HIERARCHY_INDEX_FILENAME, INDEX_FILENAME, RetrievalIndex, load_chunks, save_chunks
```

- [ ] **Step 4: Add the `ingest-hierarchy` command branch**

After the `if args.command == "ingest":` block's body, add:

```python
    elif args.command == "ingest-hierarchy":
        from codesage.hierarchy import build_hierarchy_chunks

        repo_path = Path(args.repo_path)
        chunks = build_hierarchy_chunks(repo_path)
        save_chunks(chunks, repo_path / HIERARCHY_INDEX_FILENAME)
        print(f"Indexed {len(chunks)} chunks from {repo_path} (hierarchical, no API calls)")
```

- [ ] **Step 5: Update the `ask` command branch to respect `--strategy`**

Replace the whole `elif args.command == "ask":` block with:

```python
    elif args.command == "ask":
        repo_path = Path(args.repo)
        registry = build_base_registry(repo_path)

        if args.strategy == "vector":
            index_path = repo_path / INDEX_FILENAME
            if index_path.exists():
                chunks = load_chunks(index_path)
                index = RetrievalIndex(chunks)
                registry.register(make_search_code_tool(index, llm))
            else:
                print(
                    f"(no index found at {index_path} — run 'codesage ingest {repo_path}' "
                    "for retrieval-backed answers; continuing with list_files/read_file only)",
                    file=sys.stderr,
                )
        else:
            from codesage.hierarchy import HierarchicalIndex
            from codesage.tools import make_hierarchical_search_tool

            hierarchy_index_path = repo_path / HIERARCHY_INDEX_FILENAME
            if hierarchy_index_path.exists():
                chunks = load_chunks(hierarchy_index_path)
                hierarchy_index = HierarchicalIndex(chunks)
                registry.register(make_hierarchical_search_tool(hierarchy_index, llm))
            else:
                print(
                    f"(no hierarchy index found at {hierarchy_index_path} — run "
                    f"'codesage ingest-hierarchy {repo_path}' for retrieval-backed answers; "
                    "continuing with list_files/read_file only)",
                    file=sys.stderr,
                )

        agent = Agent(llm, tools=registry)
        print(agent.ask(args.question))
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass — argparse wiring isn't unit tested directly, matching the existing convention.

- [ ] **Step 7: Manual smoke test — build the hierarchy index**

Run: `uv run codesage ingest-hierarchy target_repo/src`
Expected: prints `Indexed N chunks from target_repo/src (hierarchical, no API calls)` — should complete in well under a second since it makes zero API calls.

- [ ] **Step 8: Manual smoke test — ask with the hierarchical strategy**

Run: `uv run codesage ask "How does the Session class handle connection pooling?" --repo target_repo/src --strategy hierarchy`
Expected: a real, sensible answer citing `sessions.py` and/or `adapters.py` — read it and confirm it's actually accurate, not just present.

- [ ] **Step 9: Append a decisions log entry**

```markdown
## Task 5: ingest-hierarchy is a separate command, not folded into ingest

Decision: `codesage ingest-hierarchy` is its own subcommand rather than
a flag on the existing `codesage ingest` (e.g. `--strategy`).

Why: `ingest` costs real API quota and takes real time (rate-limited
embed calls); `ingest-hierarchy` costs nothing and is near-instant. A
shared command with a flag would blur that they have completely
different cost profiles. Two commands makes the tradeoff visible at the
command-name level, not buried in a flag someone might not read.

Real result from the smoke test: `codesage ask --strategy hierarchy`
against target_repo/src gave an accurate answer citing real files —
[fill in the actual file(s) it cited once run].
```

- [ ] **Step 10: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/cli.py
git commit -m "feat: add ingest-hierarchy command and ask --strategy flag

ingest-hierarchy is a separate command (not a flag on ingest) since it
has a completely different cost profile — zero API calls vs.
rate-limited embed calls for every chunk."
```

---

### Task 6: CLI — `eval --compare`

**Files:**
- Modify: `codesage/cli.py`

**Interfaces:**
- Consumes: `run_eval` from `codesage.eval`; `VectorRetrievalStrategy` from `codesage.index`; `HierarchicalIndex` from `codesage.hierarchy`; `make_hierarchical_search_tool` from `codesage.tools`.

- [ ] **Step 1: Add the `--compare` flag**

Change:
```python
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--repo", default=".", help="Target repo path")
    eval_parser.add_argument("--cases", default="eval_cases.json", help="Path to eval cases JSON")
```
to:
```python
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--repo", default=".", help="Target repo path")
    eval_parser.add_argument("--cases", default="eval_cases.json", help="Path to eval cases JSON")
    eval_parser.add_argument(
        "--compare", action="store_true", help="Run against both retrieval strategies and compare"
    )
```

- [ ] **Step 2: Replace the `eval` command branch**

Replace the whole `elif args.command == "eval":` block with:

```python
    elif args.command == "eval":
        from codesage.eval import load_cases, run_eval
        from codesage.index import VectorRetrievalStrategy

        repo_path = Path(args.repo)
        cases_path = Path(args.cases)
        if not cases_path.exists():
            print(f"No eval cases found at {cases_path}. See eval_cases.json for the format.", file=sys.stderr)
            sys.exit(1)
        cases = load_cases(cases_path)

        if args.compare:
            from codesage.hierarchy import HierarchicalIndex
            from codesage.tools import make_hierarchical_search_tool

            index_path = repo_path / INDEX_FILENAME
            hierarchy_index_path = repo_path / HIERARCHY_INDEX_FILENAME
            if not index_path.exists():
                print(f"No vector index found. Run 'codesage ingest {repo_path}' first.", file=sys.stderr)
                sys.exit(1)
            if not hierarchy_index_path.exists():
                print(
                    f"No hierarchy index found. Run 'codesage ingest-hierarchy {repo_path}' first.",
                    file=sys.stderr,
                )
                sys.exit(1)

            vector_index = RetrievalIndex(load_chunks(index_path))
            vector_registry = build_base_registry(repo_path)
            vector_registry.register(make_search_code_tool(vector_index, llm))
            vector_results = run_eval(
                lambda: Agent(llm, tools=vector_registry), VectorRetrievalStrategy(vector_index), llm, cases
            )

            hierarchy_index = HierarchicalIndex(load_chunks(hierarchy_index_path))
            hierarchy_registry = build_base_registry(repo_path)
            hierarchy_registry.register(make_hierarchical_search_tool(hierarchy_index, llm))
            hierarchy_results = run_eval(
                lambda: Agent(llm, tools=hierarchy_registry), hierarchy_index, llm, cases
            )

            print(f"{'Strategy':<12}{'Retrieval hit rate':<22}{'Avg answer score':<20}")
            print(
                f"{'vector':<12}"
                f"{vector_results['retrieval_hit_rate']:<22.0%}"
                f"{vector_results['avg_answer_score']:<20.0%}"
            )
            print(
                f"{'hierarchy':<12}"
                f"{hierarchy_results['retrieval_hit_rate']:<22.0%}"
                f"{hierarchy_results['avg_answer_score']:<20.0%}"
            )

        else:
            index_path = repo_path / INDEX_FILENAME
            if not index_path.exists():
                print(f"No index found. Run 'codesage ingest {repo_path}' first.", file=sys.stderr)
                sys.exit(1)

            chunks = load_chunks(index_path)
            index = RetrievalIndex(chunks)
            registry = build_base_registry(repo_path)
            registry.register(make_search_code_tool(index, llm))

            results = run_eval(lambda: Agent(llm, tools=registry), VectorRetrievalStrategy(index), llm, cases)
            print(f"Retrieval hit rate: {results['retrieval_hit_rate']:.0%}")
            print(f"Avg answer score:   {results['avg_answer_score']:.0%}")
```

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass

- [ ] **Step 4: Manual smoke test — plain eval still works (no regression)**

Run: `uv run codesage eval --repo target_repo/src`
Expected: prints `Retrieval hit rate: ...` / `Avg answer score: ...` — same output shape as before this plan started.

- [ ] **Step 5: Manual smoke test — real `--compare` run, record the actual numbers**

Run: `uv run codesage eval --repo target_repo/src --compare`
Expected: a two-row table comparing `vector` and `hierarchy` on the same `eval_cases.json` questions. **Write down the exact printed numbers** — they go into both the decisions log (this task) and the README (Task 8). Do not estimate or reconstruct these later; use the real output from this run.

- [ ] **Step 6: Append a decisions log entry**

```markdown
## Task 6: eval --compare — real numbers

Decision: --compare runs both strategies against the identical
eval_cases.json, back to back, and prints a plain-text table — no new
scoring logic, just two run_eval calls and a formatted print.

Real result from running `codesage eval --repo target_repo/src --compare`:

[PASTE THE ACTUAL TABLE OUTPUT FROM STEP 5 HERE, VERBATIM]

Why these numbers look the way they do: [one or two sentences of honest
interpretation once you have the real numbers — e.g. if hierarchy scores
lower, that's expected for a first version and worth saying plainly, not
smoothing over].
```

- [ ] **Step 7: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/cli.py
git commit -m "feat: add eval --compare — score both retrieval strategies

Runs the same eval_cases.json against vector and hierarchy strategies
back to back, prints a side-by-side table. No new scoring logic — two
run_eval calls, since both strategies now share one interface (Task 2)."
```

---

### Task 7: Integration test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add the test**

Append to `tests/test_integration.py`:

```python
from codesage.hierarchy import HierarchicalIndex, build_hierarchy_chunks
from codesage.tools import make_hierarchical_search_tool


@pytest.mark.integration
def test_hierarchical_ask_against_real_repo():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    if not TARGET_REPO_SRC.exists():
        pytest.skip("target_repo/src not present locally")

    llm = LLMClient(api_key=api_key)
    chunks = build_hierarchy_chunks(TARGET_REPO_SRC)
    index = HierarchicalIndex(chunks)

    registry = build_base_registry(TARGET_REPO_SRC)
    registry.register(make_hierarchical_search_tool(index, llm))
    agent = Agent(llm, tools=registry)

    answer = agent.ask("How does the Session class handle connection pooling?")

    assert "session" in answer.lower() or "adapter" in answer.lower()
```

- [ ] **Step 2: Run it against the real API**

Run: `uv run pytest tests/test_integration.py -m integration -v`
Expected: PASS (all 3 integration tests: fixture-repo agent test, real-repo onboarding test, and this new hierarchical test)

- [ ] **Step 3: Confirm the default run still excludes it**

Run: `uv run pytest -v`
Expected: this test doesn't appear in the output (excluded by `-m 'not integration'`)

- [ ] **Step 4: Append a decisions log entry**

```markdown
## Task 7: integration test asserts on either "session" or "adapter"

Decision: the assertion accepts the answer mentioning either "session"
or "adapter", not a single fixed keyword.

Why: this strategy's retrieval depends on the LLM's own reasoning over
the table of contents, which is less deterministic than cosine-similarity
ranking — asserting on a broader, still-meaningful signal (the topic was
actually addressed) is more robust than pinning to one exact word choice
that a slightly different but equally correct phrasing could miss.
```

- [ ] **Step 5: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add tests/test_integration.py
git commit -m "test: add integration test for hierarchical retrieval"
```

---

### Task 8: README — document both strategies with real numbers

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a new section explaining both strategies**

Add a new section after the existing `## Why brute-force retrieval, not a vector DB` section:

```markdown
## Two retrieval strategies, compared

CodeSage ships two retrieval strategies, selectable per-query:

- **`vector`** (default) — brute-force cosine similarity over embedded
  chunks (see above).
- **`hierarchy`** — "vectorless" retrieval. `codesage/hierarchy.py` uses
  Python's `ast` module to extract each file's structure (module
  docstring, every top-level function/class with its docstring and
  exact line span) with **zero LLM calls**. At query time, one
  `generate()` call shows the model that structure as a table of
  contents and asks which entries are relevant — reasoning-based
  navigation instead of nearest-neighbor search, in the spirit of
  PageIndex-style "vectorless RAG."

Both strategies expose the same `search(query, llm, k) -> list[Chunk]`
shape (see `VectorRetrievalStrategy` in `index.py`), so `eval.py` scores
either one identically, and the agent's tool-calling loop never needs to
know which is active.

Real comparison, same `eval_cases.json` questions, same target repo:

```bash
uv run codesage ingest target_repo/src
uv run codesage ingest-hierarchy target_repo/src
uv run codesage eval --repo target_repo/src --compare
```

```
[PASTE THE ACTUAL --compare TABLE OUTPUT FROM TASK 6 HERE, VERBATIM]
```
```

- [ ] **Step 2: Update the architecture table**

In the existing `## Architecture` table, add a row:

```markdown
| `hierarchy.py` | AST-based "vectorless" retrieval — `build_hierarchy_chunks` (zero LLM calls) + `HierarchicalIndex` (one reasoning call per query). |
```

Update the `index.py` row to mention the new adapter:
```markdown
| `index.py` | Brute-force cosine-similarity search over chunk vectors (`numpy`), plus JSON persistence between CLI runs. `VectorRetrievalStrategy` adapts it to the common `search(query, llm, k)` shape both strategies share. |
```

- [ ] **Step 3: Update the Usage section**

In the existing `## Usage` code block, add:

```bash
uv run codesage ingest-hierarchy <path-to-a-repo>     # zero-cost structural index
uv run codesage ask "<question>" --repo <path-to-a-repo> --strategy hierarchy
uv run codesage eval --repo <path-to-a-repo> --compare # scores both strategies
```

- [ ] **Step 4: Update the Mermaid architecture diagram**

In the existing Mermaid block (`## Architecture diagram`), change:
```
    Tools["ToolRegistry: list_files, read_file, search_code, repo_tree"]
    Index["RetrievalIndex (cosine similarity)"]
```
to:
```
    Tools["ToolRegistry: list_files, read_file, search_code, repo_tree"]
    Index["RetrievalIndex (cosine similarity)"]
    Hierarchy["HierarchicalIndex (AST + reasoning)"]
```
and add, right after the existing `Tools --> Index` line:
```
    Tools -.->|alt strategy| Hierarchy
```

- [ ] **Step 5: Run the full suite one more time**

Run: `uv run pytest -v`
Expected: all pass — this task is documentation-only.

- [ ] **Step 6: Append the final decisions log entry**

```markdown
## Task 8: README shows the real numbers, not a claim

Decision: the README's comparison section embeds the actual
`eval --compare` table output verbatim, not a summary or a claim like
"performs comparably."

Why: this whole feature exists to make retrieval strategy tradeoffs
demonstrable, not asserted. Anyone reading the README can re-run the
exact three commands shown and get the same kind of table themselves —
the claim is falsifiable, which is the point.
```

- [ ] **Step 7: Commit and push**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add README.md
git commit -m "docs: document both retrieval strategies with real comparison numbers

Includes the actual eval --compare table output, not a paraphrase —
readers can re-run the exact commands shown and reproduce it."
git push
```

---

## After all 8 tasks

- `codesage ask --strategy hierarchy` and `codesage eval --compare` both work end to end, verified against `target_repo/src`.
- Full suite green (existing + new tests), CI green on every commit.
- Decisions log has one entry per task, including the real `--compare` numbers, and exists only locally.
- README documents both strategies with real, reproducible numbers.
- The interface-unification work (Task 2) means any *third* retrieval strategy in the future only needs to implement `search(query, llm, k) -> list[Chunk]` — no changes to `eval.py` required.
