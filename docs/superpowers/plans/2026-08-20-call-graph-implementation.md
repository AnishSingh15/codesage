# Cross-File Call-Graph Traversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `find_callers`/`find_callees` agent tools that answer "who calls X" / "what does X call" across the whole repo, built and queried at zero LLM cost.

**Architecture:** A new `codesage/callgraph.py` module walks every `.py` file with an `ast.NodeVisitor` that tracks enclosing-function context on a stack, recording a `CallSite` for every call encountered — name-based, not import-resolved. Two new tools expose it; `codesage ingest-hierarchy` builds it in the same AST pass that already builds the hierarchy index.

**Tech Stack:** Python's stdlib `ast` module only — no new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-20-call-graph-design.md` — every task implements one piece of it.
- Name-based matching only — no import resolution. This is the core scoping decision, not a shortcut to fix later.
- Zero LLM calls to build the graph (pure static analysis) and zero LLM calls to query it (`find_callers`/`find_callees` are exact filters, not reasoning).
- Decisions log at `docs/superpowers/decisions/2026-08-20-call-graph-decisions.md` (gitignored). **Every task ends with an entry appended to it. Never `git add` this file — verify with `git status` before every commit.**

---

### Task 1: `codesage/callgraph.py` — build and query the call graph

**Files:**
- Create: `codesage/callgraph.py`
- Test: `tests/test_callgraph.py`

**Interfaces:**
- Consumes: `iter_source_files` from `codesage.ingest`.
- Produces: `CallSite` dataclass (`caller_name: str, caller_file: str, caller_line: int, called_name: str`); `build_call_graph(repo_path: Path) -> list[CallSite]`; `find_callers(call_sites: list[CallSite], name: str) -> list[CallSite]`; `find_callees(call_sites: list[CallSite], name: str) -> list[CallSite]`; `save_call_graph(call_sites: list[CallSite], path: Path) -> None`; `load_call_graph(path: Path) -> list[CallSite]`; `CALLGRAPH_FILENAME = ".codesage_callgraph.json"`. Task 2 (tools) and Task 3 (CLI) both depend on these exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_callgraph.py`:

```python
from pathlib import Path

import pytest

from codesage.callgraph import (
    CallSite,
    build_call_graph,
    find_callees,
    find_callers,
    load_call_graph,
    save_call_graph,
)

TARGET_REPO_SRC = Path(__file__).parent.parent / "target_repo" / "src"


def test_build_call_graph_finds_direct_calls(tmp_path: Path):
    (tmp_path / "mod.py").write_text("def foo():\n    bar()\n")

    call_sites = build_call_graph(tmp_path)

    assert any(cs.called_name == "bar" and cs.caller_name == "foo" for cs in call_sites)


def test_build_call_graph_finds_method_calls(tmp_path: Path):
    (tmp_path / "mod.py").write_text(
        "class Foo:\n"
        "    def bar(self):\n"
        "        self.baz()\n"
        "        other.qux()\n"
    )

    call_sites = build_call_graph(tmp_path)

    names = {cs.called_name for cs in call_sites if cs.caller_name == "bar"}
    assert "baz" in names
    assert "qux" in names


def test_build_call_graph_finds_nested_call_in_arguments(tmp_path: Path):
    (tmp_path / "mod.py").write_text("def foo():\n    outer(inner())\n")

    call_sites = build_call_graph(tmp_path)

    names = {cs.called_name for cs in call_sites if cs.caller_name == "foo"}
    assert "outer" in names
    assert "inner" in names


def test_build_call_graph_records_module_level_calls(tmp_path: Path):
    (tmp_path / "mod.py").write_text("setup()\n")

    call_sites = build_call_graph(tmp_path)

    assert any(cs.called_name == "setup" and cs.caller_name == "<module>" for cs in call_sites)


def test_build_call_graph_skips_files_with_syntax_errors(tmp_path: Path):
    (tmp_path / "broken.py").write_text("def foo(:\n    pass")
    (tmp_path / "good.py").write_text("def bar():\n    baz()\n")

    call_sites = build_call_graph(tmp_path)

    assert any(cs.called_name == "baz" for cs in call_sites)
    assert not any(cs.caller_file == "broken.py" for cs in call_sites)


def test_find_callers_filters_by_called_name():
    call_sites = [
        CallSite(caller_name="a", caller_file="x.py", caller_line=1, called_name="target"),
        CallSite(caller_name="b", caller_file="y.py", caller_line=2, called_name="other"),
    ]

    assert find_callers(call_sites, "target") == [call_sites[0]]


def test_find_callees_filters_by_caller_name():
    call_sites = [
        CallSite(caller_name="a", caller_file="x.py", caller_line=1, called_name="target"),
        CallSite(caller_name="b", caller_file="y.py", caller_line=2, called_name="other"),
    ]

    assert find_callees(call_sites, "a") == [call_sites[0]]


def test_save_and_load_call_graph_round_trip(tmp_path: Path):
    call_sites = [CallSite(caller_name="a", caller_file="x.py", caller_line=1, called_name="target")]
    path = tmp_path / "graph.json"

    save_call_graph(call_sites, path)
    loaded = load_call_graph(path)

    assert loaded == call_sites


def test_find_callers_against_real_repo_finds_mount_calls():
    # Not gated behind -m integration: this needs no API key, the whole
    # point of this feature is that it's LLM-free. Only needs target_repo/src
    # to exist locally (same skip pattern test_integration.py uses).
    if not TARGET_REPO_SRC.exists():
        pytest.skip("target_repo/src not present locally")

    call_sites = build_call_graph(TARGET_REPO_SRC)
    results = find_callers(call_sites, "mount")

    callers_in_init = [
        cs for cs in results if cs.caller_name == "__init__" and cs.caller_file == "requests/sessions.py"
    ]
    assert len(callers_in_init) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_callgraph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'codesage.callgraph'`

- [ ] **Step 3: Write `codesage/callgraph.py`**

```python
"""Cross-file call-graph traversal: "who calls X" and "what does X call",
built with zero LLM calls (pure static analysis, same philosophy as
hierarchy.py) and answered with zero LLM calls too (exact filtering, not
reasoning).

Name-based, not import-resolved: a call site records the simple name being
called ("foo" for foo() and for self.foo()/obj.foo()), not a fully
resolved reference to one specific definition. Two unrelated classes that
both define close() will both show up as "callers of close" — a real,
stated tradeoff, not a hidden gap. Import resolution is what tools like
Jedi exist for; it's out of scope here on purpose.
"""

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from codesage.ingest import iter_source_files

CALLGRAPH_FILENAME = ".codesage_callgraph.json"


@dataclass
class CallSite:
    caller_name: str
    caller_file: str
    caller_line: int
    called_name: str


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.call_sites: list[CallSite] = []
        self._stack: list[str] = ["<module>"]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        called_name = None
        if isinstance(node.func, ast.Name):
            called_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr

        if called_name:
            self.call_sites.append(
                CallSite(
                    caller_name=self._stack[-1],
                    caller_file=self.file_path,
                    caller_line=node.lineno,
                    called_name=called_name,
                )
            )
        self.generic_visit(node)


def build_call_graph(repo_path: Path) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for file_path in iter_source_files(repo_path):
        if file_path.suffix != ".py":
            continue
        relative_path = str(file_path.relative_to(repo_path))
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        visitor = _CallVisitor(relative_path)
        visitor.visit(tree)
        call_sites.extend(visitor.call_sites)
    return call_sites


def find_callers(call_sites: list[CallSite], name: str) -> list[CallSite]:
    return [cs for cs in call_sites if cs.called_name == name]


def find_callees(call_sites: list[CallSite], name: str) -> list[CallSite]:
    return [cs for cs in call_sites if cs.caller_name == name]


def save_call_graph(call_sites: list[CallSite], path: Path) -> None:
    path.write_text(json.dumps([asdict(cs) for cs in call_sites]))


def load_call_graph(path: Path) -> list[CallSite]:
    data = json.loads(path.read_text())
    return [CallSite(**d) for d in data]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_callgraph.py -v`
Expected: PASS (8 tests) if `target_repo/src` exists locally, or PASS (7) + SKIPPED (1) if not — either is correct, do not treat the skip as a failure.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions.

- [ ] **Step 6: Append a decisions log entry**

Append to `docs/superpowers/decisions/2026-08-20-call-graph-decisions.md` (create the file if it doesn't exist yet):

```markdown
# Cross-File Call-Graph Traversal — Decisions Log

Local-only running journal, one entry per task, written as I go. Not committed to git.

## Task 1: callgraph.py — stack-based context tracking, no import resolution

Decision: track "which function am I inside" with a plain list used as a
stack (push on FunctionDef/AsyncFunctionDef, pop on the way back out),
rather than resolving what a call actually refers to across imports.

Why: the stack is what makes nested-function attribution correct almost
for free — generic_visit's natural recursion handles the tree walk, the
stack just remembers where we are in it. Import resolution is a
fundamentally different, much harder problem (and not fully solvable in
general for Python) — explicitly out of scope, not a corner cut for time.
```

- [ ] **Step 7: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/callgraph.py tests/test_callgraph.py
git commit -m "feat: add callgraph.py — cross-file call-graph traversal

build_call_graph walks every .py file with an ast.NodeVisitor tracking
enclosing-function context on a stack, zero LLM calls. find_callers/
find_callees are plain filters over the resulting flat list — same data,
read in two directions."
```

---

### Task 2: `find_callers`/`find_callees` tools

**Files:**
- Modify: `codesage/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `CallSite`, `find_callers`, `find_callees` from `codesage.callgraph`.
- Produces: `make_find_callers_tool(call_sites: list[CallSite]) -> Tool`; `make_find_callees_tool(call_sites: list[CallSite]) -> Tool`. Task 3 (CLI) registers both.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tools.py`, add this import (the two tool factories live in `codesage.tools`, `CallSite` lives in `codesage.callgraph` — separate imports, separate modules):

```python
from codesage.callgraph import CallSite
```

And change the existing `codesage.tools` import block from:
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
to:
```python
from codesage.tools import (
    Tool,
    ToolRegistry,
    list_files_handler,
    read_file_handler,
    repo_tree_handler,
    make_hierarchical_search_tool,
    make_find_callers_tool,
    make_find_callees_tool,
)
```

Then add these test functions to the same file:

```python
def test_make_find_callers_tool_lists_call_sites():
    call_sites = [
        CallSite(caller_name="__init__", caller_file="sessions.py", caller_line=10, called_name="mount"),
        CallSite(caller_name="request", caller_file="sessions.py", caller_line=50, called_name="send"),
    ]
    tool = make_find_callers_tool(call_sites)

    result = tool.handler(name="mount")

    assert result == "sessions.py:10 (inside __init__)"


def test_make_find_callers_tool_reports_no_match():
    tool = make_find_callers_tool([])

    result = tool.handler(name="anything")

    assert result == "No callers found for 'anything'."


def test_make_find_callees_tool_lists_calls():
    call_sites = [
        CallSite(caller_name="__init__", caller_file="sessions.py", caller_line=10, called_name="mount"),
        CallSite(caller_name="__init__", caller_file="sessions.py", caller_line=11, called_name="mount"),
    ]
    tool = make_find_callees_tool(call_sites)

    result = tool.handler(name="__init__")

    assert result == "mount (at sessions.py:10)\nmount (at sessions.py:11)"


def test_make_find_callees_tool_reports_no_match():
    tool = make_find_callees_tool([])

    result = tool.handler(name="anything")

    assert result == "No calls found inside 'anything'."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_find_callers_tool'`

- [ ] **Step 3: Add the two tool factories to `codesage/tools.py`**

Add this import near the top of `codesage/tools.py`, alongside the existing `from codesage.hierarchy import HierarchicalIndex`:

```python
from codesage.callgraph import CallSite, find_callees, find_callers
```

Append to the end of `codesage/tools.py`:

```python
def make_find_callers_tool(call_sites: list[CallSite]) -> Tool:
    def handler(name: str) -> str:
        results = find_callers(call_sites, name)
        if not results:
            return f"No callers found for '{name}'."
        return "\n".join(f"{cs.caller_file}:{cs.caller_line} (inside {cs.caller_name})" for cs in results)

    return Tool(
        name="find_callers",
        description=(
            "Find every call site across the repo where a function or method "
            "named X is called. Name-based: matches by simple name, not a "
            "fully resolved reference."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The function or method name to find callers of"}},
            "required": ["name"],
        },
        handler=handler,
    )


def make_find_callees_tool(call_sites: list[CallSite]) -> Tool:
    def handler(name: str) -> str:
        results = find_callees(call_sites, name)
        if not results:
            return f"No calls found inside '{name}'."
        return "\n".join(f"{cs.called_name} (at {cs.caller_file}:{cs.caller_line})" for cs in results)

    return Tool(
        name="find_callees",
        description="Find every function/method call made inside a given function or method, across the repo.",
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The function or method name to find calls made from"}},
            "required": ["name"],
        },
        handler=handler,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: PASS (all tests, including the 4 new ones)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass, no regressions. Also sanity-check there's no import cycle: `tools.py` now also imports from `callgraph.py`, and `callgraph.py` only imports from `ingest.py` — no path back to `tools.py`.

- [ ] **Step 6: Append a decisions log entry**

```markdown
## Task 2: no llm parameter on either tool factory

Decision: `make_find_callers_tool`/`make_find_callees_tool` take only
`call_sites`, unlike `make_search_code_tool`/`make_hierarchical_search_tool`
which both take `llm`.

Why: both retrieval strategies need an LLM call at query time (embed the
query, or reason over a table of contents). find_callers/find_callees are
exact filters over already-built data — there's nothing for a model to do.
Leaving `llm` out of the signature makes that difference visible in the
type signature itself, not just in a comment.
```

- [ ] **Step 7: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/tools.py tests/test_tools.py
git commit -m "feat: add find_callers/find_callees tools

No llm parameter, unlike the retrieval tools — these are exact filters
over already-built data, zero reasoning needed at query time."
```

---

### Task 3: CLI wiring

**Files:**
- Modify: `codesage/cli.py:64-70` (the `ingest-hierarchy` branch)
- Modify: `codesage/cli.py:72-106` (the `ask` branch)

**Interfaces:**
- Consumes: `build_call_graph`, `save_call_graph`, `load_call_graph`, `CALLGRAPH_FILENAME` from `codesage.callgraph`; `make_find_callers_tool`, `make_find_callees_tool` from `codesage.tools`.

- [ ] **Step 1: Extend the `ingest-hierarchy` branch to also build the call graph**

Replace:
```python
    elif args.command == "ingest-hierarchy":
        from codesage.hierarchy import build_hierarchy_chunks

        repo_path = Path(args.repo_path)
        chunks = build_hierarchy_chunks(repo_path)
        save_chunks(chunks, repo_path / HIERARCHY_INDEX_FILENAME)
        print(f"Indexed {len(chunks)} chunks from {repo_path} (hierarchical, no API calls)")
```
with:
```python
    elif args.command == "ingest-hierarchy":
        from codesage.callgraph import CALLGRAPH_FILENAME, build_call_graph, save_call_graph
        from codesage.hierarchy import build_hierarchy_chunks

        repo_path = Path(args.repo_path)
        chunks = build_hierarchy_chunks(repo_path)
        save_chunks(chunks, repo_path / HIERARCHY_INDEX_FILENAME)

        call_sites = build_call_graph(repo_path)
        save_call_graph(call_sites, repo_path / CALLGRAPH_FILENAME)

        print(
            f"Indexed {len(chunks)} chunks and {len(call_sites)} call sites "
            f"from {repo_path} (hierarchical, no API calls)"
        )
```

- [ ] **Step 2: Extend the `ask` branch to register the call-graph tools if available**

Right before the line `agent = Agent(llm, tools=registry)` inside the `elif args.command == "ask":` block, add:

```python
        from codesage.callgraph import CALLGRAPH_FILENAME, load_call_graph
        from codesage.tools import make_find_callees_tool, make_find_callers_tool

        callgraph_path = repo_path / CALLGRAPH_FILENAME
        if callgraph_path.exists():
            call_sites = load_call_graph(callgraph_path)
            registry.register(make_find_callers_tool(call_sites))
            registry.register(make_find_callees_tool(call_sites))
```

So the end of the `ask` branch reads:

```python
        from codesage.callgraph import CALLGRAPH_FILENAME, load_call_graph
        from codesage.tools import make_find_callees_tool, make_find_callers_tool

        callgraph_path = repo_path / CALLGRAPH_FILENAME
        if callgraph_path.exists():
            call_sites = load_call_graph(callgraph_path)
            registry.register(make_find_callers_tool(call_sites))
            registry.register(make_find_callees_tool(call_sites))

        agent = Agent(llm, tools=registry)
        print(agent.ask(args.question))
```

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: all pass — argparse wiring isn't unit tested directly, matching the existing convention.

- [ ] **Step 4: Manual smoke test — build the call graph**

Run: `uv run codesage ingest-hierarchy target_repo/src`
Expected: prints `Indexed N chunks and M call sites from target_repo/src (hierarchical, no API calls)` — completes in well under a second, zero API calls.

- [ ] **Step 5: Manual smoke test — ask a call-graph-shaped question**

Run: `uv run codesage ask "What calls the mount method?" --repo target_repo/src`
Expected: the agent uses `find_callers`, reports back real call sites (e.g. inside `Session.__init__` in `requests/sessions.py`) — read the actual output and confirm it's accurate, not generic.

- [ ] **Step 6: Append a decisions log entry**

```markdown
## Task 3: call-graph tools register independently of --strategy

Decision: find_callers/find_callees get registered based purely on
whether .codesage_callgraph.json exists — not tied to --strategy vector
vs hierarchy at all.

Real result from the smoke test: `codesage ask "What calls the mount
method?" --repo target_repo/src` correctly used find_callers and reported
real call sites — [fill in the actual cited file/line once run].

Why independent of --strategy: the call graph isn't a retrieval strategy,
it's a separate, orthogonal capability — a repo could have a call graph
with no retrieval index built at all, or vice versa. Tying it to
--strategy would imply a relationship between "how do I find relevant
code" and "what calls what" that doesn't actually exist.
```

- [ ] **Step 7: Commit**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add codesage/cli.py
git commit -m "feat: wire call graph into ingest-hierarchy and ask

ingest-hierarchy now builds both the hierarchy index and the call graph
in one AST pass. ask registers find_callers/find_callees whenever a call
graph exists, independent of which retrieval --strategy is chosen."
```

---

### Task 4: README — document the new tools with the real verified example

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a new section documenting the call graph**

Add a new section after `## Two retrieval strategies, compared`:

```markdown
## Call-graph traversal — "who calls X"

Neither retrieval strategy above can answer "find every function that
calls X" — vector similarity finds things that *read* similarly, and
`hierarchy`'s table of contents only sees structure *within* one file.
`codesage/callgraph.py` fills that specific gap: it walks every file with
an `ast.NodeVisitor`, tracking which function it's currently inside on a
stack, and records a `CallSite` for every call encountered — zero LLM
calls to build, zero LLM calls to query (`find_callers`/`find_callees`
are exact filters, not reasoning).

This is name-based, not import-resolved: it matches by simple name, so
two unrelated `close()` methods in different classes both show up as
"callers of close." That's a deliberate v1 tradeoff, not a hidden gap —
real reference resolution is what tools like Jedi exist for.

```bash
uv run codesage ingest-hierarchy target_repo/src   # builds the call graph too, same pass
uv run codesage ask "What calls the mount method?" --repo target_repo/src
```

Real output against `psf/requests`: [PASTE THE ACTUAL ANSWER TEXT FROM TASK 3'S SMOKE TEST HERE, VERBATIM]
```

- [ ] **Step 2: Update the architecture table**

Add a row to the existing `## Architecture` table:

```markdown
| `callgraph.py` | Cross-file call-graph traversal — `build_call_graph` (zero LLM calls) + `find_callers`/`find_callees` (zero LLM calls at query time too — exact filters, not reasoning). |
```

- [ ] **Step 3: Update the Usage section**

In the existing `## Usage` code block, note the call graph is built alongside the hierarchy index:

```bash
uv run codesage ingest-hierarchy <path-to-a-repo>     # zero-cost structural index + call graph
```

(This replaces the existing line for `ingest-hierarchy` rather than adding a new one — same command, now documented as doing both things.)

- [ ] **Step 4: Run the full suite one more time**

Run: `uv run pytest -v`
Expected: all pass — this task is documentation-only.

- [ ] **Step 5: Append the final decisions log entry**

```markdown
## Task 4: README documents the real answer, not a paraphrase

Decision: the README embeds the actual answer text from the Task 3 smoke
test verbatim, same convention as the retrieval-strategy comparison
numbers.

Why: consistent with everything else in this project — claims here are
things a reader can go verify themselves by running the two commands
shown, not descriptions to take on faith.
```

- [ ] **Step 6: Commit and push**

```bash
git status  # confirm docs/superpowers/decisions/ is NOT listed
git add README.md
git commit -m "docs: document call-graph traversal with real verified example"
git push
```

---

## After all 4 tasks

- `codesage ingest-hierarchy` builds both the hierarchy index and the call graph in one pass, verified against `target_repo/src`.
- `find_callers("mount")` returns real, correct results — verified by a real (non-mocked) unit test and by a live `codesage ask` smoke test.
- Full suite green (existing + new tests), CI green on every commit.
- Decisions log has one entry per task, exists only locally.
- README documents the feature with a real, reproducible example.
