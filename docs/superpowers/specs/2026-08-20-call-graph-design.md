# Cross-File Call-Graph Traversal — Design Spec

Date: 2026-08-20
Status: Approved

## What this is

Two new agent tools — `find_callers(name)` and `find_callees(name)` —
that answer "who calls X" and "what does X call" across the *entire*
repo, not just one file. This is a genuine capability gap neither
existing retrieval strategy fills: vector similarity finds things that
*read* similarly, and the current vectorless strategy only sees
structure *within* one file. Neither can answer "find every function
that calls X" — that requires actually knowing the call graph.

This came directly from public feedback on the project (a LinkedIn
comment from an AI Architect, unprompted, naming this exact gap).

## Why name-based matching, not import-resolved

Fully resolving `from foo.bar import baz; baz()` to the exact function
object it refers to — handling aliases, relative imports, re-exports,
dynamic dispatch — is what real static-analysis tools (Jedi, rope,
language servers) exist to do, and it's not even fully solvable in
principle for Python specifically (duck typing and runtime dispatch
mean the "true" call target can depend on values only known at
runtime). That's not shippable this week and isn't what this project
needs to prove the point.

Instead: for every call site (`foo()` or `obj.method()`), record the
*simple name* being called. "Callers of X" = every call site anywhere
in the repo where something named `X` is invoked. This is a real,
honest simplification — if two unrelated classes both define
`close()`, asking for callers of `close` returns both. That tradeoff is
stated plainly here and in the README, the same way brute-force
retrieval and top-level-only AST extraction were — not hidden.

## Architecture

New module, `codesage/callgraph.py`:

```
@dataclass
class CallSite:
    caller_name: str   # enclosing function/method name, or "<module>" for top-level calls
    caller_file: str    # path relative to repo root
    caller_line: int    # line of the call site itself
    called_name: str    # simple name being called — "foo" for foo(), "foo" for self.foo()/obj.foo()

build_call_graph(repo_path: Path) -> list[CallSite]
    - walks .py files via ingest.py's iter_source_files (same SKIP_DIRS)
    - ast.parse each file; SyntaxError -> skip that file, continue (same
      pattern as hierarchy.py)
    - a NodeVisitor walks each file's tree tracking "which function am I
      currently inside" as a stack, and records a CallSite every time it
      encounters an ast.Call node
    - zero LLM calls — pure static analysis, same as hierarchy.py

find_callers(call_sites: list[CallSite], name: str) -> list[CallSite]
    - call_sites filtered where called_name == name

find_callees(call_sites: list[CallSite], name: str) -> list[CallSite]
    - call_sites filtered where caller_name == name
    - same flat list, read in the other direction — this is why scoping
      both directions cost almost nothing extra once the graph exists

save_call_graph / load_call_graph
    - JSON persistence, same shape as index.py's save_chunks/load_chunks
      but for CallSite instead of Chunk
```

### Why a new module, not an extension of hierarchy.py

`hierarchy.py` extracts and navigates *structure within a file*
(functions, classes, docstrings) into `Chunk`s meant to be retrieved.
A call graph is a different shape of data (edges between named things,
not retrievable text chunks) answering a different kind of question
(exact structural lookup, not "what's relevant to this query"). Keeping
them separate keeps each module's job legible — `hierarchy.py` answers
"what's here," `callgraph.py` answers "what calls what."

### Why these tools cost zero LLM calls, unlike both retrieval strategies

`find_callers`/`find_callees` are exact filters over a list — no
reasoning, no embedding, no `generate()` call at query time. This is
worth being explicit about: they're not a third "retrieval strategy"
competing with vector/vectorless, they're a categorically different,
free capability the agent can reach for when a question is
call-graph-shaped rather than "find relevant code" shaped.

## Tools

`codesage/tools.py` gains two factories, both taking a pre-loaded
`list[CallSite]` (no `llm` parameter needed, for the reason above):

- `make_find_callers_tool(call_sites: list[CallSite]) -> Tool` — tool
  name `find_callers`, one string param `name`. Output: one line per
  call site, `{caller_file}:{caller_line} (inside {caller_name})`.
- `make_find_callees_tool(call_sites: list[CallSite]) -> Tool` — tool
  name `find_callees`, one string param `name`. Output: one line per
  call site, `{called_name} (at {caller_file}:{caller_line})`.

Both registered independently of retrieval strategy — they're
available whenever a call graph has been built, regardless of whether
`ask --strategy` is `vector` or `hierarchy`. A repo can have a call
graph with no retrieval index, or vice versa; they're separate,
composable capabilities.

## CLI

**Extends the existing `codesage ingest-hierarchy` command** rather
than adding a new one: it already walks every `.py` file with
`ast.parse` at zero LLM cost — building the call graph in the same
pass reuses that walk instead of parsing every file twice. The command
now builds and persists both `.codesage_hierarchy_index.json` and
`.codesage_callgraph.json`, and prints counts for both.

`codesage ask` already conditionally registers `search_code` based on
whether an index file exists; it now *also* independently checks for
`.codesage_callgraph.json` and registers `find_callers`/`find_callees`
if present — same graceful-degradation pattern already used for the
retrieval tools (missing artifact means fewer tools, not an error).

New filename constant `CALLGRAPH_FILENAME = ".codesage_callgraph.json"`
in `codesage/callgraph.py` (not `index.py` — it's not a retrieval
index).

## Error handling

- File fails to parse (`SyntaxError`): skip that file, continue with
  the rest — matches `hierarchy.py`'s existing pattern exactly.
- No call graph built yet: `ask` prints a message pointing at
  `codesage ingest-hierarchy` and continues without the two tools,
  same as the existing missing-index messages.
- `find_callers`/`find_callees` with no matches: tool returns a plain
  "No callers/callees found for 'X'" string, not an error — matches
  `search_code`'s existing "No matching code found" pattern.
- Calls to names not defined anywhere in the repo (built-ins, stdlib,
  third-party functions) are still recorded and returned normally —
  `find_callees` answering "what does X call" should include calls to
  `len()` or `json.dumps()` just as much as calls to your own code.
  Filtering those out would require cross-referencing against the
  definitions list for no real benefit.

## Testing

- `build_call_graph`: unit tests on small fixtures — direct calls
  (`foo()`), method calls (`self.foo()` / `obj.foo()`), a call nested
  inside another call's arguments, a call at module level
  (`caller_name == "<module>"`), a file with a `SyntaxError` skipped
  without crashing the batch.
- `find_callers` / `find_callees`: unit tests on a small hand-built
  `list[CallSite]`, verifying correct filtering in both directions.
- `make_find_callers_tool` / `make_find_callees_tool`: unit tests
  verifying output formatting and the "no matches" message, using the
  existing `Tool`/`ToolRegistry` test conventions.
- One `@pytest.mark.integration`-adjacent *unit* test (no LLM call
  needed — this whole feature is LLM-free) against `target_repo/src`
  itself, confirming `find_callers("mount")` returns at least the two
  known real call sites inside `Session.__init__` (`self.mount("https://", ...)`
  and `self.mount("http://", ...)`) — a real, verifiable fact about the
  actual `requests` source, not a synthetic fixture. Since this needs
  no API key, it runs in the default suite, not gated behind
  `-m integration`.

## Decisions log

Same convention as every other feature this session: running journal
at `docs/superpowers/decisions/2026-08-20-call-graph-decisions.md`,
one entry per task, gitignored, never committed.

## Out of scope for v1

- Import resolution / precise reference resolution (see "why
  name-based" above) — the core scoping decision, not a deferred nice-to-have.
- Class-qualified names (`ClassName.method_name`) to reduce name
  collisions — a real, easy future improvement, but bare names are
  the v1 simplification, consistent with `hierarchy.py`'s
  top-level-only extraction.
- Nested function definitions (closures) as *callers* — a call inside
  a nested/inner function is currently attributed to the nearest
  enclosing named function, not tracked as its own distinct caller
  context. Matches `hierarchy.py`'s existing top-level-only stance.
- A CLI command or tool that visualizes the graph — `find_callers`/
  `find_callees` are point queries, not a graph browser.

## Success criteria

- `codesage ingest-hierarchy <repo>` builds and persists both the
  hierarchy index and the call graph in one pass, zero LLM calls,
  verified against `target_repo/src`.
- `find_callers("mount")` and `find_callees(...)` return real, correct
  results against `target_repo/src`, verified by the test described
  above and by a live `codesage ask` smoke test.
- All new code unit-tested; full suite (existing + new) stays green.
- README documents the new tools and the real verified example.
