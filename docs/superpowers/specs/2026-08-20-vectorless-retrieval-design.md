# Vectorless Retrieval — Design Spec

Date: 2026-08-20
Status: Approved

## What this is

A second retrieval strategy for CodeSage, coexisting with the original
brute-force cosine-similarity `RetrievalIndex`: a **hierarchical,
AST-based index** that a query navigates via LLM reasoning instead of
vector math — "vectorless RAG" in the PageIndex sense (structure-first
navigation instead of nearest-neighbor search over embeddings).

This fills the seam `eval.py` deliberately left open in the original
build ("a second retrieval strategy could be swapped in — Strategy
pattern") and was explicitly deferred as future work in the multi-agent
onboarding spec.

Both strategies stay in the codebase and are directly comparable via a
new `codesage eval --compare` mode — a real, demoable, numeric result
(retrieval hit rate + answer score for each strategy against the same
questions), not just a claim that vectorless "works."

## Why AST-based, not LLM-summarized

The literal PageIndex approach calls an LLM to build a document's
hierarchical map — necessary for arbitrary documents with no inherent
structure. Code isn't an arbitrary document: Python's own `ast` module
can extract exact structure (module docstring, every top-level
function/class with its docstring and precise line span) for free, with
zero LLM calls and zero risk of the model misreading the file layout.
This also produces sharper chunks than the existing fixed 40-line
windows — a semantic unit (one function, one class) instead of an
arbitrary slice that might cut a function in half.

The LLM's job in this strategy is reasoning over that structure to
answer "which of these functions/classes is relevant to this question"
— one `generate()` call per query, the same cost shape as the existing
strategy's one `embed()` call per query. This keeps a head-to-head
comparison fair: both strategies spend one LLM call per query, just a
different kind of call.

## Architecture

```
codesage/hierarchy.py

build_hierarchy_chunks(repo_path: Path) -> list[Chunk]
    - walks repo_path (reuses ingest.py's iter_source_files)
    - for each .py file: ast.parse(), extract module docstring +
      each top-level FunctionDef/ClassDef (name, docstring, line span)
      -> one Chunk per definition, vector=None
    - for each non-.py file: falls back to the existing chunk_file()
      line-window chunking (unchanged behavior for docs/config files)
    - zero LLM calls

class HierarchicalIndex:
    def __init__(self, chunks: list[Chunk])
    def search(self, query: str, llm, k: int = 5) -> list[Chunk]
        - builds a compact table-of-contents string from self._chunks,
          one entry per line: "{file_path}:{name} — {docstring_first_line}"
        - each chunk's identity for lookup purposes is its
          "{file_path}:{name}" key (name = the function/class name, or
          the literal string "module" for module-level docstring chunks,
          or the file_path itself again for non-.py line-window chunks
          that have no name)
        - one llm.generate() call: the prompt explicitly asks for the
          answer as one "{file_path}:{name}" key per line, nothing else
        - parses the response by splitting on newlines and matching each
          line against the known keys; unmatched lines are ignored
        - if zero lines matched a known key, falls back to returning the
          first k chunks rather than crashing — degraded, not broken
```

### Components

**1. `codesage/hierarchy.py`** (new module)
- `build_hierarchy_chunks(repo_path: Path) -> list[Chunk]`
- `HierarchicalIndex` class with `.search(query, llm, k) -> list[Chunk]`

**2. Persistence** — reuses `index.py`'s existing `save_chunks`/`load_chunks`
unchanged (they already handle `vector=None`). New filename constant
`HIERARCHY_INDEX_FILENAME = ".codesage_hierarchy_index.json"` in
`index.py`, alongside the existing `INDEX_FILENAME`.

**3. Tool wiring** — `codesage/tools.py` gets
`make_hierarchical_search_tool(index: HierarchicalIndex, llm) -> Tool`,
named `"search_code"` (same tool name as the vector version — a given
agent run uses exactly one strategy, chosen at `ask`/`eval` time, so
there's never a name collision within one registry).

**4. CLI**
- `codesage ingest-hierarchy --repo <path>` — builds and persists the
  hierarchy index. No `--delay` flag (no API calls to pace).
- `codesage ask --strategy vector|hierarchy` (default `vector`, so
  existing behavior/tests/docs are unaffected) — picks which index
  file to load and which tool variant to register.
- `codesage eval --compare` — loads both persisted indexes (errors
  clearly if either is missing — must run both `ingest` and
  `ingest-hierarchy` first), runs `run_eval` against each, prints a
  side-by-side table.

**5. `eval.py`** — `run_eval`'s signature is unchanged (`agent_factory,
index, llm, cases`); `--compare` in the CLI just calls it twice with
different `(agent_factory, index)` pairs and formats both results
together. No change to `eval.py` itself required — this is exactly the
payoff of the Strategy-shaped seam already being there.

## Data flow (a single hierarchical query)

1. `search_code` tool is called with a question string
2. `HierarchicalIndex.search` builds its table-of-contents text from
   the chunks it holds in memory
3. One `llm.generate()` call: system framing + table of contents +
   question, asking for the top-k relevant entry names
4. Response is parsed for entry names; matching `Chunk`s are returned
5. Tool formats them the same way `make_search_code_tool` already does
   (`file_path:line_start-line_end` + text) — output shape is
   identical between strategies, so the agent loop doesn't need to
   know which one is active

## Error handling

- Non-`.py` files: no AST to parse, silently fall back to line-window
  chunking (existing, tested code path) — not an error.
- A `.py` file that fails to parse (`SyntaxError` from `ast.parse`,
  e.g. a file with a syntax error or an unusual encoding): skip that
  file, don't crash the whole ingestion — matches the project's
  existing "one bad input doesn't take down the batch" pattern from
  `chunk_file`'s blank-window skip.
- LLM response that doesn't cleanly name valid entries: fall back to
  the first `k` chunks rather than raising — matches the project's
  general philosophy of graceful degradation over crashes.
- `eval --compare` with a missing index (either one): clear error
  message naming which `ingest`/`ingest-hierarchy` command to run,
  `sys.exit(1)` — matches the existing `eval` command's pattern for a
  missing vector index.

## Testing

- `build_hierarchy_chunks`: unit tests on a small fixture — correct
  chunks for functions/classes with docstrings, module without a
  docstring handled, a file with a syntax error skipped without
  crashing the batch, a non-`.py` file falls back to line-window
  chunking.
- `HierarchicalIndex.search`: unit tests using the existing `FakeLLM`
  pattern — correct table-of-contents text sent to the model, correct
  chunks returned when the model names valid entries, fallback to
  first-k when the model's response doesn't parse.
- One `@pytest.mark.integration` test: real ingestion + real query
  against `target_repo/src`, asserting the returned chunks are
  plausible (e.g. querying about session handling returns a chunk from
  `sessions.py`).
- Manual smoke test: `codesage eval --compare` against
  `target_repo/src` with the real `eval_cases.json`, real numbers
  recorded in the decisions log and used in the README update.

## Decisions log

Same convention as the multi-agent work: running journal at
`docs/superpowers/decisions/2026-08-20-vectorless-retrieval-decisions.md`,
one entry per task, gitignored, never committed.

## Out of scope

- Multi-provider LLM support (OpenAI/Groq/Claude) — separate spec,
  raised by the user but explicitly deferred, unrelated architecture
  change.
- Extending AST-based extraction to nested (non-top-level) functions/
  classes — top-level only for v1; nested definitions stay embedded in
  their parent's chunk text.
- A third retrieval strategy, or making the strategy choice automatic/
  adaptive — `--strategy` is an explicit user choice, not inferred.

## Success criteria

- `codesage ingest-hierarchy` builds a real hierarchy index against
  `target_repo/src` with zero LLM calls (verified: no API errors
  possible from a code path that never calls `generate`/`embed`).
- `codesage ask --strategy hierarchy` answers a real question about
  `target_repo/src` correctly, verified live.
- `codesage eval --compare` produces real, recorded numbers for both
  strategies against the same `eval_cases.json`.
- All new code unit-tested with the existing `FakeLLM`/fixture
  conventions; full suite stays green.
- README documents both strategies, includes the real `--compare`
  numbers, and updates the architecture diagram/table.
