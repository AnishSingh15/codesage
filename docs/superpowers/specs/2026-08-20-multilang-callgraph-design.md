# Multi-Language Call-Graph Traversal — Design Spec

## Problem

`codesage/callgraph.py` builds cross-file call-graph traversal
(`find_callers`/`find_callees`) using Python's `ast` module — it only
ever sees `.py` files. Verified directly by testing CodeSage against a
real non-Python project (a TypeScript backend, 83 `.js`/`.ts` files, 0
`.py` files): `build_call_graph` correctly found 0 call sites, and a
follow-up fix (shipped separately) made that scope limitation visible
rather than misleading — `ask`/`ingest-hierarchy` now print a clear
"Python (.py) files only" message and skip registering the tools
entirely when the graph is empty, instead of letting the agent discover
uselessness by trial and error.

Retrieval (`vector`/`hierarchy`) was separately verified to already work
fine on non-Python code — both strategies chunk and embed plain text,
and `hierarchy.py` already has a line-window fallback for non-Python
files. So this spec is scoped narrowly: extend call-graph traversal
specifically, not "make the whole tool support any language."

## Scope

v1 supports three languages beyond Python: **JavaScript, TypeScript,
Go** — `.js`, `.ts`, `.go` files only (`.jsx`/`.tsx` explicitly
deferred, not silently half-supported). JS/TS was chosen because it
matches a real project the user owns (immediately testable against real
code, not a toy example); Go was chosen as the third language for its
simpler grammar (no generics/overloading complexity to fight) and
because there are real, well-known public Go repos to verify against —
same testing pattern already used for Python (`psf/requests`, Django).

Mixed-language repos are a first-class case, not an edge case: a repo
with Python, Go, and TS files all get call-graph coverage in one pass,
merged into a single queryable graph. A Python function and a Go method
that happen to share a name will still both show up as "callers of
X" — the same name-based-not-import-resolved tradeoff `callgraph.py`
already states for single-language repos, now naturally extended across
languages rather than newly introduced.

## Architecture

`codesage/callgraph.py` (Python `ast`-based) is not modified — it's
shipped, tested, and correct for what it does; there's no reason to put
it at risk for this extension. A new sibling module,
`codesage/treesitter_callgraph.py`, adds JS/TS/Go support and reuses the
same `CallSite` dataclass from `callgraph.py`, so `find_callers`/
`find_callees` and the two agent tools don't need to know or care which
backend produced a given call site.

Inside the new module:

- **Per-language config** (one per language: JS, TS, Go) — the only
  place language differences live. Each config declares: which node
  types count as "function-like" and which field holds the name (e.g.
  JS: `function_declaration`/`method_definition` → field `name`); the
  call node type and which field holds the callee (`call_expression` →
  field `function`, verified identical across all three languages); and
  which node type represents a method-style call and which field holds
  the property/selector name (JS: `member_expression` → field
  `property`; Go: `selector_expression` → field `field`), so
  `obj.foo()` / `s.Mount()` resolve to a simple called name the same
  way Python's `self.foo()` does today.
- **One generic recursive tree-walker**, written once, shared across
  all three language configs. Same shape as `callgraph.py`'s existing
  `_CallVisitor`: push the current function name onto a stack when
  entering a function-like node, record a `CallSite` on a call node,
  pop on the way back out. The stack-tracking logic is identical in
  spirit to the Python version — only the node-type names differ, and
  those come from config, not code.
- **`build_treesitter_call_graph(repo_path: Path) -> list[CallSite]`**
  — walks the repo via the existing `iter_source_files` (from
  `ingest.py`, already language-agnostic), dispatches each file to its
  language config by extension, runs the walker, aggregates results.

`codesage/cli.py`'s `ingest-hierarchy` command is the merge point — it
already assembles the hierarchy index and the Python call graph in one
pass, so it grows to also call `build_treesitter_call_graph` and
concatenate its results with the Python call graph's, saving the
combined `list[CallSite]` to the same `CALLGRAPH_FILENAME`
(`.codesage_callgraph.json`) `callgraph.py` already defines — one file,
one merged graph, regardless of how many languages contributed to it.
No changes needed to `find_callers`/`find_callees`, the agent tools, or
`ask`'s wiring — they already operate on `list[CallSite]` regardless of
source.

Two configs exist for JS and TS, not one shared config, even though
their node-type names are largely identical (TypeScript's grammar is a
structural superset of JavaScript's) — because `tree-sitter-typescript`
exports its own distinct `Language` object from `tree-sitter-javascript`,
and `.ts` files must be parsed with it. Specifically:
`tree_sitter_typescript.language_typescript()` — not `.language_tsx()`,
since `.tsx` is out of scope for v1. The two configs' node-type/field
values will look near-duplicate; that's expected, not a mistake to
"clean up" during implementation.

## Algorithm detail: named vs. anonymous functions

Only *named* function-like constructs get their own stack entry (JS
`function_declaration`/`method_definition`, Go
`function_declaration`/`method_declaration`). JS arrow functions
assigned to a variable (`const foo = () => {...}`) aren't attributed
their own name in the syntax tree the same explicit way a declared
function is — their calls get attributed to whichever named scope
encloses them, or `<module>` at the top level. This is a stated v1
tradeoff, not a hidden gap, consistent with the project's existing
practice of naming tradeoffs explicitly rather than papering over them.

## Error handling

Unlike Python's `ast.parse` (raises `SyntaxError` on malformed source,
caught and the file skipped), tree-sitter never raises on malformed
source — it performs error recovery and returns a best-effort parse
tree. So there is no parse-failure branch to write for the tree-sitter
path; only file-read errors (encoding, permissions) need the same
`errors="ignore"` handling already used elsewhere in `ingest.py`.
Unrecognized extensions are skipped, the same way `build_call_graph`
already skips non-`.py` files.

## Testing

Unit tests mirror `tests/test_callgraph.py`'s existing structure, one
set per language, on small `tmp_path` fixtures: a direct call, a
method/selector call, a nested call inside call arguments, a
module-level call, and a malformed file that doesn't crash the batch.

No new vendored `target_repo`-style fixture gets added to the repo for
CI — real-world verification happens manually during implementation
(same pattern already used for the ad hoc Django test this session) and
gets written into the decisions log and README as a real, reproducible
example, rather than vendoring three more third-party repos into git
history across languages.

## Dependencies

New base dependencies in `pyproject.toml` (not an optional extras
group — call-graph traversal is a core, documented feature, same
tier as `numpy`): `tree-sitter`, `tree-sitter-javascript`,
`tree-sitter-typescript`, `tree-sitter-go`. All three verified as real,
installable PyPI packages before writing this spec (`tree-sitter==0.26.0`,
`tree-sitter-javascript==0.25.0`, `tree-sitter-typescript==0.23.2`,
`tree-sitter-go==0.25.0`), and the query/parsing API was verified
hands-on against real JS and Go source (not assumed) before this design
was written.
