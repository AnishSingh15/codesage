# Multi-Agent Onboarding Doc Generator — Design Spec

Date: 2026-08-20
Status: Approved

## What this is

A new feature on top of the existing CodeSage agent: `codesage onboard --repo <path>`
runs a small pipeline of three agents — a structure mapper, a code explorer,
and a writer — that together produce `ONBOARDING.md`, a doc combining
"what is this repo and how is it organized" with "how do I contribute."

This is CodeSage's first genuinely multi-agent feature (previously: one
agent, one loop). It follows the **sequential supervisor pattern**, which
current practice recommends as the starting point for multi-agent systems:
one orchestrating function runs sub-agents in order, each handing its
output to the next, no concurrency. Each sub-agent is literally another
instance of the existing `Agent` class — multi-agent here means
*composing* a primitive that already exists and is already tested, not
building a new one.

## Why sequential, not parallel fan-out

The code-exploration step genuinely benefits from reading the structure
summary first (it tells the explorer where to look), so the steps aren't
independent — parallelizing them would mean the explorer either runs
blind or we restructure to merge-after-parallel anyway. Sequential also
avoids introducing `asyncio`/threading, which would be a large jump in
complexity for no real benefit here. Production multi-agent systems mix
patterns; this one just doesn't need to yet.

## Why sub-agents don't get a new "role" mechanism

`Agent`/`LLMClient` are untouched by this feature. A sub-agent's "role"
(structure mapper vs. explorer vs. writer) is expressed entirely through
which tools it's given and the task text passed to `.ask()` — not through
a new system-instruction parameter. This keeps the change purely additive:
zero risk to the already-tested agent loop, and one less concept to teach
before this feature makes sense to someone who just learned `Agent`.

## Architecture

```
generate_onboarding_doc(repo_path, llm) -> str

  1. structure_agent = Agent(llm, tools=[repo_tree, list_files, read_file], max_steps=6)
     structure_summary = structure_agent.ask(STRUCTURE_PROMPT)

  2. explorer_agent = Agent(llm, tools=[list_files, read_file, search_code?], max_steps=10)
     code_summary = explorer_agent.ask(EXPLORER_PROMPT.format(structure_summary))

  3. writer_agent = Agent(llm, tools=None, max_steps=3)
     doc = writer_agent.ask(WRITER_PROMPT.format(structure_summary, code_summary))

  return doc
```

`search_code` is only added to the explorer's tools if a persisted index
exists for `repo_path` (same conditional-registration pattern the CLI's
`ask` command already uses) — this feature must work on a repo that's
never been `codesage ingest`-ed, just with less semantic search available.

### Components

**1. `repo_tree` tool** (new, added to `tools.py`)
- `list_files` only lists one directory at a time — tedious for a
  structure-mapping agent that needs the whole layout up front.
  `repo_tree` does a recursive walk (same `SKIP_DIRS` as ingestion) and
  returns an indented tree view, like the `tree` CLI command.
- Bounded output — caps at 200 entries, then stops and appends
  `"... (truncated)"` — the same "don't let unbounded growth blow up the
  context window" concern that motivated `Memory`'s sliding window
  applies here too: a huge repo could otherwise produce a tree listing
  too large to reason about.
- Same `Tool` dataclass, same registration pattern as every other tool —
  no new abstraction.

**2. `codesage/supervisor.py`** (new module)
- `generate_onboarding_doc(repo_path: Path, llm) -> str` — the pipeline
  above. Builds each sub-agent's `ToolRegistry` internally, using the
  existing tool constructors (`list_files_handler`, `read_file_handler`,
  `repo_tree_handler`, `make_search_code_tool`).
- Prompt templates for each stage live in this module as module-level
  constants (`STRUCTURE_PROMPT`, `EXPLORER_PROMPT`, `WRITER_PROMPT`) —
  plain strings, not a templating system; YAGNI.

**3. CLI: `codesage onboard --repo <path>`**
- Runs the pipeline, writes the result to `<repo_path>/ONBOARDING.md`,
  prints a confirmation line with the output path.

## Data flow

1. User runs `codesage onboard --repo target_repo/src`
2. Structure agent explores the repo via `repo_tree`/`list_files`/`read_file`
   and produces a plain-text summary of layout and entry points
3. Explorer agent receives that summary as context, explores further via
   `read_file` and (if available) `search_code`, and produces a summary
   of what the key modules/classes actually do
4. Writer agent receives both summaries (no tools — pure synthesis) and
   produces the final markdown doc: what the repo is, how it's organized,
   where to start reading, how to set up/test/contribute
5. CLI writes that doc to `ONBOARDING.md` in the target repo

## Error handling

- Same philosophy as the rest of the project: no crashes, graceful
  degradation. If a sub-agent hits its `max_steps` and returns the
  existing "I couldn't finish..." message, that text is still passed
  forward to the next stage — the writer produces a weaker doc rather
  than the pipeline failing outright. Not special-cased; this is just
  `Agent.ask()`'s existing contract working as designed.
- If no persisted index exists, the explorer silently proceeds without
  `search_code` rather than erroring (matches `cli.py`'s `ask` behavior).

## Testing

- Unit tests use the existing `FakeLLM` pattern (a list of canned
  responses popped in order) to verify: the three stages run in the
  right order, the explorer's prompt actually contains the structure
  summary, the writer's prompt contains both summaries, and the
  function's return value is the writer's output text.
- `repo_tree_handler` gets its own unit tests: correct tree shape on a
  small fixture directory, `SKIP_DIRS` respected, output bounded on a
  directory with more entries than the cap.
- One `@pytest.mark.integration` test runs the real pipeline against
  `target_repo/src` and asserts the output contains expected keywords
  (e.g. "requests", "Session") — same pattern as the existing
  integration test, real API, skipped without a key.

## Decisions log

A running journal at `docs/superpowers/decisions/2026-08-20-multi-agent-onboarding-decisions.md`,
updated with an entry at each real judgment call made during
implementation — separate from this spec (which is the plan/intent) and
from the README's "Lessons" section (which is the post-hoc summary).
**This file is gitignored — local only, never pushed to GitHub.**

## Out of scope

- Vectorless retrieval — separate spec, later, as already agreed.
- Parallel/concurrent sub-agent execution.
- Any new "role" or system-instruction mechanism on `Agent`/`LLMClient`.
- A supervisor that dynamically decides which sub-agents to run — the
  three-stage pipeline is fixed for this version.

## Success criteria

- `codesage onboard --repo <path>` produces a real `ONBOARDING.md` that
  correctly describes a repo it wasn't specifically tuned for, verified
  against `target_repo/src`.
- All new code has unit test coverage using the existing fake-LLM
  testing pattern; full suite (existing + new) stays green.
- README updated: documents the new `onboard` command, and gets a
  visual refresh — an architecture diagram (Mermaid, since GitHub
  renders it natively) and badges — not just more prose.
- Decisions log exists locally, is never committed (verified via
  `.gitignore` and a `git status` check before any commit in this work).
