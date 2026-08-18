# CodeSage — Design Spec

Date: 2026-08-18
Status: Approved

## What this is

CodeSage is an AI agent, built from scratch in plain Python (no LangGraph/CrewAI),
that answers questions about a codebase by reading its files, retrieving relevant
chunks, reasoning over them with tools, and citing exact files/lines in its answers.

Default target corpus: a real public repo (e.g. `psf/requests` or FastAPI docs) —
free, public data, no user secrets needed to demo.

Two goals drive every decision below, in order:
1. **Teach agentic AI + DSA/LLD from zero**, concept-by-concept, tied to real
   lines of code — not a lecture series, not a framework tutorial.
2. **Ship a CV-credible GitHub repo**: working code, clean history, tests, CI,
   a README that explains the architecture.

The user is a true beginner in both Python and DSA. Pace is "push through fast" —
working code first, explanations kept tight and inline, depth added in the
polish phase rather than up front.

## Why build from scratch instead of using a framework

LangGraph/CrewAI would get a demo working faster, but they hide the exact
mechanics (state machines, memory, tool routing) behind framework classes.
For a true beginner, that's *more* to learn, not less — framework abstractions
on top of concepts not yet understood. Building from scratch means every DSA/LLD
concept the user learns is a piece of code they wrote and can explain in an
interview.

Exception: we don't reinvent floating-point math or an LLM SDK. We use `numpy`
for vector math and Google's official Gemini SDK for model calls. Everything
that constitutes "the agent" — the loop, memory, tool dispatch, retrieval index,
multi-step planning — is hand-written.

## Tech stack

- **Language:** Python 3.13 (via `uv` for env/dependency management — matches
  the tool the user is already using in their `lang` learning sandbox)
- **LLM:** Google Gemini (free tier) via `google-genai` SDK
- **Vector math:** `numpy` only (no vector DB, no FAISS) — retrieval is
  brute-force cosine similarity over an in-memory index. This is a deliberate
  teaching choice (see Phase 4) and is fine at the scale of one repo's worth
  of chunks (hundreds to low thousands).
- **Testing:** `pytest`
- **CI:** GitHub Actions (lint + test on push)
- **No web framework, no database.** CodeSage is a CLI tool for v1. A UI is
  explicitly out of scope (see below).

## Architecture

```
                    ┌─────────────────┐
   user question ─▶ │   Agent Loop     │◀─── conversation Memory (bounded queue)
                    │ (observe→think→  │
                    │      act)        │
                    └───┬─────────┬────┘
                        │         │
                 ┌──────▼───┐ ┌───▼────────┐
                 │   Tool   │ │  Retrieval  │
                 │ Registry │ │   Index     │
                 │ (dict)   │ │ (chunks +   │
                 └──────────┘ │  vectors)   │
                               └─────────────┘
                                     ▲
                                     │ ingest (one-time)
                              ┌──────┴───────┐
                              │  Ingestion    │
                              │ (walk repo →  │
                              │  chunk → embed)│
                              └───────────────┘
```

### Components

**1. Ingestion** (`codesage/ingest.py`)
- Walks a target repo directory, reads text/code files (skips binaries,
  `.git`, `node_modules`, etc.)
- Splits files into chunks (by function/class where possible for code, by
  paragraph for docs; simple line-window chunking as the baseline)
- Embeds each chunk via the Gemini embeddings endpoint
- Produces a list of `Chunk(text, file_path, line_start, line_end, vector)`

**2. Retrieval Index** (`codesage/index.py`)
- Holds the list of chunks + their vectors (in-memory, no persistence in v1)
- `search(query_vector, k) -> list[Chunk]`: brute-force cosine similarity,
  returns top-k. This is the DSA teaching moment for Phase 4 (vectors,
  dot product, why brute-force is O(n) and when you'd need an ANN index
  like HNSW instead — discussed, not built).

**3. Tool Registry** (`codesage/tools.py`)
- A `dict[str, Tool]` mapping tool name → callable + schema (the "hashmap as
  registry" LLD lesson from Phase 2)
- v1 tools: `search_code` (queries the retrieval index), `read_file` (exact
  file/line range), `list_files` (directory listing)
- Adding a tool = registering a function with a schema; the agent loop never
  hardcodes tool logic (Open/Closed Principle, taught inline)

**4. Memory** (`codesage/memory.py`)
- Bounded conversation history — a deque/sliding window over past turns
  (Phase 3's DSA lesson: queue/stack, why unbounded history breaks context
  windows)

**5. Agent Loop** (`codesage/agent.py`)
- The core observe→think→act loop (ReAct-style): given a question, decide
  whether to call a tool or answer, execute, feed results back, repeat until
  done or a max-steps guard trips
- Phase 5 turns this into an explicit state machine (states: `THINKING`,
  `ACTING`, `DONE`, `ERROR`) — the graph/state-machine DSA/LLD lesson, and a
  visible defense against infinite loops

**6. Evaluation** (`codesage/eval.py`, Phase 6)
- A small fixed set of question/expected-answer pairs about the target repo
- Scores retrieval hit-rate (did we retrieve the right chunk?) and answer
  quality (simple heuristic or LLM-judge)
- Retrieval strategy is made swappable (Strategy pattern) so the eval harness
  can compare brute-force vs. a second strategy later

**7. CLI entrypoint** (`codesage/cli.py`)
- `codesage ingest <repo_path>` — builds the index
- `codesage ask "<question>"` — runs the agent loop, prints answer + citations
- `codesage eval` — runs the eval harness

## Data flow (a single question)

1. User runs `codesage ask "how does the retry logic work?"`
2. CLI loads the persisted index (or errors if `ingest` hasn't run)
3. Agent loop starts in `THINKING`: sends question + tool schemas to Gemini
4. Gemini responds with either a tool call (e.g. `search_code`) or a final answer
5. If tool call: agent looks up the tool in the registry (hashmap lookup),
   executes it, appends the result to memory, loops back to `THINKING`
6. Repeats until Gemini returns a final answer or `max_steps` is hit
   (`ERROR` state, agent reports it couldn't finish — no silent failure)
7. Final answer is printed with file/line citations pulled from the chunks
   that were actually retrieved

## Error handling

- **No API key set:** fail fast at startup with a clear message pointing to
  `.env.example`, not a stack trace mid-run
- **Index not built:** `ask` checks for the index file first and tells the
  user to run `ingest`
- **Tool execution error** (e.g. file not found): caught, turned into a tool
  result the agent can see and react to (e.g. try a different file), not a
  crash
- **Max steps exceeded:** loop terminates in `ERROR` state with a message
  showing what it tried, not an infinite hang
- **Gemini API errors** (rate limit, timeout): one retry with backoff, then
  surface a clear error

## Testing

- Unit tests per component: chunking produces expected boundaries, cosine
  similarity ranks correctly on a known small vector set, tool registry
  dispatches correctly, memory window trims correctly, state machine
  transitions correctly on mocked LLM responses
- One integration test: ingest a tiny fixture repo (checked into
  `tests/fixtures/`), ask a question with the real Gemini API (marked
  `@pytest.mark.integration`, skipped in CI by default to avoid requiring
  secrets in Actions; run locally)
- GitHub Actions runs unit tests (no API key needed) on every push

## Repo structure

```
codesage/
  codesage/
    __init__.py
    ingest.py
    index.py
    tools.py
    memory.py
    agent.py
    eval.py
    cli.py
  tests/
    fixtures/
    test_*.py
  docs/
    superpowers/specs/   (this spec + future ones)
  .github/workflows/ci.yml
  .env.example
  .gitignore
  pyproject.toml
  README.md
```

## Out of scope (v1)

- Persistent/external vector database (Chroma, FAISS, Pinecone) — brute-force
  in-memory is a deliberate teaching choice and is fast enough at this scale
- Web UI — CLI only
- Multi-agent orchestration (supervisor/sub-agents) — noted as a natural
  "v2" extension once the single-agent loop is solid, not built now
- Streaming responses
- Support for languages/frameworks beyond what's needed to demo on one
  target repo

## Success criteria

- `codesage ask` correctly answers questions about the target repo with
  accurate file/line citations, verified against the Phase 6 eval set
- Every phase (1–7) is its own commit (or small commit series) with a message
  explaining the concept it introduces
- `pytest` passes in CI on a clean checkout
- README documents architecture, setup, and how to point CodeSage at a
  different repo
- Repo is pushed to the user's GitHub as a public repository
