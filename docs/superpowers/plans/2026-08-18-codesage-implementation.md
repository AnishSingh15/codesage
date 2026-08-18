# CodeSage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build CodeSage — an agent, written from scratch in Python, that answers questions about a codebase using tool-calling, retrieval (RAG), and multi-step reasoning — teaching agentic AI + DSA/LLD concepts inline, and ship it as a tested, CI-passing, publicly deployed GitHub repo.

**Architecture:** A hand-written ReAct-style agent loop (`Agent`) drives a `ToolRegistry` (hashmap dispatch) and a bounded `Memory` (deque). Retrieval is a brute-force cosine-similarity `RetrievalIndex` over chunks embedded via Gemini. The loop is refactored into an explicit state machine once tool use exists. A thin `argparse` CLI and later a `FastAPI` wrapper are two interfaces over the same core classes.

**Tech Stack:** Python 3.13, `uv` for env/deps, `google-genai` (Gemini `gemini-2.5-flash` for generation, `gemini-embedding-001` for embeddings), `numpy` for vector math, `pytest`, GitHub Actions, `fastapi` + `uvicorn` (Phase 8 only), Hugging Face Spaces (Docker SDK) for free hosting.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-18-codesage-agent-design.md` — every task below implements one numbered phase from it.
- No LangGraph/CrewAI/other agent framework. The agent loop, memory, tool dispatch, and retrieval index are hand-written.
- No vector database (Chroma/FAISS/Pinecone). Retrieval is brute-force cosine similarity via `numpy`, held in memory and persisted to a flat JSON file between CLI runs.
- LLM: Gemini free tier only, via `google-genai`. Model `gemini-2.5-flash` for generation, `gemini-embedding-001` (768 dims) for embeddings.
- All LLM calls go through `codesage/llm.py::LLMClient` — no other module imports `google.genai` directly.
- Every task ends in a commit. Commit messages explain the concept the task introduces (not just "add file X").
- Tests that need a real Gemini API key are marked `@pytest.mark.integration` and excluded from the default `pytest` run (`addopts = "-m 'not integration'"`) so CI never needs a secret.
- Target repo (the codebase CodeSage answers questions about) is configurable via `--repo` / `CODESAGE_TARGET_REPO`, defaulting to a small public repo the user picks in Task 1 (recommendation: `psf/requests`, small enough to ingest quickly, well-documented enough to ask good questions about).

---

### Task 1: Project scaffolding, GitHub repo, and CI

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `codesage/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.github/workflows/ci.yml`
- Create: `README.md` (stub, expanded in Task 10)

**Interfaces:**
- Produces: an installable `codesage` package (`import codesage` works), a `pytest` setup that ignores `integration`-marked tests by default, a GitHub Actions job that runs `uv run pytest` on every push.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "codesage"
version = "0.1.0"
description = "An AI agent that answers questions about a codebase, built from scratch to teach agentic AI and DSA/LLD."
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "google-genai>=1.0.0",
    "numpy>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]
api = ["fastapi>=0.115.0", "uvicorn[standard]>=0.32.0"]

[project.scripts]
codesage = "codesage.cli:main"

[build-system]
requires = ["uv_build>=0.12.5,<0.13.0"]
build-backend = "uv_build"

[tool.pytest.ini_options]
markers = [
    "integration: requires a real GEMINI_API_KEY and network access",
]
addopts = "-m 'not integration'"
```

- [ ] **Step 2: Write `.env.example`, `.gitignore`, package/test init files**

`.env.example`:
```
GEMINI_API_KEY=your-api-key-here
CODESAGE_TARGET_REPO=./target_repo
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
.codesage_index.json
target_repo/
.pytest_cache/
dist/
*.egg-info/
```

`codesage/__init__.py`: empty file.
`tests/__init__.py`: empty file.

- [ ] **Step 3: Write the smoke test**

```python
# tests/test_smoke.py
import codesage


def test_package_imports():
    assert codesage is not None
```

- [ ] **Step 4: Install deps and verify the test fails cleanly first, then passes**

Run: `uv sync --all-extras`
Run: `uv run pytest -v`
Expected: `test_package_imports` PASSES (it's trivial by construction — this step confirms `uv` and `pytest` are wired correctly before we build anything real on top).

- [ ] **Step 5: Write the CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
      - name: Set up Python
        run: uv python install 3.13
      - name: Install dependencies
        run: uv sync --all-extras
      - name: Run tests
        run: uv run pytest -v
```

- [ ] **Step 6: Write the README stub**

```markdown
# CodeSage

An AI agent that answers questions about a codebase — built from scratch in
Python (no LangGraph/CrewAI) to learn agentic AI and DSA/LLD concepts by
building each piece by hand.

Full architecture writeup: `docs/superpowers/specs/2026-08-18-codesage-agent-design.md`

## Status

Work in progress — see `docs/superpowers/plans/2026-08-18-codesage-implementation.md`
for the build plan.
```

- [ ] **Step 7: Create the GitHub repo and push**

Run:
```bash
gh repo create codesage --public --source=. --remote=origin --push
```
If that fails because there's nothing committed yet on `main`, commit first (next step), then run:
```bash
gh repo create codesage --public --source=. --remote=origin
git push -u origin main
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example .gitignore codesage/__init__.py \
  tests/__init__.py tests/test_smoke.py .github/workflows/ci.yml README.md uv.lock
git commit -m "chore: project scaffolding, CI, and GitHub repo"
git push
```

Verify: open the Actions tab on the new GitHub repo (or `gh run watch`) and confirm the CI job goes green.

---

### Task 2: LLM client wrapper

**Files:**
- Create: `codesage/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `LLMClient(api_key: str | None = None, model: str = "gemini-2.5-flash", client=None)` with:
  - `generate(contents: list[types.Content], tools: list[types.Tool] | None = None) -> types.GenerateContentResponse`
  - `embed(text: str, output_dimensionality: int = 768) -> list[float]`
- This is the *only* module allowed to import `google.genai` directly (Global Constraint). Every later task depends on this signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
from types import SimpleNamespace

from codesage.llm import LLMClient


class FakeModels:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []

    def generate_content(self, model, contents, config=None):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="mocked answer", function_calls=None)

    def embed_content(self, model, contents, config=None):
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])


class FakeGenaiClient:
    def __init__(self):
        self.models = FakeModels()


def test_generate_delegates_to_client_and_returns_response():
    fake_client = FakeGenaiClient()
    llm = LLMClient(client=fake_client)

    response = llm.generate(contents=["some content"])

    assert response.text == "mocked answer"
    assert fake_client.models.generate_calls[0]["model"] == "gemini-2.5-flash"
    assert fake_client.models.generate_calls[0]["contents"] == ["some content"]


def test_embed_returns_vector_of_floats():
    fake_client = FakeGenaiClient()
    llm = LLMClient(client=fake_client)

    vector = llm.embed("why is the sky blue?")

    assert vector == [0.1, 0.2, 0.3]
    assert fake_client.models.embed_calls[0]["model"] == "gemini-embedding-001"


def test_generate_retries_once_on_transient_error_then_succeeds():
    fake_client = FakeGenaiClient()
    call_count = {"n": 0}
    real_generate_content = fake_client.models.generate_content

    def flaky_generate_content(model, contents, config=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("simulated transient failure")
        return real_generate_content(model, contents, config)

    fake_client.models.generate_content = flaky_generate_content
    llm = LLMClient(client=fake_client)

    response = llm.generate(contents=["some content"])

    assert response.text == "mocked answer"
    assert call_count["n"] == 2


def test_generate_raises_clear_error_after_retry_also_fails():
    fake_client = FakeGenaiClient()

    def always_fails(model, contents, config=None):
        raise TimeoutError("simulated persistent failure")

    fake_client.models.generate_content = always_fails
    llm = LLMClient(client=fake_client)

    with pytest.raises(RuntimeError, match="Gemini API call failed after 1 retry"):
        llm.generate(contents=["some content"])
```

Add `import pytest` to the top of the test file alongside the existing imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError: cannot import name 'LLMClient'`

- [ ] **Step 3: Write the implementation**

```python
# codesage/llm.py
"""Thin wrapper around the Gemini SDK.

This is the only module in CodeSage that imports google.genai — every other
module depends on this interface instead, so we can swap models/providers
or inject a fake client in tests without touching agent logic.
"""

import time

from google import genai
from google.genai import types

_MAX_RETRIES = 1
_RETRY_BACKOFF_SECONDS = 0.1


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        client=None,
    ):
        self._client = client or genai.Client(api_key=api_key)
        self._model = model

    def generate(
        self,
        contents: list,
        tools: list[types.Tool] | None = None,
    ) -> types.GenerateContentResponse:
        config = types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        )

    def embed(self, text: str, output_dimensionality: int = 768) -> list[float]:
        result = self._with_retry(
            lambda: self._client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
            )
        )
        return result.embeddings[0].values

    def _with_retry(self, call):
        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS)
        raise RuntimeError(
            f"Gemini API call failed after {_MAX_RETRIES} retry: {last_error}"
        ) from last_error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add codesage/llm.py tests/test_llm.py
git commit -m "feat: add LLMClient wrapper around Gemini SDK

Isolates google.genai to one module (dependency inversion) so agent
logic can be tested with a fake client instead of hitting the real API."
```

---

### Task 3: Phase 1 — single-turn Agent + CLI `ask`

**Concept taught:** what an "agent" actually is (a class wrapping an LLM call) — before any tool use or memory, so the loop added in Task 4 has an obvious "before" to compare against.

**Files:**
- Create: `codesage/agent.py`
- Create: `codesage/cli.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `LLMClient.generate(contents, tools=None)` from Task 2.
- Produces: `Agent(llm: LLMClient)` with `ask(question: str) -> str`. Task 4 will change this constructor signature (adds `tools=`) — callers in tests should use keyword args, not positional, to stay stable across that change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
from types import SimpleNamespace

from codesage.agent import Agent


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, contents, tools=None):
        self.calls.append({"contents": contents, "tools": tools})
        return self._responses.pop(0)


def test_ask_sends_question_and_returns_model_text():
    fake_llm = FakeLLM([SimpleNamespace(text="42", function_calls=None)])
    agent = Agent(llm=fake_llm)

    answer = agent.ask("what is the answer?")

    assert answer == "42"
    assert len(fake_llm.calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# codesage/agent.py
"""The agent: an observe -> think -> act loop.

Phase 1: no tools, no memory yet — just enough to prove the LLM round-trip
works end to end. Tool use lands in Task 4, memory in Task 5, the full
state machine in Task 8.
"""

from google.genai import types


class Agent:
    def __init__(self, llm):
        self._llm = llm

    def ask(self, question: str) -> str:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        response = self._llm.generate(contents)
        return response.text
```

```python
# codesage/cli.py
"""CLI entrypoint. Phase 1: just `codesage ask`."""

import argparse
import os
import sys

from dotenv import load_dotenv

from codesage.agent import Agent
from codesage.llm import LLMClient


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="codesage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    llm = LLMClient(api_key=api_key)

    if args.command == "ask":
        agent = Agent(llm)
        print(agent.ask(args.question))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Manual smoke test with the real API (requires your own `.env`)**

Run: `cp .env.example .env` then add your real `GEMINI_API_KEY`, then:
Run: `uv run codesage ask "Say hello in one word."`
Expected: prints a one-word greeting from Gemini. This is the first real, live proof the whole chain (CLI → Agent → LLMClient → Gemini) works.

- [ ] **Step 6: Commit**

```bash
git add codesage/agent.py codesage/cli.py tests/test_agent.py
git commit -m "feat: single-turn Agent and CLI ask command (Phase 1)

An agent is just a class wrapping an LLM call at this point — no tools,
no memory. Establishes the loop that later phases build on."
```

---

### Task 4: Phase 2 — Tools + tool-calling loop

**Concept taught:** hashmap-as-registry (Open/Closed Principle — adding a tool never means editing the agent loop), and the manual ReAct tool-call loop.

**Files:**
- Create: `codesage/tools.py`
- Modify: `codesage/agent.py` (constructor gains `tools=`, `ask` gains the function-call loop)
- Modify: `codesage/cli.py` (wire a base `ToolRegistry` into `ask`)
- Test: `tests/test_tools.py`
- Modify: `tests/test_agent.py` (add tool-loop test)

**Interfaces:**
- Produces (`tools.py`): `Tool(name, description, parameters_schema, handler)` dataclass with `.to_function_declaration()`; `ToolRegistry` with `register(tool)`, `get(name) -> Tool`, `call(name, **kwargs) -> str`, `has_tools() -> bool`, `as_declarations() -> list`, `as_tool() -> types.Tool`; plain functions `list_files_handler(base_dir, subdir) -> str` and `read_file_handler(base_dir, path, start_line=None, end_line=None) -> str`.
- Consumes: `types.FunctionDeclaration`, `types.Tool` from `google.genai.types` (already a dependency via Task 2).
- Produces (`agent.py`): `Agent(llm, tools: ToolRegistry | None = None)`, `ask` now loops on `response.function_calls` up to an internal safety cap of 5 iterations (formalized into a real state machine with a configurable `max_steps` in Task 8 — this cap just prevents a runaway loop in the meantime).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
from pathlib import Path

import pytest

from codesage.tools import Tool, ToolRegistry, list_files_handler, read_file_handler


def test_register_and_call_a_tool():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echoes input",
            parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
    )

    assert registry.call("echo", text="hi") == "hi"


def test_call_unknown_tool_raises_key_error():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.call("nope")


def test_has_tools_reflects_registration_state():
    registry = ToolRegistry()
    assert registry.has_tools() is False
    registry.register(
        Tool(name="a", description="", parameters_schema={"type": "object"}, handler=lambda: "")
    )
    assert registry.has_tools() is True


def test_list_files_handler_lists_directory(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.py").write_text("y = 2")

    result = list_files_handler(tmp_path, ".")

    assert "a.py" in result and "b.py" in result


def test_list_files_handler_blocks_path_escape(tmp_path: Path):
    result = list_files_handler(tmp_path, "../../etc")
    assert result.startswith("Error")


def test_read_file_handler_returns_requested_line_range(tmp_path: Path):
    (tmp_path / "f.py").write_text("line1\nline2\nline3\nline4\n")

    result = read_file_handler(tmp_path, "f.py", start_line=2, end_line=3)

    assert result == "line2\nline3"
```

Add to `tests/test_agent.py`:

```python
from codesage.tools import Tool, ToolRegistry


def test_ask_executes_a_tool_call_then_returns_final_answer():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="looks something up",
            parameters_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=lambda q: f"result for {q}",
        )
    )

    tool_call_response = SimpleNamespace(
        text=None,
        function_calls=[SimpleNamespace(name="lookup", args={"q": "foo"})],
        candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
    )
    final_response = SimpleNamespace(text="here's your answer", function_calls=None)

    fake_llm = FakeLLM([tool_call_response, final_response])
    agent = Agent(llm=fake_llm, tools=registry)

    answer = agent.ask("look up foo")

    assert answer == "here's your answer"
    assert len(fake_llm.calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: codesage.tools` and `TypeError: Agent.__init__() got an unexpected keyword argument 'tools'`

- [ ] **Step 3: Write `codesage/tools.py`**

```python
# codesage/tools.py
"""Tool registry: a hashmap mapping tool name -> callable + schema.

Adding a new tool never means editing the agent loop (agent.py) — that's
the Open/Closed Principle in practice. The registry is the seam.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google.genai import types


@dataclass
class Tool:
    name: str
    description: str
    parameters_schema: dict
    handler: Callable[..., str]

    def to_function_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.parameters_schema,
        )


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return self._tools[name]

    def call(self, name: str, **kwargs) -> str:
        return self.get(name).handler(**kwargs)

    def has_tools(self) -> bool:
        return bool(self._tools)

    def as_declarations(self) -> list[types.FunctionDeclaration]:
        return [t.to_function_declaration() for t in self._tools.values()]

    def as_tool(self) -> types.Tool:
        return types.Tool(function_declarations=self.as_declarations())


def list_files_handler(base_dir: Path, subdir: str) -> str:
    target = (base_dir / subdir).resolve()
    if not str(target).startswith(str(Path(base_dir).resolve())):
        return "Error: path escapes the target repo."
    if not target.exists():
        return f"Error: {subdir} does not exist."
    entries = sorted(p.name for p in target.iterdir())
    return "\n".join(entries) if entries else "(empty directory)"


def read_file_handler(
    base_dir: Path,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    target = (base_dir / path).resolve()
    if not str(target).startswith(str(Path(base_dir).resolve())):
        return "Error: path escapes the target repo."
    if not target.exists():
        return f"Error: {path} does not exist."
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = (start_line or 1) - 1
    end = end_line or len(lines)
    return "\n".join(lines[start:end])
```

- [ ] **Step 4: Modify `codesage/agent.py`**

Replace the whole file:

```python
# codesage/agent.py
"""The agent: an observe -> think -> act loop.

Phase 2: adds tool calling. The loop below is intentionally simple (a
while loop with a hard iteration cap) — Task 8 replaces it with an
explicit state machine once there's enough behavior to justify one.
"""

from google.genai import types

from codesage.tools import ToolRegistry

_SAFETY_MAX_ITERATIONS = 5


class Agent:
    def __init__(self, llm, tools: ToolRegistry | None = None):
        self._llm = llm
        self._tools = tools or ToolRegistry()

    def ask(self, question: str) -> str:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        tool = self._tools.as_tool() if self._tools.has_tools() else None

        response = self._llm.generate(contents, tools=[tool] if tool else None)
        iterations = 0

        while response.function_calls and iterations < _SAFETY_MAX_ITERATIONS:
            iterations += 1
            call = response.function_calls[0]
            contents.append(response.candidates[0].content)

            try:
                result = self._tools.call(call.name, **call.args)
                function_response = {"result": result}
            except Exception as exc:
                function_response = {"error": str(exc)}

            contents.append(
                types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=call.name, response=function_response)],
                )
            )
            response = self._llm.generate(contents, tools=[tool] if tool else None)

        return response.text
```

- [ ] **Step 5: Modify `codesage/cli.py`**

Replace the whole file:

```python
# codesage/cli.py
"""CLI entrypoint. Phase 2: `ask` now has list_files/read_file tools."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from codesage.agent import Agent
from codesage.llm import LLMClient
from codesage.tools import Tool, ToolRegistry, list_files_handler, read_file_handler


def build_base_registry(base_dir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="list_files",
            description="List files under a directory relative to the target repo root.",
            parameters_schema={
                "type": "object",
                "properties": {"subdir": {"type": "string", "description": "Relative subdirectory, use '.' for root"}},
                "required": ["subdir"],
            },
            handler=lambda subdir: list_files_handler(base_dir, subdir),
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a file's contents by path relative to the target repo root, optionally a line range.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=lambda path, start_line=None, end_line=None: read_file_handler(
                base_dir, path, start_line, end_line
            ),
        )
    )
    return registry


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="codesage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--repo", default=".", help="Target repo path")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    llm = LLMClient(api_key=api_key)

    if args.command == "ask":
        repo_path = Path(args.repo)
        registry = build_base_registry(repo_path)
        agent = Agent(llm, tools=registry)
        print(agent.ask(args.question))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py tests/test_agent.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Manual smoke test**

Run: `uv run codesage ask "list the files in the current directory" --repo .`
Expected: the agent calls `list_files` and reports back real filenames from your repo.

- [ ] **Step 8: Commit**

```bash
git add codesage/tools.py codesage/agent.py codesage/cli.py \
  tests/test_tools.py tests/test_agent.py
git commit -m "feat: tool registry + manual ReAct loop (Phase 2)

Tools are a dict[str, Tool] the agent dispatches through by name —
adding a tool never means touching the loop itself (Open/Closed
Principle). Agent.ask now loops: think -> call tool -> feed result
back -> think again, until the model stops requesting tools."
```

---

### Task 5: Phase 3 — Bounded memory

**Concept taught:** sliding window via `collections.deque(maxlen=...)` — why unbounded conversation history breaks context windows, and why a deque (not a list with manual trimming) is the right structure.

**Files:**
- Create: `codesage/memory.py`
- Modify: `codesage/agent.py` (use `Memory` instead of a local `contents` list; make `ask` multi-turn)
- Modify: `codesage/cli.py` (`ask` keeps one `Memory` per process run — still single-shot per CLI invocation, but the plumbing is now in place for Phase 8's long-lived API process)
- Test: `tests/test_memory.py`
- Modify: `tests/test_agent.py` (memory-aware assertions)

**Interfaces:**
- Produces: `Memory(max_turns: int = 10)` with `add(content: types.Content) -> None`, `as_contents() -> list[types.Content]`, `__len__`.
- Produces (`agent.py`): `Agent(llm, tools=None, memory: Memory | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
from google.genai import types

from codesage.memory import Memory


def _content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def test_memory_returns_added_content_in_order():
    memory = Memory(max_turns=5)
    memory.add(_content("first"))
    memory.add(_content("second"))

    contents = memory.as_contents()

    assert len(contents) == 2
    assert contents[0].parts[0].text == "first"
    assert contents[1].parts[0].text == "second"


def test_memory_drops_oldest_when_over_capacity():
    memory = Memory(max_turns=2)  # capacity = 2 turns = 4 Content entries
    for i in range(6):
        memory.add(_content(str(i)))

    contents = memory.as_contents()

    assert len(contents) == 4
    assert [c.parts[0].text for c in contents] == ["2", "3", "4", "5"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `codesage/memory.py`**

```python
# codesage/memory.py
"""Bounded conversation history.

A plain list would grow forever and eventually overflow the model's
context window. collections.deque(maxlen=...) gives us an O(1) sliding
window for free: once full, adding a new item silently drops the oldest.
"""

from collections import deque

from google.genai import types


class Memory:
    def __init__(self, max_turns: int = 10):
        self._contents: deque = deque(maxlen=max_turns * 2)

    def add(self, content: types.Content) -> None:
        self._contents.append(content)

    def as_contents(self) -> list[types.Content]:
        return list(self._contents)

    def __len__(self) -> int:
        return len(self._contents)
```

- [ ] **Step 4: Modify `codesage/agent.py`**

Replace the whole file:

```python
# codesage/agent.py
"""The agent: an observe -> think -> act loop.

Phase 3: conversation state moves from a local list into Memory, so
Agent.ask is now genuinely multi-turn if you call it more than once on
the same Agent instance.
"""

from google.genai import types

from codesage.memory import Memory
from codesage.tools import ToolRegistry

_SAFETY_MAX_ITERATIONS = 5


class Agent:
    def __init__(self, llm, tools: ToolRegistry | None = None, memory: Memory | None = None):
        self._llm = llm
        self._tools = tools or ToolRegistry()
        self._memory = memory or Memory()

    def ask(self, question: str) -> str:
        self._memory.add(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        tool = self._tools.as_tool() if self._tools.has_tools() else None

        response = self._llm.generate(self._memory.as_contents(), tools=[tool] if tool else None)
        iterations = 0

        while response.function_calls and iterations < _SAFETY_MAX_ITERATIONS:
            iterations += 1
            call = response.function_calls[0]
            self._memory.add(response.candidates[0].content)

            try:
                result = self._tools.call(call.name, **call.args)
                function_response = {"result": result}
            except Exception as exc:
                function_response = {"error": str(exc)}

            self._memory.add(
                types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=call.name, response=function_response)],
                )
            )
            response = self._llm.generate(self._memory.as_contents(), tools=[tool] if tool else None)

        self._memory.add(response.candidates[0].content if response.candidates else
                          types.Content(role="model", parts=[types.Part.from_text(text=response.text or "")]))
        return response.text
```

Note: real Gemini responses always populate `response.candidates`; the fallback branch exists purely so the `FakeLLM` test doubles (which don't set `.candidates` on the plain-text final response) don't need to be rewritten. Update `tests/test_agent.py`'s two existing `SimpleNamespace` final responses to add `function_calls=None` (already present) — no other change needed there since the fallback covers it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory.py tests/test_agent.py tests/test_tools.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Manual smoke test — multi-turn**

Run: `uv run python -c "
from dotenv import load_dotenv
load_dotenv()
import os
from codesage.agent import Agent
from codesage.llm import LLMClient

agent = Agent(LLMClient(api_key=os.environ['GEMINI_API_KEY']))
print(agent.ask('My favorite number is 7.'))
print(agent.ask('What is my favorite number?'))
"`
Expected: the second answer correctly says 7 — proving memory persists across `ask` calls on the same `Agent` instance.

- [ ] **Step 7: Commit**

```bash
git add codesage/memory.py codesage/agent.py tests/test_memory.py tests/test_agent.py
git commit -m "feat: bounded conversation memory via deque sliding window (Phase 3)

Memory replaces the per-call local contents list. deque(maxlen=...)
gives an O(1) sliding window instead of manually trimming a list —
Agent.ask is now genuinely multi-turn."
```

---

### Task 6: Phase 4a — Ingestion (chunking + embedding)

**Concept taught:** turning unstructured files into fixed-size, addressable units (chunking) — a direct precursor to the retrieval index in Task 7.

**Files:**
- Create: `codesage/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Produces: `Chunk` dataclass (`text: str, file_path: str, line_start: int, line_end: int, vector: list[float] | None = None`); `iter_source_files(repo_path: Path) -> Iterator[Path]`; `chunk_file(path: Path, lines_per_chunk: int = 40) -> list[Chunk]`; `ingest_repo(repo_path: Path, llm) -> list[Chunk]`.
- Consumes: `LLMClient.embed(text) -> list[float]` from Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
from pathlib import Path

from codesage.ingest import Chunk, chunk_file, iter_source_files, ingest_repo


def test_chunk_file_splits_by_line_window(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 91)))  # 90 lines

    chunks = chunk_file(f, lines_per_chunk=40)

    assert len(chunks) == 3
    assert chunks[0].line_start == 1 and chunks[0].line_end == 40
    assert chunks[1].line_start == 41 and chunks[1].line_end == 80
    assert chunks[2].line_start == 81 and chunks[2].line_end == 90
    assert chunks[0].text.startswith("line1\n")


def test_chunk_file_skips_blank_only_windows(tmp_path: Path):
    f = tmp_path / "blank.py"
    f.write_text("\n" * 45)  # all blank lines

    chunks = chunk_file(f, lines_per_chunk=40)

    assert chunks == []


def test_iter_source_files_skips_dot_git_and_non_text(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("ignored")

    found = {p.name for p in iter_source_files(tmp_path)}

    assert found == {"a.py"}


def test_ingest_repo_embeds_every_chunk(tmp_path: Path):
    (tmp_path / "a.py").write_text("\n".join(f"line{i}" for i in range(1, 5)))

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [float(len(text))]

    chunks = ingest_repo(tmp_path, FakeLLM())

    assert len(chunks) == 1
    assert chunks[0].vector == [float(len(chunks[0].text))]
    assert isinstance(chunks[0], Chunk)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `codesage/ingest.py`**

```python
# codesage/ingest.py
"""Turn a repo's files into embedded, addressable chunks.

Chunking is the DSA-adjacent decision here: fixed-size line windows are
simple and predictable (O(file length) to produce), at the cost of
sometimes splitting a function across two chunks. Good enough for v1;
the tradeoff is called out in the README.
"""

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


def ingest_repo(repo_path: Path, llm) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file_path in iter_source_files(repo_path):
        for chunk in chunk_file(file_path):
            chunk.vector = llm.embed(chunk.text)
            chunks.append(chunk)
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codesage/ingest.py tests/test_ingest.py
git commit -m "feat: repo ingestion — walk, chunk, embed (Phase 4a)

Fixed-size line-window chunking. Simple and predictable; the tradeoff
(may split a function across chunks) is a deliberate v1 simplification."
```

---

### Task 7: Phase 4b — Retrieval index + `search_code` tool + `ingest` CLI command

**Concept taught:** vectors and cosine similarity as a distance metric; brute-force top-k search is O(n·d) — why that's fine at hundreds/thousands of chunks and why real systems switch to an ANN index (HNSW) at scale (explained, not built — matches the spec).

**Files:**
- Create: `codesage/index.py`
- Modify: `codesage/tools.py` (add `make_search_code_tool`)
- Modify: `codesage/cli.py` (add `ingest` subcommand; `ask` loads a persisted index if present)
- Test: `tests/test_index.py`

**Interfaces:**
- Produces (`index.py`): `RetrievalIndex(chunks: list[Chunk])` with `search(query_vector: list[float], k: int = 5) -> list[Chunk]`; `save_chunks(chunks, path: Path) -> None`; `load_chunks(path: Path) -> list[Chunk]`.
- Produces (`tools.py` addition): `make_search_code_tool(index: RetrievalIndex, llm) -> Tool`.
- Consumes: `Chunk` from Task 6, `Tool`/`ToolRegistry` from Task 4, `LLMClient.embed` from Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
import json
from pathlib import Path

from codesage.ingest import Chunk
from codesage.index import RetrievalIndex, save_chunks, load_chunks


def _chunk(name: str, vector: list[float]) -> Chunk:
    return Chunk(text=f"code for {name}", file_path=f"{name}.py", line_start=1, line_end=10, vector=vector)


def test_search_ranks_by_cosine_similarity():
    chunks = [
        _chunk("auth", [1.0, 0.0, 0.0]),
        _chunk("db", [0.0, 1.0, 0.0]),
        _chunk("http", [0.9, 0.1, 0.0]),  # close to "auth"
    ]
    index = RetrievalIndex(chunks)

    results = index.search(query_vector=[1.0, 0.0, 0.0], k=2)

    assert [c.file_path for c in results] == ["auth.py", "http.py"]


def test_search_on_empty_index_returns_empty_list():
    index = RetrievalIndex([])
    assert index.search([1.0, 0.0], k=5) == []


def test_save_and_load_chunks_round_trip(tmp_path: Path):
    chunks = [_chunk("auth", [1.0, 0.0, 0.0])]
    path = tmp_path / "index.json"

    save_chunks(chunks, path)
    loaded = load_chunks(path)

    assert loaded == chunks
    assert json.loads(path.read_text())[0]["file_path"] == "auth.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `codesage/index.py`**

```python
# codesage/index.py
"""Brute-force cosine-similarity retrieval over embedded chunks.

O(n) per query where n = number of chunks. Fine at the scale of one
repo (hundreds-low thousands of chunks). At millions of vectors you'd
reach for an ANN index (e.g. HNSW, a navigable small-world graph) —
noted here, not built, per the spec's explicit v1 scope.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from codesage.ingest import Chunk


class RetrievalIndex:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._matrix = np.array([c.vector for c in chunks]) if chunks else np.empty((0, 0))

    def search(self, query_vector: list[float], k: int = 5) -> list[Chunk]:
        if not self._chunks:
            return []
        query = np.array(query_vector)
        chunk_norms = np.linalg.norm(self._matrix, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = (self._matrix @ query) / (chunk_norms * query_norm + 1e-10)
        top_k_idx = np.argsort(-similarities)[:k]
        return [self._chunks[i] for i in top_k_idx]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in chunks]))


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text())
    return [Chunk(**d) for d in data]
```

- [ ] **Step 4: Add `make_search_code_tool` to `codesage/tools.py`**

Add `from codesage.index import RetrievalIndex` to the imports at the top of the
file (no circularity — `index.py` only imports from `ingest.py`, never from
`tools.py`). Then append this function to the end of the file:

```python
def make_search_code_tool(index: RetrievalIndex, llm) -> Tool:
    def handler(query: str) -> str:
        query_vector = llm.embed(query)
        results = index.search(query_vector, k=5)
        if not results:
            return "No matching code found."
        return "\n\n".join(f"{c.file_path}:{c.line_start}-{c.line_end}\n{c.text}" for c in results)

    return Tool(
        name="search_code",
        description="Semantic search over the ingested codebase. Returns relevant code chunks with file/line citations.",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
        handler=handler,
    )
```

- [ ] **Step 5: Modify `codesage/cli.py`**

Replace the whole file:

```python
# codesage/cli.py
"""CLI entrypoint. Phase 4b: adds `ingest`; `ask` uses the index if present."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from codesage.agent import Agent
from codesage.llm import LLMClient
from codesage.tools import Tool, ToolRegistry, list_files_handler, read_file_handler, make_search_code_tool
from codesage.index import RetrievalIndex, load_chunks, save_chunks

INDEX_FILENAME = ".codesage_index.json"


def build_base_registry(base_dir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="list_files",
            description="List files under a directory relative to the target repo root.",
            parameters_schema={
                "type": "object",
                "properties": {"subdir": {"type": "string", "description": "Relative subdirectory, use '.' for root"}},
                "required": ["subdir"],
            },
            handler=lambda subdir: list_files_handler(base_dir, subdir),
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a file's contents by path relative to the target repo root, optionally a line range.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=lambda path, start_line=None, end_line=None: read_file_handler(
                base_dir, path, start_line, end_line
            ),
        )
    )
    return registry


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="codesage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("repo_path")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--repo", default=".", help="Target repo path")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    llm = LLMClient(api_key=api_key)

    if args.command == "ingest":
        from codesage.ingest import ingest_repo

        repo_path = Path(args.repo_path)
        chunks = ingest_repo(repo_path, llm)
        save_chunks(chunks, repo_path / INDEX_FILENAME)
        print(f"Indexed {len(chunks)} chunks from {repo_path}")

    elif args.command == "ask":
        repo_path = Path(args.repo)
        registry = build_base_registry(repo_path)

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

        agent = Agent(llm, tools=registry)
        print(agent.ask(args.question))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS (full suite — this is the first task that touches nearly every module, good point to confirm nothing upstream broke)

- [ ] **Step 7: Manual smoke test — real ingestion + retrieval**

Run:
```bash
git clone --depth 1 https://github.com/psf/requests.git target_repo
uv run codesage ingest target_repo
uv run codesage ask "how does the Session class handle retries?" --repo target_repo
```
Expected: the answer cites real files/line ranges from `target_repo/requests/`, not a generic LLM answer — proof retrieval is actually influencing the response.

- [ ] **Step 8: Commit**

```bash
git add codesage/index.py codesage/tools.py codesage/cli.py tests/test_index.py
git commit -m "feat: retrieval index + search_code tool + ingest CLI command (Phase 4b)

Brute-force cosine similarity via numpy — O(n) per query, which is
fine at the scale of one repo. save_chunks/load_chunks persist the
index as flat JSON between CLI runs (not a vector database)."
```

---

### Task 8: Phase 5 — Explicit state machine

**Concept taught:** the agent loop so far is an implicit state machine (a while loop with a hidden boolean). Making the states explicit (an `Enum`) turns "does this ever infinite-loop?" into a question you can answer by reading transitions, not by tracing execution.

**Files:**
- Modify: `codesage/agent.py` (full rewrite of the loop body around `AgentState`)
- Modify: `tests/test_agent.py` (add max-steps test)

**Interfaces:**
- Produces: `AgentState` enum (`THINKING`, `ACTING`, `DONE`, `ERROR`); `Agent(llm, tools=None, memory=None, max_steps: int = 8)`. `ask` signature unchanged (`str -> str`) — this is a pure internal refactor, existing callers (CLI, later API) don't change.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_ask_gives_up_after_max_steps_without_hanging():
    def always_calls_tool(*_args, **_kwargs):
        return SimpleNamespace(
            text=None,
            function_calls=[SimpleNamespace(name="loop", args={})],
            candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
        )

    class InfiniteFakeLLM:
        def __init__(self):
            self.call_count = 0

        def generate(self, contents, tools=None):
            self.call_count += 1
            return always_calls_tool()

    registry = ToolRegistry()
    registry.register(
        Tool(name="loop", description="", parameters_schema={"type": "object"}, handler=lambda: "again")
    )

    fake_llm = InfiniteFakeLLM()
    agent = Agent(llm=fake_llm, tools=registry, max_steps=3)

    answer = agent.ask("this will never resolve")

    assert "couldn't finish" in answer.lower()
    assert fake_llm.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL — with the Task 5 loop, this either runs forever (bad — but `_SAFETY_MAX_ITERATIONS=5` currently caps it) or the assertion on the "couldn't finish" message fails because that message doesn't exist yet, and `max_steps` isn't an accepted keyword. Confirm it fails on `TypeError: unexpected keyword argument 'max_steps'`.

- [ ] **Step 3: Rewrite `codesage/agent.py`**

```python
# codesage/agent.py
"""The agent: an explicit state machine over observe -> think -> act.

Phase 5: the implicit while-loop from Phase 2/3 becomes explicit states.
This is the same graph either way — making it explicit just means you
can answer "can this loop forever?" by reading the transition table
instead of tracing execution.
"""

from enum import Enum, auto

from google.genai import types

from codesage.memory import Memory
from codesage.tools import ToolRegistry


class AgentState(Enum):
    THINKING = auto()
    ACTING = auto()
    DONE = auto()
    ERROR = auto()


class Agent:
    def __init__(
        self,
        llm,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
        max_steps: int = 8,
    ):
        self._llm = llm
        self._tools = tools or ToolRegistry()
        self._memory = memory or Memory()
        self._max_steps = max_steps

    def ask(self, question: str) -> str:
        self._memory.add(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

        state = AgentState.THINKING
        response = None
        steps = 0

        while state in (AgentState.THINKING, AgentState.ACTING):
            steps += 1
            if steps > self._max_steps:
                state = AgentState.ERROR
                break

            if state is AgentState.THINKING:
                response, state = self._think()
            elif state is AgentState.ACTING:
                state = self._act(response)

        if state is AgentState.ERROR:
            return f"I couldn't finish after {self._max_steps} steps trying to answer: {question}"

        return response.text

    def _think(self):
        tool = self._tools.as_tool() if self._tools.has_tools() else None
        response = self._llm.generate(self._memory.as_contents(), tools=[tool] if tool else None)
        next_state = AgentState.ACTING if response.function_calls else AgentState.DONE
        if next_state is AgentState.DONE:
            self._memory.add(response.candidates[0].content)
        return response, next_state

    def _act(self, response) -> AgentState:
        call = response.function_calls[0]
        self._memory.add(response.candidates[0].content)

        try:
            result = self._tools.call(call.name, **call.args)
            function_response = {"result": result}
        except Exception as exc:
            function_response = {"error": str(exc)}

        self._memory.add(
            types.Content(
                role="tool",
                parts=[types.Part.from_function_response(name=call.name, response=function_response)],
            )
        )
        return AgentState.THINKING
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS (all, including the new max-steps test)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — confirms the refactor didn't break `ingest`/`index`/`tools`/`memory` tests, which don't touch `agent.py` but share fixtures.

- [ ] **Step 6: Commit**

```bash
git add codesage/agent.py tests/test_agent.py
git commit -m "refactor: agent loop as explicit state machine (Phase 5)

THINKING/ACTING/DONE/ERROR replace the implicit while-loop boolean.
max_steps is now a constructor param (was a hardcoded module constant)
and ERROR is a real, tested, reachable state instead of an assumed cap."
```

---

### Task 9: Phase 6 — Evaluation harness + `eval` CLI command

**Concept taught:** you can't claim an agent "works" without a repeatable check. Retrieval hit-rate and answer-keyword scoring are simple, cheap proxies. `RetrievalIndex.search` already has the shape (`search(query_vector, k) -> list[Chunk]`) that a second retrieval strategy would need to match — that's the Strategy pattern in practice, without building a second implementation that isn't needed yet.

**Files:**
- Create: `codesage/eval.py`
- Modify: `codesage/cli.py` (add `eval` subcommand)
- Create: `eval_cases.json` (repo root — real questions about whatever repo you ingested in Task 7)
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces: `EvalCase(question: str, expected_file_substring: str, expected_answer_keywords: list[str])`; `load_cases(path: Path) -> list[EvalCase]`; `score_retrieval(index, llm, case) -> bool`; `score_answer(answer: str, case: EvalCase) -> float`; `run_eval(agent, index, llm, cases: list[EvalCase]) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
import json
from pathlib import Path

from codesage.eval import EvalCase, load_cases, score_answer, score_retrieval, run_eval
from codesage.ingest import Chunk
from codesage.index import RetrievalIndex


def test_load_cases_parses_json(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"question": "q1", "expected_file_substring": "auth.py", "expected_answer_keywords": ["token"]}
    ]))

    cases = load_cases(path)

    assert cases == [EvalCase(question="q1", expected_file_substring="auth.py", expected_answer_keywords=["token"])]


def test_score_answer_is_fraction_of_keywords_present():
    case = EvalCase(question="q", expected_file_substring="x", expected_answer_keywords=["token", "refresh"])

    assert score_answer("uses a refresh token internally", case) == 1.0
    assert score_answer("uses nothing relevant", case) == 0.0
    assert score_answer("uses a token only", case) == 0.5


def test_score_retrieval_true_when_expected_file_is_returned():
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = RetrievalIndex([chunk])
    case = EvalCase(question="q", expected_file_substring="auth.py", expected_answer_keywords=[])

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [1.0, 0.0]

    assert score_retrieval(index, FakeLLM(), case) is True


def test_run_eval_aggregates_scores():
    case = EvalCase(question="q", expected_file_substring="auth.py", expected_answer_keywords=["token"])
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = RetrievalIndex([chunk])

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [1.0, 0.0]

    class FakeAgent:
        def ask(self, question):
            return "returns a token"

    results = run_eval(FakeAgent(), index, FakeLLM(), [case])

    assert results == {"retrieval_hit_rate": 1.0, "avg_answer_score": 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `codesage/eval.py`**

```python
# codesage/eval.py
"""A small, repeatable check that CodeSage's answers are actually grounded.

Two proxies, both cheap and deterministic:
- retrieval hit-rate: did we retrieve a chunk from the file we expected?
- answer score: fraction of expected keywords present in the final answer.

score_retrieval takes an `index` with a `.search(query_vector, k) -> list[Chunk]`
method — RetrievalIndex satisfies that shape today. A second retrieval
strategy could be swapped in here without changing this file (Strategy
pattern), if one is ever built.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalCase:
    question: str
    expected_file_substring: str
    expected_answer_keywords: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase(**d) for d in data]


def score_retrieval(index, llm, case: EvalCase) -> bool:
    query_vector = llm.embed(case.question)
    results = index.search(query_vector, k=5)
    return any(case.expected_file_substring in c.file_path for c in results)


def score_answer(answer: str, case: EvalCase) -> float:
    if not case.expected_answer_keywords:
        return 0.0
    hits = sum(1 for kw in case.expected_answer_keywords if kw.lower() in answer.lower())
    return hits / len(case.expected_answer_keywords)


def run_eval(agent, index, llm, cases: list[EvalCase]) -> dict:
    retrieval_hits = 0
    answer_scores = []
    for case in cases:
        if score_retrieval(index, llm, case):
            retrieval_hits += 1
        answer = agent.ask(case.question)
        answer_scores.append(score_answer(answer, case))

    return {
        "retrieval_hit_rate": retrieval_hits / len(cases),
        "avg_answer_score": sum(answer_scores) / len(answer_scores),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Modify `codesage/cli.py` — add the `eval` subcommand**

In the `subparsers.add_parser(...)` block, add:

```python
eval_parser = subparsers.add_parser("eval")
eval_parser.add_argument("--repo", default=".", help="Target repo path")
eval_parser.add_argument("--cases", default="eval_cases.json", help="Path to eval cases JSON")
```

In the `if args.command == ...` chain, add a final branch:

```python
    elif args.command == "eval":
        from codesage.eval import load_cases, run_eval

        repo_path = Path(args.repo)
        index_path = repo_path / INDEX_FILENAME
        if not index_path.exists():
            print(f"No index found. Run 'codesage ingest {repo_path}' first.", file=sys.stderr)
            sys.exit(1)

        cases_path = Path(args.cases)
        if not cases_path.exists():
            print(f"No eval cases found at {cases_path}. See eval_cases.json for the format.", file=sys.stderr)
            sys.exit(1)

        chunks = load_chunks(index_path)
        index = RetrievalIndex(chunks)
        registry = build_base_registry(repo_path)
        registry.register(make_search_code_tool(index, llm))
        agent = Agent(llm, tools=registry)

        results = run_eval(agent, index, llm, load_cases(cases_path))
        print(f"Retrieval hit rate: {results['retrieval_hit_rate']:.0%}")
        print(f"Avg answer score:   {results['avg_answer_score']:.0%}")
```

- [ ] **Step 6: Write `eval_cases.json` with 5 real questions**

This is the one step you write yourself rather than copy — pick 5 real questions about whatever repo you ingested in Task 7 (`target_repo`), and for each, name a file you know should be retrieved and 1-3 keywords a good answer would contain. Example shape (replace with real ones for your target repo):

```json
[
  {
    "question": "How does the Session class handle connection pooling?",
    "expected_file_substring": "sessions.py",
    "expected_answer_keywords": ["adapter", "mount"]
  }
]
```
Write 5 of these. This is also the fastest way to notice if retrieval is weak before you ship the demo.

- [ ] **Step 7: Run the eval against your ingested target repo**

Run: `uv run codesage eval --repo target_repo`
Expected: prints a retrieval hit rate and answer score. If either is low, that's real signal — either the questions are too vague, chunking is splitting relevant code awkwardly, or `k` in `search` is too small. Adjust and re-run; this loop *is* the point of Phase 6.

- [ ] **Step 8: Commit**

```bash
git add codesage/eval.py codesage/cli.py tests/test_eval.py eval_cases.json
git commit -m "feat: evaluation harness + eval CLI command (Phase 6)

Retrieval hit-rate + answer-keyword scoring against a fixed case set.
Cheap, deterministic, repeatable — the minimum bar for claiming the
agent 'works' rather than 'looked right once.'"
```

---

### Task 10: Phase 7 — Polish: integration test, README, final checks

**Files:**
- Create: `tests/fixtures/tiny_repo/` (2-3 small files)
- Create: `tests/test_integration.py`
- Modify: `README.md` (full architecture writeup)
- Modify: `.gitignore` (if anything was missed)

**Interfaces:**
- No new production interfaces — this task adds coverage and documentation over what Tasks 1-9 already built.

- [ ] **Step 1: Create the fixture repo**

```bash
mkdir -p tests/fixtures/tiny_repo
```

`tests/fixtures/tiny_repo/math_utils.py`:
```python
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def is_even(n: int) -> bool:
    """Return True if n is even."""
    return n % 2 == 0
```

`tests/fixtures/tiny_repo/README.md`:
```markdown
# Tiny Repo

A minimal fixture repo used by CodeSage's integration test.
It contains one module, math_utils.py, with add() and is_even().
```

- [ ] **Step 2: Write the integration test**

```python
# tests/test_integration.py
"""End-to-end test against the real Gemini API.

Excluded from the default pytest run (see pyproject.toml addopts).
Run explicitly with: uv run pytest tests/test_integration.py -m integration -v
Requires GEMINI_API_KEY in the environment.
"""

import os
from pathlib import Path

import pytest

from codesage.agent import Agent
from codesage.cli import build_base_registry
from codesage.index import RetrievalIndex
from codesage.ingest import ingest_repo
from codesage.llm import LLMClient
from codesage.tools import make_search_code_tool

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "tiny_repo"


@pytest.mark.integration
def test_agent_answers_question_about_fixture_repo_with_citation():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    llm = LLMClient(api_key=api_key)
    chunks = ingest_repo(FIXTURE_REPO, llm)
    index = RetrievalIndex(chunks)

    registry = build_base_registry(FIXTURE_REPO)
    registry.register(make_search_code_tool(index, llm))
    agent = Agent(llm, tools=registry)

    answer = agent.ask("What does the is_even function do?")

    assert "even" in answer.lower()
```

- [ ] **Step 3: Run it locally (requires your real API key)**

Run: `uv run pytest tests/test_integration.py -m integration -v`
Expected: PASS. If it fails, read the actual answer text the assertion prints — this is the real, non-mocked system, so a failure here means something in retrieval or the prompt is genuinely off, not a test-double mismatch.

- [ ] **Step 4: Confirm the default test run still excludes it**

Run: `uv run pytest -v`
Expected: `test_agent_answers_question_about_fixture_repo_with_citation` does NOT appear in the output (excluded by `addopts = "-m 'not integration'"` from Task 1).

- [ ] **Step 5: Write the full README**

Replace `README.md`:

```markdown
# CodeSage

An AI agent that answers questions about a codebase — citing exact
files and line ranges — built from scratch in Python (no LangGraph,
no CrewAI) so every piece of "how agents work" is code you can read,
not a framework you have to trust.

## What it does

Point CodeSage at a repo. It reads the files, chunks and embeds them,
and answers natural-language questions by retrieving relevant code and
reasoning over it with tools — the same ReAct-style loop used in
production coding assistants, hand-written here to make the mechanics
visible.

```
codesage ingest target_repo
codesage ask "how does the retry logic work?" --repo target_repo
```

## Architecture

| Module | Responsibility |
|---|---|
| `llm.py` | Only module that talks to Gemini. Wraps `generate` (with tool calling) and `embed`. |
| `tools.py` | `ToolRegistry` — a `dict[str, Tool]` the agent dispatches through by name. |
| `memory.py` | Bounded conversation history (`deque(maxlen=...)` sliding window). |
| `ingest.py` | Walks a repo, splits files into line-window chunks, embeds each one. |
| `index.py` | Brute-force cosine-similarity search over chunk vectors (`numpy`), plus JSON persistence between CLI runs. |
| `agent.py` | The agent loop, as an explicit state machine: `THINKING -> ACTING -> THINKING -> ... -> DONE` (or `ERROR` after `max_steps`). |
| `eval.py` | Retrieval hit-rate + answer-keyword scoring against a fixed case set (`eval_cases.json`). |
| `cli.py` | `codesage ingest`, `codesage ask`, `codesage eval`. |
| `api.py` | FastAPI wrapper over the same `Agent` — the deployed demo. |

## Why brute-force retrieval, not a vector DB

At the scale of one repo (hundreds to low thousands of chunks),
brute-force cosine similarity via `numpy` is O(n) per query and fast
enough. A real production system at millions of vectors would use an
ANN index (e.g. HNSW) — a deliberate simplification, not an oversight.

## Why no agent framework

LangGraph/CrewAI would hide the state machine, memory, and tool
dispatch behind framework classes. Building them by hand means every
concept here is something I can explain, not something I imported.

## Setup

```bash
uv sync --all-extras
cp .env.example .env   # add your GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
```

## Usage

```bash
uv run codesage ingest <path-to-a-repo>
uv run codesage ask "<question>" --repo <path-to-a-repo>
uv run codesage eval --repo <path-to-a-repo>   # requires eval_cases.json
```

## Testing

```bash
uv run pytest                              # unit tests, no API key needed
uv run pytest -m integration                # end-to-end, needs GEMINI_API_KEY
```

## Live demo

<!-- filled in after Task 11 -->

## Design docs

- [`docs/superpowers/specs/2026-08-18-codesage-agent-design.md`](docs/superpowers/specs/2026-08-18-codesage-agent-design.md)
- [`docs/superpowers/plans/2026-08-18-codesage-implementation.md`](docs/superpowers/plans/2026-08-18-codesage-implementation.md)
```

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures tests/test_integration.py README.md
git commit -m "test+docs: integration test against fixture repo, full README (Phase 7)"
git push
```

---

### Task 11: Phase 8 — FastAPI wrapper + deployment

**Concept taught:** separating a core library from its interface. The CLI and the API are two thin wrappers over the exact same `Agent`/`RetrievalIndex`/`ToolRegistry` classes — no logic is duplicated or reimplemented for the web.

**Files:**
- Create: `codesage/api.py`
- Create: `Dockerfile`
- Modify: `README.md` (fill in the live demo link)
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: a FastAPI `app` with `POST /ask {"question": str} -> {"answer": str}` and `GET /health -> {"status": "ok", "chunks_indexed": int}`.
- Consumes: `LLMClient`, `ingest_repo`, `RetrievalIndex`, `ToolRegistry`, `make_search_code_tool`, `Agent` — no new agent logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
from fastapi.testclient import TestClient

from codesage.api import create_app


def test_health_endpoint_reports_status_and_chunk_count():
    class FakeAgent:
        def ask(self, question):
            return "unused"

    app = create_app(agent=FakeAgent(), chunk_count=3)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks_indexed": 3}


def test_ask_endpoint_delegates_to_agent():
    class FakeAgent:
        def ask(self, question):
            return f"answer to: {question}"

    app = create_app(agent=FakeAgent(), chunk_count=1)
    client = TestClient(app)

    response = client.post("/ask", json={"question": "why?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "answer to: why?"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv sync --all-extras && uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: codesage.api`

- [ ] **Step 3: Write `codesage/api.py`**

```python
# codesage/api.py
"""FastAPI wrapper over the Agent — the deployed demo.

create_app() takes an already-constructed agent + chunk_count rather
than building them itself, so tests can inject a fake agent without a
real API key or a real repo on disk. main() below is what actually
runs in production: it does the real construction once at startup.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


def create_app(agent, chunk_count: int) -> FastAPI:
    app = FastAPI(title="CodeSage")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", chunks_indexed=chunk_count)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return AskResponse(answer=agent.ask(request.question))

    return app


def _build_production_app() -> FastAPI:
    from codesage.agent import Agent
    from codesage.ingest import ingest_repo
    from codesage.index import RetrievalIndex
    from codesage.llm import LLMClient
    from codesage.tools import make_search_code_tool
    from codesage.cli import build_base_registry

    api_key = os.environ["GEMINI_API_KEY"]
    target_repo = Path(os.environ.get("CODESAGE_TARGET_REPO", "./target_repo"))

    llm = LLMClient(api_key=api_key)
    chunks = ingest_repo(target_repo, llm)
    index = RetrievalIndex(chunks)

    registry = build_base_registry(target_repo)
    registry.register(make_search_code_tool(index, llm))
    agent = Agent(llm, tools=registry)

    return create_app(agent=agent, chunk_count=len(chunks))


app = _build_production_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Write the `Dockerfile`**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra api

COPY codesage/ codesage/
COPY target_repo/ target_repo/

ENV CODESAGE_TARGET_REPO=/app/target_repo

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "codesage.api:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 6: Test the container locally**

Run:
```bash
docker build -t codesage .
docker run -p 7860:7860 -e GEMINI_API_KEY=<your-key> codesage
```
In another terminal:
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"question": "how does the Session class work?"}'
```
Expected: `/health` returns a nonzero `chunks_indexed`; `/ask` returns a real, cited answer.

- [ ] **Step 7: Deploy to Hugging Face Spaces**

Free, no spin-down surprises, Docker SDK support out of the box. The Space is
a second git remote (separate from GitHub) that HF builds automatically on
push.

Create the Space and add it as a remote:
```bash
# requires: pip install huggingface_hub, then `huggingface-cli login` once
huggingface-cli repo create codesage --type space --space_sdk docker
git remote add space https://huggingface.co/spaces/<your-hf-username>/codesage
```

HF Spaces requires a specific YAML frontmatter block at the very top of
`README.md` to know how to build the Space. Prepend it (GitHub renders the
rest of the file normally and just ignores the unknown frontmatter, so this
is safe to keep in the GitHub copy too):

```yaml
---
title: CodeSage
emoji: 🧠
sdk: docker
app_port: 7860
---
```

In the Space's web settings (Settings → Repository secrets), add
`GEMINI_API_KEY` with your real key. Then push:

```bash
git push space main
```

Expected: the Space builds the Dockerfile and comes up at
`https://<your-hf-username>-codesage.hf.space`. Verify with:
```bash
curl https://<your-hf-username>-codesage.hf.space/health
```

- [ ] **Step 8: Fill in the README's live demo section**

```markdown
## Live demo

https://<your-hf-username>-codesage.hf.space

Try it:
\`\`\`bash
curl -X POST https://<your-hf-username>-codesage.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "how does the Session class handle retries?"}'
\`\`\`
```

- [ ] **Step 9: Commit and push to GitHub**

```bash
git add codesage/api.py tests/test_api.py Dockerfile README.md pyproject.toml uv.lock
git commit -m "feat: FastAPI wrapper + Hugging Face Spaces deployment (Phase 8)

api.py duplicates no agent logic — create_app() takes an already-built
Agent, same class the CLI uses. Deployed to HF Spaces (Docker SDK,
free tier) so the repo has a live link, not just a git clone."
git push
```

---

## After all 11 tasks

- Full commit history on `main` tells the build story phase by phase — worth linking directly in job applications ("see the commit history for how each piece was built").
- `README.md` has the live demo link, architecture table, and links to the spec + this plan.
- CI is green on every commit.
- Natural v2 ideas (explicitly out of scope here, worth mentioning in interviews as "what I'd do next"): multi-agent orchestration (a supervisor delegating to sub-agents), a persistent vector index for multi-repo support, streaming responses.
