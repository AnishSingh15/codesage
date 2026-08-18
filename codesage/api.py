"""FastAPI wrapper over the Agent — the deployed demo.

create_app() takes an agent_factory rather than a single shared agent,
so tests can inject a fake without a real API key, and so each HTTP
request gets its own Agent (own Memory). Two unrelated /ask requests
sharing one Agent's memory mixes unrelated conversations and can
corrupt the turn sequence outright — the same bug eval.py's run_eval
had, fixed the same way. get_app() is what actually runs in production:
it does the real, expensive construction (tools, index, LLM client)
once at startup, but the Agent itself is built fresh per request.
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


def create_app(agent_factory, chunk_count: int) -> FastAPI:
    app = FastAPI(title="CodeSage")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", chunks_indexed=chunk_count)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        agent = agent_factory()
        return AskResponse(answer=agent.ask(request.question))

    return app


def get_app() -> FastAPI:
    """Production entrypoint.

    Builds the real Agent from environment variables. Deliberately NOT
    called at import time — importing this module (e.g. for `create_app`
    in tests) must have zero side effects: no API key required, no
    network calls. uvicorn calls this lazily via `--factory` (see
    Dockerfile), so the expensive work only happens when the server
    actually starts.

    Reuses a persisted index (built via `codesage ingest` and baked into
    the image alongside the target repo) instead of re-ingesting on every
    startup — matches the CLI's `ask` behavior, and means restarting or
    scaling the deployed service doesn't re-spend embedding-API quota
    every time.
    """
    from codesage.agent import Agent
    from codesage.cli import build_base_registry
    from codesage.index import INDEX_FILENAME, RetrievalIndex, load_chunks
    from codesage.ingest import ingest_repo
    from codesage.llm import LLMClient
    from codesage.tools import make_search_code_tool

    api_key = os.environ["GEMINI_API_KEY"]
    target_repo = Path(os.environ.get("CODESAGE_TARGET_REPO", "./target_repo"))

    llm = LLMClient(api_key=api_key)

    index_path = target_repo / INDEX_FILENAME
    if index_path.exists():
        chunks = load_chunks(index_path)
    else:
        chunks = ingest_repo(target_repo, llm)
    index = RetrievalIndex(chunks)

    registry = build_base_registry(target_repo)
    registry.register(make_search_code_tool(index, llm))

    return create_app(agent_factory=lambda: Agent(llm, tools=registry), chunk_count=len(chunks))
