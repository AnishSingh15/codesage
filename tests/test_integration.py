"""End-to-end test against the real Gemini API.

Excluded from the default pytest run (see pyproject.toml addopts).
Run explicitly with: uv run pytest tests/test_integration.py -m integration -v
Requires GEMINI_API_KEY in the environment.
"""

import os
from pathlib import Path

import pytest

from codesage.agent import Agent
from codesage.index import RetrievalIndex
from codesage.ingest import ingest_repo
from codesage.llm import LLMClient
from codesage.tools import build_base_registry, make_search_code_tool

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "tiny_repo"


@pytest.mark.integration
def test_agent_answers_question_about_fixture_repo_with_citation():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    llm = LLMClient(api_key=api_key)
    chunks = ingest_repo(FIXTURE_REPO, llm, delay_seconds=0)
    index = RetrievalIndex(chunks)

    registry = build_base_registry(FIXTURE_REPO)
    registry.register(make_search_code_tool(index, llm))
    agent = Agent(llm, tools=registry)

    answer = agent.ask("What does the is_even function do?")

    assert "even" in answer.lower()
