"""End-to-end test against the real Gemini API.

Excluded from the default pytest run (see pyproject.toml addopts).
Run explicitly with: uv run pytest tests/test_integration.py -m integration -v
Requires GEMINI_API_KEY in the environment.
"""

import os
from pathlib import Path

import pytest

from codesage.agent import Agent
from codesage.hierarchy import HierarchicalIndex, build_hierarchy_chunks
from codesage.index import RetrievalIndex
from codesage.ingest import ingest_repo
from codesage.llm import LLMClient
from codesage.supervisor import generate_onboarding_doc
from codesage.tools import build_base_registry, make_hierarchical_search_tool, make_search_code_tool

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "tiny_repo"
TARGET_REPO_SRC = Path(__file__).parent.parent / "target_repo" / "src"


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


@pytest.mark.integration
def test_generate_onboarding_doc_against_real_repo():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    if not TARGET_REPO_SRC.exists():
        pytest.skip("target_repo/src not present locally")

    llm = LLMClient(api_key=api_key)
    doc = generate_onboarding_doc(TARGET_REPO_SRC, llm)

    assert "requests" in doc.lower()


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
