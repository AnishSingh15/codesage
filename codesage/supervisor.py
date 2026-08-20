"""Multi-agent pipeline: structure mapper -> code explorer -> writer.

A "role" here is just which tools an agent gets and what task text it's
asked — every stage is a plain Agent instance, the same class used for
`codesage ask`. Multi-agent composes an already-tested primitive; it
doesn't introduce a new one.
"""

from pathlib import Path

from codesage.agent import Agent
from codesage.index import INDEX_FILENAME, RetrievalIndex, load_chunks
from codesage.tools import (
    Tool,
    ToolRegistry,
    build_base_registry,
    make_search_code_tool,
    repo_tree_handler,
)

STRUCTURE_PROMPT = (
    "Explore this repository's directory layout using the available tools. "
    "Produce a concise plain-text summary covering: the overall structure, "
    "the main modules/packages, and likely entry points (e.g. a CLI, an "
    "__init__.py, a main script). Do not read every file in depth — this "
    "is a map, not a deep analysis."
)

EXPLORER_PROMPT_TEMPLATE = (
    "Here is a summary of this repository's structure:\n\n{structure_summary}\n\n"
    "Using that as a guide, explore the key modules and explain what they "
    "actually do — the core logic, important classes/functions, and how "
    "the pieces fit together. Focus on what someone would need to "
    "understand before making their first change."
)

WRITER_PROMPT_TEMPLATE = (
    "Write a single onboarding markdown document for this repository, "
    "combining two things: (1) orientation — what this repo is, how it's "
    "organized, where to start reading; and (2) contribution guidance — "
    "how to set it up, how to run tests, where common changes tend to "
    "happen. Use the research below. Output only the markdown document, "
    "no commentary about the task itself.\n\n"
    "## Structure summary\n{structure_summary}\n\n"
    "## Code summary\n{code_summary}"
)

_STRUCTURE_MAX_STEPS = 6
_EXPLORER_MAX_STEPS = 10
_WRITER_MAX_STEPS = 3


def build_structure_registry(repo_path: Path) -> ToolRegistry:
    registry = build_base_registry(repo_path)
    registry.register(
        Tool(
            name="repo_tree",
            description="Show a recursive directory tree of the target repo, relative to its root.",
            parameters_schema={
                "type": "object",
                "properties": {"subdir": {"type": "string", "description": "Relative subdirectory, use '.' for root"}},
                "required": ["subdir"],
            },
            handler=lambda subdir: repo_tree_handler(repo_path, subdir),
        )
    )
    return registry


def build_explorer_registry(repo_path: Path, llm) -> ToolRegistry:
    registry = build_base_registry(repo_path)
    index_path = repo_path / INDEX_FILENAME
    if index_path.exists():
        chunks = load_chunks(index_path)
        index = RetrievalIndex(chunks)
        registry.register(make_search_code_tool(index, llm))
    return registry


def generate_onboarding_doc(repo_path: Path, llm) -> str:
    structure_agent = Agent(llm, tools=build_structure_registry(repo_path), max_steps=_STRUCTURE_MAX_STEPS)
    structure_summary = structure_agent.ask(STRUCTURE_PROMPT)

    explorer_agent = Agent(llm, tools=build_explorer_registry(repo_path, llm), max_steps=_EXPLORER_MAX_STEPS)
    code_summary = explorer_agent.ask(EXPLORER_PROMPT_TEMPLATE.format(structure_summary=structure_summary))

    writer_agent = Agent(llm, max_steps=_WRITER_MAX_STEPS)
    return writer_agent.ask(
        WRITER_PROMPT_TEMPLATE.format(structure_summary=structure_summary, code_summary=code_summary)
    )
