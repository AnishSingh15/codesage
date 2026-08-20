"""Tool registry: a hashmap mapping tool name -> callable + schema.

Adding a new tool never means editing the agent loop (agent.py) — that's
the Open/Closed Principle in practice. The registry is the seam.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google.genai import types

from codesage.index import RetrievalIndex


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
