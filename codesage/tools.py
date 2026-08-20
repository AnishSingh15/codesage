"""Tool registry: a hashmap mapping tool name -> callable + schema.

Adding a new tool never means editing the agent loop (agent.py) — that's
the Open/Closed Principle in practice. The registry is the seam.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google.genai import types

from codesage.callgraph import CallSite, find_callees, find_callers
from codesage.hierarchy import HierarchicalIndex
from codesage.index import RetrievalIndex
from codesage.ingest import SKIP_DIRS


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

    def call(self, name: str, /, **kwargs) -> str:
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


_MAX_TREE_ENTRIES = 200


def repo_tree_handler(base_dir: Path, subdir: str = ".") -> str:
    target = (base_dir / subdir).resolve()
    if not str(target).startswith(str(Path(base_dir).resolve())):
        return "Error: path escapes the target repo."
    if not target.exists():
        return f"Error: {subdir} does not exist."

    lines: list[str] = []
    count = 0
    truncated = False

    def walk(dir_path: Path, prefix: str) -> None:
        nonlocal count, truncated
        if truncated:
            return
        entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for entry in entries:
            if entry.is_dir() and entry.name in SKIP_DIRS:
                continue
            if count >= _MAX_TREE_ENTRIES:
                truncated = True
                return
            lines.append(f"{prefix}{entry.name}{'/' if entry.is_dir() else ''}")
            count += 1
            if entry.is_dir():
                walk(entry, prefix + "  ")

    walk(target, "")
    if truncated:
        lines.append("... (truncated)")
    return "\n".join(lines) if lines else "(empty directory)"


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


def make_hierarchical_search_tool(index: HierarchicalIndex, llm) -> Tool:
    def handler(query: str) -> str:
        results = index.search(query, llm, k=5)
        if not results:
            return "No matching code found."
        return "\n\n".join(f"{c.file_path}:{c.line_start}-{c.line_end}\n{c.text}" for c in results)

    return Tool(
        name="search_code",
        description=(
            "Structure-based search over the ingested codebase (navigates a "
            "table of contents via reasoning, not embeddings). Returns "
            "relevant code chunks with file/line citations."
        ),
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


def make_find_callers_tool(call_sites: list[CallSite]) -> Tool:
    def handler(name: str) -> str:
        results = find_callers(call_sites, name)
        if not results:
            return f"No callers found for '{name}'."
        return "\n".join(f"{cs.caller_file}:{cs.caller_line} (inside {cs.caller_name})" for cs in results)

    return Tool(
        name="find_callers",
        description=(
            "Find every call site across the repo where a function or method "
            "named X is called. Name-based: matches by simple name, not a "
            "fully resolved reference."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The function or method name to find callers of"}},
            "required": ["name"],
        },
        handler=handler,
    )


def make_find_callees_tool(call_sites: list[CallSite]) -> Tool:
    def handler(name: str) -> str:
        results = find_callees(call_sites, name)
        if not results:
            return f"No calls found inside '{name}'."
        return "\n".join(f"{cs.called_name} (at {cs.caller_file}:{cs.caller_line})" for cs in results)

    return Tool(
        name="find_callees",
        description="Find every function/method call made inside a given function or method, across the repo.",
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The function or method name to find calls made from"}},
            "required": ["name"],
        },
        handler=handler,
    )
