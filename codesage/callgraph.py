"""Cross-file call-graph traversal: "who calls X" and "what does X call",
built with zero LLM calls (pure static analysis, same philosophy as
hierarchy.py) and answered with zero LLM calls too (exact filtering, not
reasoning).

Name-based, not import-resolved: a call site records the simple name being
called ("foo" for foo() and for self.foo()/obj.foo()), not a fully
resolved reference to one specific definition. Two unrelated classes that
both define close() will both show up as "callers of close" — a real,
stated tradeoff, not a hidden gap. Import resolution is what tools like
Jedi exist for; it's out of scope here on purpose.
"""

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from codesage.ingest import iter_source_files

CALLGRAPH_FILENAME = ".codesage_callgraph.json"


@dataclass
class CallSite:
    caller_name: str
    caller_file: str
    caller_line: int
    called_name: str


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.call_sites: list[CallSite] = []
        self._stack: list[str] = ["<module>"]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        called_name = None
        if isinstance(node.func, ast.Name):
            called_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr

        if called_name:
            self.call_sites.append(
                CallSite(
                    caller_name=self._stack[-1],
                    caller_file=self.file_path,
                    caller_line=node.lineno,
                    called_name=called_name,
                )
            )
        self.generic_visit(node)


def build_call_graph(repo_path: Path) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for file_path in iter_source_files(repo_path):
        if file_path.suffix != ".py":
            continue
        relative_path = str(file_path.relative_to(repo_path))
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        visitor = _CallVisitor(relative_path)
        visitor.visit(tree)
        call_sites.extend(visitor.call_sites)
    return call_sites


def find_callers(call_sites: list[CallSite], name: str) -> list[CallSite]:
    return [cs for cs in call_sites if cs.called_name == name]


def find_callees(call_sites: list[CallSite], name: str) -> list[CallSite]:
    return [cs for cs in call_sites if cs.caller_name == name]


def save_call_graph(call_sites: list[CallSite], path: Path) -> None:
    path.write_text(json.dumps([asdict(cs) for cs in call_sites]))


def load_call_graph(path: Path) -> list[CallSite]:
    data = json.loads(path.read_text())
    return [CallSite(**d) for d in data]
