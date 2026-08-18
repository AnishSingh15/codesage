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
