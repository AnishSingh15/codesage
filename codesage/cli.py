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
    ingest_parser.add_argument(
        "--delay", type=float, default=0.5, help="Seconds to sleep between embed calls (rate-limit mitigation)"
    )

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--repo", default=".", help="Target repo path")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--repo", default=".", help="Target repo path")
    eval_parser.add_argument("--cases", default="eval_cases.json", help="Path to eval cases JSON")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    llm = LLMClient(api_key=api_key)

    if args.command == "ingest":
        from codesage.ingest import ingest_repo

        repo_path = Path(args.repo_path)
        chunks = ingest_repo(repo_path, llm, delay_seconds=args.delay)
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

        results = run_eval(lambda: Agent(llm, tools=registry), index, llm, load_cases(cases_path))
        print(f"Retrieval hit rate: {results['retrieval_hit_rate']:.0%}")
        print(f"Avg answer score:   {results['avg_answer_score']:.0%}")


if __name__ == "__main__":
    main()
