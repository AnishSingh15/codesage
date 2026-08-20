"""CLI entrypoint. Phase 4b: adds `ingest`; `ask` uses the index if present."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from codesage.agent import Agent
from codesage.llm import LLMClient
from codesage.tools import build_base_registry, make_search_code_tool
from codesage.index import HIERARCHY_INDEX_FILENAME, INDEX_FILENAME, RetrievalIndex, load_chunks, save_chunks


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="codesage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("repo_path")
    ingest_parser.add_argument(
        "--delay", type=float, default=0.5, help="Seconds to sleep between embed calls (rate-limit mitigation)"
    )

    ingest_hierarchy_parser = subparsers.add_parser("ingest-hierarchy")
    ingest_hierarchy_parser.add_argument("repo_path")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--repo", default=".", help="Target repo path")
    ask_parser.add_argument(
        "--strategy", choices=["vector", "hierarchy"], default="vector", help="Retrieval strategy"
    )

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--repo", default=".", help="Target repo path")
    eval_parser.add_argument("--cases", default="eval_cases.json", help="Path to eval cases JSON")
    eval_parser.add_argument(
        "--compare", action="store_true", help="Run against both retrieval strategies and compare"
    )

    onboard_parser = subparsers.add_parser("onboard")
    onboard_parser.add_argument("--repo", default=".", help="Target repo path")

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

    elif args.command == "ingest-hierarchy":
        from codesage.hierarchy import build_hierarchy_chunks

        repo_path = Path(args.repo_path)
        chunks = build_hierarchy_chunks(repo_path)
        save_chunks(chunks, repo_path / HIERARCHY_INDEX_FILENAME)
        print(f"Indexed {len(chunks)} chunks from {repo_path} (hierarchical, no API calls)")

    elif args.command == "ask":
        repo_path = Path(args.repo)
        registry = build_base_registry(repo_path)

        if args.strategy == "vector":
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
        else:
            from codesage.hierarchy import HierarchicalIndex
            from codesage.tools import make_hierarchical_search_tool

            hierarchy_index_path = repo_path / HIERARCHY_INDEX_FILENAME
            if hierarchy_index_path.exists():
                chunks = load_chunks(hierarchy_index_path)
                hierarchy_index = HierarchicalIndex(chunks)
                registry.register(make_hierarchical_search_tool(hierarchy_index, llm))
            else:
                print(
                    f"(no hierarchy index found at {hierarchy_index_path} — run "
                    f"'codesage ingest-hierarchy {repo_path}' for retrieval-backed answers; "
                    "continuing with list_files/read_file only)",
                    file=sys.stderr,
                )

        agent = Agent(llm, tools=registry)
        print(agent.ask(args.question))

    elif args.command == "eval":
        from codesage.eval import load_cases, run_eval
        from codesage.index import VectorRetrievalStrategy

        repo_path = Path(args.repo)
        cases_path = Path(args.cases)
        if not cases_path.exists():
            print(f"No eval cases found at {cases_path}. See eval_cases.json for the format.", file=sys.stderr)
            sys.exit(1)
        cases = load_cases(cases_path)

        if args.compare:
            from codesage.hierarchy import HierarchicalIndex
            from codesage.tools import make_hierarchical_search_tool

            index_path = repo_path / INDEX_FILENAME
            hierarchy_index_path = repo_path / HIERARCHY_INDEX_FILENAME
            if not index_path.exists():
                print(f"No vector index found. Run 'codesage ingest {repo_path}' first.", file=sys.stderr)
                sys.exit(1)
            if not hierarchy_index_path.exists():
                print(
                    f"No hierarchy index found. Run 'codesage ingest-hierarchy {repo_path}' first.",
                    file=sys.stderr,
                )
                sys.exit(1)

            vector_index = RetrievalIndex(load_chunks(index_path))
            vector_registry = build_base_registry(repo_path)
            vector_registry.register(make_search_code_tool(vector_index, llm))
            vector_results = run_eval(
                lambda: Agent(llm, tools=vector_registry), VectorRetrievalStrategy(vector_index), llm, cases
            )

            hierarchy_index = HierarchicalIndex(load_chunks(hierarchy_index_path))
            hierarchy_registry = build_base_registry(repo_path)
            hierarchy_registry.register(make_hierarchical_search_tool(hierarchy_index, llm))
            hierarchy_results = run_eval(
                lambda: Agent(llm, tools=hierarchy_registry), hierarchy_index, llm, cases
            )

            print(f"{'Strategy':<12}{'Retrieval hit rate':<22}{'Avg answer score':<20}")
            print(
                f"{'vector':<12}"
                f"{vector_results['retrieval_hit_rate']:<22.0%}"
                f"{vector_results['avg_answer_score']:<20.0%}"
            )
            print(
                f"{'hierarchy':<12}"
                f"{hierarchy_results['retrieval_hit_rate']:<22.0%}"
                f"{hierarchy_results['avg_answer_score']:<20.0%}"
            )

        else:
            index_path = repo_path / INDEX_FILENAME
            if not index_path.exists():
                print(f"No index found. Run 'codesage ingest {repo_path}' first.", file=sys.stderr)
                sys.exit(1)

            chunks = load_chunks(index_path)
            index = RetrievalIndex(chunks)
            registry = build_base_registry(repo_path)
            registry.register(make_search_code_tool(index, llm))

            results = run_eval(lambda: Agent(llm, tools=registry), VectorRetrievalStrategy(index), llm, cases)
            print(f"Retrieval hit rate: {results['retrieval_hit_rate']:.0%}")
            print(f"Avg answer score:   {results['avg_answer_score']:.0%}")

    elif args.command == "onboard":
        from codesage.supervisor import generate_onboarding_doc

        repo_path = Path(args.repo)
        doc = generate_onboarding_doc(repo_path, llm)
        output_path = repo_path / "ONBOARDING.md"
        output_path.write_text(doc)
        print(f"Wrote onboarding doc to {output_path}")


if __name__ == "__main__":
    main()
