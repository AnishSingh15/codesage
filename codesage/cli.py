"""CLI entrypoint. Phase 1: just `codesage ask`."""

import argparse
import os
import sys

from dotenv import load_dotenv

from codesage.agent import Agent
from codesage.llm import LLMClient


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="codesage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    llm = LLMClient(api_key=api_key)

    if args.command == "ask":
        agent = Agent(llm)
        print(agent.ask(args.question))


if __name__ == "__main__":
    main()
