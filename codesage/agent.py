"""The agent: an observe -> think -> act loop.

Phase 1: no tools, no memory yet — just enough to prove the LLM round-trip
works end to end. Tool use lands next, memory after that, the full state
machine once there's enough behavior to justify one.
"""

from google.genai import types


class Agent:
    def __init__(self, llm):
        self._llm = llm

    def ask(self, question: str) -> str:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        response = self._llm.generate(contents)
        return response.text
