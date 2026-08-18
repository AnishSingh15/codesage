"""The agent: an observe -> think -> act loop.

Phase 3: conversation state moves from a local list into Memory, so
Agent.ask is now genuinely multi-turn if you call it more than once on
the same Agent instance.
"""

from google.genai import types

from codesage.memory import Memory
from codesage.tools import ToolRegistry

_SAFETY_MAX_ITERATIONS = 5


class Agent:
    def __init__(self, llm, tools: ToolRegistry | None = None, memory: Memory | None = None):
        self._llm = llm
        self._tools = tools or ToolRegistry()
        self._memory = memory or Memory()

    def ask(self, question: str) -> str:
        self._memory.add(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        tool = self._tools.as_tool() if self._tools.has_tools() else None

        response = self._llm.generate(self._memory.as_contents(), tools=[tool] if tool else None)
        iterations = 0

        while response.function_calls and iterations < _SAFETY_MAX_ITERATIONS:
            iterations += 1
            call = response.function_calls[0]
            self._memory.add(response.candidates[0].content)

            try:
                result = self._tools.call(call.name, **call.args)
                function_response = {"result": result}
            except Exception as exc:
                function_response = {"error": str(exc)}

            self._memory.add(
                types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=call.name, response=function_response)],
                )
            )
            response = self._llm.generate(self._memory.as_contents(), tools=[tool] if tool else None)

        self._memory.add(
            response.candidates[0].content
            if getattr(response, "candidates", None)
            else types.Content(role="model", parts=[types.Part.from_text(text=response.text or "")])
        )
        return response.text
