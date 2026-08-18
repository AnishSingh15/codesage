"""The agent: an observe -> think -> act loop.

Phase 2: adds tool calling. The loop below is intentionally simple (a
while loop with a hard iteration cap) — a later refactor replaces it with
an explicit state machine once there's enough behavior to justify one.
"""

from google.genai import types

from codesage.tools import ToolRegistry

_SAFETY_MAX_ITERATIONS = 5


class Agent:
    def __init__(self, llm, tools: ToolRegistry | None = None):
        self._llm = llm
        self._tools = tools or ToolRegistry()

    def ask(self, question: str) -> str:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        tool = self._tools.as_tool() if self._tools.has_tools() else None

        response = self._llm.generate(contents, tools=[tool] if tool else None)
        iterations = 0

        while response.function_calls and iterations < _SAFETY_MAX_ITERATIONS:
            iterations += 1
            call = response.function_calls[0]
            contents.append(response.candidates[0].content)

            try:
                result = self._tools.call(call.name, **call.args)
                function_response = {"result": result}
            except Exception as exc:
                function_response = {"error": str(exc)}

            contents.append(
                types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=call.name, response=function_response)],
                )
            )
            response = self._llm.generate(contents, tools=[tool] if tool else None)

        return response.text
