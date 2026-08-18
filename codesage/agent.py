"""The agent: an explicit state machine over observe -> think -> act.

Phase 5: the implicit while-loop from Phase 2/3 becomes explicit states.
This is the same graph either way — making it explicit just means you
can answer "can this loop forever?" by reading the transition table
instead of tracing execution.
"""

from enum import Enum, auto

from google.genai import types

from codesage.memory import Memory
from codesage.tools import ToolRegistry


class AgentState(Enum):
    THINKING = auto()
    ACTING = auto()
    DONE = auto()
    ERROR = auto()


class Agent:
    def __init__(
        self,
        llm,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
        max_steps: int = 8,
    ):
        self._llm = llm
        self._tools = tools or ToolRegistry()
        self._memory = memory or Memory()
        self._max_steps = max_steps

    def ask(self, question: str) -> str:
        self._memory.add(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

        state = AgentState.THINKING
        response = None
        steps = 0

        while state in (AgentState.THINKING, AgentState.ACTING):
            if state is AgentState.THINKING:
                steps += 1
                if steps > self._max_steps:
                    state = AgentState.ERROR
                    break
                response, state = self._think()
            elif state is AgentState.ACTING:
                state = self._act(response)

        if state is AgentState.ERROR:
            return f"I couldn't finish after {self._max_steps} steps trying to answer: {question}"

        return response.text

    def _think(self):
        tool = self._tools.as_tool() if self._tools.has_tools() else None
        response = self._llm.generate(self._memory.as_contents(), tools=[tool] if tool else None)
        next_state = AgentState.ACTING if response.function_calls else AgentState.DONE
        if next_state is AgentState.DONE:
            self._memory.add(
                response.candidates[0].content
                if getattr(response, "candidates", None)
                else types.Content(role="model", parts=[types.Part.from_text(text=response.text or "")])
            )
        return response, next_state

    def _act(self, response) -> AgentState:
        call = response.function_calls[0]
        self._memory.add(response.candidates[0].content)

        try:
            result = self._tools.call(call.name, **call.args)
            function_response = {"result": result}
        except Exception as exc:
            function_response = {"error": str(exc)}

        self._memory.add(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=call.name, response=function_response)],
            )
        )
        return AgentState.THINKING
