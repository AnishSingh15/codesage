from types import SimpleNamespace

from codesage.agent import Agent
from codesage.tools import Tool, ToolRegistry


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, contents, tools=None):
        self.calls.append({"contents": contents, "tools": tools})
        return self._responses.pop(0)


def test_ask_sends_question_and_returns_model_text():
    fake_llm = FakeLLM([SimpleNamespace(text="42", function_calls=None)])
    agent = Agent(llm=fake_llm)

    answer = agent.ask("what is the answer?")

    assert answer == "42"
    assert len(fake_llm.calls) == 1


def test_ask_executes_a_tool_call_then_returns_final_answer():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="looks something up",
            parameters_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=lambda q: f"result for {q}",
        )
    )

    tool_call_response = SimpleNamespace(
        text=None,
        function_calls=[SimpleNamespace(name="lookup", args={"q": "foo"})],
        candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
    )
    final_response = SimpleNamespace(text="here's your answer", function_calls=None)

    fake_llm = FakeLLM([tool_call_response, final_response])
    agent = Agent(llm=fake_llm, tools=registry)

    answer = agent.ask("look up foo")

    assert answer == "here's your answer"
    assert len(fake_llm.calls) == 2
