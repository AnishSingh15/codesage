from types import SimpleNamespace

from codesage.agent import Agent


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
