from types import SimpleNamespace

import pytest

from codesage.llm import LLMClient


class FakeModels:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []

    def generate_content(self, model, contents, config=None):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="mocked answer", function_calls=None)

    def embed_content(self, model, contents, config=None):
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])


class FakeGenaiClient:
    def __init__(self):
        self.models = FakeModels()


def test_generate_delegates_to_client_and_returns_response():
    fake_client = FakeGenaiClient()
    llm = LLMClient(client=fake_client)

    response = llm.generate(contents=["some content"])

    assert response.text == "mocked answer"
    assert fake_client.models.generate_calls[0]["model"] == "gemini-2.5-flash"
    assert fake_client.models.generate_calls[0]["contents"] == ["some content"]


def test_embed_returns_vector_of_floats():
    fake_client = FakeGenaiClient()
    llm = LLMClient(client=fake_client)

    vector = llm.embed("why is the sky blue?")

    assert vector == [0.1, 0.2, 0.3]
    assert fake_client.models.embed_calls[0]["model"] == "gemini-embedding-001"


def test_generate_retries_once_on_transient_error_then_succeeds():
    fake_client = FakeGenaiClient()
    call_count = {"n": 0}
    real_generate_content = fake_client.models.generate_content

    def flaky_generate_content(model, contents, config=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("simulated transient failure")
        return real_generate_content(model, contents, config)

    fake_client.models.generate_content = flaky_generate_content
    llm = LLMClient(client=fake_client)

    response = llm.generate(contents=["some content"])

    assert response.text == "mocked answer"
    assert call_count["n"] == 2


def test_generate_raises_clear_error_after_retry_also_fails():
    fake_client = FakeGenaiClient()

    def always_fails(model, contents, config=None):
        raise TimeoutError("simulated persistent failure")

    fake_client.models.generate_content = always_fails
    llm = LLMClient(client=fake_client)

    with pytest.raises(RuntimeError, match="Gemini API call failed after 1 retry"):
        llm.generate(contents=["some content"])
