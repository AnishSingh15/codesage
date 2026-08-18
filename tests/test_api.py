from fastapi.testclient import TestClient

from codesage.api import create_app


def test_health_endpoint_reports_status_and_chunk_count():
    class FakeAgent:
        def ask(self, question):
            return "unused"

    app = create_app(agent_factory=lambda: FakeAgent(), chunk_count=3)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks_indexed": 3}


def test_ask_endpoint_delegates_to_agent():
    class FakeAgent:
        def ask(self, question):
            return f"answer to: {question}"

    app = create_app(agent_factory=lambda: FakeAgent(), chunk_count=1)
    client = TestClient(app)

    response = client.post("/ask", json={"question": "why?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "answer to: why?"}


def test_ask_endpoint_builds_a_fresh_agent_per_request():
    # Regression test: two /ask requests are two independent questions,
    # not a conversation. Sharing one Agent's memory across requests
    # mixed unrelated context and could corrupt the turn sequence outright
    # (mirrors the same bug fixed in eval.py's run_eval).
    factory_calls = []

    class FakeAgent:
        def ask(self, question):
            return f"answer to: {question}"

    def agent_factory():
        factory_calls.append(1)
        return FakeAgent()

    app = create_app(agent_factory=agent_factory, chunk_count=1)
    client = TestClient(app)

    client.post("/ask", json={"question": "first?"})
    client.post("/ask", json={"question": "second?"})

    assert len(factory_calls) == 2
