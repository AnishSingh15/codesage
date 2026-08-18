import json
from pathlib import Path

from codesage.eval import EvalCase, load_cases, score_answer, score_retrieval, run_eval
from codesage.ingest import Chunk
from codesage.index import RetrievalIndex


def test_load_cases_parses_json(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"question": "q1", "expected_file_substring": "auth.py", "expected_answer_keywords": ["token"]}
    ]))

    cases = load_cases(path)

    assert cases == [EvalCase(question="q1", expected_file_substring="auth.py", expected_answer_keywords=["token"])]


def test_score_answer_is_fraction_of_keywords_present():
    case = EvalCase(question="q", expected_file_substring="x", expected_answer_keywords=["token", "refresh"])

    assert score_answer("uses a refresh token internally", case) == 1.0
    assert score_answer("uses nothing relevant", case) == 0.0
    assert score_answer("uses a token only", case) == 0.5


def test_score_retrieval_true_when_expected_file_is_returned():
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = RetrievalIndex([chunk])
    case = EvalCase(question="q", expected_file_substring="auth.py", expected_answer_keywords=[])

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [1.0, 0.0]

    assert score_retrieval(index, FakeLLM(), case) is True


def test_run_eval_aggregates_scores():
    case = EvalCase(question="q", expected_file_substring="auth.py", expected_answer_keywords=["token"])
    chunk = Chunk(text="auth code", file_path="src/auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    index = RetrievalIndex([chunk])

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [1.0, 0.0]

    class FakeAgent:
        def ask(self, question):
            return "returns a token"

    results = run_eval(FakeAgent(), index, FakeLLM(), [case])

    assert results == {"retrieval_hit_rate": 1.0, "avg_answer_score": 1.0}
