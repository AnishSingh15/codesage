from pathlib import Path
from types import SimpleNamespace

import pytest

from codesage.index import INDEX_FILENAME, save_chunks
from codesage.ingest import Chunk
from codesage.supervisor import build_explorer_registry, generate_onboarding_doc


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, contents, tools=None):
        self.calls.append({"contents": contents, "tools": tools})
        return self._responses.pop(0)


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_calls=None)


def test_generate_onboarding_doc_runs_three_stages_in_order(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1")

    fake_llm = FakeLLM([
        _text_response("STRUCTURE SUMMARY"),
        _text_response("CODE SUMMARY"),
        _text_response("# Final Doc"),
    ])

    doc = generate_onboarding_doc(tmp_path, fake_llm)

    assert doc == "# Final Doc"
    assert len(fake_llm.calls) == 3

    explorer_prompt = fake_llm.calls[1]["contents"][0].parts[0].text
    assert "STRUCTURE SUMMARY" in explorer_prompt

    writer_prompt = fake_llm.calls[2]["contents"][0].parts[0].text
    assert "STRUCTURE SUMMARY" in writer_prompt
    assert "CODE SUMMARY" in writer_prompt


def test_build_explorer_registry_includes_search_code_when_index_exists(tmp_path: Path):
    chunk = Chunk(text="auth code", file_path="auth.py", line_start=1, line_end=5, vector=[1.0, 0.0])
    save_chunks([chunk], tmp_path / INDEX_FILENAME)

    registry = build_explorer_registry(tmp_path, llm=FakeLLM([]))

    assert registry.get("search_code") is not None


def test_build_explorer_registry_omits_search_code_without_index(tmp_path: Path):
    registry = build_explorer_registry(tmp_path, llm=FakeLLM([]))

    with pytest.raises(KeyError):
        registry.get("search_code")
