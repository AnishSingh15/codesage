from pathlib import Path

import pytest

from codesage.callgraph import (
    CallSite,
    build_call_graph,
    find_callees,
    find_callers,
    load_call_graph,
    save_call_graph,
)

TARGET_REPO_SRC = Path(__file__).parent.parent / "target_repo" / "src"


def test_build_call_graph_finds_direct_calls(tmp_path: Path):
    (tmp_path / "mod.py").write_text("def foo():\n    bar()\n")

    call_sites = build_call_graph(tmp_path)

    assert any(cs.called_name == "bar" and cs.caller_name == "foo" for cs in call_sites)


def test_build_call_graph_finds_method_calls(tmp_path: Path):
    (tmp_path / "mod.py").write_text(
        "class Foo:\n"
        "    def bar(self):\n"
        "        self.baz()\n"
        "        other.qux()\n"
    )

    call_sites = build_call_graph(tmp_path)

    names = {cs.called_name for cs in call_sites if cs.caller_name == "bar"}
    assert "baz" in names
    assert "qux" in names


def test_build_call_graph_finds_nested_call_in_arguments(tmp_path: Path):
    (tmp_path / "mod.py").write_text("def foo():\n    outer(inner())\n")

    call_sites = build_call_graph(tmp_path)

    names = {cs.called_name for cs in call_sites if cs.caller_name == "foo"}
    assert "outer" in names
    assert "inner" in names


def test_build_call_graph_records_module_level_calls(tmp_path: Path):
    (tmp_path / "mod.py").write_text("setup()\n")

    call_sites = build_call_graph(tmp_path)

    assert any(cs.called_name == "setup" and cs.caller_name == "<module>" for cs in call_sites)


def test_build_call_graph_skips_files_with_syntax_errors(tmp_path: Path):
    (tmp_path / "broken.py").write_text("def foo(:\n    pass")
    (tmp_path / "good.py").write_text("def bar():\n    baz()\n")

    call_sites = build_call_graph(tmp_path)

    assert any(cs.called_name == "baz" for cs in call_sites)
    assert not any(cs.caller_file == "broken.py" for cs in call_sites)


def test_find_callers_filters_by_called_name():
    call_sites = [
        CallSite(caller_name="a", caller_file="x.py", caller_line=1, called_name="target"),
        CallSite(caller_name="b", caller_file="y.py", caller_line=2, called_name="other"),
    ]

    assert find_callers(call_sites, "target") == [call_sites[0]]


def test_find_callees_filters_by_caller_name():
    call_sites = [
        CallSite(caller_name="a", caller_file="x.py", caller_line=1, called_name="target"),
        CallSite(caller_name="b", caller_file="y.py", caller_line=2, called_name="other"),
    ]

    assert find_callees(call_sites, "a") == [call_sites[0]]


def test_save_and_load_call_graph_round_trip(tmp_path: Path):
    call_sites = [CallSite(caller_name="a", caller_file="x.py", caller_line=1, called_name="target")]
    path = tmp_path / "graph.json"

    save_call_graph(call_sites, path)
    loaded = load_call_graph(path)

    assert loaded == call_sites


def test_find_callers_against_real_repo_finds_mount_calls():
    # Not gated behind -m integration: this needs no API key, the whole
    # point of this feature is that it's LLM-free. Only needs target_repo/src
    # to exist locally (same skip pattern test_integration.py uses).
    if not TARGET_REPO_SRC.exists():
        pytest.skip("target_repo/src not present locally")

    call_sites = build_call_graph(TARGET_REPO_SRC)
    results = find_callers(call_sites, "mount")

    callers_in_init = [
        cs for cs in results if cs.caller_name == "__init__" and cs.caller_file == "requests/sessions.py"
    ]
    assert len(callers_in_init) >= 2
