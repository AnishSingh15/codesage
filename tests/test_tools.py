from pathlib import Path

import pytest

from codesage.tools import Tool, ToolRegistry, list_files_handler, read_file_handler


def test_register_and_call_a_tool():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echoes input",
            parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda text: text,
        )
    )

    assert registry.call("echo", text="hi") == "hi"


def test_call_unknown_tool_raises_key_error():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.call("nope")


def test_has_tools_reflects_registration_state():
    registry = ToolRegistry()
    assert registry.has_tools() is False
    registry.register(
        Tool(name="a", description="", parameters_schema={"type": "object"}, handler=lambda: "")
    )
    assert registry.has_tools() is True


def test_list_files_handler_lists_directory(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.py").write_text("y = 2")

    result = list_files_handler(tmp_path, ".")

    assert "a.py" in result and "b.py" in result


def test_list_files_handler_blocks_path_escape(tmp_path: Path):
    result = list_files_handler(tmp_path, "../../etc")
    assert result.startswith("Error")


def test_read_file_handler_returns_requested_line_range(tmp_path: Path):
    (tmp_path / "f.py").write_text("line1\nline2\nline3\nline4\n")

    result = read_file_handler(tmp_path, "f.py", start_line=2, end_line=3)

    assert result == "line2\nline3"
