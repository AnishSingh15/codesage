from google.genai import types

from codesage.memory import Memory


def _content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def test_memory_returns_added_content_in_order():
    memory = Memory(max_turns=5)
    memory.add(_content("first"))
    memory.add(_content("second"))

    contents = memory.as_contents()

    assert len(contents) == 2
    assert contents[0].parts[0].text == "first"
    assert contents[1].parts[0].text == "second"


def test_memory_drops_oldest_when_over_capacity():
    memory = Memory(max_turns=2)  # capacity = 2 turns = 4 Content entries
    for i in range(6):
        memory.add(_content(str(i)))

    contents = memory.as_contents()

    assert len(contents) == 4
    assert [c.parts[0].text for c in contents] == ["2", "3", "4", "5"]
