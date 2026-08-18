"""Bounded conversation history.

A plain list would grow forever and eventually overflow the model's
context window. collections.deque(maxlen=...) gives us an O(1) sliding
window for free: once full, adding a new item silently drops the oldest.
"""

from collections import deque

from google.genai import types


class Memory:
    def __init__(self, max_turns: int = 10):
        self._contents: deque = deque(maxlen=max_turns * 2)

    def add(self, content: types.Content) -> None:
        self._contents.append(content)

    def as_contents(self) -> list[types.Content]:
        return list(self._contents)

    def __len__(self) -> int:
        return len(self._contents)
