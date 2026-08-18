"""Thin wrapper around the Gemini SDK.

This is the only module in CodeSage that imports google.genai — every other
module depends on this interface instead, so we can swap models/providers
or inject a fake client in tests without touching agent logic.
"""

import time

from google import genai
from google.genai import types

_MAX_RETRIES = 1
_RETRY_BACKOFF_SECONDS = 0.1


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        client=None,
    ):
        self._client = client or genai.Client(api_key=api_key)
        self._model = model

    def generate(
        self,
        contents: list,
        tools: list[types.Tool] | None = None,
    ) -> types.GenerateContentResponse:
        config = types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        )

    def embed(self, text: str, output_dimensionality: int = 768) -> list[float]:
        result = self._with_retry(
            lambda: self._client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
            )
        )
        return result.embeddings[0].values

    def _with_retry(self, call):
        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS)
        raise RuntimeError(
            f"Gemini API call failed after {_MAX_RETRIES} retry: {last_error}"
        ) from last_error
