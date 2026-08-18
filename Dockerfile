FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY codesage/ codesage/
RUN uv sync --frozen --no-dev --extra api

# Only the core requests/ package, not tests/docs/ — this is the exact
# corpus validated locally (169 chunks, 100% eval hit rate). The full repo
# has 3x the files (tests/, docs/ aren't in ingest.py's SKIP_DIRS) and was
# taking multiple minutes to ingest at container startup, since get_app()
# blocks on ingestion before uvicorn can accept any connections.
COPY target_repo/src/ target_repo/

ENV CODESAGE_TARGET_REPO=/app/target_repo

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "codesage.api:get_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
