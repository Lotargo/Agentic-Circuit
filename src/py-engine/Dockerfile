FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PY_ENGINE_PORT=8823 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY src/py-engine/pyproject.toml src/py-engine/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/py-engine/src/ ./src/
RUN uv sync --frozen --no-dev

COPY config/ ./config/

EXPOSE 8823
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "agentic_circuit.main:app", "--host", "0.0.0.0", "--port", "8823"]
