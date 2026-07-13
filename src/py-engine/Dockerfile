# Python LangGraph engine
# Build context is the repo root (see docker-compose.yml) so ../config is available.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PY_ENGINE_PORT=8823

WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

COPY src/py-engine/pyproject.toml ./
RUN uv pip install --system .

# App source + shared config (providers, agents, manifests)
COPY src/py-engine/src/ ./src/
COPY config/ ./config/

EXPOSE 8823

CMD ["python", "-m", "uvicorn", "agentic_circuit.main:app", "--host", "0.0.0.0", "--port", "8823"]
