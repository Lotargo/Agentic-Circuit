# Python LangGraph engine
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PY_ENGINE_PORT=8823

WORKDIR /app

RUN pip install --no-cache-dir uv

# The project must exist before hatchling can build/install it.
COPY src/py-engine/pyproject.toml ./
COPY src/py-engine/src/ ./src/
RUN uv pip install --system --no-cache .

# Shared runtime configuration.
COPY config/ ./config/

EXPOSE 8823

CMD ["python", "-m", "uvicorn", "agentic_circuit.main:app", "--host", "0.0.0.0", "--port", "8823"]
