# agentic-circuit (chat-openwebui)

Мульти-агентный контур с единой личностью **Лиза**:

- **fast**: router сразу передаёт запрос synthesis;
- **slow**: creative, pragmatic и effective выполняются параллельно, каждый проходит phase-1 и phase-2, затем synthesis собирает единый ответ.

## Фактическая архитектура

```text
OpenWebUI
    │ OpenAI-compatible HTTP/SSE
    ▼
TS gateway (Express, transparent proxy)
    │ OpenAI-compatible HTTP/SSE
    ▼
Python engine (FastAPI + LangGraph)
    ├─ router ─ fast ───────────────┐
    └─ slow ─ creative phase1/2 ────┤
             pragmatic phase1/2 ────┼─> synthesis
             effective phase1/2 ────┘
                     │
                     ├─ Qdrant + embedding/rerank sidecars
                     └─ optional LangSearch HTTP API
```

- **Python engine** (`src/py-engine`): LangGraph, FastAPI, OpenAI-compatible provider client, Qdrant, BM25 и HTTP-клиенты embedding/rerank/web-search.
- **TS gateway** (`src/ts-gateway`): прозрачный Express proxy для `/v1/models` и `/v1/chat/completions`, а также CRUD `/v1/providers`.
- **Config** (`config/`): `providers.yaml`, `agents/*.yaml`, `manifests/<agent>/<prism>.md`.
- **OpenWebUI** подключается к TS gateway как к OpenAI-compatible backend.

Исходная спецификация упоминает LangServe, LlamaIndex и использование Vercel AI SDK в runtime. Текущая реализация их не использует. Подробности и оставшиеся расхождения находятся в `docs/superpowers/specs/2026-07-14-implementation-audit.md`.

## Локальный запуск

Скопируйте env-файл и заполните как минимум ключ провайдера:

```bash
cp .env.example .env
```

CPU-first запуск:

```bash
docker compose up --build
```

Запуск с NVIDIA GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Адреса:

- OpenWebUI: `http://localhost:3200`
- TS gateway: `http://localhost:9191`
- Python engine: `http://localhost:8823`

Основной Compose больше не требует NVIDIA runtime. GPU reservations находятся только в `docker-compose.gpu.yml`.

## Разработка без Docker

Python:

```bash
cd src/py-engine
uv venv
uv pip install -e ".[test]"
uv run pytest
```

TypeScript:

```bash
cd src/ts-gateway
npm ci
npm run typecheck
npm run build
npm run dev
```

## Автоматические проверки

GitHub Actions выполняет:

- установку Python-пакета и pytest;
- TypeScript typecheck и production build;
- `docker compose config`;
- чистую сборку Docker-образов `py-engine` и `ts-gateway`.

CI не заменяет полный smoke-test с загруженными моделями, реальным provider API и интерфейсом OpenWebUI. Актуальный статус находится в `docs/superpowers/specs/2026-07-13-agentic-circuit-todo.md`.
