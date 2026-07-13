# agentic-circuit (chat-openwebui)

Мульти-агентный «контур» с единой личностью **Лиза**. Отвечает двумя путями:

- **fast** — роутер сразу отдаёт запрос агенту синтеза (прямой ответ).
- **slow** — три изолированных контура (креативный, прагматичный, эффективный)
  работают параллельно, каждый в две фазы (сырой ответ → критика), затем агент
  синтеза собирает всё в финальный ответ, воспринимая чужие потоки как свои мысли.

## Архитектура

```
OpenWebUI ──(OpenAI-compatible /v1/chat/completions)──> TS-шлюз (AI SDK)
                                                           │  стрим
                                                           ▼
                                               Python LangGraph engine (langserve)
                                                router ─┬─(fast)─> synthesis
                                                        └─(slow)─> [creative, pragmatic, effective] параллельно
                                                                      phase-1 (raw) ─> phase-2 (critic)
                                                                      └─> synthesis (видит всё + RAG + web)
```

- **Python-движок** (`src/py-engine`): LangGraph (оркестрация), FastAPI/langserve
  (HTTP), RAG (Qdrant + TEI embedding + BM25 + ColBERT rerank), langsearch (web).
- **TS-шлюз** (`src/ts-gateway`): Vercel AI SDK, OpenAI-compatible gateway для
  OpenWebUI + REST управления провайдерами (`/v1/providers`).
- **Конфиг** (`config/`): YAML — единый `providers.yaml` и пер-агентные `agents/*.yaml`
  + индивидуальные призмы `manifests/<agent>/<prism>.md`.

## Запуск (локально, docker-compose)

```bash
cp .env.example .env        # заполните ключи (OPENCODE_ZEN_API_KEY, LANGSEARCH_API_KEY)
docker compose up --build   # поднимает qdrant, mongodb, tei, colbert, py-engine, ts-gateway, openwebui
```

- OpenWebUI: http://localhost:3200 (уже настроен на TS-шлюз)
- TS-шлюз: http://localhost:9191
- Python engine: http://localhost:8823

Инференс TEI/ColBERT работает на GPU при наличии CUDA и откатывается на CPU иначе.

## Разработка без Docker

Python-движок:

```bash
cd src/py-engine
uv venv && uv pip install -e ".[test]"
pytest                       # юнит + интегр-тесты графа на моках провайдера
```

TS-шлюз:

```bash
cd src/ts-gateway
npm install
npm run dev
```

## Тестирование

- `tests/test_config.py` — парсинг `providers.yaml` / `agents/*.yaml`, загрузка манифестов.
- `tests/test_prompts.py` — компоновка промптов (base + manifests; meta_instruction для синтеза).
- `tests/test_graph.py` — интегр-тест графа LangGraph на моках: fast/slow, изоляция
  контуров (phase-2 видит только свой phase-1), агрегация в синтезе.
