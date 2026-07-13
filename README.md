# agentic-circuit (chat-openwebui)

Мульти-агентный контур с единой личностью **Лиза**:

- **fast**: router сразу передаёт запрос synthesis;
- **slow**: creative, pragmatic и effective выполняются параллельно, каждый проходит phase-1 и phase-2, затем synthesis собирает единый ответ.

## Архитектура

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
                     ├─ Qdrant + TEI embedding
                     ├─ TEI cross-encoder rerank
                     └─ optional LangSearch HTTP API
```

Фактические решения:

- Python transport: собственный OpenAI-compatible FastAPI API, не LangServe.
- TS gateway: прозрачный Express proxy. Vercel AI SDK не используется и удалён из зависимостей.
- Rerank: поддерживаемый TEI cross-encoder `Alibaba-NLP/gte-multilingual-reranker-base`, не псевдо-ColBERT.
- MongoDB удалён из обязательного runtime, потому что не участвовал в data flow.
- Python-зависимости зафиксированы в `src/py-engine/uv.lock`, Node-зависимости — в `package-lock.json`.

## Диалог и эмоциональные призмы

В Python engine передаётся полная история `user/assistant`, а не только последнее сообщение. Запрос может содержать поле `prism`:

```json
{
  "model": "agentic-circuit",
  "prism": "joy",
  "messages": [
    {"role": "user", "content": "Меня зовут Олег"},
    {"role": "assistant", "content": "Запомнила"},
    {"role": "user", "content": "Как меня зовут?"}
  ]
}
```

Допустимые значения: `joy`, `flirt`, `resentment`, `arousal`, `anger`, `apathy`, `neutral`, `sadness`. В prompt добавляется ровно одна активная призма.

## Локальный запуск

```bash
cp .env.example .env
```

Заполните ключ provider и замените `PROVIDERS_ADMIN_TOKEN` длинным случайным секретом.

CPU-first запуск:

```bash
docker compose up --build
```

NVIDIA override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Адреса:

- OpenWebUI: `http://localhost:3200`
- TS gateway: `http://localhost:9191`
- Python engine: `http://localhost:8823`

RAG инициализируется в фоне и не блокирует запуск API. Если Qdrant или TEI ещё не готовы, LLM flow продолжает работать без RAG-контекста, а ошибки фиксируются внутри состояния графа.

## Управление провайдерами

`GET/POST/DELETE /v1/providers` требуют секрет из `PROVIDERS_ADMIN_TOKEN`:

```text
X-Admin-Token: <secret>
```

или:

```text
Authorization: Bearer <secret>
```

После изменения `providers.yaml` gateway вызывает `/v1/reload`; Python engine перестраивает registry и граф без перезапуска контейнера.

## Разработка

Python:

```bash
cd src/py-engine
uv sync --frozen --extra test
uv run --frozen pytest
```

TypeScript:

```bash
cd src/ts-gateway
npm ci
npm run typecheck
npm run build
npm run dev
```

## CI

GitHub Actions проверяет:

- актуальность `uv.lock`;
- Python pytest;
- TypeScript typecheck и production build;
- `docker compose config`;
- чистую сборку Docker-образов Python engine и TS gateway;
- реальный HTTP smoke flow: mock OpenAI-compatible provider → Python engine → TS gateway → финальный completion.

Smoke-test использует полную историю сообщений и выбранную prism. Полный запуск тяжёлых Qdrant/TEI/OpenWebUI-контейнеров с загрузкой моделей остаётся отдельной локальной эксплуатационной проверкой.
