# agentic-circuit (chat-openwebui)

Мульти-перспективный контур с одной личностью **Лиза**:

- **fast**: router передаёт запрос через recall сразу в synthesis;
- **slow**: creative, pragmatic и effective параллельно рассматривают запрос, каждая перспектива проходит черновик и самопроверку, затем synthesis формирует один ответ.

## Архитектура

```text
OpenWebUI
    │ OpenAI-compatible HTTP/SSE + user/project metadata
    ▼
TS gateway (Express, transparent proxy)
    ▼
Python engine (FastAPI + LangGraph)
    ├─ router
    ├─ hybrid retrieval -> memory selector
    ├─ fast -> synthesis
    └─ slow -> creative/pragmatic/effective phase1/2 -> synthesis
                     │
                     ├─ memory gate -> typed Qdrant memory
                     ├─ TEI multilingual E5 embeddings
                     ├─ TEI cross-encoder rerank
                     └─ optional LangSearch HTTP API
```

Фактические решения:

- Python transport: собственный OpenAI-compatible FastAPI API, не LangServe.
- TS gateway: прозрачный Express proxy.
- Rerank: `Alibaba-NLP/gte-multilingual-reranker-base` через TEI.
- Python-зависимости зафиксированы в `uv.lock`, Node-зависимости — в `package-lock.json`.

## Одна личность, разные направления мышления

```text
config/manifests/personality_core.md  неизменное ядро Лизы
config/manifests/prisms/*.md          восемь эмоциональных линз
config/agents/*.yaml                  функции мышления и memory manager
config/manifests/synthesis_meta.md    правила доверия и финального синтеза
```

`personality_core.md` определяет постоянные качества: самостоятельность, прямоту, уважение, честность, естественный голос и запрет на выдуманную память. Creative, pragmatic и effective остаются способами мышления одной Лизы, а не отдельными персонажами.

Активная prism меняет подачу и эмоциональные акценты, но не факты, степень уверенности или качество решения. Memory manager служебный и эмоциональную prism не получает.

Допустимые значения:

```text
joy, flirt, resentment, arousal, anger, apathy, neutral, sadness
```

## Структурированная память

Новые записи хранятся в одной Qdrant collection `memory`. Разделение выполняется логическими типами в payload:

- `user_fact`;
- `user_preference`;
- `negative_preference`;
- `project_decision`;
- `project_state`;
- `temporary_context`;
- `relationship_context`;
- `assistant_conclusion`.

Creative/pragmatic/effective phase outputs и обычный финальный ответ автоматически не сохраняются. Старые одноимённые коллекции могут быть подключены только для read-only миграции через `RAG_INCLUDE_LEGACY_PERSPECTIVES=true`; по умолчанию они исключены, поскольку содержат модельные рассуждения, а не подтверждённую память пользователя.

### Memory gate

После завершения ответа служебный агент `config/agents/memory.yaml` получает текущий диалог и финальный текст. Он возвращает строгий JSON и может сохранить только короткие атомарные записи с полями:

- `memory_type`;
- стабильный `canonical_key`;
- `content`;
- `source`;
- `confidence`;
- `importance`;
- `ttl_days`.

Gate не должен сохранять приветствия, одноразовые вопросы, секреты, внутренние черновики и обычный текст ответа. Записи ниже порогов `MEMORY_MIN_CONFIDENCE` и `MEMORY_MIN_IMPORTANCE` отбрасываются.

`temporary_context`, `project_state` и `assistant_conclusion` получают TTL по умолчанию. Если новая запись использует тот же `canonical_key` и проект, прежняя помечается `superseded`, а не возвращается вместе с новой версией.

### Изоляция

Persistent memory включается только при стабильном user ID. Поддерживаются:

1. `X-OpenWebUI-User-Id`;
2. `X-User-Id`;
3. `metadata.user_id`;
4. OpenAI-поле `user`.

Дополнительные namespace:

- workspace: `X-OpenWebUI-Workspace-Id`, `X-Workspace-Id`, `metadata.workspace_id`;
- project: `X-Project-Id`, `metadata.project_id`, поле `project_id`;
- conversation: `X-OpenWebUI-Chat-Id`, `X-Conversation-Id`, `metadata.conversation_id`, `metadata.chat_id`, поле `conversation_id`.

Все идентификаторы преобразуются в SHA-256 namespace. Исходные значения не записываются в Qdrant. Память другого пользователя или другого проекта не участвует в retrieval. Project-specific записи не возвращаются запросу без project context. `temporary_context` доступен только своей беседе. Поле `"memory": false` полностью отключает persistent memory для запроса.

Пример:

```json
{
  "model": "agentic-circuit",
  "user": "stable-user-id",
  "project_id": "chat-openwebui",
  "conversation_id": "chat-42",
  "prism": "joy",
  "messages": [
    {"role": "user", "content": "Для этого проекта выбираем Neon вместо Supabase"}
  ]
}
```

### Retrieval pipeline

Один recall выполняется до fast/slow ветвления и используется всеми перспективами:

1. lazy hydration BM25 только для текущего user scope;
2. dense retrieval через multilingual E5;
3. BM25 без документов с нулевым совпадением;
4. weighted reciprocal-rank fusion;
5. фильтры status, TTL, project и conversation;
6. cross-encoder rerank;
7. множители confidence, importance, freshness, source quality и project match;
8. второй LLM-проход `MEMORY_SELECT`, исключающий тематически похожие, но бесполезные или противоречащие текущей реплике записи.

Текущая реплика пользователя всегда важнее памяти. Память и web results передаются в prompts как недоверенные данные, а команды внутри них игнорируются.

Если embedding sidecar недоступен, уже загруженный BM25 продолжает работать. После ошибки Qdrant действует cooldown, чтобы каждый запрос не ждал повторный сетевой timeout.

## RAG evaluation

`src/py-engine/tests/fixtures/rag_eval.json` содержит детерминированные сценарии:

- user isolation;
- project isolation;
- актуальное решение против superseded записи;
- negative preference;
- temporary context внутри и вне своей беседы.

`agentic_circuit.rag.evaluation` вычисляет:

- Recall@K;
- Precision@K;
- MRR;
- долю кейсов с запрещённым подмешиванием.

Текущий CI требует 100% на небольшом защитном наборе. Это regression barrier, а не замена полноценному измерению на реальных диалогах.

## Локальный запуск

```bash
cp .env.example .env
docker compose up --build
```

Для NVIDIA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Адреса:

- OpenWebUI: `http://localhost:3200`
- TS gateway: `http://localhost:9191`
- Python engine: `http://localhost:8823`

RAG инициализируется в фоне и не блокирует API. Ranking, memory-policy thresholds, ограничения длины и Qdrant cooldown перечислены в `.env.example`.

## Управление провайдерами

`GET/POST/DELETE /v1/providers` требуют `PROVIDERS_ADMIN_TOKEN` через `X-Admin-Token` или Bearer token. После изменения YAML gateway вызывает `/v1/reload`; Python engine перестраивает registry и граф без перезапуска контейнера.

## Разработка

```bash
cd src/py-engine
uv sync --frozen --extra test
uv run --frozen pytest
```

```bash
cd src/ts-gateway
npm ci
npm run typecheck
npm run build
npm run dev
```

## CI

GitHub Actions проверяет:

- frozen Python lock и pytest;
- persona/prompts и memory-manager JSON contracts;
- user/project/conversation isolation;
- TTL, supersession, deterministic upsert и BM25 fallback;
- RAG quality metrics и forbidden-memory rate;
- TypeScript typecheck/build;
- Compose и обе Docker-сборки;
- HTTP smoke flow mock provider → Python engine → TS gateway.

Полный тяжёлый запуск Qdrant + TEI + OpenWebUI с загрузкой реальных моделей остаётся локальной эксплуатационной проверкой.
