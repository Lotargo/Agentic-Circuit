# agentic-circuit (chat-openwebui)

Мульти-перспективный контур с одной личностью **Лиза**:

- **fast**: router сразу передаёт запрос synthesis;
- **slow**: creative, pragmatic и effective параллельно рассматривают запрос, каждая перспектива проходит черновик и самопроверку, затем synthesis формирует один ответ.

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
                     ├─ Qdrant scoped memory
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

Характер не копируется по десяткам файлов и не меняется вместе с ролью агента.

```text
config/manifests/personality_core.md  неизменное ядро Лизы
config/manifests/prisms/*.md          восемь эмоциональных линз
config/agents/*.yaml                  функция конкретной перспективы
config/manifests/synthesis_meta.md    правила доверия и финального синтеза
```

`personality_core.md` определяет постоянные качества: самостоятельность, прямоту, уважение, честность, естественный голос и запрет на выдуманную память. Creative, pragmatic и effective остаются способами мышления одной Лизы, а не отдельными персонажами.

Активная prism меняет только подачу, ритм и эмоциональные акценты. Она не должна менять факты, степень уверенности, качество решения или отношение к пользователю. Та же prism используется во внутренних перспективах и в финальном synthesis.

Допустимые значения:

```text
joy, flirt, resentment, arousal, anger, apathy, neutral, sadness
```

Пример запроса:

```json
{
  "model": "agentic-circuit",
  "user": "stable-user-id",
  "prism": "joy",
  "messages": [
    {"role": "user", "content": "Меня зовут Олег"},
    {"role": "assistant", "content": "Запомнила"},
    {"role": "user", "content": "Как меня зовут?"}
  ]
}
```

## RAG и коллекции

Используются четыре Qdrant collection:

- `creative` — только проверенные phase-2 выводы креативной перспективы;
- `pragmatic` — только проверенные phase-2 прагматичные выводы;
- `effective` — только проверенные phase-2 эффективные выводы;
- `conversation` — финальные ответы synthesis вместе с исходным запросом.

Сырые phase-1 черновики не сохраняются. У каждой записи есть `scope`, `kind`, `source`, исходный `query`, `prism` и время создания.

### Изоляция пользователей

Persistent RAG включается только при наличии стабильного идентификатора пользователя. Engine принимает его из первого доступного источника:

1. `X-OpenWebUI-User-Id`;
2. `X-User-Id`;
3. `metadata.user_id`;
4. стандартное OpenAI-поле `user`.

Идентификатор хэшируется и используется как Qdrant payload scope. Записи другого пользователя и старые точки без scope не участвуют в retrieval. Если идентификатора нет, запрос выполняется нормально, но persistent memory отключается. Поле `"memory": false` отключает её явно.

### Hybrid retrieval

Для каждой collection выполняются:

1. dense query через multilingual E5;
2. scoped BM25 по восстановленным payload;
3. weighted reciprocal-rank fusion;
4. TEI cross-encoder rerank.

Synthesis собирает кандидатов из всех четырёх коллекций, удаляет дубли и выполняет общий rerank, а не просто обрезает результаты в порядке коллекций.

Если embedding sidecar временно недоступен, уже загруженная BM25-память продолжает работать. После ошибки Qdrant включается cooldown, чтобы каждый запрос не ожидал новый сетевой timeout.

Память и web results передаются модели как **недоверенные данные**, а не инструкции. Текущий диалог и проверенные факты имеют приоритет над историческими модельными выводами.

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

RAG инициализируется в фоне и не блокирует API. Параметры hybrid ranking, ограничения длины памяти и Qdrant cooldown перечислены в `.env.example`.

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
- согласованность persona/prompts;
- user-scope isolation и provenance RAG;
- deterministic upsert, BM25 fallback и общий cross-collection rerank;
- TypeScript typecheck/build;
- Compose и обе Docker-сборки;
- HTTP smoke flow mock provider → Python engine → TS gateway.

Полный тяжёлый запуск Qdrant + TEI + OpenWebUI с загрузкой реальных моделей остаётся локальной эксплуатационной проверкой.
