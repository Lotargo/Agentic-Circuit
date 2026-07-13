# Дизайн: агентский контур (chat-openwebui)

Дата: 2026-07-13

## 1. Цель

Построить мульти-агентный «контур», который отвечает пользователю двумя путями:

1. **Быстрый путь** — роутер сразу отдаёт запрос агенту синтеза, который формулирует
   итоговый ответ без прогона контуров.
2. **Долгий путь** — три изолированных контура (креативный, прагматичный, эффективный)
   работают параллельно, каждый в две фазы (сырой ответ → критика), затем агент синтеза
   собирает всё в финальный ответ.

Фронтенд — OpenWebUI, подключённый к нашему OpenAI-compatible шлюзу.

## 2. Стек

- **Python-движок**: LangGraph (оркестрация графа), LangServe (HTTP-сервер графа),
  LlamaIndex (RAG), langsearch (web-поиск и реранкинг). Управление пакетами — `uv`.
- **TS-слой**: Vercel AI SDK + `@ai-sdk/openai-compatible` — OpenAI-compatible gateway для
  OpenWebUI и REST API управления провайдерами.
- **Векторный стор**: Qdrant (гибридный ретрив: dense + BM25 + ColBERT late-interaction).
- **NoSQL**: MongoDB — сырые текстовые данные и настройки OpenWebUI.
- **Эмбеддинги/реранк**: sidecar-сервис на базе TEI/vLLM в docker-compose
  (Intfloat Multilingual E5 Small для dense, Answer.AI ColBERT Small V1 для late-interaction;
  BM25 — лексический, через Qdrant/локальный индекс).
- **Фронтенд**: OpenWebUI (docker), подключён к TS-шлюзу как OpenAI-compatible backend.

## 3. Архитектура и поток

```
OpenWebUI ──(OpenAI-compatible /v1/chat/completions)──> TS-шлюз (AI SDK)
                                                          │  стрим ответа
                                                          ▼
                                              Python LangGraph engine (langserve)
                                               router ─┬─(fast)─> synthesis (прямо)
                                                       └─(slow)─> [creative, pragmatic, effective] параллельно
                                                                     каждый: phase-1 (raw) ─> phase-2 (critic)
                                                                     └─> synthesis (видит всё + фаза-1 + все RAG)
```

- Роутер **не отвечает** пользователю, только принимает решение fast/slow по эвристике из
  своего промпта/конфига (например, оценка сложности/типа запроса).
- **Fast** = агент синтеза напрямую (тот же агент, что и на долгом пути, но без контуров).
- **Slow** = 3 контура параллельно через LangGraph `Send()`, затем агент синтеза.

## 4. Конфигурация

Единый провайдер-конфиг + пер-агентные промпты. Все конфиги — YAML, читаются Python-движком.

### 4.1 `config/providers.yaml` (единый провайдер-конфиг)

Список провайдеров. Основной — `opencode-zen`:

```yaml
providers:
  opencode-zen:
    type: openai-compatible
    base_url: https://opencode.ai/zen/v1/chat/completions
    api_key_env: OPENCODE_ZEN_API_KEY
    models:
      - big-pickle
      - mimo-v2.5-free
      - north-mini-code-free
      - nemotron-3-ultra-free
      - deepseek-v4-flash-free
      - hy3-free
  # доп. провайдеры добавляются отсюда (в т.ч. из UI OpenWebUI через TS-слой)
```

### 4.2 Единая личность «Лиза» (организм)

Все агенты — это одна личность по имени **Лиза**. Каждый агент = Лиза, которая мыслит в
своём ключе (креативная Лиза, прагматичная Лиза, эффективная Лиза, Лиза-критик,
Лиза-синтез). Финальный агент синтеза — тоже Лиза, но она воспринимает ответы других
агентов **как свои собственные мысли**, а не как продукт отдельных агентов. Так система
ведёт себя как единый цельный организм, а не ансамбль ботов.

Базовая инструкция каждого агента закрепляет идентичность Лизы и ведёт на список
доступных манифестов-призм.

### 4.3 `config/agents/<agent>.yaml`

Один файл на агента. Агенты: `router`, `creative-1`, `creative-2`, `pragmatic-1`,
`pragmatic-2`, `effective-1`, `effective-2`, `synthesis`.

```yaml
name: creative-1
base_prompt: |-
  Ты — Лиза. Ты креативная ипостась своего «я». Веди себя согласно базовым паттернам
  личности и раскрывайся через призмы настроения ниже. Это сырой, первый поток мыслей.
manifests:            # список призм-настроений (индивидуален для этого агента)
  - joy.md
  - flirt.md
  - resentment.md
  - arousal.md
  - anger.md
  - apathy.md
  - neutral.md
  - sadness.md
model:
  provider: opencode-zen
  model: hy3-free
  temperature: 0.9
  max_tokens: 2048
  top_p: 0.95
  thinking_level: off   # off | low | medium | high
tools:
  web_search: false
  rag: true             # пишет/читает только свою коллекцию контура
collection: creative    # имя RAG-коллекции в Qdrant
```

- `base_prompt` — основная инструкция: закрепляет идентичность Лизы, задаёт базовые
  паттерны поведения и **ведёт на список манифестов** (`manifests`).
- `manifests` — список призм-настроений. Каждая призма **индивидуальна для агента**:
  манифест `joy.md` у креативщика отличается от `joy.md` у прагматика, а манифест
  прагматика-фазы-1 отличается от манифеста прагматика-критика (фазы-2).
- Агент **синтеза** не имеет манифестов — вместо них у него большая `meta_instruction`
  (см. 4.5).
- Параметры модели (`temperature`, `max_tokens`, `top_p`, `thinking_level`) задаются
  индивидуально на каждого агента.

### 4.4 `config/manifests/<agent>/<prism>.md`

Текстовые инструкции-призмы, **индивидуальные для каждого агента**. Набор призм:
`joy` (радость), `flirt` (флирт), `resentment` (обида), `arousal` (возбуждение),
`anger` (злость), `apathy` (апатия), `neutral` (нейтральность), `sadness` (грусть) и т.д.

Пример структуры:

```
config/manifests/
  creative-1/joy.md      # креативная радость
  creative-1/anger.md
  creative-2/joy.md      # креативный критик в радости (другой текст)
  pragmatic-1/joy.md     # прагматичная радость (отличается от creative-1/joy.md)
  pragmatic-2/joy.md
  effective-1/joy.md
  effective-2/joy.md
  ...
```

Один и тот же файл не переиспользуется между агентами — содержимое призмы пишется под
роль агента (сырой ответ vs критика, креатив vs прагматика vs эффективность).

### 4.5 Агент синтеза: мета-инструкция

У `synthesis` нет манифестов. Вместо них — одна большая `meta_instruction`, которая:

- закрепляет, что синтез — это тоже Лиза;
- учит правильно **синтезировать** ответы из потоков других ипостасей;
- прямо разрешает **отклонять** те или иные решения/мысли других ипостасей, если они
  ошибочны, противоречивы или не соответствуют личности;
- трактует входные ответы контуров как **собственные мысли** Лизы, а не как вывод
  внешних агентов.

```yaml
name: synthesis
base_prompt: |-
  Ты — Лиза. Ты собираешь свои же размышления (креативные, прагматичные, эффективные
  потоки) и формируешь единый цельный ответ. Это твои мысли, не чужие.
meta_instruction: synthesis_meta.md   # большая мета-инструкция вместо manifests
model: { provider: opencode-zen, model: hy3-free, temperature: 0.7, max_tokens: 4096, top_p: 0.9, thinking_level: medium }
tools: { web_search: true, rag: true }   # видит ВСЕ коллекции
```

## 5. Граф LangGraph

- `router` (LLMCallNode) -> conditional edge `fast | slow`.
- **Slow**: `Send()` × 3 в подграф контура. Контур = два последовательных узла:
  - `phase1` (raw answer) — пишет ответ в свою RAG-коллекцию (`collection: <circuit>`).
  - `phase2` (critic) — видит только свой phase-1 ответ + запрос пользователя
    (не видит другие контуры); читает только свою коллекцию.
- `synthesis` (LLMCallNode): на вход — запрос пользователя + все phase-2 ответы + все
  phase-1 ответы + ретрив из **всех** коллекций + (опц.) web-поиск. Память синтеза = все коллекции.

Изоляция: контуры не видят ответы друг друга ни в фазе-1, ни в фазе-2. Только синтез
агрегирует всё.

## 6. RAG и память

- **Qdrant** — векторный стор. Каждый контур = отдельная collection. Фаза-1/2 агент контура
  пишет и читает только свою коллекцию.
- **Гибридный ретрив**: dense (E5 small) + BM25 (лексический) + ColBERT small v1
  (late-interaction реранк). Эмбеддинги/реранк считаются в sidecar TEI/vLLM, Python-движок
  зовёт его по HTTP.
- **Агенты «помнят прошлые похожие ответы»** — ретрив по эмбеддингам при каждом вызове.
- **Синтез** читает все 3 коллекции + web (langsearch + реранк).
- **MongoDB** — сырые текстовые данные (исходники документов, логи) и настройки OpenWebUI.

## 7. TS-слой (Vercel AI SDK)

- `POST /v1/chat/completions` — OpenAI-compatible endpoint, к которому подключается OpenWebUI.
  Принимает запрос, пересылает в Python-движок (langserve), стримит ответ обратно.
  `@ai-sdk/openai-compatible` используется для формирования ответа в формате OpenWebUI.
- `GET/POST/DELETE /v1/providers` — управление провайдерами из UI. Пишет в
  `config/providers.yaml`. OpenWebUI вызывает через кастомный tool/админку.
- TS-слой не делает LLM-вызовов сам — провайдер-вызовы в Python-движке.

## 8. Инструменты (langsearch) — только synthesis

Web-поиск + реранкинг включены только у агента синтеза (`tools.web_search: true`).
Контуры — только RAG, без внешних инструментов.

## 9. Запуск / Dev (контейнеризация)

Каждый компонент — **отдельный Docker-контейнер**, связаны через **общую Docker-сеть** и
общаются по **HTTP или gRPC**. Это нужно, чтобы позже удобно деплоить по отдельности.

- `uv` для Python-движка (`pyproject.toml` + `uv.lock`).
- `package.json` для TS-слоя.
- `docker-compose.yml` поднимает независимые сервисы в одной сети:
  - `qdrant` — векторный стор (в проде заменяется на Qdrant Cloud)
  - `mongodb` — сырые тексты + настройки OpenWebUI
  - `openwebui` — фронтенд (отдельный контейнер), стучится в TS-шлюз
  - `tei-embedding` — инференс E5 small (отдельный контейнер)
  - `colbert-rerank` — инференс ColBERT small v1 (отдельный контейнер)
  - `py-engine` — Python-движок LangGraph (отдельный контейнер, langserve)
  - `ts-gateway` — TS-шлюз (отдельный контейнер)
- Общая сеть (`docker network`), порты см. раздел 10.
- `.env.example` (см. раздел 10).

### Границы общения (HTTP/gRPC)

- OpenWebUI → TS-шлюз: **HTTP** (`/v1/chat/completions`, OpenAI-compatible).
- TS-шлюз → Python-движок: **HTTP** (langserve invoke/stream).
- Python-движок → Qdrant: **gRPC/HTTP** (qdrant-client).
- Python-движок → MongoDB: **HTTP** (pymongo).
- Python-движок → TEI/ColBERT: **HTTP** (REST инференс эмбеддингов/реранка).

## 13. Развёртывание (target)

Контейнерная изоляция позволяет переносить компоненты по отдельности:

- **Vercel** — TS-шлюз (бессерверный/edge), точка входа OpenWebUI-compatible.
- **Render** (или аналог) — Python-движок LangGraph (langserve).
- **Облачные зависимости**: Qdrant заменяется на **Qdrant Cloud**; инференс-сервисы
  (TEI embedding / ColBERT rerank) переносятся на облачный инференс. MongoDB — на
  облачный MongoDB (Atlas) или Render-аддон.
- OpenWebUI при деплое конфигурируется на URL TS-шлюза (Vercel).

> Контракты между сервисами (URL/порты) берутся только из env, чтобы один и тот же образ
> работал и локально, и в облаке.

## 10. Переменные окружения (`.env.example`)

Порты намеренно нестандартные, чтобы не конфликтовать с уже поднятой инфраструктурой.

```
# Провайдеры (OpenAI-compatible)
OPENCODE_ZEN_API_KEY=sk-...
# доп. провайдеры: их ключи по мере добавления

# Векторный стор (Qdrant, стд. 6333/6334 -> 6633/6634)
QDRANT_URL=http://localhost:6633
QDRANT_API_KEY=

# NoSQL (MongoDB, стд. 27017 -> 27617)
MONGODB_URL=mongodb://localhost:27617
MONGODB_DB=chat_openwebui

# Эмбеддинг/реранк sidecar (TEI/vLLM, стд. 8080/8081 -> 8899/8898)
EMBEDDING_SIDECAR_URL=http://localhost:8899
RERANK_SIDECAR_URL=http://localhost:8898
EMBEDDING_MODEL=intfloat/multilingual-e5-small
RERANK_MODEL=answerdotai/colbert-small-v1

# Web-поиск
LANGSEARCH_API_KEY=

# Сервисы
PY_ENGINE_URL=http://localhost:8823   # langserve (стд. 8123 -> 8823)
TS_GATEWAY_PORT=9191                  # OpenAI-compatible gateway (стд. 8000 -> 9191)
OPENWEBUI_URL=http://localhost:3200   # OpenWebUI (стд. 3000 -> 3200)
```

### Карта портов (внешний -> контейнер)

| Сервис            | Хост    | Контейнер |
|-------------------|---------|-----------|
| Qdrant REST       | 6633    | 6333      |
| Qdrant gRPC       | 6634    | 6334      |
| MongoDB           | 27617   | 27017     |
| TEI embedding     | 8899    | 80        |
| TEI/ColBERT rerank| 8898    | 80        |
| Python engine     | 8823    | 8123      |
| TS gateway        | 9191    | 9191      |
| OpenWebUI         | 3200    | 8080      |

## 11. Тестирование

- Юнит-тесты: компоновка промптов (base + skills), парсинг `providers.yaml` и `agents/*.yaml`.
- Интегр-тест графа LangGraph на моках провайдера: fast/slow ветки, изоляция RAG-коллекций
  (контуры не видят чужие ответы), агрегация в синтезе.
- Smoke-тест: запрос из OpenWebUI через TS-шлюз -> Python-движок -> ответ.

## 12. Out of scope (YAGNI)

- Аутентификация пользователей (используем встроенную OpenWebUI).
- Постоянное хранилище истории чатов (берёт OpenWebUI).
- Горизонтальное масштабирование контуров (пока один процесс Python-движка).
