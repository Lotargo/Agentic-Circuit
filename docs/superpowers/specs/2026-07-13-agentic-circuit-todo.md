# Чек-лист реализации: агентский контур (chat-openwebui)

Дата создания: 2026-07-13  
Актуализировано: 2026-07-14  
Связанный дизайн: `docs/superpowers/specs/2026-07-13-agentic-circuit-design.md`  
Аудит: `docs/superpowers/specs/2026-07-14-implementation-audit.md`

Легенда: `[ ]` — не начато, `[~]` — реализовано частично или не проверено в живом стеке, `[x]` — реализовано и проверено доступными автоматическими тестами.

> Текущий статус: чистая установка Python-зависимостей, pytest, TypeScript typecheck/build,
> `docker compose config` и сборка обоих собственных Docker-образов проходят в GitHub Actions.
> Полный запуск OpenWebUI + Qdrant + TEI embedding/rerank с реальным provider API пока не
> выполнялся, поэтому проект ещё нельзя считать готовым к эксплуатации.

## 0. Репозиторий и базовый каркас

- [x] Структура каталогов (`config/`, `src/py-engine/`, `src/ts-gateway/`, `docs/`)
- [x] README с фактической архитектурой и командами запуска
- [x] GitHub Actions CI для Python, TypeScript, Compose и Docker build

## 1. Управление пакетами и зависимости

- [x] Python-пакет устанавливается из `pyproject.toml`
- [ ] Создать и закоммитить `uv.lock`
- [~] Зафиксировать версии Python-зависимостей вместо широких диапазонов
- [x] TS-зависимости зафиксированы через `package-lock.json`
- [x] Удалены неиспользуемые и несовместимые зависимости `langsearch`, `langserve`, `llama-index`

## 2. Конфигурация

- [x] `config/providers.yaml` и Pydantic-валидация
- [x] YAML-конфиги router, synthesis и шести фаз контуров
- [x] Индивидуальные markdown-манифесты призм
- [x] Router использует собственный `base_prompt` из YAML
- [ ] Реализовать выбор одной активной призмы; сейчас все манифесты агента добавляются в prompt одновременно
- [~] CRUD провайдеров изменяет YAML, но Python-движку нужен рестарт для перечитывания кэшированного конфига

## 3. Python-движок: клиент провайдеров

- [x] OpenAI-compatible клиент на базе `providers.yaml`
- [x] Нормализация полного `/chat/completions` URL для `AsyncOpenAI`
- [x] Проверка отсутствующего API-ключа с понятной ошибкой
- [x] Маппинг `temperature`, `max_tokens`, `top_p`, `thinking_level`
- [x] Структурированная фиксация ошибки, latency и token usage

## 4. Python-движок: LangGraph

- [x] Router и условный `fast | slow` путь
- [x] Три параллельных контура через `Send()`
- [x] Последовательные phase-1 и phase-2 внутри каждого контура
- [x] Изоляция контуров до synthesis
- [x] Reducer для объединения результатов и ошибок параллельных веток
- [x] Текущий phase-1 не дублируется в phase-2 через RAG текущего же хода
- [x] Ошибки RAG не обрушивают основной LLM flow
- [x] Synthesis для fast и slow путей
- [ ] Передавать полную историю диалога; сейчас используется только последнее сообщение пользователя

## 5. OpenAI-compatible API

- [x] `POST /v1/chat/completions`
- [x] Валидный non-stream response с `id`, `object`, `created`, `model`, `choices`, `usage`
- [x] Валидные SSE chunks без двойного `data:` и с `[DONE]`
- [x] `GET /v1/models` для обнаружения модели OpenWebUI
- [x] Контрактные тесты API
- [~] Ответ вычисляется полностью и затем нарезается на chunks; настоящего token streaming нет
- [~] Python-сервис является собственным FastAPI API, а не LangServe runtime из исходной спецификации

## 6. RAG и память

- [x] Per-circuit коллекции Qdrant
- [x] Dense embedding через TEI HTTP sidecar
- [x] Корректные `query:` / `passage:` префиксы для multilingual E5
- [x] In-process BM25 с idempotent update
- [x] Восстановление текстов и BM25 из Qdrant payload после рестарта
- [x] Reciprocal-rank fusion dense + BM25
- [x] TEI cross-encoder rerank через контракт `query + texts + return_text`
- [x] Поддерживаемый мультиязычный reranker `Alibaba-NLP/gte-multilingual-reranker-base`
- [x] Composite retrieval всех коллекций для synthesis
- [~] `MongoStore` существует, но не подключён к основному графу
- [ ] Настоящий ColBERT late interaction через token-level multi-vectors и MaxSim
- [ ] Проверить живой Qdrant/TEI flow на реальных контейнерах и моделях

## 7. Web-поиск

- [x] Web-поиск включён только у synthesis
- [x] Вызов LangSearch API реализован напрямую через `httpx`
- [x] Ошибка web-поиска не обрушает весь граф
- [ ] Добавить rerank результатов web-поиска; текущий cross-encoder применяется только к RAG-документам

## 8. TS-шлюз

- [x] Прозрачный proxy `POST /v1/chat/completions`
- [x] Проксирование `GET /v1/models`
- [x] SSE backpressure и отмена upstream при закрытии клиента
- [x] `GET/POST/DELETE /v1/providers`
- [x] Корректный путь к `providers.yaml` локально и в контейнере
- [x] Атомарная запись YAML через временный файл
- [~] `/v1/providers` не имеет отдельной авторизации
- [~] Vercel AI SDK не участвует в runtime; исходное утверждение о формировании OpenAI-ответа через него было неверным

## 9. Docker Compose

- [x] Общая сеть и persistent volumes
- [x] CPU-first основной `docker-compose.yml`
- [x] Отдельный `docker-compose.gpu.yml` с NVIDIA reservations
- [x] Актуальные TEI image tags серии 1.9
- [x] Persistent model cache volumes для embedding и rerank sidecars
- [x] Healthchecks `py-engine` и `ts-gateway`
- [x] OpenWebUI ожидает здоровый TS gateway, TS gateway ожидает здоровый Python engine
- [x] Сборка `py-engine` в чистом Docker build
- [x] Сборка `ts-gateway` в multi-stage Docker build
- [x] `docker compose config` проходит в CI
- [~] Qdrant и TEI sidecars пока подключены через start-order, без собственных Compose health conditions
- [ ] Полный `docker compose up` и health-check всех сервисов
- [ ] Smoke-test OpenWebUI -> TS gateway -> Python engine -> реальный provider

## 10. Автоматические проверки

- [x] Загрузка конфигурации и prompt assembly
- [x] Fast/slow, изоляция контуров и synthesis
- [x] OpenAI-compatible API и SSE
- [x] Нормализация provider endpoint
- [x] TEI embedding payload и E5 prefixes
- [x] TEI rerank request/response contract
- [x] Persistent Qdrant payload и BM25 hydration
- [x] TypeScript typecheck и production build
- [x] Docker build обоих собственных сервисов

## 11. До состояния рабочего MVP

- [ ] Выбрать одну транспортную архитектуру: LangServe invoke/stream или собственный OpenAI-compatible FastAPI API
- [ ] Реализовать hot reload provider config либо явно возвращать `restart_required`
- [ ] Реализовать выбор активной эмоциональной призмы
- [ ] Передавать историю чата, а не только последнее сообщение
- [ ] Подключить MongoDB к реальному data flow либо убрать её из обязательного стека
- [ ] Добавить readiness Qdrant/TEI вместо одного start-order
- [ ] Определиться: настоящий ColBERT multi-vector или поддерживаемый cross-encoder
- [ ] Провести полный живой smoke-test и зафиксировать результат
