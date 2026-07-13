# Чек-лист реализации: агентский контур (chat-openwebui)

Дата создания: 2026-07-13  
Актуализировано: 2026-07-14  
Связанный дизайн: `docs/superpowers/specs/2026-07-13-agentic-circuit-design.md`  
Аудит: `docs/superpowers/specs/2026-07-14-implementation-audit.md`

Легенда: `[ ]` — не начато, `[~]` — реализовано частично или не проверено в живом стеке, `[x]` — реализовано и проверено доступными автоматическими тестами.

> Текущий статус: чистая установка Python-зависимостей, pytest, TypeScript typecheck/build,
> `docker compose config` и сборка обоих собственных Docker-образов проходят в GitHub Actions.
> Полный запуск OpenWebUI + Qdrant + MongoDB + TEI + ColBERT с реальными ключами и моделями
> пока не выполнялся, поэтому проект ещё нельзя считать готовым к эксплуатации.

## 0. Репозиторий и базовый каркас

- [x] Структура каталогов (`config/`, `src/py-engine/`, `src/ts-gateway/`, `docs/`)
- [x] README с базовым описанием и командами запуска
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
- [x] Synthesis для fast и slow путей
- [ ] Передавать полную историю диалога; сейчас используется только последнее сообщение пользователя

## 5. OpenAI-compatible API

- [x] `POST /v1/chat/completions`
- [x] Валидный non-stream response с `id`, `object`, `created`, `model`, `choices`, `usage`
- [x] Валидные SSE chunks без двойного `data:` и с `[DONE]`
- [x] `GET /v1/models` для обнаружения модели OpenWebUI
- [x] Контрактные тесты API
- [~] Python-сервис является собственным FastAPI API, а не LangServe runtime, заявленным в исходной спецификации

## 6. RAG и память

- [x] Per-circuit коллекции Qdrant
- [x] Dense embedding через HTTP sidecar
- [x] In-process BM25
- [x] Опциональный rerank через HTTP sidecar
- [x] Reciprocal-rank fusion dense + BM25
- [x] Чтение `text` из Qdrant payload после рестарта процесса
- [x] Composite retrieval всех коллекций для synthesis
- [~] `MongoStore` существует, но не подключён к основному графу
- [ ] Проверить живой Qdrant/TEI/ColBERT flow на реальных контейнерах и моделях

## 7. Web-поиск

- [x] Web-поиск включён только у synthesis
- [x] Вызов LangSearch API реализован напрямую через `httpx`
- [x] Ошибка web-поиска не обрушает весь граф и попадает в список ошибок
- [ ] Добавить фактический rerank результатов web-поиска; текущий ColBERT rerank применяется только к RAG-документам

## 8. TS-шлюз

- [x] Прозрачный proxy `POST /v1/chat/completions`
- [x] Проксирование `GET /v1/models`
- [x] SSE backpressure и отмена upstream при закрытии клиента
- [x] `GET/POST/DELETE /v1/providers`
- [x] Корректный путь к `providers.yaml` локально и в контейнере
- [x] Атомарная запись YAML через временный файл
- [~] Vercel AI SDK не участвует в runtime; исходное утверждение о «формировании OpenAI-ответа через `@ai-sdk/openai-compatible`» было неверным

## 9. Docker Compose

- [x] Общая сеть, volumes и отдельные контейнеры
- [x] CPU-first основной `docker-compose.yml`
- [x] Отдельный `docker-compose.gpu.yml` с NVIDIA reservations
- [x] Сборка `py-engine` в чистом Docker build
- [x] Сборка скомпилированного `ts-gateway` в multi-stage Docker build
- [x] `docker compose config` проходит в CI
- [ ] Полный `docker compose up` и health-check всех сервисов
- [ ] Smoke-test OpenWebUI -> TS gateway -> Python engine -> реальный provider

## 10. Автоматические проверки

- [x] Тесты загрузки конфигурации и prompt assembly
- [x] Интеграционные тесты fast/slow и изоляции контуров
- [x] Тест OpenAI-compatible API и SSE
- [x] Тест нормализации provider endpoint
- [x] Тест persistent Qdrant payload retrieval
- [x] TypeScript typecheck и production build
- [x] Docker build обоих собственных сервисов

## 11. До состояния рабочего MVP

- [ ] Выбрать и закрепить одну транспортную архитектуру: LangServe invoke/stream или собственный OpenAI-compatible FastAPI API
- [ ] Реализовать hot reload provider config либо явно возвращать `restart_required`
- [ ] Реализовать выбор активной эмоциональной призмы
- [ ] Передавать историю чата, а не только последнее сообщение
- [ ] Подключить MongoDB к реальному data flow либо убрать её из обязательного стека
- [ ] Добавить healthchecks/readiness и ожидание готовности зависимостей вместо одного `depends_on`
- [ ] Провести полный живой smoke-test и зафиксировать результат
