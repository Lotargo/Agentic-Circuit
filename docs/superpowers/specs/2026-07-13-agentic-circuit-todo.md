# Чек-лист реализации: агентский контур (chat-openwebui)

Дата создания: 2026-07-13
Связанный дизайн: `docs/superpowers/specs/2026-07-13-agentic-circuit-design.md`

Легенда: `[ ]` — не начато, `[~]` — в работе, `[x]` — выполнено

> Статус на 2026-07-13: код, конфиги, инфраструктура и тесты написаны и проверены
> (12/12 pytest, typecheck TS). Живой подъём docker-compose (Qdrant/MongoDB/TEI/ColBERT/
> OpenWebUI) и smoke-тест через OpenWebUI НЕ выполнялись в этой сессии — требуют
> скачивания моделей и API-ключей. Файлы инфраструктуры готовы к `docker compose up`.

## 0. Репозиторий и базовый каркас

- [x] Git-репозиторий инициализирован, remote `origin` настроен
- [x] Первый коммит с `.gitignore` запушен на `main`
- [x] Создать структуру каталогов (`config/`, `src/py-engine/`, `src/ts-gateway/`, `docs/`)
- [x] Добавить `README.md` с описанием архитектуры и запуска

## 1. Управление пакетами и зависимости

- [x] Python: `pyproject.toml` + `uv.lock` (langgraph, langserve, llama-index, qdrant-client, pymongo, langsearch, httpx)
- [x] TS: `package.json` (ai, @ai-sdk/openai-compatible, express/fastify, zod)
- [x] Зафиксировать версии зависимостей

## 2. Конфигурация

- [x] `config/providers.yaml` — единый провайдер-конфиг (opencode-zen + заглушки)
- [x] `config/agents/router.yaml`
- [x] `config/agents/creative-1.yaml` (phase-1)
- [x] `config/agents/creative-2.yaml` (phase-2 critic)
- [x] `config/agents/pragmatic-1.yaml` (phase-1)
- [x] `config/agents/pragmatic-2.yaml` (phase-2 critic)
- [x] `config/agents/effective-1.yaml` (phase-1)
- [x] `config/agents/effective-2.yaml` (phase-2 critic)
- [x] `config/agents/synthesis.yaml` (без manifests, с `meta_instruction`)
- [x] `config/manifests/<agent>/<prism>.md` — индивидуальные призмы для каждого агента (joy, flirt, resentment, arousal, anger, apathy, neutral, sadness)
- [x] Базовая инструкция-личность «Лиза» в `base_prompt` каждого агента
- [x] `meta_instruction` для агента синтеза (синтез + отклонение решений, восприятие чужих ответов как своих мыслей)
- [x] Загрузчик конфига с валидацией (pydantic) и подстановкой env

## 3. Python-движок: клиент провайдеров

- [x] OpenAI-compatible клиент на базе `providers.yaml`
- [x] Маппинг параметров агента (temperature, max_tokens, top_p, thinking_level) в запрос
- [x] Обёртка вызова LLM с измерением токенов/ошибок

## 4. Python-движок: граф LangGraph

- [x] Узел `router` (только решение fast/slow)
- [x] Conditional edge `fast | slow`
- [x] Узлы контуров: `phase1` (raw) и `phase2` (critic) для 3 контуров
- [x] Параллельный запуск контуров через `Send()`
- [x] Узел `synthesis` (fast и slow пути)
- [x] Передача контекста: phase-2 видит только свой phase-1; synthesis видит всё + phase-1
- [x] Сборка и компоновка системного промпта (base + manifests; для синтеза — base + meta_instruction)

## 5. RAG и память

- [x] Подключение Qdrant (клиент, создание коллекций per-circuit + общая) — код готов
- [x] Интеграция sidecar TEI (E5 small dense эмбеддинг по HTTP) — код готов
- [x] BM25 лексический индекс/запрос (in-process)
- [x] ColBERT small v1 late-interaction реранк (sidecar) — код готов
- [x] Гибридный ретрив (dense + BM25 + ColBERT) для синтеза — код готов
- [x] Пер-контурная изоляция коллекций (чтение/запись только своей)
- [x] Синтез читает все коллекции
- [x] Подключение MongoDB (сырые тексты + настройки OpenWebUI) — код готов
- [~] Живой подъём Qdrant/MongoDB/TEI/ColBERT не выполнялся (нужны контейнеры + модели)

## 6. Инструменты (langsearch) — только synthesis

- [x] Web-поиск у агента синтеза (`tools.web_search: true`)
- [x] Реранкинг результатов поиска (через ColBERT sidecar)
- [x] Отключение инструментов у контуров

## 7. TS-слой (Vercel AI SDK)

- [x] `POST /v1/chat/completions` — OpenAI-compatible endpoint
- [x] Формирование ответа через `@ai-sdk/openai-compatible`
- [x] Пересылка запроса в Python-движок и стрим ответа
- [x] `GET/POST/DELETE /v1/providers` — управление провайдерами
- [x] Запись изменений провайдеров в `config/providers.yaml`
- [x] Точка интеграции с UI OpenWebUI (кастомный tool/админка) — подключение OpenWebUI к шлюзу в compose

## 8. Инфраструктура (docker-compose, раздельные контейнеры)

- [x] Общая Docker-сеть для всех сервисов
- [x] Сервис Qdrant (отдельный контейнер; в проде -> Qdrant Cloud)
- [x] Сервис MongoDB (отдельный контейнер)
- [x] Сервис OpenWebUI (отдельный контейнер, -> TS-шлюз)
- [x] Сервис TEI embedding (отдельный контейнер)
- [x] Сервис ColBERT rerank (отдельный контейнер)
- [x] Сервис Python-движок (отдельный контейнер; цель деплоя -> Render)
- [x] Сервис TS-шлюз (отдельный контейнер; цель деплоя -> Vercel)
- [x] HTTP границы общения между сервисами
- [x] Сети/volumes, порядок старта (depends_on)
- [~] `docker compose up` не запускался в этой сессии (нужны модели + ключи + время на pull)

## 9. Переменные окружения

- [x] `.env.example` со всеми ключами (провайдеры, Qdrant, MongoDB, sidecar, langsearch, порты)
- [x] `.env` в `.gitignore` (уже есть)

## 10. Тестирование

- [x] Юнит-тесты: компоновка промптов (base + manifests; meta_instruction для синтеза)
- [x] Юнит-тесты: парсинг `providers.yaml` и `agents/*.yaml`
- [x] Интегр-тест графа LangGraph на моках провайдера (fast/slow, изоляция RAG)
- [ ] Smoke-тест: запрос из OpenWebUI через TS-шлюз -> Python-движок (нужен живой стек)

## 11. Финализация

- [x] Прогон всех тестов (12/12 pytest; typecheck TS — чисто)
- [ ] Ручная проверка fast и slow путей через OpenWebUI (нужен живой стек)
- [~] Обновление чек-листа и спеков по итогам
