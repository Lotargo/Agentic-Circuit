# Чек-лист реализации: агентский контур (chat-openwebui)

Дата создания: 2026-07-13
Связанный дизайн: `docs/superpowers/specs/2026-07-13-agentic-circuit-design.md`

Легенда: `[ ]` — не начато, `[~]` — в работе, `[x]` — выполнено

## 0. Репозиторий и базовый каркас

- [x] Git-репозиторий инициализирован, remote `origin` настроен
- [x] Первый коммит с `.gitignore` запушен на `main`
- [ ] Создать структуру каталогов (`config/`, `src/py-engine/`, `src/ts-gateway/`, `docs/`)
- [ ] Добавить `README.md` с описанием архитектуры и запуска

## 1. Управление пакетами и зависимости

- [ ] Python: `pyproject.toml` + `uv.lock` (langgraph, langserve, llama-index, qdrant-client, pymongo, langsearch, httpx)
- [ ] TS: `package.json` (ai, @ai-sdk/openai-compatible, express/fastify, zod)
- [ ] Зафиксировать версии зависимостей

## 2. Конфигурация

- [ ] `config/providers.yaml` — единый провайдер-конфиг (opencode-zen + заглушки)
- [ ] `config/agents/router.yaml`
- [ ] `config/agents/creative-1.yaml` (phase-1)
- [ ] `config/agents/creative-2.yaml` (phase-2 critic)
- [ ] `config/agents/pragmatic-1.yaml` (phase-1)
- [ ] `config/agents/pragmatic-2.yaml` (phase-2 critic)
- [ ] `config/agents/effective-1.yaml` (phase-1)
- [ ] `config/agents/effective-2.yaml` (phase-2 critic)
- [ ] `config/agents/synthesis.yaml`
- [ ] `config/skills/*.md` — текстовые инструкции для агентов
- [ ] Загрузчик конфига с валидацией (pydantic/zod) и подстановкой env

## 3. Python-движок: клиент провайдеров

- [ ] OpenAI-compatible клиент на базе `providers.yaml`
- [ ] Маппинг параметров агента (temperature, max_tokens, top_p, thinking_level) в запрос
- [ ] Обёртка вызова LLM с измерением токенов/ошибок

## 4. Python-движок: граф LangGraph

- [ ] Узел `router` (LLMCallNode, только решение fast/slow)
- [ ] Conditional edge `fast | slow`
- [ ] Узлы контуров: `phase1` (raw) и `phase2` (critic) для 3 контуров
- [ ] Параллельный запуск контуров через `Send()`
- [ ] Узел `synthesis` (fast и slow пути)
- [ ] Передача контекста: phase-2 видит только свой phase-1; synthesis видит всё + phase-1
- [ ] Сборка и компоновка системного промпта (base + skills)

## 5. RAG и память

- [ ] Подключение Qdrant (клиент, создание коллекций per-circuit + общая)
- [ ] Интеграция sidecar TEI/vLLM (E5 small dense эмбеддинг по HTTP)
- [ ] BM25 лексический индекс/запрос
- [ ] ColBERT small v1 late-interaction реранк (sidecar)
- [ ] Гибридный ретрив (dense + BM25 + ColBERT) для синтеза
- [ ] Пер-контурная изоляция коллекций (чтение/запись только своей)
- [ ] Синтез читает все коллекции
- [ ] Подключение MongoDB (сырые тексты + настройки OpenWebUI)

## 6. Инструменты (langsearch) — только synthesis

- [ ] Web-поиск у агента синтеза (`tools.web_search: true`)
- [ ] Реранкинг результатов поиска
- [ ] Отключение инструментов у контуров

## 7. TS-слой (Vercel AI SDK)

- [ ] `POST /v1/chat/completions` — OpenAI-compatible endpoint
- [ ] Формирование ответа через `@ai-sdk/openai-compatible`
- [ ] Пересылка запроса в Python-движок (langserve) и стрим ответа
- [ ] `GET/POST/DELETE /v1/providers` — управление провайдерами
- [ ] Запись изменений провайдеров в `config/providers.yaml`
- [ ] Точка интеграции с UI OpenWebUI (кастомный tool/админка)

## 8. Инфраструктура (docker-compose)

- [ ] Сервис Qdrant
- [ ] Сервис MongoDB
- [ ] Сервис OpenWebUI (-> TS-шлюз)
- [ ] Сервис TEI/vLLM sidecar (E5 small + ColBERT small v1)
- [ ] Сервис Python-движок (langserve)
- [ ] Сервис TS-шлюз
- [ ] Сети/volumes, порядок старта (depends_on)

## 9. Переменные окружения

- [ ] `.env.example` со всеми ключами (провайдеры, Qdrant, MongoDB, sidecar, langsearch, порты)
- [ ] `.env` в `.gitignore` (уже есть)

## 10. Тестирование

- [ ] Юнит-тесты: компоновка промптов (base + skills)
- [ ] Юнит-тесты: парсинг `providers.yaml` и `agents/*.yaml`
- [ ] Интегр-тест графа LangGraph на моках провайдера (fast/slow, изоляция RAG)
- [ ] Smoke-тест: запрос из OpenWebUI через TS-шлюз -> Python-движок

## 11. Финализация

- [ ] Прогон всех тестов
- [ ] Ручная проверка fast и slow путей через OpenWebUI
- [ ] Обновление чек-листа и спеков по итогам
