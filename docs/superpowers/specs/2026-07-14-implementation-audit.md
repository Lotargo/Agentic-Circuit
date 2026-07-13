# Аудит реализации chat-openwebui

Дата: 2026-07-14

## Исправленные запускающие дефекты

- Python Dockerfile больше не пытается установить пакет до копирования исходников.
- Удалена неразрешимая зависимость: пакет `langsearch` требовал `openai==0.27.0`, а движок использует `openai>=1.40.0`.
- OpenAI-compatible streaming использует корректные SSE-кадры без двойного `data:`.
- Добавлен `GET /v1/models` для обнаружения логической модели интерфейсом OpenWebUI.
- TS-шлюз стал прозрачным HTTP/SSE proxy и больше не имитирует использование AI SDK пустым вызовом.
- Путь к `config/providers.yaml` корректно разрешается локально и в контейнере; запись выполняется атомарно.
- Полный URL `/chat/completions` нормализуется перед передачей в `AsyncOpenAI`.
- Отсутствующий API-ключ даёт понятную ошибку, а не используется как буквальное значение ключа.
- Ошибки параллельных веток LangGraph агрегируются reducer-ом.
- Router использует собственный YAML `base_prompt`.
- Текущая phase-1 больше не попадает в phase-2 дважды через немедленный RAG upsert.
- Ошибки RAG retrieve/upsert не обрушивают весь LLM-граф.
- E5 получает обязательные `query:` и `passage:` префиксы.
- Тексты и BM25-индекс восстанавливаются из Qdrant payload после рестарта Python-процесса.
- TEI rerank client приведён к реальному контракту `query + texts + return_text`.
- Неподдерживаемый TEI-моделью `answerdotai/colbert-small-v1` заменён на поддерживаемый мультиязычный cross-encoder `Alibaba-NLP/gte-multilingual-reranker-base`.
- Базовый Compose стал CPU-first; GPU вынесен в отдельный override.
- TEI images обновлены до серии 1.9, добавлены volumes кэша моделей и healthchecks собственных сервисов.
- Добавлен CI: pytest, TypeScript typecheck/build, проверка Compose и Docker-сборки.

## Подтверждённые расхождения со спецификацией

Эти пункты нельзя считать выполненными:

1. `LangServe` заявлен как транспорт Python-движка, но runtime использует собственный FastAPI endpoint `/v1/chat/completions`.
2. `LlamaIndex` и Python-пакет `langsearch` заявлены в стеке, но runtime их не использует. LangSearch API вызывается напрямую через `httpx`.
3. `@ai-sdk/openai-compatible` является provider adapter для исходящих LLM-вызовов, а не formatter OpenAI-compatible server responses. TS-шлюз фактически является proxy.
4. Provider CRUD изменяет YAML, но Python-движок кэширует конфигурацию; нужен рестарт `py-engine`.
5. `MongoStore` написан, но не подключён к основному графу.
6. Все манифесты настроений одного агента одновременно добавляются в prompt; выбора активной призмы нет.
7. В граф передаётся только последнее пользовательское сообщение; история диалога теряется.
8. Настоящий ColBERT late interaction не реализован. Он требует token-level multi-vectors и MaxSim, а не TEI cross-encoder `/rerank`.
9. Ответ сначала полностью вычисляется графом и только потом нарезается на SSE chunks. Это совместимый, но не настоящий token streaming.
10. `/v1/providers` не имеет отдельной авторизации; безопасен только в доверенной локальной сети.
11. `uv.lock` отсутствует, Python-зависимости не зафиксированы lock-файлом.
12. Полный smoke flow OpenWebUI -> gateway -> engine -> provider + Qdrant/TEI ещё не выполнялся.

## Критерий готовности рабочего MVP

- CI зелёный на чистом checkout.
- CPU Compose полностью поднимается без NVIDIA runtime и проходит healthchecks.
- OpenWebUI видит `agentic-circuit` через `/v1/models` и получает валидный ответ.
- Проверены fast и slow ветки с реальным provider API.
- Provider config либо перечитывается горячо, либо API явно возвращает `restart_required: true`.
- Выбрана одна транспортная архитектура: LangServe invoke/stream или внутренний OpenAI-compatible FastAPI API.
- Для ColBERT принято отдельное решение: реализовать Qdrant multi-vector MaxSim либо официально оставить cross-encoder rerank.
