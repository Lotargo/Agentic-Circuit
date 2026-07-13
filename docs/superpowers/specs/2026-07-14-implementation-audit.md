# Аудит реализации chat-openwebui

Дата: 2026-07-14  
Статус: исправленный и автоматически проверенный MVP-каркас.

## Исправлено

- Выбрана одна транспортная архитектура: собственный OpenAI-compatible FastAPI engine за прозрачным Express gateway.
- LangServe, LlamaIndex, Python-пакет `langsearch`, Vercel AI SDK и MongoDB удалены из обязательного runtime как неиспользуемые или неверно заявленные компоненты.
- Python и Node зависимости зафиксированы lock-файлами; Docker builds используют frozen Python lock.
- Полная история `user/assistant` проходит через router, circuit phase-1/2 и synthesis.
- Для запроса выбирается ровно одна активная эмоциональная призма.
- Provider CRUD защищён `PROVIDERS_ADMIN_TOKEN`, пишет YAML атомарно и вызывает hot reload Python engine.
- Конфигурация отслеживается fingerprint-ом и применяется без рестарта контейнера.
- Финальный synthesis использует настоящий upstream provider stream. Токены проходят через LangGraph custom events и немедленно превращаются в OpenAI SSE chunks.
- TS gateway проксирует SSE с backpressure и явно преобразует Web Stream chunks в Node `Buffer`.
- Router использует YAML prompt; параллельные состояния и ошибки объединяются reducer-ами.
- RAG не блокирует старт API и деградирует безопасно при недоступных sidecars.
- E5 использует `query:` / `passage:`; Qdrant payload восстанавливает тексты и BM25 после рестарта.
- TEI rerank приведён к реальному cross-encoder контракту; неподдерживаемый псевдо-ColBERT заменён на `Alibaba-NLP/gte-multilingual-reranker-base`.
- Текущий phase-1 не загрязняет RAG-контекст phase-2 того же хода.
- CPU/GPU Compose разделены; добавлены volumes моделей и readiness для собственных сервисов.

## Проверено в GitHub Actions

Финальный clean run проверяет:

- актуальность `uv.lock` и frozen install;
- весь pytest suite;
- TypeScript typecheck и production build;
- `docker compose config`;
- чистые Docker builds Python engine и TS gateway;
- HTTP flow mock OpenAI provider -> Python engine -> TS gateway -> completion с историей сообщений и выбранной prism.

Все четыре job завершились успешно.

## Что остаётся эксплуатационной проверкой

Автоматический smoke-test намеренно не скачивает тяжёлые TEI-модели и OpenWebUI image. Поэтому перед реальной эксплуатацией всё ещё нужен один локальный прогон полного Compose с настоящим provider key:

1. дождаться readiness Qdrant и обоих TEI sidecars;
2. убедиться, что OpenWebUI обнаруживает `agentic-circuit`;
3. вручную проверить fast и slow запросы;
4. проверить сохранение и извлечение RAG после перезапуска;
5. проверить выбранный GPU image tag на конкретной видеокарте.

Настоящий ColBERT late interaction не является частью текущего MVP. Фактическое и документированное решение — TEI cross-encoder rerank. Если ColBERT понадобится позднее, это отдельная функция с token-level multi-vectors и MaxSim, а не замена имени модели в существующем sidecar.
