# Структурированная память и RAG evaluation

Дата: 2026-07-14

## Цель

Перестать использовать каждый ответ модели как долговременную память. Система должна сохранять только устойчивые сведения, отделять проекты и рабочие пространства, заменять устаревшие решения и измеримо проверять retrieval.

## Реализованная архитектура

### Одна физическая collection

Новые записи хранятся в Qdrant collection `memory`. Логическое разделение выполняется полями payload:

- `memory_type`;
- `canonical_key`;
- `scope`;
- `project_id`;
- `conversation_id`;
- `confidence`;
- `importance`;
- `source_quality`;
- `status`;
- `created_at`, `updated_at`, `expires_at`;
- `superseded_by`.

Поддерживаемые типы:

- `user_fact`;
- `user_preference`;
- `negative_preference`;
- `project_decision`;
- `project_state`;
- `temporary_context`;
- `relationship_context`;
- `assistant_conclusion`.

Старые collections creative, pragmatic и effective не получают новые записи. Их можно включить только как read-only источник миграции через `RAG_INCLUDE_LEGACY_PERSPECTIVES=true`.

### Memory recall

Recall выполняется один раз перед fast/slow ветвлением:

1. scoped BM25 и dense retrieval;
2. weighted RRF;
3. фильтрация workspace/user, project, conversation, status и TTL;
4. cross-encoder rerank;
5. пересчёт по confidence, importance, freshness, source quality и project match;
6. второй LLM-проход `MEMORY_SELECT`.

Выбранные записи передаются всем внутренним перспективам и synthesis как недоверенные данные. Текущая реплика пользователя имеет больший приоритет.

### Memory gate

После synthesis служебный агент `memory` возвращает строгий JSON. Сохраняются только атомарные записи, прошедшие:

- `should_store=true`;
- явное `sensitive=false`;
- минимальный confidence;
- минимальную importance;
- Pydantic-валидацию типа и canonical key.

Поле `sensitive` работает fail-closed: значение по умолчанию `true`. Повреждённый элемент JSON не уничтожает другие корректные элементы ответа.

Обычный финальный ответ, phase-1 и phase-2 автоматически не сохраняются.

### Актуальность и противоречия

Записи с одинаковым `canonical_key` внутри одного проекта представляют одну логическую сущность. Новое значение сохраняется как active, предыдущие получают:

```json
{
  "status": "superseded",
  "superseded_by": "<new-id>"
}
```

Expired и superseded записи исключаются до rerank. Временный контекст доступен только исходной беседе.

### Изоляция

Persistent scope включает tenant, workspace и user. Исходные идентификаторы хэшируются и не записываются в Qdrant.

Дополнительно используются отдельные project и conversation namespaces. Gateway пересылает только разрешённый список headers, необходимый Python engine для построения этих namespaces.

Без стабильного user ID память отключается. Поле `memory=false` отключает её явно.

## Деградация

- При недоступном embedding sidecar retrieval продолжает работать по уже загруженному BM25.
- После ошибки Qdrant действует cooldown.
- При ошибке `MEMORY_SELECT` используются уже жёстко отфильтрованные top candidates.
- При ошибке `MEMORY_EXTRACT` ответ пользователю сохраняет доступность, новые записи просто не создаются.

## Проверки

Unit tests покрывают:

- user/workspace/project/conversation isolation;
- TTL и temporary context;
- deterministic IDs;
- supersession;
- ranking policy;
- BM25 fallback;
- malformed JSON sibling isolation;
- fail-closed sensitive flag;
- memory gate thresholds;
- отсутствие persistence внутренних рассуждений.

Детерминированный набор `tests/fixtures/rag_eval.json` измеряет:

- Recall@K;
- Precision@K;
- MRR;
- forbidden-case rate.

Текущий защитный набор требует 100% по всем позитивным кейсам и 0% запрещённых подмешиваний.

CI smoke выполняет два HTTP-запроса через Express gateway. Первый создаёт user fact, второй извлекает его, вызывает `MEMORY_SELECT` и проверяет полный OpenAI-compatible flow.

## Ограничения

- LLM memory gate добавляет дополнительный вызов модели после synthesis.
- `MEMORY_SELECT` добавляет вызов только когда retrieval нашёл кандидатов.
- Качество на синтетическом наборе не заменяет evaluation на реальной истории диалогов.
- Полный Qdrant + TEI benchmark с настоящими моделями остаётся отдельной эксплуатационной проверкой.
