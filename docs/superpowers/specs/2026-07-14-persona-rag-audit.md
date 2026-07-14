# Аудит личности и RAG

Дата: 2026-07-14

## Проблемы исходных инструкций

1. Базовые prompts называли каждого агента Лизой, но почти не определяли устойчивый характер. Поэтому модель могла звучать как разные персонажи.
2. 48 per-agent manifest-файлов были копиями одного шаблона с заменой названия роли и эмоции. Они не выполняли требование индивидуальности, но создавали шесть независимых источников дрейфа.
3. Эмоция описывалась как доминирующая манера поведения. В `apathy`, `resentment` и `anger` это могло ухудшать помощь, уважение и точность.
4. Synthesis не имел эмоциональных manifests, поэтому active prism влияла на внутренние ответы, но могла исчезнуть в финальном тексте.
5. Память, web results и внутренние черновики не были достаточно явно обозначены как недоверенные данные, что повышало риск prompt injection и повторного закрепления ошибок модели.

## Новая модель личности

- `personality_core.md` — единственный источник постоянного характера Лизы.
- `prisms/*.md` — восемь общих эмоциональных линз. Они меняют выражение, но не факты, компетентность, усилие и уважение.
- `agents/*.yaml` — разные функции одной личности: расширение идей, проверка реальности, сокращение пути к результату и редактура.
- `synthesis_meta.md` — приоритет источников, разрешение противоречий и запрет раскрывать внутреннюю механику.

Старые per-agent prism-копии удалены. Synthesis теперь использует ту же active prism, что и внутренние перспективы.

## Проблемы исходного RAG

1. Все пользователи читали и записывали одну глобальную память.
2. Сохранялись и сырой phase-1, и phase-2, поэтому случайные идеи становились долговременным контекстом.
3. В payload хранился только `text`: не было user scope, типа записи, источника, запроса, prism и времени.
4. Имена Qdrant collections фактически строились из circuit name и могли расходиться с YAML.
5. Synthesis последовательно складывал результаты коллекций и обрезал список без общего rerank.
6. BM25 возвращал документы с нулевым совпадением.
7. Вся память всех пользователей загружалась в локальный BM25 при startup.
8. Ошибка embedding sidecar отключала не только dense retrieval, но и уже рабочий BM25.
9. Недоступный Qdrant мог повторять сетевой timeout на каждом запросе.
10. Случайные UUID создавали дубли одинаковой памяти.

## Новая модель памяти

Collections:

- `creative` — refined creative perspectives;
- `pragmatic` — refined pragmatic perspectives;
- `effective` — refined effective perspectives;
- `conversation` — final synthesis answers.

Каждая запись содержит:

- `scope` — SHA-256 namespace пользователя и tenant;
- `kind` — `refined_perspective` или `assistant_answer`;
- `source`;
- исходный `query`;
- `prism`;
- `created_at`;
- `text`.

Phase-1 не сохраняется. При отсутствии стабильного user id persistent RAG отключается вместо использования общего anonymous scope. Legacy points без scope игнорируются.

Retrieval:

1. lazy hydration BM25 только для текущего scope;
2. scoped dense query в Qdrant;
3. BM25 без zero-score документов;
4. weighted RRF;
5. cross-encoder rerank;
6. global deduplication и rerank для synthesis по всем четырём collections.

Если embeddings недоступны, retrieval деградирует до BM25. После ошибки Qdrant действует retry cooldown. Длина сохранённого ответа и исходного запроса ограничена env-настройками.

## Проверки

Добавлены тесты на:

- одно personality core для всех nodes;
- одинаковую canonical prism для всех направлений и synthesis;
- trust boundaries памяти и web content;
- user-scope isolation и игнорирование legacy unscoped points;
- provenance payload;
- deterministic upsert;
- refined-only persistence;
- сохранение final synthesis answer;
- BM25 fallback при падении embeddings;
- global cross-collection rerank и deduplication.

Полная проверка реальных TEI и Qdrant containers с загруженными моделями всё ещё должна выполняться как отдельный эксплуатационный smoke-test.
