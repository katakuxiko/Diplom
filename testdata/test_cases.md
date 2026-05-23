# Тест-кейсы для доказательной проверки качества RAG

## Формат фиксации результата по каждому вопросу

Обязательно сохранять:

1. Вопрос
2. Найденный фрагмент документа
3. Ответ модели
4. Экспертную оценку (0/1/2) + комментарий
5. Время ответа (мс)
6. Флаг expected_no_answer

## Case 1: Загрузка и индексация документа

Вход:

- Endpoint: POST /documents/upload
- Form-data:
  - chat_id: <UUID>
  - file: sample_regulations.pdf

Ожидаемый результат:

- HTTP 200
- chunks_total > 0
- chunks_saved > 0

## Case 2: Batch-загрузка контрольных вопросов

Вход:

- Endpoint: POST /chats/{chat_id}/test-questions/batch
- Body: testdata/control_questions_seed_40.json

Ожидаемый результат:

- HTTP 201
- count = 40

## Case 3: Запуск полного evaluation run

Вход:

- Endpoint: POST /evaluations/runs
- Body:
  {
    "chat_id": "<UUID>",
    "top_k": 5,
    "model": "qwen2.5-7b-instruct"
  }

Ожидаемый результат:

- HTTP 201
- total_questions = 40
- results.length = 40
- Для каждого result есть question + response_time_ms

## Case 4: Экспертная оценка ответа

Вход:

- Endpoint: PUT /evaluations/results/{result_id}/score
- Body:
  {
    "expert_score": 2,
    "expert_feedback": "Ответ совпадает с регламентом",
    "is_correct": true
  }

Ожидаемый результат:

- HTTP 200
- expert_score установлен
- evaluated_at заполнен

## Case 5: Получение KPI метрик

Вход:

- Endpoint: GET /evaluations/runs/{run_id}/metrics

Ожидаемый результат:

- HTTP 200
- Поля:
  - correct_answer_rate
  - correct_refusal_rate
  - hallucination_rate
  - avg_latency_ms
  - p95_latency_ms
  - error_rate

## Case 6: Сравнение с обычным поиском

Вход:

- Endpoint: GET /evaluations/runs/{run_id}/baseline?limit=1

Ожидаемый результат:

- HTTP 200
- Поля:
  - rag_context_hit_rate
  - baseline_context_hit_rate
  - baseline_avg_search_ms
  - baseline_p95_search_ms

## Case 7: Негативный сценарий без контекста

Подготовка:

- Вопрос с expected_no_answer = true

Вход:

- Запуск evaluation run

Ожидаемый результат:

- В метриках учитывается correct_refusal_rate
- Некорректный ответ при expected_no_answer увеличивает hallucination_rate

## Рекомендуемый порядок запуска

1. Загрузить документы
2. Импортировать вопросы
3. Запустить run
4. Проставить экспертные оценки
5. Получить /metrics
6. Получить /baseline
