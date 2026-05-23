To use LLM need LM STUDIO

DB PostgresSQL

if you have own config, use .env file
```
 SERVER_ADDR=":8080"
```

Start with refresh

```
go install github.com/air-verse/air@latest
```
```
air
```

swagger initialization:

```
swag init -g main.go -d ./ -o ./docs
~/go/bin/swag init -g main.go -d ./ -o ./docs
```

minio download https://www.min.io/download?platform=windows

minio start
```
 .\minio.exe server C:\minio\data --console-address ":9090"

 /opt/homebrew/opt/minio/bin/minio server --certs-dir\=/opt/homebrew/etc/minio/certs --address\=:9000 /opt/homebrew/var/minio

 docker compose up -d postgres minio minio-client

```

To extract text need install 
```
https://github.com/oschwartz10612/poppler-windows/releases
```

pg vector

```
https://github.com/pgvector/pgvector
```

docker compose up -d postgres minio minio-client

## Evaluation API (контрольные вопросы и экспертная оценка)

Новые защищенные JWT эндпоинты:

- POST /chats/:chat_id/test-questions
- POST /chats/:chat_id/test-questions/batch
- GET /chats/:chat_id/test-questions?page=1&limit=20
- POST /evaluations/runs
- GET /evaluations/runs/:run_id
- GET /evaluations/runs/:run_id/metrics
- GET /evaluations/runs/:run_id/baseline
- PUT /evaluations/results/:result_id/score

Пример запуска тестового прогона:

```bash
curl -X POST http://localhost:8080/evaluations/runs \
	-H "Authorization: Bearer <JWT>" \
	-H "Content-Type: application/json" \
	-d '{"chat_id":"<CHAT_UUID>","top_k":5,"model":"qwen2.5-7b-instruct"}'
```

Шкала экспертной оценки:

- 0: некорректный ответ
- 1: частично корректный
- 2: корректный

Для вопросов, где в документах заведомо не должно быть ответа, передавайте:

- expected_no_answer: true

Это поле используется для метрик Correct Refusal Rate и Hallucination Rate.

```bash
curl -X PUT http://localhost:8080/evaluations/results/<RESULT_UUID>/score \
	-H "Authorization: Bearer <JWT>" \
	-H "Content-Type: application/json" \
	-d '{"expert_score":2,"expert_feedback":"Ответ корректный","is_correct":true}'
```

Получение KPI прогона:

```bash
curl -X GET http://localhost:8080/evaluations/runs/<RUN_UUID>/metrics \
	-H "Authorization: Bearer <JWT>"
```

Сравнение с обычным keyword-поиском:

```bash
curl -X GET "http://localhost:8080/evaluations/runs/<RUN_UUID>/baseline?limit=1" \
	-H "Authorization: Bearer <JWT>"
```

Готовые воспроизводимые тест-кейсы:

- back/testdata/test_cases.md

Автоматический прогон цепочки upload -> questions -> run -> metrics -> baseline:

```powershell
./scripts/run_evaluation.ps1 \
	-BaseUrl "http://localhost:8080" \
	-Jwt "<JWT>" \
	-ChatId "<CHAT_UUID>" \
	-PdfPath "C:\\path\\to\\sample.pdf"
```

Seed-набор контрольных вопросов (40 штук):

- back/testdata/control_questions_seed_40.json

Загрузка seed одним запросом:

```bash
curl -X POST http://localhost:8080/chats/<CHAT_UUID>/test-questions/batch \
	-H "Authorization: Bearer <JWT>" \
	-H "Content-Type: application/json" \
	--data-binary @testdata/control_questions_seed_40.json
```

## Security controls (реализовано)

- Rate limit для `/ask`: 60 запросов в минуту.
- Rate limit для `/evaluations/runs`: 10 запусков в минуту.
- Валидация загрузок документов:
	- только `.pdf`,
	- проверка MIME по сигнатуре файла,
	- ограничение размера до 20MB.

Поля с секретами API (externalApiKey, embedExternalApiKey) уже шифруются при сохранении настроек.