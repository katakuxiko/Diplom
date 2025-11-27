# Diplom Project

## Описание

API для дипломного проекта с поддержкой обработки PDF документов и чат-функциональности.

## Технологии

- Go 1.24.1
- PostgreSQL с pgvector расширением
- MinIO для хранения файлов
- Docker & Docker Compose
- Swagger/OpenAPI документация

## Запуск через Docker

### Предварительные требования

- Docker Desktop установлен и запущен
- Docker Compose

### Инструкции по запуску

1. Клонируйте репозиторий:

```bash
git clone <repository-url>
cd Diplom
```

2. Запустите все сервисы:

```bash
docker-compose up --build
```

Это запустит:

- PostgreSQL (порт 5432) с базой данных `pdf_ai` и pgvector расширением
- MinIO (порт 9000, веб-интерфейс 9001) для хранения файлов
- Go API приложение (порт 8080)

3. Проверьте работоспособность:

- API: http://localhost:8080
- Swagger документация: http://localhost:8080/swagger/
- MinIO веб-интерфейс: http://localhost:9001 (admin/password123)

### Переменные окружения

Все переменные настроены в `docker-compose.yml` и `.env` файле:

```env
# Database
PG_CONN=host=postgres port=5432 user=postgres password=123123 dbname=pdf_ai sslmode=disable

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password123
MINIO_BUCKET=documents

# JWT
JWT_SECRET=sadadasdasd
```

### Остановка сервисов

```bash
docker-compose down
```

Для полной очистки (удаление volumes):

```bash
docker-compose down -v
```

## Структура проекта

```
├── internal/           # Внутренняя логика приложения
│   ├── api/           # API обработчики и маршруты
│   ├── config/        # Конфигурация
│   ├── dto/           # Data Transfer Objects
│   ├── handlers/      # HTTP обработчики
│   ├── middleware/    # Middleware (JWT и др.)
│   ├── models/        # Модели данных
│   ├── repository/    # Слой доступа к данным
│   ├── service/       # Бизнес-логика
│   ├── storage/       # Работа с файловым хранилищем
│   └── utils/         # Утилиты
├── docs/              # Swagger документация
├── init-db/           # SQL скрипты для инициализации БД
├── docker-compose.yml # Конфигурация Docker Compose
├── Dockerfile         # Образ для Go приложения
└── .env               # Переменные окружения
```
