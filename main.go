package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"os/exec"

	"github.com/gofiber/fiber/v2"
	_ "github.com/lib/pq"
	"github.com/sashabaranov/go-openai"
)

// ---------------- CONFIG ----------------

const (
	embedModelDefault  = "text-embedding-nomic-embed-text-v1.5" // LM Studio embeddings
	chatModelDefault   = "google/gemma-3n-e4b:2"                 // LM Studio chat model
	embedDim           = 768                                     // размерность вектора embeddings
	chunkSizeTokens    = 220                                     // грубая оценка "токенов" (по словам)
	chunkOverlapTokens = 40
	topK               = 5
	maxContextChars    = 12000 // приблизительно ~3000 токенов
)

// ---------------- TYPES ----------------

type Chunk struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type AskRequest struct {
	Query string `json:"query"`
	Model string `json:"model,omitempty"` // необязательно: выбрать runtime-модель
	TopK  int    `json:"topK,omitempty"`  // необязательно: сколько чанков возвращать
}

type PgStore struct {
	db *sql.DB
}

// ---------------- GLOBALS ----------------

var (
	ctx       = context.Background()
	dbStore   *PgStore
	oaicl     *openai.Client
	embedName string
	chatName  string
)

// ---------------- MAIN ----------------

func main() {
	// ENV
	pgConn := getenv("PG_CONN", "host=localhost port=5432 user=postgres password=123123 dbname=pdf_ai sslmode=disable")
	baseURL := getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
	embedName = getenv("EMBED_MODEL", embedModelDefault)
	chatName = getenv("LLM_MODEL", chatModelDefault)

	// DB
	db, err := sql.Open("postgres", pgConn)
	if err != nil {
		log.Fatal("Ошибка подключения к БД:", err)
	}
	dbStore = &PgStore{db: db}
	if err := ensureSchema(db); err != nil {
		log.Fatal("Ошибка инициализации схемы:", err)
	}

	// LM Studio client (OpenAI-compatible)
	cfg := openai.DefaultConfig("not-needed")
	cfg.BaseURL = baseURL
	oaicl = openai.NewClientWithConfig(cfg)

	// API
	app := fiber.New()

	app.Get("/health", func(c *fiber.Ctx) error { return c.SendString("ok") })
	app.Get("/models", listModels) // посмотреть какие модели подняты в LM Studio
	app.Post("/ingest", ingestPDF) // загрузить PDF и проиндексировать
	app.Post("/ask", askQuestion)  // RAG: поиск релевантных чанков + ответ LLM

	log.Println("🚀 Server started at http://localhost:8080")
	log.Fatal(app.Listen(":8080"))
}

// ---------------- HANDLERS ----------------

// /models — вернуть список моделей LM Studio
func listModels(c *fiber.Ctx) error {
	resp, err := oaicl.ListModels(ctx)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(resp.Models)
}

// /ingest — принять PDF, распарсить и сохранить чанки+векторы
func ingestPDF(c *fiber.Ctx) error {
	file, err := c.FormFile("file")
	if err != nil {
		return c.Status(400).SendString("Файл обязателен (form field: file)")
	}
	savePath := filepath.Join("data", "pdfs", timestamped(file.Filename))
	if err := os.MkdirAll(filepath.Dir(savePath), 0o755); err != nil {
		return c.Status(500).SendString("Ошибка подготовки папки")
	}
	if err := c.SaveFile(file, savePath); err != nil {
		return c.Status(500).SendString("Ошибка сохранения файла")
	}

	text, err := extractText(savePath)
	if err != nil {
		return c.Status(500).SendString("Ошибка чтения PDF")
	}
	text = sanitize(text)
	print(len(text), "символов извлечено из PDF")

	chunks := chunkByWords(text, chunkSizeTokens, chunkOverlapTokens)
	print(len(chunks), "чанков создано")
	print(chunks[0].Text) // пример
	if len(chunks) == 0 {
		return c.Status(400).SendString("Не удалось извлечь текст из PDF")
	}

	docName := filepath.Base(savePath)
	okCnt := 0
	for i := range chunks {
		chunks[i].ID = fmt.Sprintf("%s_chunk_%d", docName, i)

		emb, err := getEmbedding(chunks[i].Text)
		if err != nil {
			log.Printf("embedding error for %s: %v", chunks[i].ID, err)
			continue
		}
		if len(emb) != embedDim {
			log.Printf("embedding dim mismatch for %s: got %d want %d", chunks[i].ID, len(emb), embedDim)
			continue
		}
		if err := dbStore.Add(docName, chunks[i], emb); err != nil {
			log.Printf("db insert error for %s: %v", chunks[i].ID, err)
			continue
		}
		okCnt++
	}

	return c.JSON(fiber.Map{
		"status":        "ok",
		"doc":           docName,
		"chunks_total":  len(chunks),
		"chunks_saved":  okCnt,
		"embed_model":   embedName,
		"vector_dim":    embedDim,
		"storage":       "pgvector",
		"table":         "chunks",
		"index":         "ivfflat cosine",
	})
}

// /ask — выполнить RAG: поиск + ответ LLM
func askQuestion(c *fiber.Ctx) error {
	var req AskRequest
	if err := c.BodyParser(&req); err != nil || strings.TrimSpace(req.Query) == "" {
		return c.Status(400).SendString("Неверный запрос. Ожидается JSON: {\"query\":\"...\"}")
	}

	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		modelName = chatName
	}
	k := req.TopK
	if k <= 0 || k > 20 {
		k = topK
	}

	// 1) embedding для запроса
	qVec, err := getEmbedding(req.Query)
	if err != nil {
		return c.Status(500).SendString("Ошибка получения embedding запроса")
	}

	// 2) поиск релевантных чанков
	top, err := dbStore.Search(qVec, k)
	if err != nil {
		return c.Status(500).SendString("Ошибка поиска релевантных чанков")
	}
	if len(top) == 0 {
		return c.JSON(fiber.Map{
			"answer":  "Не нашёл релевантный контекст в базе.",
			"context": []Chunk{},
		})
	}
	println(len(top), "релевантных чанков найдено")

	// 3) сбор контекста с ограничением по длине
	var b strings.Builder
	for _, t := range top {
		// добавим явные маркеры источника
		blk := fmt.Sprintf("[%s]\n%s\n\n", t.ID, t.Text)
		if b.Len()+len(blk) > maxContextChars {
			break
		}
		b.WriteString(blk)
	}
	contextText := b.String()
	if contextText == "" && len(top) > 0 {
		// хотя бы кусок от первого
		first := fmt.Sprintf("[%s]\n%s\n\n", top[0].ID, truncateRunes(top[0].Text, maxContextChars/2))
		contextText = first
	}

	// 4) запрос к chat-модели
	ans, err := askLLM(modelName, req.Query, contextText)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": fmt.Sprintf("Ошибка LLM: %v", err)})
	}

	return c.JSON(fiber.Map{
		"answer":  ans,
		"context": top,
		"model":   modelName,
	})
}

// ---------------- PDF / TEXT UTILS ----------------

// Извлечение текста из PDF (rsc.io/pdf)
func extractText(path string) (string, error) {
    cmd := exec.Command("pdftotext", "-enc", "UTF-8", path, "-")
    out, err := cmd.Output()
    if err != nil {
        return "", err
    }
    return string(out), nil
}
// Простая очистка текста
func sanitize(s string) string {
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.Join(strings.Fields(s), " ")
	return s
}

// Разбиение по словам с overlap
func chunkByWords(text string, size, overlap int) []Chunk {
	words := strings.Fields(text)
	if size <= 0 {
		size = 200
	}
	if overlap < 0 {
		overlap = 0
	}
	var out []Chunk
	for i := 0; i < len(words); i += max(1, size-overlap) {
		end := i + size
		if end > len(words) {
			end = len(words)
		}
		out = append(out, Chunk{Text: strings.Join(words[i:end], " ")})
		if end == len(words) {
			break
		}
	}
	return out
}

func truncateRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n])
}

// ---------------- EMBEDDINGS ----------------

func getEmbedding(text string) ([]float32, error) {
	model := embedName
	resp, err := oaicl.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(model), // явное приведение string → EmbeddingModel
		Input: []string{text},
	})

	if err != nil {
		return nil, err
	}
	emb := resp.Data[0].Embedding
	// openai lib отдаёт []float32 уже, но проверим
	out := make([]float32, len(emb))
	copy(out, emb)
	return out, nil
}

// ---------------- POSTGRES + PGVECTOR ----------------

func ensureSchema(db *sql.DB) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE TABLE IF NOT EXISTS chunks (
			id SERIAL PRIMARY KEY,
			doc_name TEXT,
			chunk_id TEXT,
			text TEXT,
			embedding vector(768)
		)`,
		// ivfflat требует ANALYZE; создадим, если нет
		`DO $$
		BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM pg_class c
				JOIN pg_namespace n ON n.oid=c.relnamespace
				WHERE c.relname='chunks_embedding_ivfflat_idx'
			) THEN
				EXECUTE 'CREATE INDEX chunks_embedding_ivfflat_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)';
			END IF;
		END $$;`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}
	// Рекомендуется: ANALYZE для ivfflat
	_, _ = db.Exec(`ANALYZE chunks`)
	return nil
}

func (s *PgStore) Add(doc string, c Chunk, v []float32) error {
	vec := floatsToPgVectorLiteral(v) // формируем строку вида [0.1,0.2,...]
	_, err := s.db.Exec(`
		INSERT INTO chunks (doc_name, chunk_id, text, embedding)
		VALUES ($1, $2, $3, $4::vector)
	`, doc, c.ID, c.Text, vec)
	return err
}

func (s *PgStore) Search(q []float32, k int) ([]Chunk, error) {
	if k <= 0 {
		k = 5
	}
	vec := floatsToPgVectorLiteral(q)
	rows, err := s.db.Query(`
		SELECT chunk_id, text
		FROM chunks
		ORDER BY embedding <-> $1::vector
		LIMIT $2
	`, vec, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var res []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.Text); err != nil {
			return nil, err
		}
		res = append(res, c)
	}
	return res, rows.Err()
}

// pgvector ожидает формат: [0.123,0.456,...]
func floatsToPgVectorLiteral(v []float32) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, f := range v {
		// используем %.6f — достаточно для pgvector и экономит размер
		sb.WriteString(fmt.Sprintf("%.6f", float64(f)))
		if i < len(v)-1 {
			sb.WriteString(",")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

// ---------------- LLM (Chat) ----------------

func askLLM(modelName, query, context string) (string, error) {
	// Упрощаем: всё в user-сообщение (LM Studio иногда не любит role=system)
	prompt := fmt.Sprintf(
		"Ты университетский помощник. Отвечай строго на основе контекста.\n\nКонтекст:\n%s\n\nВопрос: %s\n\nЕсли информации недостаточно, скажи об этом честно.",
		context, query,
	)

	resp, err := oaicl.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: modelName,
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.2,
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("LLM вернул пустой ответ")
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

// ---------------- HELPERS ----------------

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func timestamped(name string) string {
	ts := time.Now().Format("20060102_150405")
	return fmt.Sprintf("%s__%s", ts, name)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

//
// (опционально) если когда-нибудь понадобится downcast/round:
//
//nolint:unused
func f32round(x float32, p int) float32 {
	scale := float32(math.Pow10(p))
	return float32(math.Round(float64(x*scale))) / scale
}
