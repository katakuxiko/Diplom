package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gofiber/fiber/v2"
	_ "github.com/lib/pq"
	"github.com/sashabaranov/go-openai"
	"rsc.io/pdf"
)

type Chunk struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type PgStore struct {
	db *sql.DB
}

var (
	store   *PgStore
	oaicl   *openai.Client
	ctx     = context.Background()
	embedSz = 768 // размер вектора для embeddings
)

// ---------------- MAIN ----------------
func main() {
	// Подключение к Postgres
	connStr := "host=localhost port=5432 user=postgres password=123123 dbname=pdf_ai sslmode=disable"
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		log.Fatal("Ошибка подключения к БД:", err)
	}
	store = &PgStore{db: db}

	// LM Studio client (OpenAI совместимый)
	config := openai.DefaultConfig("not-needed")
	config.BaseURL = "http://localhost:1234/v1" // LM Studio API
	oaicl = openai.NewClientWithConfig(config)

	// API
	app := fiber.New()
	app.Post("/ingest", ingestPDF)
	app.Post("/ask", askQuestion)

	log.Println("🚀 Server started at http://localhost:8080")
	log.Fatal(app.Listen(":8080"))
}

// ---------------- HANDLERS ----------------

// Загрузка PDF и сохранение чанков
func ingestPDF(c *fiber.Ctx) error {
	file, err := c.FormFile("file")
	if err != nil {
		return c.Status(400).SendString("Файл обязателен")
	}

	savePath := filepath.Join("data", "pdfs", file.Filename)
	os.MkdirAll(filepath.Dir(savePath), os.ModePerm)
	if err := c.SaveFile(file, savePath); err != nil {
		return c.Status(500).SendString("Ошибка сохранения файла")
	}

	text, err := extractText(savePath)
	if err != nil {
		return c.Status(500).SendString("Ошибка чтения PDF")
	}

	chunks := chunkText(text, 800, 200)
    for i, ch := range chunks {
        emb, err := getEmbedding(ch.Text)
		fmt.Println("Embedding length:", len(emb))
        if err != nil || len(emb) == 0 {
            log.Println("Ошибка получения embedding:", err)
            continue // пропускаем этот чанк
        }
        ch.ID = fmt.Sprintf("%s_chunk_%d", file.Filename, i)
        if err := store.Add(ch, emb); err != nil {
            log.Println("Ошибка вставки:", err)
        }
    }

	return c.JSON(fiber.Map{"status": "ok", "chunks": len(chunks)})
}

// Вопрос пользователя
type AskRequest struct {
	Query string `json:"query"`
}

func askQuestion(c *fiber.Ctx) error {
	var req AskRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).SendString("Неверный запрос")
	}

	qVec, _ := getEmbedding(req.Query)
	top, _ := store.Search(qVec, 5)

	var contextText strings.Builder
	for _, t := range top {
		contextText.WriteString("[" + t.ID + "]\n" + t.Text + "\n\n")
	}

	answer, err := askGemma(req.Query, contextText.String())
	if err != nil {
		fmt.Println("Ошибка Gemma:", err)
		return c.Status(500).SendString("Ошибка Gemma")
	}

	return c.JSON(fiber.Map{
		"answer":  answer,
		"context": top,
	})
}

// ---------------- PDF UTILS ----------------
func extractText(path string) (string, error) {
    r, err := pdf.Open(path)
    if err != nil {
        return "", err
    }
    var sb strings.Builder
    for i := 1; i <= r.NumPage(); i++ {
        p := r.Page(i)
        if p.V.IsNull() {
            continue
        }
        texts := p.Content().Text
        for _, t := range texts {
            // Удаляем нулевые байты
            clean := strings.ReplaceAll(t.S, "\x00", "")
            sb.WriteString(clean)
        }
        sb.WriteString("\n")
    }
    return sb.String(), nil
}
// ...existing code...

func chunkText(text string, size, overlap int) []Chunk {
	var chunks []Chunk
	words := strings.Fields(text)
	for i := 0; i < len(words); i += (size - overlap) {
		end := i + size
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, Chunk{Text: strings.Join(words[i:end], " ")})
		if end == len(words) {
			break
		}
	}
	return chunks
}

// ---------------- EMBEDDINGS ----------------
func getEmbedding(text string) ([]float32, error) {
	resp, err := oaicl.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: "nomic-embed-text", // должна быть загружена в LM Studio
		Input: []string{text},
	})
	if err != nil {
		return nil, err
	}
	return resp.Data[0].Embedding, nil
}

// ---------------- POSTGRES STORE ----------------
func (s *PgStore) Add(c Chunk, v []float32) error {
	vec := floatsToPgArray(v)
	_, err := s.db.Exec(`
		INSERT INTO chunks (doc_name, chunk_id, text, embedding)
		VALUES ($1, $2, $3, $4::vector)
	`, "doc", c.ID, c.Text, vec)
	return err
}

func (s *PgStore) Search(q []float32, topK int) ([]Chunk, error) {
	vec := floatsToPgArray(q)
	rows, err := s.db.Query(`
		SELECT chunk_id, text
		FROM chunks
		ORDER BY embedding <-> $1::vector
		LIMIT $2
	`, vec, topK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.Text); err != nil {
			return nil, err
		}
		results = append(results, c)
	}
	return results, nil
}

func floatsToPgArray(v []float32) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, val := range v {
		// Лучше использовать %.6f, чтобы не перегружать лишними знаками
		sb.WriteString(fmt.Sprintf("%.6f", val))
		if i < len(v)-1 {
			sb.WriteString(",")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

// ---------------- GEMMA ----------------
func askGemma(query, context string) (string, error) {
	resp, err := oaicl.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: "google/gemma-3n-e4b", // модель должна быть запущена в LM Studio
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: "Ты университетский помощник. Отвечай только на основе контекста."},
			{Role: "user", Content: fmt.Sprintf("Контекст:ТЕСТ\n\nВопрос: %s", context)},
		},
		Temperature: 0.2,
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}
