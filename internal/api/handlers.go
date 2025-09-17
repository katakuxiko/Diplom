package api

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/pdf"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/store"
	"github.com/katakuxiko/Diplom/internal/util"
	"github.com/pgvector/pgvector-go"
)

// Handler хранит зависимости для обработчиков
type Handler struct {
	rag   *service.RAGService
	llm   *service.LLMClient
	store *store.PgStore
}

// NewHandler конструктор
func NewHandler(rag *service.RAGService, llm *service.LLMClient, s *store.PgStore) *Handler {
	return &Handler{rag: rag, llm: llm, store: s}
}

// Health — простая проверка
func (h *Handler) Health(c *fiber.Ctx) error {
	return c.SendString("ok")
}

// ListModels — проксирование к LM Studio (список моделей)
func (h *Handler) ListModels(c *fiber.Ctx) error {
	models, err := h.llm.ListModels()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(models)
}

// IngestPDF — загрузка PDF, извлечение текста, разбиение, embeddings, сохранение в pgvector
func (h *Handler) IngestPDF(c *fiber.Ctx) error {
	// получаем файл
	file, err := c.FormFile("file")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "file is required (form field: file)"})
	}

	// сохраняем локально
	saveDir := filepath.Join("data", "pdfs")
	if err := os.MkdirAll(saveDir, 0o755); err != nil {
		log.Printf("mkdir error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to prepare storage"})
	}
	saveName := util.Timestamped(file.Filename)
	savePath := filepath.Join(saveDir, saveName)
	if err := c.SaveFile(file, savePath); err != nil {
		log.Printf("save file error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to save file"})
	}

	// extract text
	txt, err := pdf.ExtractText(savePath)
	if err != nil {
		log.Printf("extract error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to extract text from pdf"})
	}
	txt = pdf.Sanitize(txt)
	if len(txt) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "no text extracted from PDF"})
	}

	// chunk
	// берём параметры по умолчанию (можно вынести в config/env)
	const chunkSize = 220
	const chunkOverlap = 40
	parts := pdf.ChunkByWords(txt, chunkSize, chunkOverlap)
	if len(parts) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "no chunks created"})
	}

	docName := filepath.Base(savePath)
	saved := 0
	for i, p := range parts {
		chunk_name := fmt.Sprintf("%s_chunk_%d", docName, i)
		ch := models.Chunk{Text: p, Filepath: savePath, DocName: chunk_name, ChunkName: chunk_name}

		emb, err := h.llm.Embedding(p)
		if err != nil {
			log.Printf("embedding error (%s): %v", chunk_name, err)
			continue
		}

		vec := pgvector.NewVector(emb) // ✨ конвертация

		if err := h.store.Add(docName, ch, vec); err != nil {
			log.Printf("db insert error (%s): %v", chunk_name, err)
			continue
		}
		saved++
	}

	return c.JSON(fiber.Map{
		"status":       "ok",
		"doc":          docName,
		"chunks_total": len(parts),
		"chunks_saved": saved,
	})
}

// AskQuestion — RAG: поиск + LLM
func (h *Handler) AskQuestion(c *fiber.Ctx) error {
	var req models.AskRequest
	if err := c.BodyParser(&req); err != nil || len(req.Query) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "invalid request, expected JSON: {\"query\":\"...\"}"})
	}

	modelName := req.Model
	// topK fallback
	k := req.TopK
	if k <= 0 {
		k = 5
	}
	// если модель указана, используем её; иначе — дефолт внутри LLMClient/Service
	ans, ctxChunks, err := h.rag.Ask(req.Query, k)
	if err != nil {
		log.Printf("rag ask error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.JSON(fiber.Map{
		"answer":  ans,
		"context": ctxChunks,
		"model":   modelName,
	})
}
