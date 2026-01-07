package api

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/pdf"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
)

// Handler хранит зависимости для обработчиков
type Handler struct {
	rag          *service.RAGService
	llm          *service.LLMClient
	chunkService *service.ChunkService
}

// NewHandler конструктор
func NewHandler(rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService) *Handler {
	return &Handler{rag: rag, llm: llm, chunkService: chunkService}
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
	saveName := utils.Timestamped(file.Filename)
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
	parts := pdf.ChunkBySentences(txt, chunkSize, chunkOverlap)
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

		if err := h.chunkService.SaveChunk(ch, emb); err != nil {
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

// AskQuestion godoc
// @Summary Ask a question to the RAG system (LLM + search)
// @Description Получение ответа на вопрос с использованием Retrieval-Augmented Generation (RAG)
// @Tags RAG
// @Accept json
// @Produce json
// @Param request body models.AskRequest true "Request payload, e.g., {\"query\":\"...\", \"top_k\":5, \"model\":\"gpt-4\"}"
// @Success 200 {object} map[string]interface{} "Answer and context chunks"
// @Failure 400 {object} map[string]string "Invalid request"
// @Failure 500 {object} map[string]string "Internal server error"
// @Router /ask [post]
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

	if req.ChatID == uuid.Nil {
		return c.Status(400).JSON(fiber.Map{"error": "chat_id is required"})
	}

	// Собираем настройки LLM
	settings := req.Settings
	if settings == nil {
		settings = &models.AskSettings{}
	}
	if settings.Model == "" && req.Model != "" {
		settings.Model = req.Model
	}

	// если модель указана, используем её; иначе — дефолт внутри LLMClient/Service
	ans, ctxChunks, err := h.rag.Ask(req.Query, k, req.ChatID, settings)
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
