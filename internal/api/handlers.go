package api

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/pdf"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
)

// Handler хранит зависимости для обработчиков
type Handler struct {
	rag             *service.RAGService
	llm             *service.LLMClient
	chunkService    *service.ChunkService
	chatHistoryRepo *repository.ChatHistoryRepository
	messageRepo     *repository.MessageRepository
}

// NewHandler конструктор
func NewHandler(rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService, chatHistoryRepo *repository.ChatHistoryRepository, messageRepo *repository.MessageRepository) *Handler {
	return &Handler{rag: rag, llm: llm, chunkService: chunkService, chatHistoryRepo: chatHistoryRepo, messageRepo: messageRepo}
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

	claims, ok := c.Locals("user").(jwt.MapClaims)
	if !ok {
		return c.Status(401).JSON(fiber.Map{"error": "unauthorized"})
	}

	accessLevel := 0
	if al, ok := claims["access_level"].(float64); ok {
		accessLevel = int(al)
	}

	if role, ok := claims["role"].(string); ok {
		if role == "superuser" {
			accessLevel = 100
		}
		if role == "chat_user" {
			if chatStr, ok := claims["chat_id"].(string); ok && chatStr != "" {
				if claimChat, err := uuid.Parse(chatStr); err == nil {
					if claimChat != req.ChatID {
						return c.Status(403).JSON(fiber.Map{"error": "chat mismatch"})
					}
				}
			}
		}
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
	ans, ctxChunks, err := h.rag.Ask(req.Query, k, req.ChatID, settings, accessLevel)
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

// GetChatHistoryForAdmin — получить истории и сообщения чата для админа
// @Summary      Get chat histories with messages
// @Tags         chat
// @Produce      json
// @Param        chat_id path string true "Chat ID"
// @Success      200 {array} dto.ChatHistoryWithMessagesResponse
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /chats/{chat_id}/history [get]
// @Security     BearerAuth
func (h *Handler) GetChatHistoryForAdmin(c *fiber.Ctx) error {
	chatID := c.Params("chat_id")
	if chatID == "" {
		return c.Status(400).JSON(fiber.Map{"error": "chat_id is required"})
	}
	histories, err := h.chatHistoryRepo.GetHistoriesWithMessages(chatID)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	response := make([]dto.ChatHistoryWithMessagesResponse, 0, len(histories))
	for _, hst := range histories {
		messages := make([]dto.MessageResponse, 0, len(hst.Messages))
		for _, m := range hst.Messages {
			messages = append(messages, dto.MessageResponse{
				ID:            m.ID,
				ChatHistoryID: m.ChatHistoryID,
				Text:          m.Text,
				Role:          m.Role,
				CreatedDate:   m.CreatedDate,
			})
		}

		response = append(response, dto.ChatHistoryWithMessagesResponse{
			ID:       hst.ID,
			ChatID:   hst.ChatID,
			UserID:   hst.UserID,
			Username: hst.User.Username,
			Messages: messages,
		})
	}

	return c.JSON(response)
}

// CreateChatHistory — создать историю чата для пользователя
// @Summary      Create chat history
// @Description  Создать историю чата для конкретного пользователя
// @Tags         chat
// @Accept       json
// @Produce      json
// @Param        request body map[string]string true "{chat_id, user_id}"
// @Success      201 {object} models.ChatHistory
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /chat_histories [post]
// @Security     BearerAuth
func (h *Handler) CreateChatHistory(c *fiber.Ctx) error {
	var req struct {
		ChatID string `json:"chat_id"`
		UserID string `json:"user_id"`
	}
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	chatID, err := uuid.Parse(req.ChatID)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}
	userID, err := uuid.Parse(req.UserID)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid user_id"})
	}
	history := models.ChatHistory{ChatID: chatID, UserID: userID}
	if err := h.chatHistoryRepo.Create(&history); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(201).JSON(history)
}

// CreateMessage — добавить сообщение в историю
// @Summary      Create message
// @Description  Добавить сообщение в историю чата
// @Tags         chat
// @Accept       json
// @Produce      json
// @Param        request body map[string]string true "{chat_history_id, text, role}"
// @Success      201 {object} models.Message
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /messages [post]
// @Security     BearerAuth
func (h *Handler) CreateMessage(c *fiber.Ctx) error {
	var req struct {
		ChatHistoryID string `json:"chat_history_id"`
		Text          string `json:"text"`
		Role          string `json:"role"`
	}
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	historyID, err := uuid.Parse(req.ChatHistoryID)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_history_id"})
	}
	if req.Text == "" {
		return c.Status(400).JSON(fiber.Map{"error": "text is required"})
	}
	msg := models.Message{ChatHistoryID: historyID, Text: req.Text, Role: req.Role}
	if msg.Role == "" {
		msg.Role = "user"
	}
	if err := h.messageRepo.Create(&msg); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(201).JSON(msg)
}
