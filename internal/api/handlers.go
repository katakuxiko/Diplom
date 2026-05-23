package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

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
	chatSettings    *service.ChatSettingsService
	chatHistoryRepo *repository.ChatHistoryRepository
	messageRepo     *repository.MessageRepository
	evaluation      *service.EvaluationService
}

// NewHandler конструктор
func NewHandler(rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService, chatSettings *service.ChatSettingsService, chatHistoryRepo *repository.ChatHistoryRepository, messageRepo *repository.MessageRepository, evaluation *service.EvaluationService) *Handler {
	return &Handler{rag: rag, llm: llm, chunkService: chunkService, chatSettings: chatSettings, chatHistoryRepo: chatHistoryRepo, messageRepo: messageRepo, evaluation: evaluation}
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
	if err := utils.ValidatePDFUpload(file); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
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
	// Попробуем получить chat_id из формы (опционально) и загрузить per-chat настройки
	chatIDStr := c.FormValue("chat_id")
	var chatID uuid.UUID
	if chatIDStr != "" {
		if cid, err := uuid.Parse(chatIDStr); err == nil {
			chatID = cid
		}
	}

	for i, p := range parts {
		chunk_name := fmt.Sprintf("%s_chunk_%d", docName, i)
		ch := models.Chunk{Text: p, Filepath: savePath, DocName: chunk_name, ChunkName: chunk_name}

		var emb []float32
		var err error
		if chatID != uuid.Nil && h.chatSettings != nil {
			if cs, serr := h.chatSettings.GetByChatID(context.Background(), chatID); serr == nil && cs != nil && cs.Settings != nil {
				raw, _ := json.Marshal(cs.Settings)
				var dbSettings models.AskSettings
				if jerr := json.Unmarshal(raw, &dbSettings); jerr == nil {
					// Попробуем дешифровать ключи, если они были сохранены зашифрованными
					if dbSettings.ExternalAPIKey != "" {
						if dec, derr := utils.DecryptString(dbSettings.ExternalAPIKey); derr == nil {
							dbSettings.ExternalAPIKey = dec
						}
					}
					if dbSettings.EmbedExternalAPIKey != "" {
						if dec2, derr2 := utils.DecryptString(dbSettings.EmbedExternalAPIKey); derr2 == nil {
							dbSettings.EmbedExternalAPIKey = dec2
						}
					}
					emb, err = h.llm.EmbeddingWithSettings(p, &dbSettings)
				} else {
					emb, err = h.llm.Embedding(p)
				}
			} else {
				emb, err = h.llm.Embedding(p)
			}
		} else {
			emb, err = h.llm.Embedding(p)
		}

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

	// Попробуем получить claims из контекста; если их нет — разрешаем анонимный доступ с accessLevel=0
	accessLevel := 0
	if v := c.Locals("user"); v != nil {
		if claims, ok := v.(jwt.MapClaims); ok {
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
		}
	}

	// Собираем настройки LLM
	settings := req.Settings
	if settings == nil {
		settings = &models.AskSettings{}
	}
	// если модель в теле запроса не указана, используем параметр model из query
	if settings.Model == "" && req.Model != "" {
		settings.Model = req.Model
	}
	// Попробуем получить настройки чата из БД и слить их (db -> override by request)
	if h.chatSettings != nil {
		if cs, err := h.chatSettings.GetByChatID(context.Background(), req.ChatID); err == nil && cs != nil && cs.Settings != nil {
			// marshal JSONB -> bytes
			raw, _ := json.Marshal(cs.Settings)
			var dbSettings models.AskSettings
			if err := json.Unmarshal(raw, &dbSettings); err == nil {
				// Попробуем дешифровать ключи, если они были сохранены зашифрованными
				if dbSettings.ExternalAPIKey != "" {
					if dec, derr := utils.DecryptString(dbSettings.ExternalAPIKey); derr == nil {
						dbSettings.ExternalAPIKey = dec
					}
				}
				if dbSettings.EmbedExternalAPIKey != "" {
					if dec2, derr2 := utils.DecryptString(dbSettings.EmbedExternalAPIKey); derr2 == nil {
						dbSettings.EmbedExternalAPIKey = dec2
					}
				}
				// Применяем только те поля из dbSettings, которые не заданы в request (request имеет приоритет)
				if settings.Provider == "" {
					settings.Provider = dbSettings.Provider
				}
				if settings.ExternalAPIKey == "" {
					settings.ExternalAPIKey = dbSettings.ExternalAPIKey
				}
				if settings.ExternalBaseURL == "" {
					settings.ExternalBaseURL = dbSettings.ExternalBaseURL
				}
				if settings.EmbedProvider == "" {
					settings.EmbedProvider = dbSettings.EmbedProvider
				}
				if settings.EmbedExternalAPIKey == "" {
					settings.EmbedExternalAPIKey = dbSettings.EmbedExternalAPIKey
				}
				if settings.EmbedExternalBaseURL == "" {
					settings.EmbedExternalBaseURL = dbSettings.EmbedExternalBaseURL
				}
				if settings.Model == "" {
					settings.Model = dbSettings.Model
				}
				if settings.EmbedModel == "" {
					settings.EmbedModel = dbSettings.EmbedModel
				}
				if settings.SystemPrompt == "" {
					settings.SystemPrompt = dbSettings.SystemPrompt
				}
				if settings.MaxTokens == 0 {
					settings.MaxTokens = dbSettings.MaxTokens
				}
				if settings.Temperature == 0 {
					settings.Temperature = dbSettings.Temperature
				}
			}
		}
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

		var userID *uuid.UUID
		var username string
		if hst.UserID != nil {
			userID = hst.UserID
		}
		if hst.User != nil {
			username = hst.User.Username
		}

		response = append(response, dto.ChatHistoryWithMessagesResponse{
			ID:       hst.ID,
			ChatID:   hst.ChatID,
			UserID:   userID,
			Username: username,
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
	var userID *uuid.UUID
	if req.UserID != "" {
		uid, err := uuid.Parse(req.UserID)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid user_id"})
		}
		userID = &uid
	} else {
		// anonymous user: keep userID nil
		userID = nil
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

func (h *Handler) CreateTestQuestion(c *fiber.Ctx) error {
	chatID, err := uuid.Parse(c.Params("chat_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}

	var req dto.TestQuestionCreateRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}

	if strings.TrimSpace(req.Text) == "" {
		return c.Status(400).JSON(fiber.Map{"error": "text is required"})
	}

	question := &models.TestQuestion{
		ChatID:           chatID,
		Text:             strings.TrimSpace(req.Text),
		Category:         strings.TrimSpace(req.Category),
		ExpectedAnswer:   strings.TrimSpace(req.ExpectedAnswer),
		ExpectedNoAnswer: req.ExpectedNoAnswer,
		SourceHint:       strings.TrimSpace(req.SourceHint),
		OrderNum:         req.OrderNum,
	}

	if err := h.evaluation.CreateTestQuestion(context.Background(), question); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(201).JSON(mapTestQuestionToDTO(*question))
}

func (h *Handler) CreateTestQuestionsBatch(c *fiber.Ctx) error {
	chatID, err := uuid.Parse(c.Params("chat_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}

	var req []dto.TestQuestionCreateRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}

	if len(req) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "empty question list"})
	}

	created := make([]dto.TestQuestionResponse, 0, len(req))
	for _, item := range req {
		if strings.TrimSpace(item.Text) == "" {
			return c.Status(400).JSON(fiber.Map{"error": "each question must contain text"})
		}

		question := &models.TestQuestion{
			ChatID:           chatID,
			Text:             strings.TrimSpace(item.Text),
			Category:         strings.TrimSpace(item.Category),
			ExpectedAnswer:   strings.TrimSpace(item.ExpectedAnswer),
			ExpectedNoAnswer: item.ExpectedNoAnswer,
			SourceHint:       strings.TrimSpace(item.SourceHint),
			OrderNum:         item.OrderNum,
		}

		if err := h.evaluation.CreateTestQuestion(context.Background(), question); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}

		created = append(created, mapTestQuestionToDTO(*question))
	}

	return c.Status(201).JSON(fiber.Map{"data": created, "count": len(created)})
}

func (h *Handler) ListTestQuestions(c *fiber.Ctx) error {
	chatID, err := uuid.Parse(c.Params("chat_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}

	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 20)

	questions, total, err := h.evaluation.ListTestQuestionsByChat(context.Background(), chatID, page, limit)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	resp := dto.PaginatedTestQuestions{
		Data:  make([]dto.TestQuestionResponse, 0, len(questions)),
		Page:  page,
		Limit: limit,
		Total: total,
	}

	for _, q := range questions {
		resp.Data = append(resp.Data, mapTestQuestionToDTO(q))
	}

	return c.JSON(resp)
}

func (h *Handler) DeleteTestQuestion(c *fiber.Ctx) error {
	chatID, err := uuid.Parse(c.Params("chat_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}

	questionID, err := uuid.Parse(c.Params("question_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid question_id"})
	}

	if err := h.evaluation.DeleteTestQuestion(context.Background(), chatID, questionID); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.SendStatus(204)
}

func (h *Handler) StartEvaluationRun(c *fiber.Ctx) error {
	var req dto.EvaluationRunCreateRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}

	if req.ChatID == uuid.Nil {
		return c.Status(400).JSON(fiber.Map{"error": "chat_id is required"})
	}

	topK := req.TopK
	if topK <= 0 {
		topK = 5
	}

	run := &models.EvaluationRun{
		ChatID:    req.ChatID,
		Status:    "in_progress",
		Model:     strings.TrimSpace(req.Model),
		TopK:      topK,
		StartedAt: time.Now(),
	}

	if err := h.evaluation.CreateRun(context.Background(), run); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	questions, err := h.evaluation.ListAllTestQuestionsByChat(context.Background(), req.ChatID)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	if len(questions) == 0 {
		run.Status = "failed"
		completed := time.Now()
		run.CompletedAt = &completed
		_ = h.evaluation.UpdateRun(context.Background(), run)
		return c.Status(400).JSON(fiber.Map{"error": "no test questions found for chat"})
	}

	settings := &models.AskSettings{}
	if req.Settings != nil {
		settings = req.Settings
	}
	if req.Model != "" && settings.Model == "" {
		settings.Model = req.Model
	}

	if h.chatSettings != nil {
		if cs, err := h.chatSettings.GetByChatID(context.Background(), req.ChatID); err == nil && cs != nil && cs.Settings != nil {
			raw, _ := json.Marshal(cs.Settings)
			var dbSettings models.AskSettings
			if err := json.Unmarshal(raw, &dbSettings); err == nil {
				if dbSettings.ExternalAPIKey != "" {
					if dec, derr := utils.DecryptString(dbSettings.ExternalAPIKey); derr == nil {
						dbSettings.ExternalAPIKey = dec
					}
				}
				if dbSettings.EmbedExternalAPIKey != "" {
					if dec2, derr2 := utils.DecryptString(dbSettings.EmbedExternalAPIKey); derr2 == nil {
						dbSettings.EmbedExternalAPIKey = dec2
					}
				}

				if settings.Provider == "" {
					settings.Provider = dbSettings.Provider
				}
				if settings.ExternalAPIKey == "" {
					settings.ExternalAPIKey = dbSettings.ExternalAPIKey
				}
				if settings.ExternalBaseURL == "" {
					settings.ExternalBaseURL = dbSettings.ExternalBaseURL
				}
				if settings.EmbedProvider == "" {
					settings.EmbedProvider = dbSettings.EmbedProvider
				}
				if settings.EmbedExternalAPIKey == "" {
					settings.EmbedExternalAPIKey = dbSettings.EmbedExternalAPIKey
				}
				if settings.EmbedExternalBaseURL == "" {
					settings.EmbedExternalBaseURL = dbSettings.EmbedExternalBaseURL
				}
				if settings.Model == "" {
					settings.Model = dbSettings.Model
				}
				if settings.EmbedModel == "" {
					settings.EmbedModel = dbSettings.EmbedModel
				}
				if settings.SystemPrompt == "" {
					settings.SystemPrompt = dbSettings.SystemPrompt
				}
				if settings.MaxTokens == 0 {
					settings.MaxTokens = dbSettings.MaxTokens
				}
				if settings.Temperature == 0 {
					settings.Temperature = dbSettings.Temperature
				}
			}
		}
	}

	if run.Model == "" && settings.Model != "" {
		run.Model = settings.Model
	}

	accessLevel := 100
	if v := c.Locals("user"); v != nil {
		if claims, ok := v.(jwt.MapClaims); ok {
			if role, ok := claims["role"].(string); ok && role == "chat_user" {
				if al, ok := claims["access_level"].(float64); ok {
					accessLevel = int(al)
				}
			}
		}
	}

	results := make([]models.EvaluationResult, 0, len(questions))
	for _, question := range questions {
		start := time.Now()
		answer, chunks, askErr := h.rag.Ask(question.Text, topK, req.ChatID, settings, accessLevel)
		duration := time.Since(start).Milliseconds()

		retrieved := ""
		if len(chunks) > 0 {
			retrieved = chunks[0].Text
		}

		result := models.EvaluationResult{
			RunID:             run.ID,
			QuestionID:        question.ID,
			RetrievedFragment: retrieved,
			ModelAnswer:       answer,
			ResponseTimeMs:    duration,
			FallbackUsed:      len(chunks) == 0,
		}
		if askErr != nil {
			result.ErrorMessage = askErr.Error()
			if result.ModelAnswer == "" {
				result.ModelAnswer = ""
			}
		}

		results = append(results, result)
	}

	if err := h.evaluation.CreateResults(context.Background(), results); err != nil {
		run.Status = "failed"
		completed := time.Now()
		run.CompletedAt = &completed
		_ = h.evaluation.UpdateRun(context.Background(), run)
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	run.TotalQuestions = len(results)
	run.Status = "completed"
	completed := time.Now()
	run.CompletedAt = &completed

	if err := h.evaluation.UpdateRun(context.Background(), run); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	runWithResults, err := h.evaluation.GetRunByID(context.Background(), run.ID)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(201).JSON(mapEvaluationRunToDTO(*runWithResults))
}

func (h *Handler) GetEvaluationRun(c *fiber.Ctx) error {
	runID, err := uuid.Parse(c.Params("run_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid run_id"})
	}

	run, err := h.evaluation.GetRunByID(context.Background(), runID)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "run not found"})
	}

	return c.JSON(mapEvaluationRunToDTO(*run))
}

func (h *Handler) ListEvaluationRuns(c *fiber.Ctx) error {
	chatID, err := uuid.Parse(c.Params("chat_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}

	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 20)

	runs, total, err := h.evaluation.ListRunsByChat(context.Background(), chatID, page, limit)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	resp := dto.PaginatedEvaluationRuns{
		Data:  make([]dto.EvaluationRunListItem, 0, len(runs)),
		Page:  page,
		Limit: limit,
		Total: total,
	}

	for _, run := range runs {
		resp.Data = append(resp.Data, dto.EvaluationRunListItem{
			ID:             run.ID,
			ChatID:         run.ChatID,
			Status:         run.Status,
			Model:          run.Model,
			TopK:           run.TopK,
			TotalQuestions: run.TotalQuestions,
			EvaluatedCount: run.EvaluatedCount,
			CorrectCount:   run.CorrectCount,
			AvgScore:       run.AvgScore,
			StartedAt:      run.StartedAt,
			CompletedAt:    run.CompletedAt,
		})
	}

	return c.JSON(resp)
}

func (h *Handler) ScoreEvaluationResult(c *fiber.Ctx) error {
	resultID, err := uuid.Parse(c.Params("result_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid result_id"})
	}

	var req dto.EvaluationResultScoreRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}

	if req.ExpertScore < 0 || req.ExpertScore > 2 {
		return c.Status(400).JSON(fiber.Map{"error": "expert_score must be in [0..2]"})
	}

	var evaluatorID *uuid.UUID
	if v := c.Locals("user"); v != nil {
		if claims, ok := v.(jwt.MapClaims); ok {
			if idStr, ok := claims["id"].(string); ok {
				if parsed, err := uuid.Parse(idStr); err == nil {
					evaluatorID = &parsed
				}
			}
		}
	}

	result, err := h.evaluation.UpdateResultScore(context.Background(), resultID, req.ExpertScore, strings.TrimSpace(req.ExpertFeedback), req.IsCorrect, evaluatorID)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.JSON(mapEvaluationResultToDTO(*result))
}

func (h *Handler) GetEvaluationRunMetrics(c *fiber.Ctx) error {
	runID, err := uuid.Parse(c.Params("run_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid run_id"})
	}

	run, err := h.evaluation.GetRunByID(context.Background(), runID)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "run not found"})
	}

	metrics := calculateRunMetrics(*run)
	return c.JSON(metrics)
}

func (h *Handler) CompareRunWithBaseline(c *fiber.Ctx) error {
	runID, err := uuid.Parse(c.Params("run_id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid run_id"})
	}

	run, err := h.evaluation.GetRunByID(context.Background(), runID)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "run not found"})
	}

	searchLimit := c.QueryInt("limit", 1)
	if searchLimit <= 0 {
		searchLimit = 1
	}

	accessLevel := 100
	if v := c.Locals("user"); v != nil {
		if claims, ok := v.(jwt.MapClaims); ok {
			if role, ok := claims["role"].(string); ok && role == "chat_user" {
				if al, ok := claims["access_level"].(float64); ok {
					accessLevel = int(al)
				}
			}
		}
	}

	questionTotal := len(run.Results)
	if questionTotal == 0 {
		return c.JSON(dto.EvaluationBaselineCompareResponse{RunID: run.ID})
	}

	ragHits := 0
	baselineHits := 0
	searchTimes := make([]int64, 0, questionTotal)

	for _, result := range run.Results {
		if strings.TrimSpace(result.RetrievedFragment) != "" {
			ragHits++
		}

		start := time.Now()
		chunks, err := h.chunkService.SearchByKeyword(result.Question.Text, searchLimit, run.ChatID, accessLevel)
		elapsed := time.Since(start).Milliseconds()
		if elapsed < 0 {
			elapsed = 0
		}
		searchTimes = append(searchTimes, elapsed)

		if err == nil && len(chunks) > 0 {
			baselineHits++
		}
	}

	sort.Slice(searchTimes, func(i, j int) bool { return searchTimes[i] < searchTimes[j] })
	var sum int64
	for _, t := range searchTimes {
		sum += t
	}
	avgSearch := float64(sum) / float64(len(searchTimes))
	p95Search := percentile95(searchTimes)

	resp := dto.EvaluationBaselineCompareResponse{
		RunID:                  run.ID,
		QuestionsTotal:         questionTotal,
		RAGContextHitRate:      safeRate(float64(ragHits), float64(questionTotal)),
		BaselineContextHitRate: safeRate(float64(baselineHits), float64(questionTotal)),
		BaselineAvgSearchMs:    avgSearch,
		BaselineP95SearchMs:    p95Search,
	}

	return c.JSON(resp)
}

func mapTestQuestionToDTO(q models.TestQuestion) dto.TestQuestionResponse {
	return dto.TestQuestionResponse{
		ID:               q.ID,
		ChatID:           q.ChatID,
		Text:             q.Text,
		Category:         q.Category,
		ExpectedAnswer:   q.ExpectedAnswer,
		ExpectedNoAnswer: q.ExpectedNoAnswer,
		SourceHint:       q.SourceHint,
		OrderNum:         q.OrderNum,
		CreatedAt:        q.CreatedAt,
	}
}

func mapEvaluationResultToDTO(r models.EvaluationResult) dto.EvaluationResultResponse {
	return dto.EvaluationResultResponse{
		ID:                r.ID,
		RunID:             r.RunID,
		Question:          mapTestQuestionToDTO(r.Question),
		RetrievedFragment: r.RetrievedFragment,
		ModelAnswer:       r.ModelAnswer,
		ResponseTimeMs:    r.ResponseTimeMs,
		FallbackUsed:      r.FallbackUsed,
		ErrorMessage:      r.ErrorMessage,
		ExpertScore:       r.ExpertScore,
		ExpertFeedback:    r.ExpertFeedback,
		IsCorrect:         r.IsCorrect,
		EvaluatorAdminID:  r.EvaluatorAdminID,
		EvaluatedAt:       r.EvaluatedAt,
		CreatedAt:         r.CreatedAt,
	}
}

func mapEvaluationRunToDTO(run models.EvaluationRun) dto.EvaluationRunResponse {
	resultItems := make([]dto.EvaluationResultResponse, 0, len(run.Results))
	for _, item := range run.Results {
		resultItems = append(resultItems, mapEvaluationResultToDTO(item))
	}

	return dto.EvaluationRunResponse{
		ID:             run.ID,
		ChatID:         run.ChatID,
		Status:         run.Status,
		Model:          run.Model,
		TopK:           run.TopK,
		TotalQuestions: run.TotalQuestions,
		EvaluatedCount: run.EvaluatedCount,
		CorrectCount:   run.CorrectCount,
		AvgScore:       run.AvgScore,
		StartedAt:      run.StartedAt,
		CompletedAt:    run.CompletedAt,
		Results:        resultItems,
	}
}

func calculateRunMetrics(run models.EvaluationRun) dto.EvaluationMetricsResponse {
	total := len(run.Results)
	if total == 0 {
		return dto.EvaluationMetricsResponse{RunID: run.ID}
	}

	evaluatedCount := 0
	errorCount := 0
	fallbackCount := 0

	answerExpectedTotal := 0
	answerCorrect := 0
	refusalExpectedTotal := 0
	refusalCorrect := 0
	hallucinations := 0

	latencies := make([]int64, 0, total)
	var sumLatency int64

	for _, result := range run.Results {
		latencies = append(latencies, result.ResponseTimeMs)
		sumLatency += result.ResponseTimeMs

		if strings.TrimSpace(result.ErrorMessage) != "" {
			errorCount++
		}
		if result.FallbackUsed {
			fallbackCount++
		}

		if result.ExpertScore != nil || result.IsCorrect != nil {
			evaluatedCount++
		}

		if result.Question.ExpectedNoAnswer {
			refusalExpectedTotal++
			if (result.IsCorrect != nil && *result.IsCorrect) || (result.IsCorrect == nil && result.FallbackUsed) {
				refusalCorrect++
			}
			if !result.FallbackUsed && strings.TrimSpace(result.ModelAnswer) != "" && (result.IsCorrect == nil || (result.IsCorrect != nil && !*result.IsCorrect)) {
				hallucinations++
			}
		} else {
			answerExpectedTotal++
			if result.IsCorrect != nil && *result.IsCorrect {
				answerCorrect++
			}
		}
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	return dto.EvaluationMetricsResponse{
		RunID:              run.ID,
		TotalQuestions:     total,
		EvaluatedCount:     evaluatedCount,
		CorrectAnswerRate:  safeRate(float64(answerCorrect), float64(answerExpectedTotal)),
		CorrectRefusalRate: safeRate(float64(refusalCorrect), float64(refusalExpectedTotal)),
		HallucinationRate:  safeRate(float64(hallucinations), float64(refusalExpectedTotal)),
		FallbackRate:       safeRate(float64(fallbackCount), float64(total)),
		ErrorRate:          safeRate(float64(errorCount), float64(total)),
		AvgLatencyMs:       safeRate(float64(sumLatency), float64(total)),
		P95LatencyMs:       percentile95(latencies),
	}
}

func safeRate(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return num / den
}

func percentile95(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	idx := int(float64(len(values)-1) * 0.95)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(values) {
		idx = len(values) - 1
	}
	return values[idx]
}
