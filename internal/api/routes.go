package api

import (
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/handlers"
	"github.com/katakuxiko/Diplom/internal/middleware"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/routes"
	"github.com/katakuxiko/Diplom/internal/service"
)

func RegisterRoutes(app *fiber.App, cfg *config.Config, rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService, adminService *service.AdminService, chatService *service.ChatService, documentService *service.DocumentService, chatuserService *service.ChatUserService, chatSettingsService *service.ChatSettingsService, chatHistoryRepo *repository.ChatHistoryRepository, messageRepo *repository.MessageRepository, evaluationService *service.EvaluationService) {

	h := NewHandler(rag, llm, chunkService, chatSettingsService, chatHistoryRepo, messageRepo, evaluationService)
	docH := handlers.NewDocumentHandler(documentService, chunkService, llm, cfg, chatSettingsService)
	middleware.JwtSecret = []byte(cfg.JWTSecret)
	handlers.RegisterAuthRoutes(app, adminService, chatuserService, cfg)

	handlers.RegisterAdminRoutes(app, adminService)
	handlers.RegisterDocumentRoutes(app, documentService, cfg)
	routes.RegisterChatRoutes(app, chatService)
	handlers.RegisterChatUserRoutes(app, chatuserService)
	chatSettingsHandler := &handlers.ChatSettingsHandler{Service: chatSettingsService}
	routes.RegisterChatSettingsRoutes(app, chatSettingsHandler)

	askLimiter := limiter.New(limiter.Config{
		Max:        60,
		Expiration: time.Minute,
	})

	evaluationRunLimiter := limiter.New(limiter.Config{
		Max:        10,
		Expiration: time.Minute,
	})

	// Публичные эндпоинты (анонимный доступ) — создание истории, отправка сообщений и запрос к RAG
	app.Post("/ask", askLimiter, h.AskQuestion)
	app.Post("/chat_histories", h.CreateChatHistory)
	app.Post("/messages", h.CreateMessage)

	// Публичный просмотр истории чатов (доступен без JWT)
	app.Get("/chats/:chat_id/history", h.GetChatHistoryForAdmin)

	// Защищенные (только с JWT) эндпоинты
	newApp := app.Group("", middleware.JWTProtected())
	newApp.Post("/documents/upload", docH.UploadAndIngestPDF)
	newApp.Get("/health", h.Health)
	newApp.Get("/models", h.ListModels)
	newApp.Post("/ingest", h.IngestPDF)
	newApp.Post("/chats/:chat_id/test-questions", h.CreateTestQuestion)
	newApp.Post("/chats/:chat_id/test-questions/batch", h.CreateTestQuestionsBatch)
	newApp.Get("/chats/:chat_id/test-questions", h.ListTestQuestions)
	newApp.Delete("/chats/:chat_id/test-questions/:question_id", h.DeleteTestQuestion)
	newApp.Get("/chats/:chat_id/evaluations/runs", h.ListEvaluationRuns)
	newApp.Post("/evaluations/runs", evaluationRunLimiter, h.StartEvaluationRun)
	newApp.Get("/evaluations/runs/:run_id", h.GetEvaluationRun)
	newApp.Get("/evaluations/runs/:run_id/metrics", h.GetEvaluationRunMetrics)
	newApp.Get("/evaluations/runs/:run_id/baseline", h.CompareRunWithBaseline)
	newApp.Put("/evaluations/results/:result_id/score", h.ScoreEvaluationResult)
}
