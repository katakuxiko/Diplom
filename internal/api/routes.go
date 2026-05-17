package api

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/handlers"
	"github.com/katakuxiko/Diplom/internal/middleware"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/routes"
	"github.com/katakuxiko/Diplom/internal/service"
)

func RegisterRoutes(app *fiber.App, cfg *config.Config, rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService, adminService *service.AdminService, chatService *service.ChatService, documentService *service.DocumentService, chatuserService *service.ChatUserService, chatSettingsService *service.ChatSettingsService, chatHistoryRepo *repository.ChatHistoryRepository, messageRepo *repository.MessageRepository) {

	h := NewHandler(rag, llm, chunkService, chatSettingsService, chatHistoryRepo, messageRepo)
	docH := handlers.NewDocumentHandler(documentService, chunkService, llm, cfg, chatSettingsService)
	middleware.JwtSecret = []byte(cfg.JWTSecret)
	handlers.RegisterAuthRoutes(app, adminService, chatuserService, cfg)

	handlers.RegisterAdminRoutes(app, adminService)
	handlers.RegisterDocumentRoutes(app, documentService, cfg)
	routes.RegisterChatRoutes(app, chatService)
	handlers.RegisterChatUserRoutes(app, chatuserService)
	chatSettingsHandler := &handlers.ChatSettingsHandler{Service: chatSettingsService}
	routes.RegisterChatSettingsRoutes(app, chatSettingsHandler)

    // Публичные эндпоинты (анонимный доступ) — создание истории, отправка сообщений и запрос к RAG
    app.Post("/ask", h.AskQuestion)
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
}
