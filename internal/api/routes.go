package api

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/handlers"
	"github.com/katakuxiko/Diplom/internal/routes"
	"github.com/katakuxiko/Diplom/internal/service"
)

func RegisterRoutes(app *fiber.App, cfg *config.Config, rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService, adminService *service.AdminService, chatService *service.ChatService, documentService *service.DocumentService) {

	h := NewHandler(rag, llm, chunkService)
	docH := handlers.NewDocumentHandler(documentService, chunkService, llm, cfg)

	// Публичные маршруты
	app.Get("/health", h.Health)
	app.Get("/models", h.ListModels)
	// Эндпоинт авторизации уже зарегистрирован в RegisterAdminRoutes

	// Группа защищённых маршрутов
	api := app.Group("/", AuthMiddleware)
	handlers.RegisterAdminRoutes(api, adminService)
	handlers.RegisterDocumentRoutes(api, documentService, cfg)
	routes.RegisterChatRoutes(api, chatService)
	api.Post("/documents/upload", docH.UploadAndIngestPDF)
	api.Post("/ingest", h.IngestPDF)
	api.Post("/ask", h.AskQuestion)
}
