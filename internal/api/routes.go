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

	handlers.RegisterAdminRoutes(app, adminService)
	handlers.RegisterDocumentRoutes(app, documentService, cfg)
	routes.RegisterChatRoutes(app, chatService)

	app.Get("/health", h.Health)
	app.Get("/models", h.ListModels)
	app.Post("/ingest", h.IngestPDF)
	app.Post("/ask", h.AskQuestion)
}
