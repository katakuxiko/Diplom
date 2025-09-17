package api

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/service"
)

func RegisterRoutes(app *fiber.App, rag *service.RAGService, llm *service.LLMClient, chunkService *service.ChunkService, adminService *service.AdminService) {

	h := NewHandler(rag, llm, chunkService)

	RegisterAdminRoutes(app, adminService)

	app.Get("/health", h.Health)
	app.Get("/models", h.ListModels)
	app.Post("/ingest", h.IngestPDF)
	app.Post("/ask", h.AskQuestion)
}
