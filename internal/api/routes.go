package api

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/store"
)

func RegisterRoutes(app *fiber.App, rag *service.RAGService, llm *service.LLMClient, s *store.PgStore) {
	h := NewHandler(rag, llm, s)

	app.Get("/health", h.Health)
	app.Get("/models", h.ListModels)
	app.Post("/ingest", h.IngestPDF)
	app.Post("/ask", h.AskQuestion)
}
