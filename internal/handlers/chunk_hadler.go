package handlers

import (
	"net/http"

	"github.com/gofiber/fiber/v2" // можно gin/echo, я беру fiber как пример
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
)

type ChunkHandler struct {
	service *service.ChunkService
}

func NewChunkHandler(service *service.ChunkService) *ChunkHandler {
	return &ChunkHandler{service: service}
}

// POST /chunks
func (h *ChunkHandler) CreateChunk(c *fiber.Ctx) error {
	var req struct {
		Text     string    `json:"text"`
		DocName  string    `json:"docName"`
		Filepath string    `json:"filepath"`
		Emb      []float32 `json:"embedding"`
	}

	if err := c.BodyParser(&req); err != nil {
		return c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "invalid input"})
	}

	chunk := models.Chunk{
		Text:     req.Text,
		DocName:  req.DocName,
		Filepath: req.Filepath,
	}

	if err := h.service.SaveChunk(chunk, req.Emb); err != nil {
		return c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	return c.JSON(fiber.Map{"status": "ok"})
}

// POST /chunks/search
func (h *ChunkHandler) SearchChunks(c *fiber.Ctx) error {
	var req struct {
		Vector []float32 `json:"vector"`
		Limit  int       `json:"limit"`
	}

	if err := c.BodyParser(&req); err != nil {
		return c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "invalid input"})
	}

	chunks, err := h.service.SearchSimilar(req.Vector, req.Limit)
	if err != nil {
		return c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	return c.JSON(chunks)
}
