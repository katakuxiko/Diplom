package dto

import (
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
)

type CreateDocumentDTO struct {
	ChatID uuid.UUID `form:"chat_id" validate:"required"`
}

type DocumentResponseDTO struct {
	ID          uuid.UUID `json:"id"`
	ChatID      uuid.UUID `json:"chat_id"`
	Name        string    `json:"name"`
	Tags        []string  `json:"tags"`
	Path        string    `json:"path"`
	Protected   bool      `json:"protected"`
	AccessLevel int       `json:"access_level"`
	CreatedDate time.Time `json:"created_date"`
}

type PaginatedDocuments struct {
	Documents   []models.Document `json:"documents"`
	Total       int64             `json:"total"`
	TotalPages  int               `json:"total_pages"`
	CurrentPage int               `json:"current_page"`
}

type DocumentIngestResponse struct {
	Status      string      `json:"status"`
	Document    interface{} `json:"doc"` // можно заменить на конкретный DTO DocumentResponseDTO
	ChunksTotal int         `json:"chunks_total"`
	ChunksSaved int         `json:"chunks_saved"`
}

type DocumentTagsResponse struct {
	Tags []string `json:"tags"`
}
