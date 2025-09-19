package dto

import (
	"time"

	"github.com/google/uuid"
)

type CreateDocumentDTO struct {
	ChatID uuid.UUID `form:"chat_id" validate:"required"`
	Name   string    `form:"name" validate:"required"`
}

type DocumentResponseDTO struct {
	ID          uuid.UUID `json:"id"`
	ChatID      uuid.UUID `json:"chat_id"`
	Name        string    `json:"name"`
	Path        string    `json:"path"`
	Protected   bool      `json:"protected"`
	AccessLevel int       `json:"access_level"`
	CreatedDate time.Time `json:"created_date"`
}
