package dto

import (
	"time"

	"github.com/google/uuid"
)

type MessageResponse struct {
	ID            uuid.UUID `json:"id"`
	ChatHistoryID uuid.UUID `json:"chat_history_id"`
	Text          string    `json:"text"`
	Role          string    `json:"role"`
	CreatedDate   time.Time `json:"created_date"`
}

type ChatHistoryWithMessagesResponse struct {
	ID       uuid.UUID         `json:"id"`
	ChatID   uuid.UUID         `json:"chat_id"`
	UserID   uuid.UUID         `json:"user_id"`
	Username string            `json:"username"`
	Messages []MessageResponse `json:"messages"`
}
