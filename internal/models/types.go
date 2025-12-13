package models

import "github.com/google/uuid"

type AskRequest struct {
	Query  string `json:"query"`
	Model  string `json:"model,omitempty"`
	TopK   int    `json:"topK,omitempty"`
	ChatID uuid.UUID `json:"chat_id"`
}

type Rule struct {
	ID          int    `json:"id" db:"id"`
	Name        string `json:"name" db:"name"`
	Description string `json:"description" db:"description"`
}
