package dto

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
)

// ChatSettingCreateRequest используется для создания настроек чата
type ChatSettingCreateRequest struct {
	ChatID    uuid.UUID    `json:"chatID" binding:"required"`
	HelloText string       `json:"helloText"`
	Name      string       `json:"name"`
	Descr     string       `json:"descr"`
	URL       string       `json:"url"`
	Settings  models.JSONB `json:"settings"`
}

// ChatSettingUpdateRequest используется для обновления настроек чата
type ChatSettingUpdateRequest struct {
	HelloText string       `json:"helloText"`
	Name      string       `json:"name"`
	Descr     string       `json:"descr"`
	URL       string       `json:"url"`
	Settings  models.JSONB `json:"settings"`
}

// ChatSettingResponse используется для ответа с настройками чата
type ChatSettingResponse struct {
	ID          uuid.UUID    `json:"id"`
	ChatID      uuid.UUID    `json:"chatID"`
	HelloText   string       `json:"helloText"`
	Name        string       `json:"name"`
	Descr       string       `json:"descr"`
	URL         string       `json:"url"`
	CreatedDate string       `json:"createdDate"`
	Settings    models.JSONB `json:"settings"`
}
