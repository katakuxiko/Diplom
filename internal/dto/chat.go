package dto

import (
	"time"

	"github.com/google/uuid"
)

// ChatCreateRequest DTO для создания
type ChatCreateRequest struct {
	Name  string `json:"name" example:"Project Chat"`
	Descr string `json:"descr,omitempty" example:"Обсуждение проекта"`
}

// ChatResponse DTO для ответа (базовый)
type ChatResponse struct {
	ID          uuid.UUID  `json:"id"`
	AdminID     *uuid.UUID `json:"admin_id,omitempty"`
	Name        string     `json:"name"`
	Descr       string     `json:"descr,omitempty"`
	CreatedDate time.Time  `json:"created_date"`
}

// ChatDetailedResponse для детального просмотра
type ChatDetailedResponse struct {
	ID          uuid.UUID  `json:"id"`
	AdminID     *uuid.UUID `json:"admin_id,omitempty"`
	Name        string     `json:"name"`
	Descr       string     `json:"descr,omitempty"`
	CreatedDate time.Time  `json:"created_date"`
	// Documents   []DocumentBrief `json:"documents"`
	// Users       []ChatUserBrief `json:"users"`
}

// ChatUpdateRequest DTO для обновления чата
type ChatUpdateRequest struct {
	Name  *string `json:"name,omitempty" example:"Updated Project Chat"`
	Descr *string `json:"descr,omitempty" example:"Обновленное описание проекта"`
}

// ChatInviteAdminRequest DTO для приглашения админа в чат
type ChatInviteAdminRequest struct {
	AdminID string `json:"admin_id" example:"550e8400-e29b-41d4-a716-446655440000"`
}
