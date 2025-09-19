package dto

import (
	"time"

	"github.com/google/uuid"
)

// ChatCreateRequest DTO для создания
type ChatCreateRequest struct {
	AdminID *uuid.UUID `json:"admin_id,omitempty"`
	Name    string     `json:"name" example:"Project Chat"`
	Descr   string     `json:"descr,omitempty" example:"Обсуждение проекта"`
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
