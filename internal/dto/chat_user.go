package dto

import "github.com/google/uuid"

// DTO для создания
type ChatUserCreateRequest struct {
	Username string `json:"username" example:"chatuser"`
	Password string `json:"password" example:"secret123"`
	User_Role string `json:"user_role" example:"member"`
	User_Info string `json:"user_info" example:"info"`
}

// DTO для ответа
type ChatUserResponse struct {
	ID          uuid.UUID `json:"id"`
	Username    string    `json:"username"`
	User_Role string `json:"user_role" example:"member"`
	User_Info string `json:"user_info" example:"info"`
}