package dto

import "github.com/google/uuid"

// DTO для создания
type ChatUserCreateRequest struct {
	ChatID      string `json:"chat_id" example:"52a563cc-e7a3-4aaf-8abb-5a20890d5e01"`
	Username    string `json:"username" example:"chatuser"`
	Password    string `json:"password" example:"secret123"`
	User_Role   string `json:"user_role" example:"member"`
	AccessLevel int    `json:"access_level" example:"0"`
	User_Info   string `json:"user_info" example:"info"`
}

// DTO для ответа
type ChatUserResponse struct {
	ID          uuid.UUID `json:"id"`
	ChatID      uuid.UUID `json:"chat_id"`
	Username    string    `json:"username"`
	User_Role   string    `json:"user_role" example:"member"`
	AccessLevel int       `json:"access_level" example:"0"`
	User_Info   string    `json:"user_info" example:"info"`
}
