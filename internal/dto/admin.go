package dto

import "github.com/google/uuid"

// DTO для создания
type AdminCreateRequest struct {
	Username string `json:"username" example:"admin"`
	Password string `json:"password" example:"secret123"`
	IsSuper  bool   `json:"is_super_user"`
}

// DTO для ответа
type AdminResponse struct {
	ID          uuid.UUID `json:"id"`
	Username    string    `json:"username"`
	IsSuperUser bool      `json:"is_super_user"`
}

// (See AdminStatsResponse below - contains global counts and per-chat stats)

// ChatStatsResponse представляет статистику для одного чата
type ChatStatsResponse struct {
	ChatID         uuid.UUID `json:"chat_id"`
	Name           string    `json:"name"`
	UsersCount     int64     `json:"users_count"`
	DocumentsCount int64     `json:"documents_count"`
	MessagesCount  int64     `json:"messages_count"`
	ChunksCount    int64     `json:"chunks_count"`
}

// AdminStatsResponse теперь содержит также статистику по каждому чату
type AdminStatsResponse struct {
	UsersCount     int64                `json:"users_count"`
	ChatsCount     int64                `json:"chats_count"`
	DocumentsCount int64                `json:"documents_count"`
	MessagesCount  int64                `json:"messages_count"`
	ChunksCount    int64                `json:"chunks_count"`
	Chats          []ChatStatsResponse  `json:"chats"`
}
