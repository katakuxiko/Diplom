package dto

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
	ChatID   string `json:"chat_id,omitempty"`
}
