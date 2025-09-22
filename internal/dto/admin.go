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

// DTO для запроса авторизации
type AdminAuthRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// DTO для ответа с токенами
type AuthTokensResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
}
