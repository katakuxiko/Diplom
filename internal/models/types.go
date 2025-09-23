package models

type AskRequest struct {
	Query string `json:"query"`
	Model string `json:"model,omitempty"`
	TopK  int    `json:"topK,omitempty"`
}

type Rule struct {
	ID          int    `json:"id" db:"id"`
	Name        string `json:"name" db:"name"`
	Description string `json:"description" db:"description"`
}
