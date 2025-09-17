package models

type AskRequest struct {
	Query string `json:"query"`
	Model string `json:"model,omitempty"`
	TopK  int    `json:"topK,omitempty"`
}
