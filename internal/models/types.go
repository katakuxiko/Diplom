package models

import "github.com/google/uuid"

type AskSettings struct {
	EnableHistory  bool    `json:"enableHistory"`
	MaxTokens      int     `json:"maxTokens"`
	Model          string  `json:"model"`
	RequestsLimit  int     `json:"requestsLimit"`
	RequestsWindow int     `json:"requestsWindow"`
	SystemPrompt   string  `json:"systemPrompt"`
	Temperature    float32 `json:"temperature"`
	// Provider settings
	Provider        string `json:"provider,omitempty"`        // "local" or "external"
	ExternalAPIKey  string `json:"externalApiKey,omitempty"`  // api key for external provider
	ExternalBaseURL string `json:"externalBaseUrl,omitempty"` // base url for external OpenAI-compatible API
}

type AskRequest struct {
	Query    string       `json:"query"`
	Model    string       `json:"model,omitempty"`
	TopK     int          `json:"topK,omitempty"`
	ChatID   uuid.UUID    `json:"chat_id"`
	Settings *AskSettings `json:"settings,omitempty"`
}

type Rule struct {
	ID          int    `json:"id" db:"id"`
	Name        string `json:"name" db:"name"`
	Description string `json:"description" db:"description"`
}
