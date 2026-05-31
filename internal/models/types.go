package models

import "github.com/google/uuid"

type AskSettings struct {
	EnableHistory     bool    `json:"enableHistory"`
	MaxTokens         int     `json:"maxTokens"`
	Model             string  `json:"model"`
	EmbedModel        string  `json:"embedModel,omitempty"`
	RetrievalMode     string  `json:"retrievalMode,omitempty"` // "vector", "keyword", "hybrid"
	MaxCosineDistance float32 `json:"maxCosineDistance,omitempty"`
	MaxDistanceGap    float32 `json:"maxDistanceGap,omitempty"`
	MinChunkChars     int     `json:"minChunkChars,omitempty"`
	// Embedding provider specific settings
	EmbedProvider        string  `json:"embedProvider,omitempty"`
	EmbedExternalAPIKey  string  `json:"embedExternalApiKey,omitempty"`
	EmbedExternalBaseURL string  `json:"embedExternalBaseUrl,omitempty"`
	RequestsLimit        int     `json:"requestsLimit"`
	RequestsWindow       int     `json:"requestsWindow"`
	SystemPrompt         string  `json:"systemPrompt"`
	Temperature          float32 `json:"temperature"`
	// Provider settings
	Provider        string `json:"provider,omitempty"`        // "local" or "external"
	ExternalAPIKey  string `json:"externalApiKey,omitempty"`  // api key for external provider
	ExternalBaseURL string `json:"externalBaseUrl,omitempty"` // base url for external OpenAI-compatible API
}

type AskRequest struct {
	Query         string       `json:"query"`
	Model         string       `json:"model,omitempty"`
	TopK          int          `json:"topK,omitempty"`
	Stream        bool         `json:"stream,omitempty"`
	ChatID        uuid.UUID    `json:"chat_id"`
	ChatHistoryID *uuid.UUID   `json:"chat_history_id,omitempty"`
	Settings      *AskSettings `json:"settings,omitempty"`
}

// ChatContextMessage хранит краткую историю диалога для генерации ответа в LLM.
// Используется только как дополнительный контекст и передаётся в ограниченном объёме.
type ChatContextMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Rule struct {
	ID          int    `json:"id" db:"id"`
	Name        string `json:"name" db:"name"`
	Description string `json:"description" db:"description"`
}
