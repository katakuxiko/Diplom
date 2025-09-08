package service

import (
	"context"
	"fmt"
	"strings"

	"github.com/sashabaranov/go-openai"
	"github.com/katakuxiko/Diplom/internal/config"
)

// LLMClient — клиент для LM Studio / OpenAI совместимых моделей
type LLMClient struct {
	client    *openai.Client
	embedName string
	chatName  string
}

// NewLLMClient создаёт новый клиент с настройками из config
func NewLLMClient(cfg *config.Config) *LLMClient {
	oaiCfg := openai.DefaultConfig("not-needed")
	oaiCfg.BaseURL = cfg.LMBaseURL
	client := openai.NewClientWithConfig(oaiCfg)

	return &LLMClient{
		client:    client,
		embedName: cfg.EmbedModel,
		chatName:  cfg.ChatModel,
	}
}

// Embedding получает embedding текста
func (l *LLMClient) Embedding(text string) ([]float32, error) {
	resp, err := l.client.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{
			Model: openai.EmbeddingModel(l.embedName),
			Input: []string{text},
		},
	)
	if err != nil {
		return nil, err
	}
	return resp.Data[0].Embedding, nil
}

// Ask выполняет RAG/LLM запрос с контекстом
func (l *LLMClient) Ask(query, contextText string) (string, error) {
	prompt := fmt.Sprintf(
		"Ты университетский помощник. Отвечай строго на основе контекста.\n\nКонтекст:\n%s\n\nВопрос: %s\n\nЕсли информации недостаточно, скажи об этом честно.",
		contextText, query,
	)

	resp, err := l.client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: l.chatName,
			Messages: []openai.ChatCompletionMessage{
				{Role: "user", Content: prompt},
			},
			Temperature: 0.2,
		},
	)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("LLM вернул пустой ответ")
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

// ListModels возвращает список моделей LM Studio
func (l *LLMClient) ListModels() ([]openai.Model, error) {
	resp, err := l.client.ListModels(context.Background())
	if err != nil {
		return nil, err
	}
	return resp.Models, nil
}
