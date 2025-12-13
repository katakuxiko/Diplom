package service

import (
	"context"
	"fmt"
	"strings"

	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/sashabaranov/go-openai"
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
	// Проверка на приветствие
	greetings := []string{
		"привет", "привет!", "здравствуй", "здравствуйте",
		"hi", "hello", "hey", "привета", "хай", "хелло",
	}
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	for _, greeting := range greetings {
		if lowerQuery == greeting {
			return "Привет! 👋", nil
		}
	}

	prompt := fmt.Sprintf(
		"Ты ассистент для ответов на вопросы по материалам.\n\n"+
			"ОСНОВНЫЕ ПРАВИЛА:\n"+
			"- Используй ТОЛЬКО информацию из контекста\n"+
			"- Четко отличай факты из контекста от предположений\n"+
			"- Не додумывай, не полагайся на общие знания\n"+
			"- Если информация неполная - сообщи об этом\n"+
			"- Не пиши, что ты ИИ или модель\n"+
			"- Не упоминай существование контекста явно в ответе\n"+
			"- Не добавляй в ответ ссылки и источники\n"+
			"- Не добавляй в ответ информацию о правилах и формате ответа\n"+
			"- Ответ должен быть КРАТКИМ и по существу\n\n"+

			"ПРОВЕРКИ:\n"+
			"- Если контекст не относится к вопросу → 'Информация недоступна'\n"+
			"- Если контекст пустой → 'Информация недоступна'\n"+
			"- Если ответ требует информации вне контекста → 'Информация недоступна'\n\n"+

			"КОНТЕКСТ:\n%s\n\n"+
			"ВОПРОС ПОЛЬЗОВАТЕЛЯ: %s",
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
