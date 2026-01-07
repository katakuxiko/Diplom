package service

import (
	"context"
	"fmt"
	"strings"

	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/models"
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

// Ask выполняет RAG/LLM запрос с контекстом и настраиваемыми параметрами
func (l *LLMClient) Ask(query, contextText string, settings *models.AskSettings) (string, error) {
	// Проверка на приветствие
	greetings := []string{
		"привет", "привет!", "здравствуй", "здравствуйте",
		"hi", "hello", "hey", "привета", "хай", "хелло",
	}
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	for _, greeting := range greetings {
		if lowerQuery == greeting {
			return "Привет! 👋 Я помогу найти информацию в загруженных документах. Задайте ваш вопрос.", nil
		}
	}

	// Проверка на пустой контекст
	if strings.TrimSpace(contextText) == "" {
		return "К сожалению, в загруженных документах не найдено информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите дополнительные материалы.", nil
	}

	// Базовые дефолты, могут быть переопределены настройками
	modelName := l.chatName
	temperature := float32(0.7)
	maxTokens := 2000
	systemPrompt := `Ты - профессиональный аналитик документов. Твоя задача - давать точные, структурированные ответы на основе предоставленных материалов.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из КОНТЕКСТА ниже
2. Не используй знания, полученные во время обучения
3. Если информация неполная или неоднозначная - явно укажи это
4. При отсутствии релевантной информации отвечай: "Информация по данному вопросу отсутствует в документах"

ФОРМАТИРОВАНИЕ ОТВЕТА:
- Структурируй ответ (используй списки, подзаголовки при необходимости)
- Будь конкретным и информативным
- Если в контексте есть несколько релевантных фрагментов - синтезируй целостный ответ
- Избегай упоминаний о "контексте", "документах" или своей природе как ИИ
- Отвечай прямо на вопрос, без лишних вступлений`

	if settings != nil {
		if settings.Model != "" {
			modelName = settings.Model
		}
		if settings.SystemPrompt != "" {
			systemPrompt = settings.SystemPrompt
		}
		if settings.MaxTokens > 0 {
			maxTokens = settings.MaxTokens
		}
		if settings.Temperature > 0 {
			temperature = settings.Temperature
		}
	}

	userPrompt := fmt.Sprintf(
		"КОНТЕКСТ:\n%s\n\n"+
			"ВОПРОС: %s\n\n"+
			"ОТВЕТ:",
		contextText, query,
	)

	resp, err := l.client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: modelName,
			Messages: []openai.ChatCompletionMessage{
				{Role: "system", Content: systemPrompt},
				{Role: "user", Content: userPrompt},
			},
			Temperature:     temperature,
			TopP:            0.9, // Фокус на наиболее вероятных токенах
			MaxTokens:       maxTokens,
			PresencePenalty: 0.1, // Небольшое разнообразие
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
