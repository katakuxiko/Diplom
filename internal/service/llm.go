package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"

	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/sashabaranov/go-openai"
)

// LLMClient — клиент для LM Studio / OpenAI совместимых моделей
type LLMClient struct {
	client    *openai.Client
	embedName string
	chatName  string
	baseURL   string
}

const (
	maxAutoContinuationParts = 2
	continuePrompt           = "Продолжи ответ с того места, где остановился. Не повторяй уже сказанное и сохрани структуру ответа."
	maxHistoryMessages       = 8
	historyMessageMaxChars   = 1200
	defaultHistoryCharBudget = 3500
	maxHistoryCharBudget     = 7000
	maxTranslateTokens       = 128
	translateQueryPrompt     = "You rewrite a user search query into concise Russian for retrieval over Russian documents. Preserve names, abbreviations, numbers, dates, and domain terms. Return only the rewritten Russian query without explanations."
	answerLanguageConstraint = "Answer strictly in the same language as the user's question. Do not switch language unless the user explicitly requests it."
)

func createChatCompletionWithContinuation(client *openai.Client, req openai.ChatCompletionRequest) (string, error) {
	parts := make([]string, 0, maxAutoContinuationParts+1)
	workingReq := req

	for attempt := 0; attempt <= maxAutoContinuationParts; attempt++ {
		resp, err := client.CreateChatCompletion(context.Background(), workingReq)
		if err != nil {
			return "", err
		}
		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("LLM вернул пустой ответ")
		}

		choice := resp.Choices[0]
		content := strings.TrimSpace(choice.Message.Content)
		if content != "" {
			parts = append(parts, content)
		}

		if strings.ToLower(strings.TrimSpace(string(choice.FinishReason))) != "length" {
			break
		}
		if attempt == maxAutoContinuationParts || content == "" {
			break
		}

		workingReq.Messages = append(workingReq.Messages,
			openai.ChatCompletionMessage{Role: "assistant", Content: content},
			openai.ChatCompletionMessage{Role: "user", Content: continuePrompt},
		)
	}

	result := strings.TrimSpace(strings.Join(parts, "\n"))
	if result == "" {
		return "", fmt.Errorf("LLM вернул пустой ответ")
	}
	return result, nil
}

func createChatCompletionStreamWithContinuation(client *openai.Client, req openai.ChatCompletionRequest, onDelta func(string) error) (string, error) {
	if onDelta == nil {
		return "", fmt.Errorf("stream callback is required")
	}

	streamStartedAt := time.Now()
	log.Printf(
		"LLM stream start: model=%s messages=%d max_tokens=%d temperature=%.3f top_p=%.3f",
		req.Model,
		len(req.Messages),
		req.MaxTokens,
		req.Temperature,
		req.TopP,
	)

	parts := make([]string, 0, maxAutoContinuationParts+1)
	workingReq := req
	attemptsUsed := 0
	totalDeltaChunks := 0
	totalDeltaChars := 0
	firstTokenLogged := false

	for attempt := 0; attempt <= maxAutoContinuationParts; attempt++ {
		attemptsUsed++
		attemptStartedAt := time.Now()
		attemptDeltaChunks := 0
		attemptDeltaChars := 0
		attemptFirstTokenAt := time.Time{}
		log.Printf("LLM stream attempt=%d started: messages=%d", attempt+1, len(workingReq.Messages))

		stream, err := client.CreateChatCompletionStream(context.Background(), workingReq)
		if err != nil {
			log.Printf("LLM stream attempt=%d init error: %v", attempt+1, err)
			return "", err
		}

		var partBuilder strings.Builder
		finishReason := ""

		for {
			resp, recvErr := stream.Recv()
			if errors.Is(recvErr, io.EOF) {
				break
			}
			if recvErr != nil {
				_ = stream.Close()
				return "", recvErr
			}
			if len(resp.Choices) == 0 {
				continue
			}

			choice := resp.Choices[0]
			if choice.FinishReason != "" {
				finishReason = strings.ToLower(strings.TrimSpace(string(choice.FinishReason)))
			}

			delta := choice.Delta.Content
			if delta == "" {
				continue
			}

			if attemptFirstTokenAt.IsZero() {
				attemptFirstTokenAt = time.Now()
				if !firstTokenLogged {
					firstTokenLogged = true
					log.Printf(
						"LLM stream first token: attempt=%d latency=%s",
						attempt+1,
						attemptFirstTokenAt.Sub(streamStartedAt).Round(time.Millisecond),
					)
				}
			}

			deltaChars := len([]rune(delta))
			attemptDeltaChunks++
			attemptDeltaChars += deltaChars
			totalDeltaChunks++
			totalDeltaChars += deltaChars

			if attemptDeltaChunks == 1 || attemptDeltaChunks%40 == 0 {
				log.Printf(
					"LLM stream attempt=%d progress: chunks=%d chars=%d elapsed=%s",
					attempt+1,
					attemptDeltaChunks,
					attemptDeltaChars,
					time.Since(attemptStartedAt).Round(time.Millisecond),
				)
			}

			partBuilder.WriteString(delta)
			if cbErr := onDelta(delta); cbErr != nil {
				_ = stream.Close()
				log.Printf("LLM stream attempt=%d callback error after chunks=%d: %v", attempt+1, attemptDeltaChunks, cbErr)
				return "", cbErr
			}
		}

		_ = stream.Close()

		content := strings.TrimSpace(partBuilder.String())
		if content != "" {
			parts = append(parts, content)
		}

		effectiveFinishReason := finishReason
		if effectiveFinishReason == "" {
			effectiveFinishReason = "unknown"
		}
		if attemptFirstTokenAt.IsZero() {
			log.Printf(
				"LLM stream attempt=%d completed without content: finish_reason=%s elapsed=%s",
				attempt+1,
				effectiveFinishReason,
				time.Since(attemptStartedAt).Round(time.Millisecond),
			)
		} else {
			log.Printf(
				"LLM stream attempt=%d completed: finish_reason=%s first_token=%s chunks=%d chars=%d content_chars=%d elapsed=%s",
				attempt+1,
				effectiveFinishReason,
				attemptFirstTokenAt.Sub(attemptStartedAt).Round(time.Millisecond),
				attemptDeltaChunks,
				attemptDeltaChars,
				len([]rune(content)),
				time.Since(attemptStartedAt).Round(time.Millisecond),
			)
		}

		if finishReason != "length" {
			break
		}
		if attempt == maxAutoContinuationParts || content == "" {
			break
		}

		log.Printf("LLM stream continuation requested: next_attempt=%d", attempt+2)

		workingReq.Messages = append(workingReq.Messages,
			openai.ChatCompletionMessage{Role: "assistant", Content: content},
			openai.ChatCompletionMessage{Role: "user", Content: continuePrompt},
		)
	}

	result := strings.TrimSpace(strings.Join(parts, "\n"))
	log.Printf(
		"LLM stream finished: attempts=%d total_chunks=%d total_chars=%d result_chars=%d total_elapsed=%s",
		attemptsUsed,
		totalDeltaChunks,
		totalDeltaChars,
		len([]rune(result)),
		time.Since(streamStartedAt).Round(time.Millisecond),
	)
	if result == "" {
		return "", fmt.Errorf("LLM вернул пустой ответ")
	}
	return result, nil
}

func normalizeHistoryRole(role string) (string, bool) {
	r := strings.ToLower(strings.TrimSpace(role))
	switch r {
	case "user":
		return "user", true
	case "assistant", "ai", "bot":
		return "assistant", true
	default:
		return "", false
	}
}

func truncateByRunes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit])
}

func historyCharBudget(settings *models.AskSettings) int {
	budget := defaultHistoryCharBudget
	if settings != nil && settings.MaxTokens > 0 {
		dynamic := settings.MaxTokens * 2
		if dynamic > budget {
			budget = dynamic
		}
	}
	if budget > maxHistoryCharBudget {
		budget = maxHistoryCharBudget
	}
	if budget < 1200 {
		budget = 1200
	}
	return budget
}

func sanitizeHistoryMessages(history []models.ChatContextMessage, settings *models.AskSettings) []openai.ChatCompletionMessage {
	if len(history) == 0 {
		return nil
	}
	if settings != nil && !settings.EnableHistory {
		return nil
	}

	budget := historyCharBudget(settings)
	selected := make([]openai.ChatCompletionMessage, 0, maxHistoryMessages)
	usedChars := 0

	for i := len(history) - 1; i >= 0; i-- {
		if len(selected) >= maxHistoryMessages || usedChars >= budget {
			break
		}

		role, ok := normalizeHistoryRole(history[i].Role)
		if !ok {
			continue
		}

		content := strings.TrimSpace(history[i].Content)
		if content == "" {
			continue
		}

		content = truncateByRunes(content, historyMessageMaxChars)
		pieceLen := len([]rune(content))
		if usedChars+pieceLen > budget {
			remaining := budget - usedChars
			if remaining <= 0 {
				break
			}
			content = strings.TrimSpace(truncateByRunes(content, remaining))
			pieceLen = len([]rune(content))
			if content == "" {
				break
			}
		}

		selected = append(selected, openai.ChatCompletionMessage{Role: role, Content: content})
		usedChars += pieceLen
	}

	for left, right := 0, len(selected)-1; left < right; left, right = left+1, right-1 {
		selected[left], selected[right] = selected[right], selected[left]
	}

	return selected
}

func buildAnswerMessages(systemPrompt, userPrompt string, settings *models.AskSettings, history []models.ChatContextMessage) []openai.ChatCompletionMessage {
	messages := make([]openai.ChatCompletionMessage, 0, 2+maxHistoryMessages)
	messages = append(messages, openai.ChatCompletionMessage{Role: "system", Content: systemPrompt})

	historyMessages := sanitizeHistoryMessages(history, settings)
	if len(historyMessages) > 0 {
		messages = append(messages, historyMessages...)
	}

	messages = append(messages, openai.ChatCompletionMessage{Role: "user", Content: userPrompt})
	return messages
}

func scriptLetterCounts(query string) (latinCount, cyrillicCount int) {
	for _, r := range query {
		if !unicode.IsLetter(r) {
			continue
		}
		if unicode.In(r, unicode.Latin) {
			latinCount++
		}
		if unicode.In(r, unicode.Cyrillic) {
			cyrillicCount++
		}
	}
	return latinCount, cyrillicCount
}

func detectPrimaryQuestionLanguage(query string) string {
	latinCount, cyrillicCount := scriptLetterCounts(query)

	switch {
	case latinCount >= 3 && latinCount >= cyrillicCount*2:
		return "en"
	case cyrillicCount >= 3 && cyrillicCount >= latinCount*2:
		return "ru"
	case latinCount > 0 && cyrillicCount == 0:
		return "en"
	case cyrillicCount > 0 && latinCount == 0:
		return "ru"
	default:
		return "same"
	}
}

func isLikelyEnglishQuery(query string) bool {
	return detectPrimaryQuestionLanguage(query) == "en"
}

func buildLanguagePolicyInstruction(query string) string {
	switch detectPrimaryQuestionLanguage(query) {
	case "en":
		return "LANGUAGE POLICY:\nYou MUST answer in English. Do not answer in Russian unless the user explicitly asks for Russian."
	case "ru":
		return "ПРАВИЛО ЯЗЫКА ОТВЕТА:\nВы ДОЛЖНЫ отвечать на русском языке. Не переходите на английский без явной просьбы пользователя."
	default:
		return "LANGUAGE POLICY:\n" + answerLanguageConstraint
	}
}

func applyLanguagePolicyToSystemPrompt(systemPrompt, query string) string {
	policy := strings.TrimSpace(buildLanguagePolicyInstruction(query))
	base := strings.TrimSpace(systemPrompt)
	if base == "" {
		return policy
	}
	return base + "\n\n" + policy
}

func localizedStaticReply(query, ru, en string) string {
	if isLikelyEnglishQuery(query) {
		return en
	}
	return ru
}

func buildLanguageAwareUserPrompt(contextText, query string) string {
	return fmt.Sprintf(
		"%s\n\nCONTEXT:\n%s\n\nQUESTION:\n%s\n\nANSWER:",
		buildLanguagePolicyInstruction(query),
		contextText,
		query,
	)
}

// TranslateQueryToRussianForRetrieval переводит короткий запрос на русский для fallback retrieval.
// Используется только когда поиск по исходному запросу не вернул релевантных чанков.
func (l *LLMClient) TranslateQueryToRussianForRetrieval(query string, settings *models.AskSettings) (string, error) {
	input := strings.TrimSpace(query)
	if input == "" {
		return "", nil
	}

	modelName := l.chatName
	if settings != nil && strings.TrimSpace(settings.Model) != "" {
		modelName = settings.Model
	}

	client := l.clientForSettings(settings)
	req := openai.ChatCompletionRequest{
		Model: modelName,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: translateQueryPrompt},
			{Role: "user", Content: input},
		},
		Temperature:     0,
		TopP:            1,
		MaxTokens:       maxTranslateTokens,
		PresencePenalty: 0,
	}
	if effort := reasoningEffortForModel(modelName); effort != "" {
		req.ReasoningEffort = effort
	}

	resp, err := client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("LLM translation returned empty response")
	}

	translated := strings.TrimSpace(resp.Choices[0].Message.Content)
	translated = strings.Trim(translated, "\"'")
	translated = strings.TrimSpace(translated)
	return translated, nil
}

func reasoningEffortForModel(modelName string) string {
	m := strings.ToLower(strings.TrimSpace(modelName))
	if m == "" {
		return ""
	}
	if strings.Contains(m, "reasoner") ||
		strings.Contains(m, "thinking") ||
		strings.Contains(m, "deepseek-r1") ||
		strings.Contains(m, "DeepSeek-V4-Pro") ||
		strings.Contains(m, "o3") ||
		strings.Contains(m, "o4") {
		return "low"
	}
	return ""
}

func resolveChatProvider(s *models.AskSettings) string {
	if s == nil {
		return "local"
	}
	p := strings.ToLower(strings.TrimSpace(s.Provider))
	if p != "" {
		return p
	}
	if strings.TrimSpace(s.ExternalBaseURL) != "" || strings.TrimSpace(s.ExternalAPIKey) != "" {
		return "external"
	}
	return "local"
}

func resolveEmbeddingProvider(s *models.AskSettings) string {
	if s == nil {
		return "local"
	}
	p := strings.ToLower(strings.TrimSpace(s.EmbedProvider))
	if p != "" {
		return p
	}
	if strings.TrimSpace(s.EmbedExternalBaseURL) != "" || strings.TrimSpace(s.EmbedExternalAPIKey) != "" {
		return "external"
	}
	return resolveChatProvider(s)
}

// authTransport добавляет Authorization Bearer заголовок при необходимости
type authTransport struct {
	apiKey string
	base   http.RoundTripper
}

func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Header.Get("Authorization") == "" && t.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+t.apiKey)
	}
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", "diplom-llm-client/1.0")
	}
	if t.base == nil {
		t.base = http.DefaultTransport
	}
	return t.base.RoundTrip(req)
}

// NewLLMClient создаёт новый клиент с настройками из config
func NewLLMClient(cfg *config.Config) *LLMClient {
	oaiCfg := openai.DefaultConfig("not-needed")
	oaiCfg.BaseURL = normalizeAPIBase(cfg.LMBaseURL)
	client := openai.NewClientWithConfig(oaiCfg)

	return &LLMClient{
		client:    client,
		embedName: cfg.EmbedModel,
		chatName:  cfg.ChatModel,
		baseURL:   cfg.LMBaseURL,
	}
}

// Embedding получает embedding текста (глобальный клиент)
func (l *LLMClient) Embedding(text string) ([]float32, error) {
	resp, err := l.client.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{Model: openai.EmbeddingModel(l.embedName), Input: []string{text}},
	)
	if err != nil {
		return nil, err
	}
	return resp.Data[0].Embedding, nil
}

// clientForSettings возвращает openai.Client, учитывая настройки провайдера (локальный или внешний)
func (l *LLMClient) clientForSettings(s *models.AskSettings) *openai.Client {
	provider := resolveChatProvider(s)
	if s != nil && provider == "external" {
		key := strings.TrimSpace(s.ExternalAPIKey)
		if key == "" {
			key = "not-needed"
		}
		cfg := openai.DefaultConfig(key)
		if strings.TrimSpace(s.ExternalBaseURL) != "" {
			cfg.BaseURL = normalizeAPIBase(s.ExternalBaseURL)
		}
		cfg.HTTPClient = &http.Client{Transport: &authTransport{apiKey: strings.TrimSpace(s.ExternalAPIKey), base: http.DefaultTransport}}
		return openai.NewClientWithConfig(cfg)
	}
	return l.client
}

// clientForEmbeddingSettings возвращает openai.Client, учитывая настройки провайдера для эмбеддингов
func (l *LLMClient) clientForEmbeddingSettings(s *models.AskSettings) *openai.Client {
	if s == nil {
		return l.client
	}
	provider := resolveEmbeddingProvider(s)
	if provider == "external" {
		key := strings.TrimSpace(s.EmbedExternalAPIKey)
		if key == "" {
			key = strings.TrimSpace(s.ExternalAPIKey)
		}
		if key == "" {
			key = "not-needed"
		}
		cfg := openai.DefaultConfig(key)
		if strings.TrimSpace(s.EmbedExternalBaseURL) != "" {
			cfg.BaseURL = normalizeAPIBase(s.EmbedExternalBaseURL)
		} else if strings.TrimSpace(s.ExternalBaseURL) != "" {
			cfg.BaseURL = normalizeAPIBase(s.ExternalBaseURL)
		}
		diagKey := strings.TrimSpace(s.EmbedExternalAPIKey)
		if diagKey == "" {
			diagKey = strings.TrimSpace(s.ExternalAPIKey)
		}
		cfg.HTTPClient = &http.Client{Transport: &authTransport{apiKey: diagKey, base: http.DefaultTransport}}
		return openai.NewClientWithConfig(cfg)
	}
	return l.client
}

// diagGET делает быстрый GET к указанному URL и возвращает статус и короткую часть тела
func diagGET(rawURL string) (string, string) {
	if rawURL == "" {
		return "", ""
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(rawURL)
	if err != nil {
		return "ERR", err.Error()
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	return resp.Status, string(b)
}

// diagGETWithAuth делает GET с передачей Authorization Bearer заголовка (если ключ передан)
func diagGETWithAuth(rawURL, apiKey string) (string, string) {
	if rawURL == "" {
		return "", ""
	}
	client := &http.Client{Timeout: 3 * time.Second}
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return "ERR", err.Error()
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "ERR", err.Error()
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	return resp.Status, string(b)
}

// normalizeAPIBase возвращает корректный baseURL для OpenAI-совместимого клиента.
// Если передан полный путь к endpoint (например "/v1/chat/completions"),
// функция обрежет строку до корня API включая "/v1".
func normalizeAPIBase(raw string) string {
	if raw == "" {
		return ""
	}
	u := strings.TrimSpace(raw)
	parsed, err := url.Parse(u)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		if idx := strings.Index(u, "/v1/"); idx != -1 {
			return strings.TrimRight(u[:idx+3], "/")
		}
		if strings.HasSuffix(u, "/v1") {
			return strings.TrimRight(u, "/")
		}
		return strings.TrimRight(u, "/") + "/v1"
	}
	base := parsed.Scheme + "://" + parsed.Host
	if idx := strings.Index(parsed.Path, "/v1/"); idx != -1 {
		base = base + strings.TrimRight(parsed.Path[:idx+3], "/")
		return base
	}
	if strings.HasSuffix(parsed.Path, "/v1") {
		base = base + strings.TrimRight(parsed.Path, "/")
		return base
	}
	return base + "/v1"
}

// hfEmbedding выполняет вызов к Hugging Face Inference/Router API для получения эмбеддингов
func hfEmbedding(usedBase, modelName, text, apiKey string) ([]float32, error) {
	if modelName == "" {
		return nil, fmt.Errorf("model name is empty for HF embedding")
	}
	lb := strings.ToLower(usedBase)
	var endpoint string
	if strings.Contains(lb, "/hf-inference/") || strings.Contains(lb, "/pipeline/") {
		endpoint = usedBase
	} else if strings.Contains(lb, "router.huggingface.co") {
		esc := url.PathEscape(modelName)
		esc = strings.ReplaceAll(esc, "%2F", "/")
		endpoint = fmt.Sprintf("https://router.huggingface.co/hf-inference/models/%s/pipeline/feature-extraction", esc)
	} else if strings.Contains(lb, "api-inference.huggingface.co") {
		esc := url.PathEscape(modelName)
		esc = strings.ReplaceAll(esc, "%2F", "/")
		endpoint = fmt.Sprintf("https://api-inference.huggingface.co/models/%s/pipeline/feature-extraction", esc)
	} else {
		esc := url.PathEscape(modelName)
		esc = strings.ReplaceAll(esc, "%2F", "/")
		endpoint = strings.TrimRight(usedBase, "/") + "/hf-inference/models/" + esc + "/pipeline/feature-extraction"
	}

	bodyMap := map[string]interface{}{"inputs": text}
	bodyBytes, _ := json.Marshal(bodyMap)
	req, err := http.NewRequest("POST", endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("hf embed error, status: %s, body: %s", resp.Status, string(b))
	}

	var parsed interface{}
	if err := json.Unmarshal(b, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse hf embedding response: %v body=%s", err, string(b))
	}

	var floats []float32
	var tryExtract func(interface{}) bool
	tryExtract = func(v interface{}) bool {
		switch x := v.(type) {
		case []interface{}:
			if len(x) == 0 {
				return false
			}
			switch x[0].(type) {
			case float64:
				floats = make([]float32, len(x))
				for i, vv := range x {
					if num, ok := vv.(float64); ok {
						floats[i] = float32(num)
					} else {
						return false
					}
				}
				return true
			default:
				for _, item := range x {
					if tryExtract(item) {
						return true
					}
				}
				return false
			}
		default:
			return false
		}
	}

	if !tryExtract(parsed) {
		return nil, fmt.Errorf("unexpected hf embedding response format: %s", string(b))
	}
	return floats, nil
}

// EmbeddingWithSettings позволяет получать embedding, используя провайдера из настроек
func (l *LLMClient) EmbeddingWithSettings(text string, s *models.AskSettings) ([]float32, error) {
	client := l.clientForEmbeddingSettings(s)
	modelName := l.embedName
	if s != nil && s.EmbedModel != "" {
		modelName = s.EmbedModel
	}
	usedBase := l.baseURL
	provider := resolveEmbeddingProvider(s)
	if s != nil {
		if provider == "external" && strings.TrimSpace(s.EmbedExternalBaseURL) != "" {
			usedBase = s.EmbedExternalBaseURL
		} else if provider == "external" && strings.TrimSpace(s.ExternalBaseURL) != "" {
			usedBase = s.ExternalBaseURL
		}
	}
	apiKey := ""
	if s != nil {
		if provider == "external" && strings.TrimSpace(s.EmbedExternalAPIKey) != "" {
			apiKey = s.EmbedExternalAPIKey
		} else if provider == "external" && strings.TrimSpace(s.ExternalAPIKey) != "" {
			apiKey = s.ExternalAPIKey
		}
	}

	log.Printf("Embedding request: model=%s provider=%s baseURL=%s", modelName, provider, usedBase)

	// Если указан Hugging Face в usedBase — вызываем HF inference напрямую
	if strings.Contains(strings.ToLower(usedBase), "huggingface") {
		emb, err := hfEmbedding(usedBase, modelName, text, apiKey)
		if err != nil {
			log.Printf("Embedding error (hf): %v", err)
			if usedBase != "" {
				status, body := diagGETWithAuth(usedBase, apiKey)
				log.Printf("Embedding diagnostic GET %s -> status=%s body=%s", usedBase, status, body)
			}
			return nil, err
		}
		return emb, nil
	}

	resp, err := client.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{Model: openai.EmbeddingModel(modelName), Input: []string{text}},
	)
	if err != nil {
		log.Printf("Embedding error: %v", err)
		if usedBase != "" {
			status, body := diagGETWithAuth(usedBase, apiKey)
			log.Printf("Embedding diagnostic GET %s -> status=%s body=%s", usedBase, status, body)
		}
		return nil, err
	}
	return resp.Data[0].Embedding, nil
}

// Ask выполняет RAG/LLM запрос с контекстом и настраиваемыми параметрами
func (l *LLMClient) Ask(query, contextText string, settings *models.AskSettings, history []models.ChatContextMessage) (string, error) {
	greetings := []string{"привет", "привет!", "здравствуй", "здравствуйте", "hi", "hello", "hey", "привета", "хай", "хелло"}
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	for _, greeting := range greetings {
		if lowerQuery == greeting {
			return localizedStaticReply(
				query,
				"Привет! 👋 Я помогу найти информацию в загруженных документах. Задайте ваш вопрос.",
				"Hi! 👋 I can help you find information in the uploaded documents. Ask your question.",
			), nil
		}
	}

	if strings.TrimSpace(contextText) == "" {
		return localizedStaticReply(
			query,
			"К сожалению, в загруженных документах не найдено информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите дополнительные материалы.",
			"Unfortunately, no relevant information was found in the uploaded documents for your request. Try rephrasing the question or upload additional materials.",
		), nil
	}

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

	systemPrompt = applyLanguagePolicyToSystemPrompt(systemPrompt, query)

	userPrompt := buildLanguageAwareUserPrompt(contextText, query)

	req := openai.ChatCompletionRequest{
		Model:           modelName,
		Messages:        buildAnswerMessages(systemPrompt, userPrompt, settings, history),
		Temperature:     temperature,
		TopP:            0.9,
		MaxTokens:       maxTokens,
		PresencePenalty: 0.1,
	}
	if effort := reasoningEffortForModel(modelName); effort != "" {
		req.ReasoningEffort = effort
	}

	return createChatCompletionWithContinuation(l.client, req)
}

// AskWithSettings выполняет запрос к модели с учётом per-chat провайдера (локальный или внешний)
func (l *LLMClient) AskWithSettings(query, contextText string, settings *models.AskSettings, history []models.ChatContextMessage) (string, error) {
	greetings := []string{"привет", "привет!", "здравствуй", "здравствуйте", "hi", "hello", "hey", "привета", "хай", "хелло"}
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	for _, greeting := range greetings {
		if lowerQuery == greeting {
			return localizedStaticReply(
				query,
				"Привет! 👋 Я помогу найти информацию в загруженных документах. Задайте ваш вопрос.",
				"Hi! 👋 I can help you find information in the uploaded documents. Ask your question.",
			), nil
		}
	}

	if strings.TrimSpace(contextText) == "" {
		return localizedStaticReply(
			query,
			"К сожалению, в загруженных документах не найдено информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите дополнительные материалы.",
			"Unfortunately, no relevant information was found in the uploaded documents for your request. Try rephrasing the question or upload additional materials.",
		), nil
	}

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

	systemPrompt = applyLanguagePolicyToSystemPrompt(systemPrompt, query)

	userPrompt := buildLanguageAwareUserPrompt(contextText, query)

	client := l.clientForSettings(settings)

	usedBase := l.baseURL
	provider := resolveChatProvider(settings)
	if settings != nil {
		if provider == "external" && strings.TrimSpace(settings.ExternalBaseURL) != "" {
			usedBase = settings.ExternalBaseURL
		}
	}
	log.Printf("Chat request: model=%s provider=%s baseURL=%s", modelName, provider, usedBase)

	req := openai.ChatCompletionRequest{
		Model:           modelName,
		Messages:        buildAnswerMessages(systemPrompt, userPrompt, settings, history),
		Temperature:     temperature,
		TopP:            0.9,
		MaxTokens:       maxTokens,
		PresencePenalty: 0.1,
	}
	if effort := reasoningEffortForModel(modelName); effort != "" {
		req.ReasoningEffort = effort
	}

	answer, err := createChatCompletionWithContinuation(client, req)
	if err != nil {
		log.Printf("Chat error: %v", err)
		if usedBase != "" {
			diagKey := ""
			if settings != nil && provider == "external" && settings.ExternalAPIKey != "" {
				diagKey = settings.ExternalAPIKey
			}
			status, body := diagGETWithAuth(usedBase, diagKey)
			log.Printf("Chat diagnostic GET %s -> status=%s body=%s", usedBase, status, body)
		}
		return "", err
	}
	return answer, nil
}

// AskWithSettingsStream выполняет потоковый запрос к модели с учётом per-chat провайдера.
func (l *LLMClient) AskWithSettingsStream(query, contextText string, settings *models.AskSettings, history []models.ChatContextMessage, onDelta func(string) error) (string, error) {
	if onDelta == nil {
		return "", fmt.Errorf("stream callback is required")
	}

	greetings := []string{"привет", "привет!", "здравствуй", "здравствуйте", "hi", "hello", "hey", "привета", "хай", "хелло"}
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	for _, greeting := range greetings {
		if lowerQuery == greeting {
			staticAnswer := localizedStaticReply(
				query,
				"Привет! 👋 Я помогу найти информацию в загруженных документах. Задайте ваш вопрос.",
				"Hi! 👋 I can help you find information in the uploaded documents. Ask your question.",
			)
			if err := onDelta(staticAnswer); err != nil {
				return "", err
			}
			return staticAnswer, nil
		}
	}

	if strings.TrimSpace(contextText) == "" {
		staticAnswer := localizedStaticReply(
			query,
			"К сожалению, в загруженных документах не найдено информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите дополнительные материалы.",
			"Unfortunately, no relevant information was found in the uploaded documents for your request. Try rephrasing the question or upload additional materials.",
		)
		if err := onDelta(staticAnswer); err != nil {
			return "", err
		}
		return staticAnswer, nil
	}

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

	systemPrompt = applyLanguagePolicyToSystemPrompt(systemPrompt, query)

	userPrompt := buildLanguageAwareUserPrompt(contextText, query)

	client := l.clientForSettings(settings)

	usedBase := l.baseURL
	provider := resolveChatProvider(settings)
	if settings != nil {
		if provider == "external" && strings.TrimSpace(settings.ExternalBaseURL) != "" {
			usedBase = settings.ExternalBaseURL
		}
	}
	log.Printf("Chat stream request: model=%s provider=%s baseURL=%s", modelName, provider, usedBase)

	req := openai.ChatCompletionRequest{
		Model:           modelName,
		Messages:        buildAnswerMessages(systemPrompt, userPrompt, settings, history),
		Temperature:     temperature,
		TopP:            0.9,
		MaxTokens:       maxTokens,
		PresencePenalty: 0.1,
		Stream:          true,
	}
	if effort := reasoningEffortForModel(modelName); effort != "" {
		req.ReasoningEffort = effort
	}

	answer, err := createChatCompletionStreamWithContinuation(client, req, onDelta)
	if err != nil {
		log.Printf("Chat stream error: %v", err)
		if usedBase != "" {
			diagKey := ""
			if settings != nil && provider == "external" && settings.ExternalAPIKey != "" {
				diagKey = settings.ExternalAPIKey
			}
			status, body := diagGETWithAuth(usedBase, diagKey)
			log.Printf("Chat stream diagnostic GET %s -> status=%s body=%s", usedBase, status, body)
		}
		return "", err
	}
	return answer, nil
}

// ListModels возвращает список моделей LM Studio
func (l *LLMClient) ListModels() ([]openai.Model, error) {
	resp, err := l.client.ListModels(context.Background())
	if err != nil {
		return nil, err
	}
	return resp.Models, nil
}
