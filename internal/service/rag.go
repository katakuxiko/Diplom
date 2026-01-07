package service

import (
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/utils"
	"github.com/pgvector/pgvector-go"
)

type RAGService struct {
	ChunkRepository *repository.ChunkRepository
	llm             *LLMClient
}

func NewRAGService(ChunkRepository *repository.ChunkRepository, llm *LLMClient) *RAGService {
	return &RAGService{ChunkRepository: ChunkRepository, llm: llm}
}

func (s *RAGService) Ask(query string, topK int, chatID uuid.UUID, settings *models.AskSettings) (string, []models.Chunk, error) {
	// Получаем больше чанков для последующей фильтрации
	expandedTopK := topK * 2
	if expandedTopK > 20 {
		expandedTopK = 20
	}

	v, err := s.llm.Embedding(query)
	if err != nil {
		return "", nil, fmt.Errorf("embedding error: %w", err)
	}
	vec := pgvector.NewVector(v)

	chunks, err := s.ChunkRepository.SearchByVector(vec, expandedTopK, chatID)
	if err != nil {
		return "", nil, fmt.Errorf("search error: %w", err)
	}

	// Фильтруем чанки по relevance threshold
	filteredChunks := s.filterRelevantChunks(chunks, topK)

	// Если после фильтрации не осталось чанков
	if len(filteredChunks) == 0 {
		return "К сожалению, в загруженных документах не найдено релевантной информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите дополнительные материалы.", nil, nil
	}

	// Build normalized, compact context with a character budget
	// Heuristic: cap context to ~8000 chars (~2k tokens), enough for fast responses
	contextBudget := 8000
	if settings != nil && settings.MaxTokens > 0 {
		// Leave room for answer; use ~1.5x of answer tokens as context chars (approx 4 chars per token)
		budget := int(float32(settings.MaxTokens) * 1.5 * 4)
		if budget > 0 && budget < 20000 { // clamp to avoid oversized prompts
			contextBudget = budget
		}
	}

	var b strings.Builder
	used := 0
	for i, ch := range filteredChunks {
		normalized := utils.NormalizeText(ch.Text)
		header := fmt.Sprintf("Фрагмент %d: ", i+1)
		piece := header + normalized + "\n"
		if used+len(piece) > contextBudget {
			remaining := contextBudget - used
			if remaining <= 0 {
				break
			}
			piece = utils.TruncateByChars(piece, remaining)
		}
		b.WriteString(piece)
		used += len(piece)
		if used >= contextBudget {
			break
		}
	}
	ctx := b.String()

	startTime := time.Now()
	answer, err := s.llm.Ask(query, ctx, settings)
	fmt.Printf("⏱️  LLM response time: %v\n", time.Since(startTime))
	if err != nil {
		return "", nil, fmt.Errorf("llm error: %w", err)
	}

	return answer, filteredChunks, nil
}

// filterRelevantChunks фильтрует чанки по релевантности
func (s *RAGService) filterRelevantChunks(chunks []models.Chunk, maxChunks int) []models.Chunk {
	if len(chunks) == 0 {
		return chunks
	}

	// Простая эвристика: берем только чанки с длиной текста > 50 символов
	// и ограничиваем количество maxChunks
	filtered := make([]models.Chunk, 0, maxChunks)
	for _, ch := range chunks {
		if len(strings.TrimSpace(ch.Text)) > 50 {
			filtered = append(filtered, ch)
			if len(filtered) >= maxChunks {
				break
			}
		}
	}

	return filtered
}
