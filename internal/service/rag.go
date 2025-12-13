package service

import (
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/pgvector/pgvector-go"
)

type RAGService struct {
	ChunkRepository *repository.ChunkRepository
	llm             *LLMClient
}

func NewRAGService(ChunkRepository *repository.ChunkRepository, llm *LLMClient) *RAGService {
	return &RAGService{ChunkRepository: ChunkRepository, llm: llm}
}

func (s *RAGService) Ask(query string, topK int, chatID uuid.UUID) (string, []models.Chunk, error) {
	v, err := s.llm.Embedding(query)
	if err != nil {
		return "", nil, fmt.Errorf("embedding error: %w", err)
	}
	vec := pgvector.NewVector(v)

	chunks, err := s.ChunkRepository.SearchByVector(vec, topK, chatID)
	if err != nil {
		return "", nil, fmt.Errorf("search error: %w", err)
	}

	var b strings.Builder
	for _, ch := range chunks {
		b.WriteString(fmt.Sprintf("[%s]\n%s\n\n", ch.ChunkName, ch.Text))
	}
	ctx := b.String()
	startTime := time.Now()
	answer, err := s.llm.Ask(query, ctx)
	fmt.Println(time.Since(startTime)) // разница во времени
	if err != nil {
		return "", nil, fmt.Errorf("llm error: %w", err)
	}

	return answer, chunks, nil
}
