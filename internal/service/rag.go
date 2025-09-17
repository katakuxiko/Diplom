package service

import (
	"fmt"
	"strings"
	"time"

	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/store"
)

type RAGService struct {
	store *store.PgStore
	llm   *LLMClient
}

func NewRAGService(store *store.PgStore, llm *LLMClient) *RAGService {
	return &RAGService{store: store, llm: llm}
}

func (s *RAGService) Ask(query string, topK int) (string, []models.Chunk, error) {
	vec, err := s.llm.Embedding(query)
	if err != nil {
		return "", nil, fmt.Errorf("embedding error: %w", err)
	}

	chunks, err := s.store.Search(vec, topK)
	if err != nil {
		return "", nil, fmt.Errorf("search error: %w", err)
	}

	var b strings.Builder
	for _, ch := range chunks {
		b.WriteString(fmt.Sprintf("[%s]\n%s\n\n", ch.ID, ch.Text))
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
