package service

import (
	"fmt"
	"strings"

	"github.com/katakuxiko/Diplom/internal/model"
	"github.com/katakuxiko/Diplom/internal/store"
)

type RAGService struct {
	store *store.PgStore
	llm   *LLMClient
}

func NewRAGService(store *store.PgStore, llm *LLMClient) *RAGService {
	return &RAGService{store: store, llm: llm}
}

func (s *RAGService) Ask(query string, topK int) (string, []model.Chunk, error) {
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

	answer, err := s.llm.Ask(query, ctx)
	if err != nil {
		return "", nil, fmt.Errorf("llm error: %w", err)
	}

	return answer, chunks, nil
}
