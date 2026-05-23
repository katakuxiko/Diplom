package service

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/pgvector/pgvector-go"
)

type ChunkService struct {
	repo *repository.ChunkRepository
}

func NewChunkService(repo *repository.ChunkRepository) *ChunkService {
	return &ChunkService{repo: repo}
}

func (s *ChunkService) SaveChunk(c models.Chunk, embedding []float32) error {
	vec := pgvector.NewVector(embedding)

	c.Embedding = vec
	return s.repo.Add(c)
}

func (s *ChunkService) SearchSimilar(vec []float32, limit int, chatID uuid.UUID, accessLevel int) ([]models.Chunk, error) {
	return s.repo.SearchByVector(pgvector.NewVector(vec), limit, chatID, accessLevel)
}

func (s *ChunkService) SearchByKeyword(query string, limit int, chatID uuid.UUID, accessLevel int) ([]models.Chunk, error) {
	return s.repo.SearchByKeyword(query, limit, chatID, accessLevel)
}
