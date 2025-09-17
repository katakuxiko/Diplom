package repository

import (
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"
)

type ChunkRepository struct {
	db *gorm.DB
}

func NewChunkRepository(db *gorm.DB) *ChunkRepository {
	return &ChunkRepository{db: db}
}

func (r *ChunkRepository) Add(chunk models.Chunk) error {
	return r.db.Create(&chunk).Error
}

func (r *ChunkRepository) FindByDocID(docID string) ([]models.Chunk, error) {
	var chunks []models.Chunk
	err := r.db.Where("doc_id = ?", docID).Find(&chunks).Error
	return chunks, err
}

func (r *ChunkRepository) SearchByVector(vec pgvector.Vector, limit int) ([]models.Chunk, error) {
	var chunks []models.Chunk
	err := r.db.Raw(`
        SELECT * FROM chunks
        ORDER BY embedding <-> ?
        LIMIT ?
    `, vec, limit).Scan(&chunks).Error
	return chunks, err
}
