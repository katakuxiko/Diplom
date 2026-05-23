package repository

import (
	"strings"

	"github.com/google/uuid"
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

func (r *ChunkRepository) SearchByVector(vec pgvector.Vector, limit int, chatID uuid.UUID, accessLevel int) ([]models.Chunk, error) {
	var chunks []models.Chunk
	err := r.db.Raw(`
		SELECT c.* FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		WHERE c.chat_id = ? AND d.access_level <= ?
		ORDER BY c.embedding <-> ?
		LIMIT ?
	`, chatID, accessLevel, vec, limit).Scan(&chunks).Error
	return chunks, err
}

func (r *ChunkRepository) SearchByKeyword(query string, limit int, chatID uuid.UUID, accessLevel int) ([]models.Chunk, error) {
	if limit <= 0 {
		limit = 5
	}

	terms := strings.Fields(strings.TrimSpace(query))
	if len(terms) == 0 {
		return []models.Chunk{}, nil
	}
	if len(terms) > 6 {
		terms = terms[:6]
	}

	var conditions []string
	args := make([]interface{}, 0, 2+len(terms)+1)
	args = append(args, chatID, accessLevel)

	for _, term := range terms {
		clean := strings.TrimSpace(term)
		if len(clean) < 3 {
			continue
		}
		conditions = append(conditions, "LOWER(c.text) LIKE LOWER(?)")
		args = append(args, "%"+clean+"%")
	}

	if len(conditions) == 0 {
		return []models.Chunk{}, nil
	}

	args = append(args, limit)
	querySQL := `
		SELECT c.* FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		WHERE c.chat_id = ? AND d.access_level <= ? AND (` + strings.Join(conditions, " OR ") + `)
		ORDER BY c.doc_name ASC, c.chunk_name ASC
		LIMIT ?
	`

	var chunks []models.Chunk
	err := r.db.Raw(querySQL, args...).Scan(&chunks).Error
	return chunks, err
}
