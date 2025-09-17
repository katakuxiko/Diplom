package store

import (
	"fmt"
	"strings"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	"github.com/katakuxiko/Diplom/internal/models"
)

type PgStore struct {
	db *gorm.DB
}

func NewPgStore(conn string) (*PgStore, error) {
	// Можно передавать conn напрямую, если он уже DSN
	db, err := gorm.Open(postgres.Open(conn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect database: %w", err)
	}

	// Автомиграция всех моделей
	if err := db.AutoMigrate(
		&models.Admin{},
		&models.Chat{},
		&models.Document{},
		&models.Chunk{},
		&models.ChatSetting{},
		&models.Role{},
		&models.ChatUser{},
		&models.ChatHistory{},
		&models.Message{},
	); err != nil {
		return nil, err
	}

	if err := ensureSchema(db); err != nil {
		return nil, err
	}

	return &PgStore{db: db}, nil
}

// Добавление чанка с вектором
func (s *PgStore) Add(doc string, c models.Chunk, v []float32) error {
	vec := floatsToPgVectorLiteral(v)
	return s.db.Exec(`
		INSERT INTO chunks (doc_name, id, text, embedding)
		VALUES (?, ?, ?, ?::vector)
	`, doc, c.ID, c.Text, vec).Error
}

// Поиск по вектору
func (s *PgStore) Search(q []float32, k int) ([]models.Chunk, error) {
	vec := floatsToPgVectorLiteral(q)

	var res []models.Chunk
	err := s.db.Raw(`
		SELECT id, text
		FROM chunks
		ORDER BY embedding <-> ?::vector
		LIMIT ?
	`, vec, k).Scan(&res).Error
	if err != nil {
		return nil, err
	}
	return res, nil
}

func floatsToPgVectorLiteral(v []float32) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, f := range v {
		sb.WriteString(fmt.Sprintf("%.6f", f))
		if i < len(v)-1 {
			sb.WriteString(",")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

// ensureSchema для pgvector и индексов
func ensureSchema(db *gorm.DB) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`,
		`DO $$
		BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM pg_class c
				JOIN pg_namespace n ON n.oid=c.relnamespace
				WHERE c.relname='chunks_embedding_ivfflat_idx'
			) THEN
				EXECUTE 'CREATE INDEX chunks_embedding_ivfflat_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)';
			END IF;
		END $$;`,
	}

	for _, s := range stmts {
		if err := db.Exec(s).Error; err != nil {
			return err
		}
	}

	// ANALYZE
	_ = db.Exec(`ANALYZE chunks`).Error
	return nil
}
