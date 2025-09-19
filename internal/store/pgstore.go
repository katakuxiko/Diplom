package store

import (
	"fmt"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	"github.com/katakuxiko/Diplom/internal/models"
)

func NewPgStore(conn string) (*gorm.DB, error) {
	// Можно передавать conn напрямую, если он уже DSN
	db, err := gorm.Open(postgres.Open(conn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect database: %w", err)
	}
	if err := ensureExtension(db); err != nil {
		return nil, err
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

	return db, nil
}

// ensureSchema для pgvector и индексов
func ensureSchema(db *gorm.DB) error {
	stmts := []string{

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

func ensureExtension(db *gorm.DB) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`,
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
