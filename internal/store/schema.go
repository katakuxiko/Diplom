package store

import (
	"database/sql"
)

// ensureSchema создаёт расширение, таблицу и индекс для pgvector
func ensureSchema(db *sql.DB) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE TABLE IF NOT EXISTS chunks (
			id SERIAL PRIMARY KEY,
			doc_name TEXT,
			chunk_id TEXT,
			text TEXT,
			embedding vector(768)
		)`,
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
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}

	// ANALYZE для корректной работы ivfflat
	_, _ = db.Exec(`ANALYZE chunks`)
	return nil
}
