package store

import (
	"database/sql"
)

// ensureSchema создаёт расширение, таблицу и индекс для pgvector
func ensureSchema(db *sql.DB) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`,
		`CREATE TABLE IF NOT EXISTS admins (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			username VARCHAR(255) NOT NULL UNIQUE,
			password_hash TEXT NOT NULL,
			is_super_user BOOLEAN DEFAULT FALSE
		);`,
		`CREATE TABLE IF NOT EXISTS chats (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    		admin_id UUID REFERENCES admins(id) ON DELETE SET NULL,
			name TEXT NOT NULL,
			descr TEXT,
			created_date TIMESTAMP DEFAULT now()
		);`,
		`CREATE TABLE IF NOT EXISTS documents (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
			name TEXT NOT NULL,
			path TEXT,
			protected BOOLEAN DEFAULT FALSE,
			access_level INT DEFAULT 0,
			created_date TIMESTAMP DEFAULT now()
		);`,
		`CREATE TABLE IF NOT EXISTS chunks (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
			doc_name TEXT,
			text TEXT,
			embedding VECTOR(768),
			filepath TEXT
		);`,
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
		`CREATE TABLE IF NOT EXISTS chat_settings (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
			hello_text TEXT,
			name TEXT,
			descr TEXT,
			url TEXT,
			created_date TIMESTAMP DEFAULT now(),
			settings JSONB
		);`,
		`CREATE TABLE IF NOT EXISTS roles (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
			name TEXT,
			access_level INT DEFAULT 0
		);`,
		`CREATE TABLE IF NOT EXISTS chat_users (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
			user_role UUID REFERENCES roles(id),
			username VARCHAR(255),
			user_info TEXT,
			password_hash TEXT
		);`,
		`CREATE TABLE IF NOT EXISTS chat_history (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
			user_id UUID REFERENCES chat_users(id)
		);`,
		`CREATE TABLE IF NOT EXISTS messages (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			chat_history_id UUID REFERENCES chat_history(id) ON DELETE CASCADE,
			text TEXT,
			role VARCHAR(50),
			created_date TIMESTAMP DEFAULT now()
		);`,
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
