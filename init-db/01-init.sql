-- Initialize database with pgvector extension
-- This script runs automatically when PostgreSQL container starts for the first time

-- Create the vector extension (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the uuid extension for UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify extensions are installed
\dx

-- Optional: Create some basic schema if needed
-- You can add your initial table creation scripts here

-- Example: Create a table with vector column for embeddings
-- CREATE TABLE IF NOT EXISTS documents (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     title VARCHAR(255) NOT NULL,
--     content TEXT,
--     embedding vector(1536), -- Adjust dimensions based on your embedding model
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- Create index on vector column for faster similarity search
-- CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

GRANT ALL PRIVILEGES ON DATABASE pdf_ai TO postgres;