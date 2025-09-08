package config

import (
	"os"
)

type Config struct {
	PgConn     string
	ServerAddr string
	EmbedModel string
	ChatModel  string
	LMBaseURL  string
}

func Load() *Config {
	return &Config{
		PgConn:     getenv("PG_CONN", "host=localhost port=5432 user=postgres password=123123 dbname=pdf_ai sslmode=disable"),
		ServerAddr: getenv("SERVER_ADDR", ":8080"),
		EmbedModel: getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
		ChatModel:  getenv("LLM_MODEL", "google/gemma-3n-e4b"),
		LMBaseURL:  getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
	}
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
