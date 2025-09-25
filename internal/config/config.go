package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	PgConn     string
	ServerAddr string
	EmbedModel string
	ChatModel  string
	LMBaseURL  string

	// MinIO
	MinioEndpoint string
	MinioAccess   string
	MinioSecret   string
	MinioBucket   string
	MinioUseSSL   bool

	JWTSecret []byte
}

func Load() *Config {
	if err := godotenv.Load(); err != nil {
		log.Println("  .env файл не найден, используем системные переменные")
	}

	return &Config{
		PgConn:     getenv("PG_CONN", "host=localhost port=5432 user=postgres password=123123 dbname=pdf_ai sslmode=disable"),
		ServerAddr: getenv("SERVER_ADDR", ":8080"),
		EmbedModel: getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
		ChatModel:  getenv("LLM_MODEL", "liquid/lfm2-1.2b"),
		LMBaseURL:  getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),

		MinioEndpoint: getenv("MINIO_ENDPOINT", "localhost:9000"),
		MinioAccess:   getenv("MINIO_ACCESS_KEY", "admin"),
		MinioSecret:   getenv("MINIO_SECRET_KEY", "password123"),
		MinioBucket:   getenv("MINIO_BUCKET", "documents"),
		MinioUseSSL:   getenvBool("MINIO_USE_SSL", false),
		JWTSecret:     []byte(getenv("JWT_SECRET", "sadadasdasd")),
	}
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func getenvBool(k string, def bool) bool {
	if v := os.Getenv(k); v != "" {
		if v == "true" || v == "1" {
			return true
		}
		return false
	}
	return def
}
