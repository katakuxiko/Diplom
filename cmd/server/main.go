package main

import (
	"log"

	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/api"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/store"
)

func main() {
	// config
	cfg := config.Load()

	// store
	dbStore, err := store.NewPgStore(cfg.PgConn)
	if err != nil {
		log.Fatal(err)
	}

	// services
	llm := service.NewLLMClient(cfg)
	rag := service.NewRAGService(dbStore, llm)

	// api
	app := fiber.New()
	api.RegisterRoutes(app, rag, llm, dbStore)

	log.Printf("🚀 Server started at %s", cfg.ServerAddr)
	log.Fatal(app.Listen(cfg.ServerAddr))
}
