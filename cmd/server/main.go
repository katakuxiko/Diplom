// @title			Diplom API
// @version		1.0
// @description	REST API для дипломного проекта
// @host			localhost:8080
// @BasePath		/
package main

import (
	"fmt"
	"log"

	"github.com/gofiber/fiber/v2"
	_ "github.com/katakuxiko/Diplom/docs"
	"github.com/katakuxiko/Diplom/internal/api"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/store"

	swagger "github.com/swaggo/fiber-swagger"
)

// @title			Diplom API
// @version		1.0
// @description	API для дипломного проекта
// @host			localhost:8080
// @BasePath		/
func main() {
	// config
	cfg := config.Load()

	// store
	db, err := store.NewPgStore(cfg.PgConn)
	if err != nil {
		log.Fatal(err)
	}

	// repo
	chunkRepo := repository.NewChunkRepository(db)
	adminRepo := repository.NewAdminRepository(db)
	// services
	llm := service.NewLLMClient(cfg)
	rag := service.NewRAGService(chunkRepo, llm)
	chunkService := service.NewChunkService(chunkRepo)
	adminService := service.NewAdminService(adminRepo)

	// api
	app := fiber.New()
	api.RegisterRoutes(app, rag, llm, chunkService, adminService)

	app.Get("/swagger/*", swagger.WrapHandler)
	fmt.Print(cfg)

	log.Printf("🚀 Server started at %s", cfg.ServerAddr)
	log.Fatal(app.Listen(cfg.ServerAddr))
}
