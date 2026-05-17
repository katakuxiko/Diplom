package main

import (
	"log"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/katakuxiko/Diplom/internal/api"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/storage"
	"github.com/katakuxiko/Diplom/internal/store"

	swagger "github.com/swaggo/fiber-swagger"
)

// @title			Diplom API
// @version		1.0
// @description	API для дипломного проекта asd
// @host			localhost:8080
// @BasePath		/
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description Введите JWT ток ен в формате: Bearer {your token}

func main() {
	// config
	cfg := config.Load()

	// store
	db, err := store.NewPgStore(cfg.PgConn)
	if err != nil {
		log.Fatal(err)
	}

	//storage
	minioClient, err := storage.NewMinioStorage(cfg.MinioEndpoint, cfg.MinioAccess, cfg.MinioSecret, cfg.MinioBucket, cfg.MinioUseSSL)
	if err != nil {
		log.Fatal(err)
	}
	cfg.MinioStorage = minioClient

	// repo
	chunkRepo := repository.NewChunkRepository(db)
	adminRepo := repository.NewAdminRepository(db)
	chatRepo := repository.NewChatRepository(db)
	documentRepo := repository.NewDocumentRepository(db)
	chatuserRepo := repository.NewChatUserRepository(db)
	chatSettingsRepo := repository.NewChatSettingsRepository(db)
	chatHistoryRepo := repository.NewChatHistoryRepository(db)
	messageRepo := repository.NewMessageRepository(db)
	// services
	llm := service.NewLLMClient(cfg)
	rag := service.NewRAGService(chunkRepo, llm)
	chunkService := service.NewChunkService(chunkRepo)
	adminService := service.NewAdminService(adminRepo)

	chatService := service.NewChatService(chatRepo)
	chatService.SetDB(db) // Передаем БД для удаления документов при удалении чата
	documentService := service.NewDocumentService(documentRepo, minioClient)
	chatUserService := service.NewChatUserService(chatuserRepo)
	chatSettingsService := service.NewChatSettingsService(chatSettingsRepo)

	// api
	app := fiber.New(fiber.Config{
		BodyLimit: 100 * 1024 * 1024, // 100 MB лимит для загрузки файлов
	})

	// Initialize default admin if not exists
	existingAdmin, err := adminService.GetByUsername("admin")
	if err != nil {
		log.Fatal(err)
	}
	if existingAdmin == nil {
		_, err := adminService.Create(&dto.AdminCreateRequest{
			Username: "admin",
			Password: "admin",
			IsSuper:  true,
		})
		if err != nil {
			log.Fatal(err)
		}
		log.Println("✓ Default admin created (login: admin, password: admin)")
	}

	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET,POST,PUT,DELETE,OPTIONS",
		AllowHeaders: "Origin, Content-Type, Accept, Authorization",
	}))

	app.Get("/swagger/*", swagger.WrapHandler)
	api.RegisterRoutes(app, cfg, rag, llm, chunkService, adminService, chatService, documentService, chatUserService, chatSettingsService, chatHistoryRepo, messageRepo)

	log.Printf("🚀 Server started at %s", cfg.ServerAddr)
	log.Fatal(app.Listen(cfg.ServerAddr))
}
