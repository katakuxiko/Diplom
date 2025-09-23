package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/handlers"
	"github.com/katakuxiko/Diplom/internal/service"
)

var chatService *service.ChatService

func RegisterChatRoutes(router fiber.Router, svc *service.ChatService) {
	chatService = svc
	r := router.Group("/chats")

	r.Get("/", handlers.ListChats(chatService))
	r.Get("/:id", handlers.GetChat(chatService))
	r.Post("/", handlers.CreateChat(chatService))
	r.Delete("/:id", handlers.DeleteChat(chatService))
}
