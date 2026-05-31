package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/handlers"
	"github.com/katakuxiko/Diplom/internal/middleware"
	"github.com/katakuxiko/Diplom/internal/service"
)

var chatService *service.ChatService

func RegisterChatRoutes(app *fiber.App, svc *service.ChatService) {
	chatService = svc
	r := app.Group("/chats")

	r.Get("/", middleware.JWTProtected(), handlers.ListChats(chatService))
	r.Get("/:id", middleware.JWTProtected(), handlers.GetChat(chatService))
	r.Get("/:id/admins", middleware.JWTProtected(), handlers.ListChatAdmins(chatService))
	r.Post("/", middleware.JWTProtected(), handlers.CreateChat(chatService))
	r.Post("/:id/invite-admin", middleware.JWTProtected(), handlers.InviteAdminToChat(chatService))
	r.Delete("/:id", middleware.JWTProtected(), handlers.DeleteChat(chatService))
	r.Put("/:id", middleware.JWTProtected(), handlers.UpdateChat(chatService))
}
