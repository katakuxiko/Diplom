package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/handlers"
)

func RegisterChatSettingsRoutes(app *fiber.App, handler *handlers.ChatSettingsHandler) {
	app.Get("/chat-settings", handler.ListChatSettings)
	app.Get("/chat-settings/:id", handler.GetChatSettingsByID)
	app.Post("/chat-settings", handler.CreateChatSettings)
	app.Put("/chat-settings/:id", handler.UpdateChatSettings)
	app.Delete("/chat-settings/:id", handler.DeleteChatSettings)
}
