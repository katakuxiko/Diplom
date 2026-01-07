package routes

import (
	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/handlers"
)

func RegisterChatSettingsRoutes(app *fiber.App, handler *handlers.ChatSettingsHandler) {
	settings := app.Group("/chat-settings")

	settings.Get("/", handler.ListChatSettings)
	settings.Get("/chat/:chatId", handler.GetChatSettingsByChatID)
	settings.Get("/:id", handler.GetChatSettingsByID)
	settings.Post("/", handler.CreateOrUpdateChatSettings)
	settings.Put("/:id", handler.UpdateChatSettings)
	settings.Delete("/:id", handler.DeleteChatSettings)
}
