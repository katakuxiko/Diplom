package handlers

import (
	"context"
	"strconv"

	"github.com/gofiber/fiber/v2"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
)

type ChatSettingsHandler struct {
	Service *service.ChatSettingsService // Сервис для работы с настройками чата
}

// CreateChatSettings создает новые настройки чата
func (h *ChatSettingsHandler) CreateChatSettings(c *fiber.Ctx) error {
	settings := new(models.ChatSettings)
	// Парсим тело запроса в структуру
	if err := c.BodyParser(settings); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request"})
	}
	// Вызываем сервис для создания
	if err := h.Service.Create(context.Background(), settings); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusCreated).JSON(settings)
}

// GetChatSettingsByID возвращает настройки чата по ID
func (h *ChatSettingsHandler) GetChatSettingsByID(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id"})
	}
	settings, err := h.Service.GetByID(context.Background(), id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "not found"})
	}
	return c.JSON(settings)
}

// UpdateChatSettings обновляет настройки чата
func (h *ChatSettingsHandler) UpdateChatSettings(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id"})
	}
	settings := new(models.ChatSettings)
	if err := c.BodyParser(settings); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request"})
	}
	settings.ID = id // Устанавливаем ID из параметра
	if err := h.Service.Update(context.Background(), settings); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(settings)
}

// DeleteChatSettings удаляет настройки чата
func (h *ChatSettingsHandler) DeleteChatSettings(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id"})
	}
	if err := h.Service.Delete(context.Background(), id); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.SendStatus(fiber.StatusNoContent)
}

// ListChatSettings возвращает все настройки чата
func (h *ChatSettingsHandler) ListChatSettings(c *fiber.Ctx) error {
	settingsList, err := h.Service.List(context.Background())
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(settingsList)
}
