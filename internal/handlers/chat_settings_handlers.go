package handlers

import (
	"context"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
)

type ChatSettingsHandler struct {
	Service *service.ChatSettingsService // Сервис для работы с настройками чата
}

// CreateOrUpdateChatSettings godoc
// @Summary		Создать или обновить настройки чата
// @Description	Создает новые настройки чата или обновляет существующие по chat_id
// @Tags			chat-settings
// @Accept			json
// @Produce		json
// @Param			settings	body		dto.ChatSettingCreateRequest	true	"Настройки чата"
// @Success		200			{object}	dto.ChatSettingResponse
// @Failure		400			{object}	map[string]string
// @Failure		500			{object}	map[string]string
// @Router			/chat-settings [post]
// @Security		BearerAuth
func (h *ChatSettingsHandler) CreateOrUpdateChatSettings(c *fiber.Ctx) error {
	req := new(dto.ChatSettingCreateRequest)
	// Парсим тело запроса в структуру
	if err := c.BodyParser(req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request: " + err.Error()})
	}

	// Создаем модель из DTO
	settings := &models.ChatSetting{
		ChatID:    req.ChatID,
		HelloText: req.HelloText,
		Name:      req.Name,
		Descr:     req.Descr,
		URL:       req.URL,
		Settings:  req.Settings,
	}

	// Вызываем сервис для создания или обновления
	if err := h.Service.CreateOrUpdate(context.Background(), settings); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	// Формируем ответ
	response := &dto.ChatSettingResponse{
		ID:          settings.ID,
		ChatID:      settings.ChatID,
		HelloText:   settings.HelloText,
		Name:        settings.Name,
		Descr:       settings.Descr,
		URL:         settings.URL,
		CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
		Settings:    settings.Settings,
	}

	return c.Status(fiber.StatusOK).JSON(response)
}

// GetChatSettingsByChatID godoc
// @Summary		Получить настройки по ID чата
// @Description	Возвращает настройки для указанного чата
// @Tags			chat-settings
// @Produce		json
// @Param			chatId	path		string	true	"ID чата (UUID)"
// @Success		200	{object}	dto.ChatSettingResponse
// @Failure		400	{object}	map[string]string
// @Failure		404	{object}	map[string]string
// @Router			/chat-settings/chat/{chatId} [get]
// @Security		BearerAuth
func (h *ChatSettingsHandler) GetChatSettingsByChatID(c *fiber.Ctx) error {
	chatID, err := uuid.Parse(c.Params("chatId"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid chat id format"})
	}

	settings, err := h.Service.GetByChatID(context.Background(), chatID)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "not found"})
	}

	// Конвертируем в DTO
	response := &dto.ChatSettingResponse{
		ID:          settings.ID,
		ChatID:      settings.ChatID,
		HelloText:   settings.HelloText,
		Name:        settings.Name,
		Descr:       settings.Descr,
		URL:         settings.URL,
		CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
		Settings:    settings.Settings,
	}

	return c.JSON(response)
}

// GetChatSettingsByID godoc
// @Summary		Получить настройки чата по ID настройки
// @Description	Возвращает настройки чата по ID записи настроек
// @Tags			chat-settings
// @Produce		json
// @Param			id	path		string	true	"ID настроек чата (UUID)"
// @Success		200	{object}	dto.ChatSettingResponse
// @Failure		400	{object}	map[string]string
// @Failure		404	{object}	map[string]string
// @Router			/chat-settings/{id} [get]
// @Security		BearerAuth
func (h *ChatSettingsHandler) GetChatSettingsByID(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id format"})
	}
	settings, err := h.Service.GetByID(context.Background(), id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "not found"})
	}

	response := &dto.ChatSettingResponse{
		ID:          settings.ID,
		ChatID:      settings.ChatID,
		HelloText:   settings.HelloText,
		Name:        settings.Name,
		Descr:       settings.Descr,
		URL:         settings.URL,
		CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
		Settings:    settings.Settings,
	}

	return c.JSON(response)
}

// UpdateChatSettings godoc
// @Summary		Обновить настройки чата
// @Description	Обновляет настройки чата по ID
// @Tags			chat-settings
// @Accept			json
// @Produce		json
// @Param			id			path		string						true	"ID настроек чата (UUID)"
// @Param			settings	body		dto.ChatSettingUpdateRequest	true	"Обновленные настройки"
// @Success		200			{object}	dto.ChatSettingResponse
// @Failure		400			{object}	map[string]string
// @Failure		500			{object}	map[string]string
// @Router			/chat-settings/{id} [put]
// @Security		BearerAuth
func (h *ChatSettingsHandler) UpdateChatSettings(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id format"})
	}

	req := new(dto.ChatSettingUpdateRequest)
	if err := c.BodyParser(req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request: " + err.Error()})
	}

	// Получаем существующие настройки
	settings, err := h.Service.GetByID(context.Background(), id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "not found"})
	}

	// Обновляем поля
	settings.HelloText = req.HelloText
	settings.Name = req.Name
	settings.Descr = req.Descr
	settings.URL = req.URL
	settings.Settings = req.Settings

	if err := h.Service.Update(context.Background(), settings); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	response := &dto.ChatSettingResponse{
		ID:          settings.ID,
		ChatID:      settings.ChatID,
		HelloText:   settings.HelloText,
		Name:        settings.Name,
		Descr:       settings.Descr,
		URL:         settings.URL,
		CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
		Settings:    settings.Settings,
	}

	return c.JSON(response)
}

// DeleteChatSettings godoc
// @Summary		Удалить настройки чата
// @Description	Удаляет настройки чата по ID
// @Tags			chat-settings
// @Param			id	path	string	true	"ID настроек чата (UUID)"
// @Success		204
// @Failure		400	{object}	map[string]string
// @Failure		500	{object}	map[string]string
// @Router			/chat-settings/{id} [delete]
// @Security		BearerAuth
func (h *ChatSettingsHandler) DeleteChatSettings(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id format"})
	}
	if err := h.Service.Delete(context.Background(), id); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.SendStatus(fiber.StatusNoContent)
}

// ListChatSettings godoc
// @Summary		Получить все настройки чатов
// @Description	Возвращает список всех настроек чатов
// @Tags			chat-settings
// @Produce		json
// @Success		200	{array}		dto.ChatSettingResponse
// @Failure		500	{object}	map[string]string
// @Router			/chat-settings [get]
// @Security		BearerAuth
func (h *ChatSettingsHandler) ListChatSettings(c *fiber.Ctx) error {
	settingsList, err := h.Service.List(context.Background())
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	// Конвертируем в DTO
	response := make([]dto.ChatSettingResponse, len(settingsList))
	for i, settings := range settingsList {
		response[i] = dto.ChatSettingResponse{
			ID:          settings.ID,
			ChatID:      settings.ChatID,
			HelloText:   settings.HelloText,
			Name:        settings.Name,
			Descr:       settings.Descr,
			URL:         settings.URL,
			CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
			Settings:    settings.Settings,
		}
	}

	return c.JSON(response)
}
