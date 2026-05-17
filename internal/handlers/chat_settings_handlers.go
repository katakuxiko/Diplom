package handlers

import (
	"context"
	"errors"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
	"gorm.io/gorm"
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

	// Перед сохранением шифруем externalApiKey (если есть)
	if settings.Settings != nil {
		if v, ok := settings.Settings["externalApiKey"]; ok {
			if s, ok2 := v.(string); ok2 && s != "" {
				enc, err := utils.EncryptString(s)
				if err == nil {
					settings.Settings["externalApiKey"] = enc
				}
			}
		}
		// шифруем ключ для embedding провайдера, если указан
		if v2, ok := settings.Settings["embedExternalApiKey"]; ok {
			if s2, ok22 := v2.(string); ok22 && s2 != "" {
				enc2, err := utils.EncryptString(s2)
				if err == nil {
					settings.Settings["embedExternalApiKey"] = enc2
				}
			}
		}
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
		if errors.Is(err, gorm.ErrRecordNotFound) {
			// Возвращаем 200 и пустой объект, если настроек для данного чата нет
			return c.JSON(fiber.Map{})
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	// Перед возвратом дешифруем externalApiKey если он зашифрован
	if settings.Settings != nil {
		if v, ok := settings.Settings["externalApiKey"]; ok {
			if s, ok2 := v.(string); ok2 && s != "" {
				if dec, err := utils.DecryptString(s); err == nil {
					settings.Settings["externalApiKey"] = dec
				}
			}
		}
		// дешифруем ключ embed-провайдера
		if v2, ok := settings.Settings["embedExternalApiKey"]; ok {
			if s2, ok2 := v2.(string); ok2 && s2 != "" {
				if dec2, err := utils.DecryptString(s2); err == nil {
					settings.Settings["embedExternalApiKey"] = dec2
				}
			}
		}
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

	// Дешифруем ключ если нужно
	if settings.Settings != nil {
		if v, ok := settings.Settings["externalApiKey"]; ok {
			if s, ok2 := v.(string); ok2 && s != "" {
				if dec, err := utils.DecryptString(s); err == nil {
					settings.Settings["externalApiKey"] = dec
				}
			}
		}
		if v2, ok := settings.Settings["embedExternalApiKey"]; ok {
			if s2, ok2 := v2.(string); ok2 && s2 != "" {
				if dec2, err := utils.DecryptString(s2); err == nil {
					settings.Settings["embedExternalApiKey"] = dec2
				}
			}
		}
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

	// Шифруем ключ перед сохранением
	if settings.Settings != nil {
		if v, ok := settings.Settings["externalApiKey"]; ok {
			if s, ok2 := v.(string); ok2 && s != "" {
				enc, err := utils.EncryptString(s)
				if err == nil {
					settings.Settings["externalApiKey"] = enc
				}
			}
		}
		if v2, ok := settings.Settings["embedExternalApiKey"]; ok {
			if s2, ok2 := v2.(string); ok2 && s2 != "" {
				enc2, err := utils.EncryptString(s2)
				if err == nil {
					settings.Settings["embedExternalApiKey"] = enc2
				}
			}
		}
	}

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
		// дешифруем ключи
		if settings.Settings != nil {
			if v, ok := settings.Settings["externalApiKey"]; ok {
				if s, ok2 := v.(string); ok2 && s != "" {
					if dec, err := utils.DecryptString(s); err == nil {
						settings.Settings["externalApiKey"] = dec
					}
				}
			}
			if v2, ok := settings.Settings["embedExternalApiKey"]; ok {
				if s2, ok2 := v2.(string); ok2 && s2 != "" {
					if dec2, err := utils.DecryptString(s2); err == nil {
						settings.Settings["embedExternalApiKey"] = dec2
					}
				}
			}
		}
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
