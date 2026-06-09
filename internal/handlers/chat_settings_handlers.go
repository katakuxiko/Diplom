package handlers

import (
	"context"
	"errors"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/middleware"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
	"gorm.io/gorm"
)

type ChatSettingsHandler struct {
	Service *service.ChatSettingsService // Сервис для работы с настройками чата
}

// sanitizeSettings возвращает копию settings без полей с секретами,
// но с флагами наличия ключей: externalApiKeySet, embedExternalApiKeySet.
// sanitizeSettings возвращает копию settings без полей с секретами,
// но с флагами наличия ключей: externalApiKeySet, embedExternalApiKeySet.
// Если запрос содержит JWT с ролью superuser — возвращаем полные (дешифрованные) ключи.
func sanitizeSettings(c *fiber.Ctx, s models.JSONB) models.JSONB {
	// пустые настройки
	if s == nil {
		return models.JSONB{
			"externalApiKeySet":      false,
			"embedExternalApiKeySet": false,
		}
	}

	// определим, является ли запрос админским
	isAdmin := false
	if c != nil {
		// сначала проверим, положил ли middleware.JWTProtected() claims в Locals
		if v := c.Locals("user"); v != nil {
			if claims, ok := v.(jwt.MapClaims); ok {
				if r, ok2 := claims["role"].(string); ok2 && r == "superuser" {
					isAdmin = true
				}
			}
		} else {
			// если middleware не сработал (эндпоинт публичный), попробуем распарсить токен из заголовка
			tokenStr := c.Get("Authorization")
			if tokenStr != "" {
				if len(tokenStr) > 7 && tokenStr[:7] == "Bearer " {
					tokenStr = tokenStr[7:]
				}
				if tok, err := jwt.ParseWithClaims(tokenStr, jwt.MapClaims{}, func(t *jwt.Token) (interface{}, error) {
					if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
						return nil, fiber.ErrUnauthorized
					}
					return middleware.JwtSecret, nil
				}); err == nil && tok != nil && tok.Valid {
					if claims, ok := tok.Claims.(jwt.MapClaims); ok {
						if r, ok2 := claims["role"].(string); ok2 && r == "superuser" {
							isAdmin = true
						}
					}
				}
			}
		}
	}

	if isAdmin {
		// вернуть полные настройки, попытаемся дешифровать ключи
		out := make(models.JSONB)
		for k, v := range s {
			out[k] = v
		}
		if v, ok := out["externalApiKey"].(string); ok && v != "" {
			if dec, derr := utils.DecryptString(v); derr == nil {
				out["externalApiKey"] = dec
			}
		}
		if v2, ok := out["embedExternalApiKey"].(string); ok && v2 != "" {
			if dec2, derr2 := utils.DecryptString(v2); derr2 == nil {
				out["embedExternalApiKey"] = dec2
			}
		}
		return out
	}

	// не админ — скрываем секреты и выставляем флаги наличия
	out := make(models.JSONB)
	for k, v := range s {
		if k == "externalApiKey" || k == "embedExternalApiKey" {
			continue
		}
		out[k] = v
	}
	if v, ok := s["externalApiKey"]; ok {
		if str, ok2 := v.(string); ok2 && str != "" {
			out["externalApiKeySet"] = true
		} else {
			out["externalApiKeySet"] = false
		}
	} else {
		out["externalApiKeySet"] = false
	}
	if v2, ok := s["embedExternalApiKey"]; ok {
		if str2, ok2 := v2.(string); ok2 && str2 != "" {
			out["embedExternalApiKeySet"] = true
		} else {
			out["embedExternalApiKeySet"] = false
		}
	} else {
		out["embedExternalApiKeySet"] = false
	}
	return out
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

	// Попробуем получить существующие настройки, чтобы безопасно замерджить поля
	var settings *models.ChatSetting
	existing, err := h.Service.GetByChatID(context.Background(), req.ChatID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			// создаём новые
			settings = &models.ChatSetting{
				ChatID:    req.ChatID,
				HelloText: req.HelloText,
				Name:      req.Name,
				Descr:     req.Descr,
				URL:       req.URL,
				Settings:  req.Settings,
			}
		} else {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
	} else {
		// обновляем существующие поля и мерджим Settings
		settings = existing
		settings.HelloText = req.HelloText
		settings.Name = req.Name
		settings.Descr = req.Descr
		settings.URL = req.URL
		if settings.Settings == nil {
			settings.Settings = make(models.JSONB)
		}
		if req.Settings != nil {
			for k, v := range req.Settings {
				settings.Settings[k] = v
			}
		}
	}

	// Перед сохранением шифруем внешние ключи, если они были переданы
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
		Settings:    sanitizeSettings(c, settings.Settings),
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

	// Не возвращаем реальные API-ключи клиенту — отдаем sanitized settings
	response := &dto.ChatSettingResponse{
		ID:          settings.ID,
		ChatID:      settings.ChatID,
		HelloText:   settings.HelloText,
		Name:        settings.Name,
		Descr:       settings.Descr,
		URL:         settings.URL,
		CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
		Settings:    sanitizeSettings(c, settings.Settings),
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

	// Не возвращаем реальные API-ключи клиенту — отдаем sanitized settings
	response := &dto.ChatSettingResponse{
		ID:          settings.ID,
		ChatID:      settings.ChatID,
		HelloText:   settings.HelloText,
		Name:        settings.Name,
		Descr:       settings.Descr,
		URL:         settings.URL,
		CreatedDate: settings.CreatedDate.Format("2006-01-02T15:04:05Z07:00"),
		Settings:    sanitizeSettings(c, settings.Settings),
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
	// Мерджим Settings, не затирая существующие ключи
	if settings.Settings == nil {
		settings.Settings = make(models.JSONB)
	}
	if req.Settings != nil {
		for k, v := range req.Settings {
			settings.Settings[k] = v
		}
	}

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
 		Settings:    sanitizeSettings(c, settings.Settings),
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
			Settings:    sanitizeSettings(c, settings.Settings),
		}
	}

	return c.JSON(response)
}
