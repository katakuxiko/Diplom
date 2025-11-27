package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/service"
)

// CreateChat godoc
// @Summary      Создать чат
// @Tags         chats
// @Accept       json
// @Produce      json
// @Param        chat body dto.ChatCreateRequest true "Chat object"
// @Success      201 {object} dto.ChatResponse
// @Router       /chats [post]
// @Security     BearerAuth
func CreateChat(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req dto.ChatCreateRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
		}

		chat, err := chatService.Create(&req)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}

		return c.Status(201).JSON(dto.ChatResponse{
			ID:          chat.ID,
			AdminID:     chat.AdminID,
			Name:        chat.Name,
			Descr:       chat.Descr,
			CreatedDate: chat.CreatedDate,
		})
	}
}

// GetChat godoc
// @Summary      Получить чат по ID
// @Tags         chats
// @Produce      json
// @Param        id path string true "Chat ID"
// @Success      200 {object} dto.ChatResponse
// @Router       /chats/{id} [get]
// @Security     BearerAuth
func GetChat(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		id := c.Params("id")

		chat, err := chatService.GetByID(id)
		if err != nil {
			return c.Status(404).JSON(fiber.Map{"error": "chat not found"})
		}

		return c.JSON(dto.ChatResponse{
			ID:          chat.ID,
			AdminID:     chat.AdminID,
			Name:        chat.Name,
			Descr:       chat.Descr,
			CreatedDate: chat.CreatedDate,
		})
	}
}

// ListChats godoc
// @Summary      Получить список чатов
// @Tags         chats
// @Produce      json
// @Success      200 {array} dto.ChatResponse
// @Router       /chats [get]
// @Security     BearerAuth
func ListChats(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		claims := c.Locals("user").(jwt.MapClaims)
		adminID, ok := claims["id"].(string)
		if !ok {
			return c.Status(401).JSON(fiber.Map{"error": "invalid token claims"})
		}

		chats, err := chatService.ListByAdmin(adminID)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}

		res := make([]dto.ChatResponse, 0, len(chats))
		for _, chat := range chats {
			res = append(res, dto.ChatResponse{
				ID:          chat.ID,
				AdminID:     chat.AdminID,
				Name:        chat.Name,
				Descr:       chat.Descr,
				CreatedDate: chat.CreatedDate,
			})
		}
		return c.JSON(res)
	}
}

// DeleteChat godoc
// @Summary      Удалить чат
// @Tags         chats
// @Param        id path string true "Chat ID"
// @Success      204
// @Router       /chats/{id} [delete]
// @Security     BearerAuth
func DeleteChat(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		id := c.Params("id")

		if err := chatService.Delete(id); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}

		return c.SendStatus(204)
	}
}
// UpdateChat godoc
// @Summary      Обновить чат
// @Tags         chats
// @Accept       json
// @Produce      json
// @Param        id path string true "Chat ID"
// @Param        chat body dto.ChatCreateRequest true "Chat object"
// @Success      200 {object} dto.ChatResponse
// @Router       /chats/{id} [put]
// @Security     BearerAuth
func UpdateChat(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		id := c.Params("id")
		
		var req dto.ChatUpdateRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
		}

		chat, err := chatService.Update(id, &req)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}

		return c.JSON(dto.ChatResponse{
			ID:          chat.ID,
			AdminID:     chat.AdminID,
			Name:        chat.Name,
			Descr:       chat.Descr,
			CreatedDate: chat.CreatedDate,
		})
	}
}