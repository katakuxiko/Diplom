package handlers

import (
	"errors"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/service"
)

func currentAdminID(c *fiber.Ctx) (string, bool) {
	claims, ok := c.Locals("user").(jwt.MapClaims)
	if !ok {
		_ = c.Status(401).JSON(fiber.Map{"error": "invalid token claims"})
		return "", false
	}

	adminID, ok := claims["id"].(string)
	if !ok {
		_ = c.Status(401).JSON(fiber.Map{"error": "invalid token claims"})
		return "", false
	}

	return adminID, true
}

func writeChatServiceError(c *fiber.Ctx, err error) error {
	switch {
	case errors.Is(err, service.ErrChatNotFound):
		return c.Status(404).JSON(fiber.Map{"error": "chat not found"})
	case errors.Is(err, service.ErrChatAccessDenied):
		return c.Status(403).JSON(fiber.Map{"error": "access denied"})
	case errors.Is(err, service.ErrOnlyCreatorCanInvite):
		return c.Status(403).JSON(fiber.Map{"error": "only chat creator can invite admins"})
	case errors.Is(err, service.ErrAdminNotFound):
		return c.Status(404).JSON(fiber.Map{"error": "admin not found"})
	case errors.Is(err, service.ErrAdminAlreadyInChat):
		return c.Status(409).JSON(fiber.Map{"error": "admin already in chat"})
	default:
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
}

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
		adminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		var req dto.ChatCreateRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
		}

		chat, err := chatService.CreateForAdmin(adminID, &req)
		if err != nil {
			return writeChatServiceError(c, err)
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
		adminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		id := c.Params("id")
		if err := chatService.EnsureAdminAccess(id, adminID); err != nil {
			return writeChatServiceError(c, err)
		}

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
		adminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		chats, err := chatService.ListByAdmin(adminID)
		if err != nil {
			return writeChatServiceError(c, err)
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

// ListPublicChats godoc
// @Summary      Получить публичный список всех чатов
// @Tags         chats
// @Produce      json
// @Success      200 {array} dto.ChatResponse
// @Router       /public/chats [get]
func ListPublicChats(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		chats, err := chatService.ListAll()
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
		adminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		id := c.Params("id")
		if err := chatService.EnsureAdminAccess(id, adminID); err != nil {
			return writeChatServiceError(c, err)
		}

		if err := chatService.Delete(id); err != nil {
			return writeChatServiceError(c, err)
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
		adminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		id := c.Params("id")
		if err := chatService.EnsureAdminAccess(id, adminID); err != nil {
			return writeChatServiceError(c, err)
		}

		var req dto.ChatUpdateRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
		}

		chat, err := chatService.Update(id, &req)
		if err != nil {
			return writeChatServiceError(c, err)
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

// InviteAdminToChat godoc
// @Summary      Пригласить админа в чат
// @Tags         chats
// @Accept       json
// @Produce      json
// @Param        id path string true "Chat ID"
// @Param        invite body dto.ChatInviteAdminRequest true "Admin invite payload"
// @Success      201 {object} map[string]string
// @Router       /chats/{id}/invite-admin [post]
// @Security     BearerAuth
func InviteAdminToChat(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		inviterAdminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		chatID := c.Params("id")

		var req dto.ChatInviteAdminRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
		}
		if req.AdminID == "" {
			return c.Status(400).JSON(fiber.Map{"error": "admin_id is required"})
		}

		if err := chatService.InviteAdmin(chatID, inviterAdminID, req.AdminID); err != nil {
			return writeChatServiceError(c, err)
		}

		return c.Status(201).JSON(fiber.Map{"status": "ok"})
	}
}

// ListChatAdmins godoc
// @Summary      Получить админов чата
// @Tags         chats
// @Produce      json
// @Param        id path string true "Chat ID"
// @Success      200 {array} dto.AdminResponse
// @Router       /chats/{id}/admins [get]
// @Security     BearerAuth
func ListChatAdmins(chatService service.ChatServiceInterface) fiber.Handler {
	return func(c *fiber.Ctx) error {
		adminID, ok := currentAdminID(c)
		if !ok {
			return nil
		}

		chatID := c.Params("id")
		admins, err := chatService.ListAdmins(chatID, adminID)
		if err != nil {
			return writeChatServiceError(c, err)
		}

		result := make([]dto.AdminResponse, 0, len(admins))
		for _, admin := range admins {
			result = append(result, dto.AdminResponse{
				ID:          admin.ID,
				Username:    admin.Username,
				IsSuperUser: admin.IsSuperUser,
			})
		}

		return c.JSON(result)
	}
}
