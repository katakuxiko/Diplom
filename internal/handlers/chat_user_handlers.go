package handlers

import (
	"bytes"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
)

var ChatUserService *service.ChatUserService

// RegisterChatUserRoutes регистрирует CRUD эндпоинты для пользователей чата

func RegisterChatUserRoutes(app *fiber.App, svc *service.ChatUserService) {
	ChatUserService = svc
	r := app.Group("/chatusers")

	r.Get("/", GetChatUsers)
	r.Get("/:id", GetChatUserByID)
	r.Post("/", CreateChatUser)
	r.Put("/:id", UpdateChatUser)
	r.Delete("/:id", DeleteChatUser)
	r.Post("/import", ImportChatUsers)

}

// GetChatUsers godoc
// @Summary		Получить всех пользователей чата
// @Description	Возвращает список пользователей чата
// @Tags			chatusers
// @Produce		json
// @Success		200	{array}		[]dto.ChatUserResponse
// @Failure		500	{object}	map[string]string
// @Router			/chatusers [get]
// @Security     BearerAuth
func GetChatUsers(c *fiber.Ctx) error {
	chatUser, err := ChatUserService.GetAll()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(chatUser)
}

// GetChatUserByID godoc
// @Summary		Получить пользователя чата по ID
// @Description	Возвращает пользователя чата по UUID
// @Tags			chatusers
// @Param			id	path	string	true	"ChatUser ID"
// @Produce		json
// @Success		200	{object}	dto.ChatUserResponse
// @Failure		400	{object}	map[string]string
// @Failure		404	{object}	map[string]string
// @Router			/chatusers/{id} [get]
func GetChatUserByID(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	chatuser, err := ChatUserService.GetByID(id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "chatuser not found"})
	}
	return c.JSON(chatuser)
}

// CreateChatUser godoc
// @Summary		Создать пользователя чата
// @Description	Создаёт нового пользователя чата
// @Tags			chatusers
// @Accept			json
// @Produce		json
// @Param			chatuser	body		dto.ChatUserCreateRequest	true	"chatuser object"
// @Success		201		{object}	dto.ChatUserResponse
// @Failure		400		{object}	map[string]string
// @Failure		500		{object}	map[string]string
// @Router			/chatusers [post]
func CreateChatUser(c *fiber.Ctx) error {
	var chatuser dto.ChatUserCreateRequest
	if err := c.BodyParser(&chatuser); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	created, err := ChatUserService.Create(&chatuser)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	resp := dto.ChatUserResponse{
		ID:          created.ID,
		ChatID:      created.ChatID,
		Username:    created.Username,
		User_Role:   chatuser.User_Role,
		User_Info:   chatuser.User_Info,
		AccessLevel: chatuser.AccessLevel,
	}
	return c.Status(201).JSON(resp)
}

// Updatechatuser godoc
// @Summary		Обновить пользователя чата
// @Description	Обновляет данные пользователя чата по UUID
// @Tags			chatusers
// @Accept			json
// @Produce		json
// @Param			id		path		string			true	"chatuser ID"
// @Param			chatuser	body		dto.ChatUserCreateRequest	true	"ChatUser object"
// @Success		200		{object}	dto.ChatUserResponse
// @Failure		400		{object}	map[string]string
// @Failure		500		{object}	map[string]string
// @Router			/chatusers/{id} [put]
func UpdateChatUser(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	existing, err := ChatUserService.GetByID(id)
	if err != nil || existing == nil {
		return c.Status(404).JSON(fiber.Map{"error": "chatuser not found"})
	}

	var payload dto.ChatUserCreateRequest
	if err := c.BodyParser(&payload); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}

	// preserve chat_id if not provided
	if payload.ChatID != "" {
		if parsed, err := uuid.Parse(payload.ChatID); err == nil {
			existing.ChatID = parsed
		}
	}

	existing.Username = payload.Username
	existing.UserInfo = payload.User_Info
	existing.RoleName = payload.User_Role
	existing.AccessLevel = payload.AccessLevel

	if payload.Password != "" {
		hash, err := utils.HashPassword(payload.Password)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": "failed to hash password"})
		}
		existing.PasswordHash = hash
	}

	if err := ChatUserService.Update(existing); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(existing)
}

// DeleteChatUser godoc
// @Summary		пользователя чата
// @Description	пользователя чата по UUID
// @Tags			chatusers
// @Param			id	path	string	true	"chatuser ID"
// @Success		204	"No Content"
// @Failure		400	{object}	map[string]string
// @Failure		500	{object}	map[string]string
// @Router			/chatusers/{id} [delete]
func DeleteChatUser(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	if err := ChatUserService.Delete(id); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.SendStatus(204)
}

// ImportChatUsers godoc
// @Summary      Импорт пользователей из Excel/CSV
// @Description  Загружает файл (.xlsx или .csv) и добавляет пользователей в базу. Колонки: username,password,role,chat_id
// @Tags         chatusers
// @Accept       multipart/form-data
// @Produce      json
// @Param        file  formData  file  true  "Файл (.xlsx или .csv)"
// @Success      200   {object}  map[string]string
// @Failure      400   {object}  map[string]string
// @Failure      500   {object}  map[string]string
// @Router       /chatusers/import [post]
// @Security     BearerAuth
func ImportChatUsers(c *fiber.Ctx) error {
	file, err := c.FormFile("file")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "файл не найден"})
	}

	f, err := file.Open()
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "не удалось открыть файл"})
	}
	defer f.Close()

	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(f); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "ошибка чтения файла"})
	}

	// Определяем тип по расширению
	name := file.Filename
	var importErr error
	if strings.HasSuffix(strings.ToLower(name), ".csv") {
		importErr = ChatUserService.ImportFromCSV(buf.Bytes())
	} else {
		importErr = ChatUserService.ImportFromExcel(buf.Bytes())
	}
	if importErr != nil {
		return c.Status(500).JSON(fiber.Map{"error": importErr.Error()})
	}

	return c.JSON(fiber.Map{"message": "Пользователи успешно импортированы"})
}
