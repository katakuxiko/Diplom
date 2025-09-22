package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
)

// Обновление access токена по refresh токену
type RefreshTokenRequest struct {
	RefreshToken string `json:"refresh_token"`
}

func RefreshAccessToken(c *fiber.Ctx) error {
	var req RefreshTokenRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid request"})
	}
	claims, err := service.ParseJWT(req.RefreshToken)
	if err != nil {
		return c.Status(401).JSON(fiber.Map{"error": "invalid refresh token"})
	}
	// Генерируем новый access token
	accessToken, err := service.GenerateJWT(claims.AdminID, 15*60) // 15 минут
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "token generation error"})
	}
	return c.JSON(fiber.Map{"access_token": accessToken})
}

var adminService *service.AdminService

// Авторизация администратора
func AdminLogin(c *fiber.Ctx) error {
	var req dto.AdminAuthRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid request"})
	}
	tokens, err := adminService.Authenticate(&req)
	if err != nil {
		return c.Status(401).JSON(fiber.Map{"error": "invalid credentials"})
	}
	return c.JSON(tokens)
}

// RegisterAdminRoutes регистрирует CRUD эндпоинты для админов
func RegisterAdminRoutes(router fiber.Router, svc *service.AdminService) {
	adminService = svc
	r := router.Group("/admins")

	r.Get("/", GetAdmins)
	r.Get("/:id", GetAdminByID)
	r.Post("/", CreateAdmin)
	r.Put("/:id", UpdateAdmin)
	r.Delete("/:id", DeleteAdmin)

	// Эндпоинт авторизации только для публичного app, не для защищённой группы
	if app, ok := router.(*fiber.App); ok {
		app.Post("/api/login", AdminLogin)
		app.Post("/api/refresh", RefreshAccessToken)
	}
}

// GetAdmins godoc
// @Summary		Получить всех админов 1
// @Description	Возвращает список админов
// @Tags			admins
// @Produce		json
// @Success		200	{array}		[]dto.AdminResponse
// @Failure		500	{object}	map[string]string
// @Router			/admins [get]
func GetAdmins(c *fiber.Ctx) error {
	admins, err := adminService.GetAll()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(admins)
}

// GetAdminByID godoc
// @Summary		Получить админа по ID
// @Description	Возвращает админа по UUID
// @Tags			admins
// @Param			id	path	string	true	"Admin ID"
// @Produce		json
// @Success		200	{object}	dto.AdminResponse
// @Failure		400	{object}	map[string]string
// @Failure		404	{object}	map[string]string
// @Router			/admins/{id} [get]
func GetAdminByID(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	admin, err := adminService.GetByID(id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "admin not found"})
	}
	return c.JSON(admin)
}

// CreateAdmin godoc
// @Summary		Создать админа
// @Description	Создаёт нового админа
// @Tags			admins
// @Accept			json
// @Produce		json
// @Param			admin	body		dto.AdminCreateRequest	true	"Admin object"
// @Success		201		{object}	dto.AdminResponse
// @Failure		400		{object}	map[string]string
// @Failure		500		{object}	map[string]string
// @Router			/admins [post]
func CreateAdmin(c *fiber.Ctx) error {
	var admin dto.AdminCreateRequest
	if err := c.BodyParser(&admin); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	if _, err := adminService.Create(&admin); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(201).JSON(admin)
}

// UpdateAdmin godoc
// @Summary		Обновить админа
// @Description	Обновляет данные админа по UUID
// @Tags			admins
// @Accept			json
// @Produce		json
// @Param			id		path		string			true	"Admin ID"
// @Param			admin	body		dto.AdminCreateRequest	true	"Admin object"
// @Success		200		{object}	dto.AdminResponse
// @Failure		400		{object}	map[string]string
// @Failure		500		{object}	map[string]string
// @Router			/admins/{id} [put]
func UpdateAdmin(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	var admin models.Admin
	if err := c.BodyParser(&admin); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	admin.ID = id
	if err := adminService.Update(&admin); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(admin)
}

// DeleteAdmin godoc
// @Summary		Удалить админа
// @Description	Удаляет админа по UUID
// @Tags			admins
// @Param			id	path	string	true	"Admin ID"
// @Success		204	"No Content"
// @Failure		400	{object}	map[string]string
// @Failure		500	{object}	map[string]string
// @Router			/admins/{id} [delete]
func DeleteAdmin(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	if err := adminService.Delete(id); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.SendStatus(204)
}
