package api

import (
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
)

// RegisterAdminRoutes регистрирует CRUD эндпоинты для админов
func RegisterAdminRoutes(app *fiber.App, adminService *service.AdminService) {
	r := app.Group("/admins")

	//	@Summary		Получить всех админов
	//	@Description	Возвращает список админов
	//	@Tags			admins
	//	@Produce		json
	//	@Success		200	{array}		models.Admin
	//	@Failure		500	{object}	map[string]string
	//	@Router			/admins [get]
	r.Get("/", func(c *fiber.Ctx) error {
		admins, err := adminService.GetAll()
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(admins)
	})

	//	@Summary		Получить админа по ID
	//	@Description	Возвращает админа по UUID
	//	@Tags			admins
	//	@Param			id	path	string	true	"Admin ID"
	//	@Produce		json
	//	@Success		200	{object}	models.Admin
	//	@Failure		400	{object}	map[string]string
	//	@Failure		404	{object}	map[string]string
	//	@Router			/admins/{id} [get]
	r.Get("/:id", func(c *fiber.Ctx) error {
		id, err := uuid.Parse(c.Params("id"))
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
		}
		admin, err := adminService.GetByID(id)
		if err != nil {
			return c.Status(404).JSON(fiber.Map{"error": "admin not found"})
		}
		return c.JSON(admin)
	})

	//	@Summary		Создать админа
	//	@Description	Создаёт нового админа
	//	@Tags			admins
	//	@Accept			json
	//	@Produce		json
	//	@Param			admin	body		models.Admin	true	"Admin object"
	//	@Success		201		{object}	models.Admin
	//	@Failure		400		{object}	map[string]string
	//	@Failure		500		{object}	map[string]string
	//	@Router			/admins [post]
	r.Post("/", func(c *fiber.Ctx) error {
		var admin models.Admin
		if err := c.BodyParser(&admin); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
		}
		if err := adminService.Create(&admin); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}
		return c.Status(201).JSON(admin)
	})

	//	@Summary		Обновить админа
	//	@Description	Обновляет данные админа по UUID
	//	@Tags			admins
	//	@Accept			json
	//	@Produce		json
	//	@Param			id		path		string			true	"Admin ID"
	//	@Param			admin	body		models.Admin	true	"Admin object"
	//	@Success		200		{object}	models.Admin
	//	@Failure		400		{object}	map[string]string
	//	@Failure		500		{object}	map[string]string
	//	@Router			/admins/{id} [put]
	r.Put("/:id", func(c *fiber.Ctx) error {
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
	})

	//	@Summary		Удалить админа
	//	@Description	Удаляет админа по UUID
	//	@Tags			admins
	//	@Param			id	path	string	true	"Admin ID"
	//	@Success		204	"No Content"
	//	@Failure		400	{object}	map[string]string
	//	@Failure		500	{object}	map[string]string
	//	@Router			/admins/{id} [delete]
	r.Delete("/:id", func(c *fiber.Ctx) error {
		id, err := uuid.Parse(c.Params("id"))
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
		}
		if err := adminService.Delete(id); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
		}
		return c.SendStatus(204)
	})
}
