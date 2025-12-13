package handlers

import (
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/service"
)

// лучше хранить секрет в cfg.Config, а не в коде

// LoginHandler godoc
// @Summary Авторизация администратора
// @Description Проверяет логин/пароль и возвращает JWT токен
// @Tags auth
// @Accept json
// @Produce json
// @Param request body dto.LoginRequest true "Данные для входа"
// @Success 200 {object} map[string]string "Токен"
// @Failure 400 {object} map[string]string "Некорректное тело запроса"
// @Failure 401 {object} map[string]string "Неверные данные авторизации"
// @Failure 500 {object} map[string]string "Ошибка при создании токена"
// @Router /auth/login [post]
func LoginHandler(svc *service.AdminService, cfg *config.Config) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req dto.LoginRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(fiber.StatusBadRequest).
				JSON(fiber.Map{"error": "invalid body"})
		}

		admin, err := svc.GetByUsername(req.Username)
		if err != nil {
			return c.Status(fiber.StatusUnauthorized).
				JSON(fiber.Map{"error": "invalid credentials"})
		}
		if admin == nil {
			return c.Status(fiber.StatusUnauthorized).
				JSON(fiber.Map{"error": "invalid credentials"})
		}

		if !svc.CheckPassword(req.Password, admin.PasswordHash) {
			return c.Status(fiber.StatusUnauthorized).
				JSON(fiber.Map{"error": "invalid credentials"})
		}

		role := "user"

		if admin.IsSuperUser {
			role = "superuser"
		}

		// Создание токена
		claims := jwt.MapClaims{
			"id":       admin.ID.String(),
			"role":     role,
			"exp":      time.Now().Add(24 * time.Hour).Unix(),
			"username": admin.Username,
		}

		token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
		signed, err := token.SignedString(cfg.JWTSecret)
		print(cfg.JWTSecret)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).
				JSON(fiber.Map{"error": "could not sign token"})
		}

		return c.JSON(fiber.Map{"token": signed})
	}
}

func RegisterAuthRoutes(app *fiber.App, svc *service.AdminService, cfg *config.Config) {
	r := app.Group("/auth")
	r.Post("/login", LoginHandler(svc, cfg))
}
