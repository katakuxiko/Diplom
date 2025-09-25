package middleware

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
)

var JwtSecret []byte

func JWTProtected() fiber.Handler {
	return func(c *fiber.Ctx) error {
		fmt.Printf("JWTProtected middleware called %v\n", JwtSecret)
		tokenStr := c.Get("Authorization")
		if tokenStr == "" {
			return c.Status(401).JSON(fiber.Map{"error": "missing token"})
		}

		// убираем "Bearer "
		if len(tokenStr) > 7 && tokenStr[:7] == "Bearer " {
			tokenStr = tokenStr[7:]
		}

		token, err := jwt.Parse(tokenStr, func(t *jwt.Token) (interface{}, error) {
			if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fiber.ErrUnauthorized
			}
			return JwtSecret, nil
		})
		if err != nil || !token.Valid {
			fmt.Print(err)
			return c.Status(401).JSON(fiber.Map{"error": "invalid token"})
		}

		// можно сохранить claims в контекст
		if claims, ok := token.Claims.(jwt.MapClaims); ok {
			c.Locals("user", claims)
		}

		return c.Next()
	}
}