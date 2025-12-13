package middleware

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
)

var JwtSecret []byte

func JWTProtected() fiber.Handler {
	return func(c *fiber.Ctx) error {
		claims, err := parseClaimsFromHeader(c)
		if err != nil {
			return err
		}

		// сохранить claims в контекст
		c.Locals("user", claims)

		return c.Next()
	}
}

// SuperadminProtected возвращает middleware, который проверяет JWT и роль пользователя.
// Доступ разрешается только если claim "role" == "superuser". Иначе 403.
func SuperadminProtected() fiber.Handler {
	return func(c *fiber.Ctx) error {
		// если предыдущий middleware уже положил user в Locals, используем его
		if v := c.Locals("user"); v != nil {
			if claims, ok := v.(jwt.MapClaims); ok {
				if hasSuperuserRole(claims) {
					return c.Next()
				}
				return c.Status(403).JSON(fiber.Map{"error": "forbidden"})
			}
		}

		// иначе парсим токен из заголовка и проверяем роль
		claims, err := parseClaimsFromHeader(c)
		fmt.Println("1", claims)
		if err != nil {
			return err
		}

		if !hasSuperuserRole(claims) {
			return c.Status(403).JSON(fiber.Map{"error": "forbidden"})
		}

		// сохраняем и продолжаем
		c.Locals("user", claims)
		return c.Next()
	}
}

// parseClaimsFromHeader читает Authorization header, парсит JWT и возвращает MapClaims.
func parseClaimsFromHeader(c *fiber.Ctx) (jwt.MapClaims, error) {
	tokenStr := c.Get("Authorization")
	if tokenStr == "" {
		return nil, c.Status(401).JSON(fiber.Map{"error": "missing token"})
	}

	// убираем "Bearer "
	if len(tokenStr) > 7 && tokenStr[:7] == "Bearer " {
		tokenStr = tokenStr[7:]
	}

	// Parse with explicit MapClaims so we always get jwt.MapClaims type
	token, err := jwt.ParseWithClaims(tokenStr, jwt.MapClaims{}, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fiber.ErrUnauthorized
		}
		return JwtSecret, nil
	})

	if err != nil {
		fmt.Print(err)
		return nil, c.Status(401).JSON(fiber.Map{"error": "invalid token"})
	}

	if !token.Valid {
		return nil, c.Status(401).JSON(fiber.Map{"error": "invalid token"})
	}

	if claims, ok := token.Claims.(jwt.MapClaims); ok {
		return claims, nil
	}

	return nil, c.Status(401).JSON(fiber.Map{"error": "invalid token claims"})
}

// hasSuperuserRole проверяет поле role в claims и сравнивает со строкой "superuser".
func hasSuperuserRole(claims jwt.MapClaims) bool {
	fmt.Println("hassuper")

	fmt.Println(claims["role"])
	if claims == nil {
		return false
	}

	if r, ok := claims["role"].(string); ok {
		return r == "superuser"
	}
	// иногда роль может быть массивом
	if arr, ok := claims["role"].([]interface{}); ok && len(arr) > 0 {
		if s, ok := arr[0].(string); ok {
			return s == "superuser"
		}
	}
	return false
}
