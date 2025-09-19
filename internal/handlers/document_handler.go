package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/service"
)

var documentService *service.DocumentService

// RegisterDocumentRoutes регистрирует CRUD эндпоинты для документов
func RegisterDocumentRoutes(app *fiber.App, svc *service.DocumentService) {
	documentService = svc
	r := app.Group("/documents")

	r.Post("/", CreateDocument)
	r.Get("/", GetDocuments)
	r.Get("/:id", GetDocumentByID)
	r.Delete("/:id", DeleteDocument)
}

// CreateDocument godoc
// @Summary      Загрузить документ
// @Description  Загружает файл в MinIO и сохраняет метаданные в БД
// @Tags         documents
// @Accept       multipart/form-data
// @Produce      json
// @Param        chat_id formData string true "Chat ID (UUID)"
// @Param        name    formData string true "Document name"
// @Param        file    formData file   true "Document file"
// @Success      201 {object} dto.DocumentResponseDTO
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /documents [post]
func CreateDocument(c *fiber.Ctx) error {
	fileHeader, err := c.FormFile("file")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "file is required"})
	}

	name := c.FormValue("name")
	if name == "" {
		return c.Status(400).JSON(fiber.Map{"error": "name is required"})
	}

	chatIDStr := c.FormValue("chat_id")
	chatID, err := uuid.Parse(chatIDStr)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}

	file, err := fileHeader.Open()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed to open file"})
	}
	defer file.Close()

	doc, err := documentService.CreateDocument(chatID, name, file, fileHeader)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(201).JSON(doc)
}

// GetDocuments godoc
// @Summary      Получить все документы
// @Description  Возвращает список документов
// @Tags         documents
// @Produce      json
// @Success      200 {array} dto.DocumentResponseDTO
// @Failure      500 {object} map[string]string
// @Router       /documents [get]
func GetDocuments(c *fiber.Ctx) error {
	docs, err := documentService.GetAllDocuments()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(docs)
}

// GetDocumentByID godoc
// @Summary      Получить документ по ID
// @Description  Возвращает один документ по UUID
// @Tags         documents
// @Param        id path string true "Document ID"
// @Produce      json
// @Success      200 {object} dto.DocumentResponseDTO
// @Failure      400 {object} map[string]string
// @Failure      404 {object} map[string]string
// @Router       /documents/{id} [get]
func GetDocumentByID(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	doc, err := documentService.GetDocument(id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "document not found"})
	}

	return c.JSON(doc)
}

// DeleteDocument godoc
// @Summary      Удалить документ
// @Description  Удаляет документ по UUID
// @Tags         documents
// @Param        id path string true "Document ID"
// @Success      204 "No Content"
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /documents/{id} [delete]
func DeleteDocument(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	if err := documentService.DeleteDocument(id); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.SendStatus(204)
}
