package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/service"
)

var documentService *service.DocumentService
var cfg *config.Config

// RegisterDocumentRoutes регистрирует CRUD эндпоинты для документов
func RegisterDocumentRoutes(app *fiber.App, svc *service.DocumentService, cfgo *config.Config) {
	documentService = svc
	cfg = cfgo
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

	doc, err := documentService.CreateDocument(chatID, file, fileHeader, cfg)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(201).JSON(doc)
}

// GetDocuments godoc
// @Summary      Получить документы с пагинацией
// @Description  Возвращает список документов с пагинацией
// @Tags         documents
// @Produce      json
// @Param        page  query     int  false  "Номер страницы"  default(1)
// @Param        limit query     int  false  "Количество документов на странице"  default(10)
// @Success      200 {object} dto.PaginatedDocuments
// @Failure      500 {object} map[string]string
// @Router       /documents [get]
func GetDocuments(c *fiber.Ctx) error {
	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 10)

	paginatedDocs, err := documentService.GetAllDocumentsPaginated(limit, page)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.JSON(paginatedDocs)
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
