package handlers

import (
	"io"
	"log"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/middleware"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
)

var documentService *service.DocumentService
var cfg *config.Config

// RegisterDocumentRoutes регистрирует CRUD эндпоинты для документов
func RegisterDocumentRoutes(app *fiber.App, svc *service.DocumentService, cfgo *config.Config) {
	documentService = svc
	cfg = cfgo
	r := app.Group("/documents", middleware.JWTProtected())

	r.Post("/", CreateDocument)
	r.Get("/", GetDocuments)
	r.Get("/:id", GetDocumentByID)
	r.Delete("/:id", DeleteDocument)
	r.Put("/:id/access", UpdateDocumentAccess)

	// Публичный эндпоинт для скачивания (без JWT)
	app.Get("/documents/:id/download", DownloadDocument)
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
// @Security     BearerAuth
func CreateDocument(c *fiber.Ctx) error {
	fileHeader, err := c.FormFile("file")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "file is required"})
	}
	if err := utils.ValidatePDFUpload(fileHeader); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
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
// @Summary      Получить документы чата с пагинацией
// @Description  Возвращает список документов для конкретного чата с пагинацией
// @Tags         documents
// @Produce      json
// @Param        chat_id query     string  true   "Chat ID (UUID)"
// @Param        page    query     int     false  "Номер страницы"  default(1)
// @Param        limit   query     int     false  "Количество документов на странице"  default(10)
// @Success      200 {object} dto.PaginatedDocuments
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /documents [get]
// @Security     BearerAuth
func GetDocuments(c *fiber.Ctx) error {
	page := c.QueryInt("page", 1)
	limit := c.QueryInt("limit", 10)
	chatIDStr := c.Query("chat_id")
	chatID, err := uuid.Parse(chatIDStr)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid chat_id"})
	}
	maxAccess := -1
	if claims, ok := c.Locals("user").(jwt.MapClaims); ok {
		if role, rok := claims["role"].(string); rok && role == "chat_user" {
			if al, ok := claims["access_level"].(float64); ok {
				maxAccess = int(al)
			}
		}
	}

	paginatedDocs, err := documentService.GetAllDocumentsPaginated(limit, page, chatID, maxAccess)
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
// @Security     BearerAuth
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

// DownloadDocument godoc
// @Summary      Скачать файл документа
// @Description  Возвращает файл документа для скачивания
// @Tags         documents
// @Param        id path string true "Document ID"
// @Produce      application/pdf
// @Success      200 {file} binary
// @Failure      400 {object} map[string]string
// @Failure      404 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /documents/{id}/download [get]
// @Security     BearerAuth
func DownloadDocument(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		log.Printf("Invalid document ID: %v", err)
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	doc, err := documentService.GetFile(id)
	if err != nil {
		log.Printf("Document not found: %v", err)
		return c.Status(404).JSON(fiber.Map{"error": "document not found"})
	}

	// Получаем файл из MinIO через storage
	file, _, contentType, err := cfg.MinioStorage.GetFile(doc.Path)
	if err != nil {
		log.Printf("Failed to get file from storage: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to get file from storage"})
	}
	defer file.Close()

	// Читаем весь файл в память
	data, err := io.ReadAll(file)
	if err != nil {
		log.Printf("Failed to read file: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to read file"})
	}

	// Устанавливаем заголовки
	c.Set("Content-Type", contentType)
	c.Set("Content-Disposition", "attachment; filename=\""+doc.Name+"\"")
	c.Set("Access-Control-Expose-Headers", "Content-Disposition")

	log.Printf("Sending file: %s, size: %d bytes", doc.Name, len(data))

	// Отдаем файл
	return c.Send(data)
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
// @Security     BearerAuth
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

// UpdateDocumentAccess godoc
// @Summary      Обновить уровень доступа документа
// @Description  Меняет поле access_level для документа
// @Tags         documents
// @Param        id    path   string true "Document ID"
// @Param        body  body   map[string]int true "{\"access_level\":1}"
// @Success      200   {object} dto.DocumentResponseDTO
// @Failure      400   {object} map[string]string
// @Failure      500   {object} map[string]string
// @Router       /documents/{id}/access [put]
// @Security     BearerAuth
func UpdateDocumentAccess(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	var payload struct {
		AccessLevel int `json:"access_level"`
	}
	if err := c.BodyParser(&payload); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}

	doc, err := documentService.UpdateAccessLevel(id, payload.AccessLevel)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	return c.JSON(doc)
}
