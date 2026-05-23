package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/pdf"
	"github.com/katakuxiko/Diplom/internal/service"
	"github.com/katakuxiko/Diplom/internal/utils"
)

type DocumentHandler struct {
	documentService *service.DocumentService
	chunkService    *service.ChunkService
	llm             *service.LLMClient
	cfg             *config.Config
	chatSettings    *service.ChatSettingsService
}

// NewDocumentHandler конструктор с DI
func NewDocumentHandler(
	documentService *service.DocumentService,
	chunkService *service.ChunkService,
	llm *service.LLMClient,
	cfg *config.Config,
	chatSettings *service.ChatSettingsService,
) *DocumentHandler {
	return &DocumentHandler{
		documentService: documentService,
		chunkService:    chunkService,
		llm:             llm,
		cfg:             cfg,
		chatSettings:    chatSettings,
	}
}

// UploadAndIngestPDF загружает PDF, сохраняет в MinIO, извлекает текст,
// дробит на chunks и сохраняет в БД вместе с embeddings.
//
// @Summary      Upload and ingest documents
// @Description  Загружает документ, сохраняет его в MinIO, извлекает текст,
// @Description  дробит на части и сохраняет embeddings в базе данных.
// @Tags         documents
// @Accept       multipart/form-data
// @Produce      json
// @Param        chat_id formData string true "Chat ID (uuid)"
// @Param        file formData file true "document file"
// @Success      200 {object} dto.DocumentIngestResponse
// @Failure      400 {object} map[string]string
// @Failure      500 {object} map[string]string
// @Router       /documents/upload [post]
// @Security     BearerAuth
func (h *DocumentHandler) UploadAndIngestPDF(c *fiber.Ctx) error {
	// --- 1. Получаем файл
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

	// --- 2. Сохраняем документ через сервис
	doc, err := h.documentService.CreateDocument(chatID, file, fileHeader, h.cfg)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	// --- 3. Сохраняем временно для обработки PDF
	tmpFile := filepath.Join(os.TempDir(), fileHeader.Filename)
	if err := c.SaveFile(fileHeader, tmpFile); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed to save temp file"})
	}
	defer os.Remove(tmpFile)

	// --- 4. Извлекаем текст из PDF
	txt, err := pdf.ExtractText(tmpFile)
	if err != nil {
		log.Printf("extract error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to extract text from pdf"})
	}
	txt = pdf.Sanitize(txt)
	if len(txt) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "no text extracted from PDF"})
	}

	// --- 5. Дробим на chunks
	const chunkSize = 220
	const chunkOverlap = 40
	parts := pdf.ChunkBySentences(txt, chunkSize, chunkOverlap)
	if len(parts) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "no chunks created"})
	}

	// --- 6. Сохраняем chunks
	saved := 0
	for i, p := range parts {
		chunkName := fmt.Sprintf("%s_chunk_%d", doc.Name, i)
		ch := models.Chunk{
			Text:      p,
			Filepath:  doc.Path, // ссылка на MinIO
			DocName:   doc.Name,
			ChunkName: chunkName,
			DocID:     doc.ID,
			ChatID:    chatID,
		}

		// Попробуем получить настройки чата и использовать их для эмбеддингов
		var emb []float32
		if h.chatSettings != nil {
			if cs, err := h.chatSettings.GetByChatID(context.Background(), chatID); err == nil && cs != nil && cs.Settings != nil {
				// marshal JSONB -> bytes -> AskSettings
				raw, _ := json.Marshal(cs.Settings)
				var dbSettings models.AskSettings
				if err := json.Unmarshal(raw, &dbSettings); err == nil {
					// Попробуем дешифровать ключи, если они были сохранены зашифрованными
					if dbSettings.ExternalAPIKey != "" {
						if dec, derr := utils.DecryptString(dbSettings.ExternalAPIKey); derr == nil {
							dbSettings.ExternalAPIKey = dec
						}
					}
					if dbSettings.EmbedExternalAPIKey != "" {
						if dec2, derr2 := utils.DecryptString(dbSettings.EmbedExternalAPIKey); derr2 == nil {
							dbSettings.EmbedExternalAPIKey = dec2
						}
					}
					emb, err = h.llm.EmbeddingWithSettings(p, &dbSettings)
				} else {
					emb, err = h.llm.Embedding(p)
				}
			} else {
				emb, err = h.llm.Embedding(p)
			}
		} else {
			emb, err = h.llm.Embedding(p)
		}
		if err != nil {
			log.Printf("embedding error (%s): %v", chunkName, err)
			continue
		}

		if err := h.chunkService.SaveChunk(ch, emb); err != nil {
			log.Printf("db insert error (%s): %v", chunkName, err)
			continue
		}
		saved++
	}

	return c.JSON(dto.DocumentIngestResponse{
		Status:      "ok",
		Document:    doc,
		ChunksTotal: len(parts),
		ChunksSaved: saved,
	})
}
