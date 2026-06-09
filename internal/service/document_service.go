package service

import (
	"fmt"
	"mime/multipart"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/storage"
	"github.com/lib/pq"
)

type DocumentService struct {
	repo    *repository.DocumentRepository
	storage *storage.MinioStorage
}

func NewDocumentService(repo *repository.DocumentRepository, storage *storage.MinioStorage) *DocumentService {
	return &DocumentService{repo: repo, storage: storage}
}

func (s *DocumentService) CreateDocument(chatID uuid.UUID, file multipart.File, fileHeader *multipart.FileHeader, cfg *config.Config, tags []string) (*models.Document, error) {
	docID := uuid.New()
	objectName := fmt.Sprintf("%s/%s", chatID.String(), fileHeader.Filename)
	fullPath := fmt.Sprintf("%s/%s/%s", cfg.MinioEndpoint, cfg.MinioBucket, objectName)
	print(objectName)
	normalizedTags := normalizeDocumentTags(tags)
	// загрузка в MinIO через storage
	if _, err := s.storage.UploadFile(objectName, file, fileHeader); err != nil {
		return nil, err
	}

	doc := &models.Document{
		ID:          docID,
		ChatID:      chatID,
		Name:        fileHeader.Filename,
		Tags:        pq.StringArray(normalizedTags),
		Path:        objectName,
		FullPath:    fullPath,
		CreatedDate: time.Now(),
	}
	if err := s.repo.Create(doc); err != nil {
		return nil, err
	}
	return doc, nil
}

func (s *DocumentService) GetFile(id uuid.UUID) (*models.Document, error) {
	return s.repo.GetByID(id)
}

// GetAllDocuments — список документов
func (s *DocumentService) GetAllDocumentsPaginated(limit, page int, chatID uuid.UUID, maxAccessLevel int, tags []string) (*dto.PaginatedDocuments, error) {
	if page < 1 {
		page = 1
	}
	offset := (page - 1) * limit
	normalizedTags := normalizeDocumentTags(tags)

	docs, total, err := s.repo.GetAllPaginated(limit, offset, chatID, maxAccessLevel, normalizedTags)
	if err != nil {
		return nil, err
	}

	totalPages := int((total + int64(limit) - 1) / int64(limit)) // округляем вверх

	return &dto.PaginatedDocuments{
		Documents:   docs,
		Total:       total,
		TotalPages:  totalPages,
		CurrentPage: page,
	}, nil
}

// GetDocument — один документ по ID
func (s *DocumentService) GetDocument(id uuid.UUID) (*models.Document, error) {
	return s.repo.GetByID(id)
}

// DeleteDocument — удаление документа из БД и MinIO
func (s *DocumentService) DeleteDocument(id uuid.UUID) error {
	doc, err := s.repo.GetByID(id)
	if err != nil {
		return err
	}

	// Удаляем файл из MinIO
	if doc.Path != "" {
		err = s.storage.DeleteFile(doc.Path)
		if err != nil {
			fmt.Println(err)
			return err
		}
	}

	return s.repo.Delete(id)
}

func (s *DocumentService) UpdateAccessLevel(id uuid.UUID, level int) (*models.Document, error) {
	doc, err := s.repo.GetByID(id)
	if err != nil {
		return nil, err
	}
	doc.AccessLevel = level
	if err := s.repo.Update(doc); err != nil {
		return nil, err
	}
	return doc, nil
}

func (s *DocumentService) UpdateTags(id uuid.UUID, tags []string) (*models.Document, error) {
	doc, err := s.repo.GetByID(id)
	if err != nil {
		return nil, err
	}

	doc.Tags = pq.StringArray(normalizeDocumentTags(tags))
	if err := s.repo.Update(doc); err != nil {
		return nil, err
	}

	return doc, nil
}

func (s *DocumentService) GetDocumentTags(chatID uuid.UUID, maxAccessLevel int) ([]string, error) {
	tags, err := s.repo.GetDistinctTags(chatID, maxAccessLevel)
	if err != nil {
		return nil, err
	}

	return normalizeDocumentTags(tags), nil
}

func normalizeDocumentTags(tags []string) []string {
	if len(tags) == 0 {
		return []string{}
	}

	seen := make(map[string]struct{}, len(tags))
	normalized := make([]string, 0, len(tags))
	for _, raw := range tags {
		tag := strings.ToLower(strings.TrimSpace(raw))
		if tag == "" {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		normalized = append(normalized, tag)
	}

	sort.Strings(normalized)
	return normalized
}
