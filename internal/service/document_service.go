package service

import (
	"fmt"
	"mime/multipart"
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/config"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/storage"
)

type DocumentService struct {
	repo    *repository.DocumentRepository
	storage *storage.MinioStorage
}

func NewDocumentService(repo *repository.DocumentRepository, storage *storage.MinioStorage) *DocumentService {
	return &DocumentService{repo: repo, storage: storage}
}

func (s *DocumentService) CreateDocument(chatID uuid.UUID, file multipart.File, fileHeader *multipart.FileHeader, cfg *config.Config) (*models.Document, error) {
	docID := uuid.New()
	objectName := fmt.Sprintf("%s/%s", chatID.String(), fileHeader.Filename)
	path := fmt.Sprintf("%s/%s/%s", cfg.MinioEndpoint, cfg.MinioBucket, objectName)
	print(objectName)
	// загрузка в MinIO через storage
	if _, err := s.storage.UploadFile(objectName, file, fileHeader); err != nil {
		return nil, err
	}

	doc := &models.Document{
		ID:          docID,
		ChatID:      chatID,
		Name:        fileHeader.Filename,
		Path:        path,
		CreatedDate: time.Now(),
	}
	if err := s.repo.Create(doc); err != nil {
		return nil, err
	}
	return doc, nil
}

func (s *DocumentService) GetDownloadLink(id uuid.UUID) (string, error) {
	doc, err := s.repo.GetByID(id)
	if err != nil {
		return "", err
	}
	return s.storage.DownloadFile(doc.Path)
}

// GetAllDocuments — список документов
func (s *DocumentService) GetAllDocuments() ([]models.Document, error) {
	return s.repo.GetAll()
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
		_ = s.storage.DeleteFile(doc.Name)
	}

	return s.repo.Delete(id)
}
