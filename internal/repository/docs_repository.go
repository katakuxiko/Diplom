package repository

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type DocumentRepository struct {
	db *gorm.DB
}

func NewDocumentRepository(db *gorm.DB) *DocumentRepository {
	return &DocumentRepository{db: db}
}

func (r *DocumentRepository) Create(doc *models.Document) error {
	return r.db.Create(doc).Error
}

func (r *DocumentRepository) GetByID(id uuid.UUID) (*models.Document, error) {
	var doc models.Document
	err := r.db.Preload("Chunks").First(&doc, "id = ?", id).Error
	return &doc, err
}

func (r *DocumentRepository) GetAll() ([]models.Document, error) {
	var docs []models.Document
	err := r.db.Find(&docs).Error
	return docs, err
}

func (r *DocumentRepository) Update(doc *models.Document) error {
	return r.db.Save(doc).Error
}

func (r *DocumentRepository) Delete(id uuid.UUID) error {
	return r.db.Delete(&models.Document{}, "id = ?", id).Error
}
