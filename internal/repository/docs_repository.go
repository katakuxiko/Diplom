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

func (r *DocumentRepository) GetAllPaginated(limit, offset int, chatID uuid.UUID, maxAccessLevel int) ([]models.Document, int64, error) {
	var docs []models.Document
	var total int64

	// Считаем общее количество документов по чату
	query := r.db.Model(&models.Document{}).Where("chat_id = ?", chatID)
	if maxAccessLevel >= 0 {
		query = query.Where("access_level <= ?", maxAccessLevel)
	}
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	// Получаем страницу документов по чату
	query = r.db.Where("chat_id = ?", chatID)
	if maxAccessLevel >= 0 {
		query = query.Where("access_level <= ?", maxAccessLevel)
	}
	err := query.Limit(limit).Offset(offset).Find(&docs).Error
	return docs, total, err
}

func (r *DocumentRepository) Update(doc *models.Document) error {
	return r.db.Save(doc).Error
}

func (r *DocumentRepository) Delete(id uuid.UUID) error {
	return r.db.Delete(&models.Document{}, "id = ?", id).Error
}
