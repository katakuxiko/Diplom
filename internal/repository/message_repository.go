package repository

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type MessageRepository struct {
	db *gorm.DB
}

func NewMessageRepository(db *gorm.DB) *MessageRepository {
	return &MessageRepository{db: db}
}

func (r *MessageRepository) Create(msg *models.Message) error {
	return r.db.Create(msg).Error
}

func (r *MessageRepository) GetRecentByChatHistoryID(chatHistoryID uuid.UUID, limit int) ([]models.Message, error) {
	if limit <= 0 {
		limit = 20
	}

	var messages []models.Message
	err := r.db.Where("chat_history_id = ?", chatHistoryID).
		Order("created_date desc").
		Limit(limit).
		Find(&messages).Error
	return messages, err
}
