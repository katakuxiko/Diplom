package repository

import (
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type ChatRepository struct {
	db *gorm.DB
}

func NewChatRepository(db *gorm.DB) *ChatRepository {
	return &ChatRepository{db: db}
}

func (r *ChatRepository) Create(chat *models.Chat) error {
	return r.db.Create(chat).Error
}

func (r *ChatRepository) GetByID(id string) (*models.Chat, error) {
	var chat models.Chat
	if err := r.db.First(&chat, "id = ?", id).Error; err != nil {
		return nil, err
	}
	return &chat, nil
}

func (r *ChatRepository) List() ([]models.Chat, error) {
	var chats []models.Chat
	if err := r.db.Find(&chats).Error; err != nil {
		return nil, err
	}
	return chats, nil
}

func (r *ChatRepository) Delete(id string) error {
	return r.db.Delete(&models.Chat{}, "id = ?", id).Error
}
