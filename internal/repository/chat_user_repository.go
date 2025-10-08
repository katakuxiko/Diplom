package repository

import (
	"errors"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type ChatUserRepository struct {
	db *gorm.DB
}

func NewChatUserRepository(db *gorm.DB) *ChatUserRepository {
	return &ChatUserRepository{db: db}
}

func (r *ChatUserRepository) GetAll() ([]models.ChatUser, error) {
	var chatusers []models.ChatUser
	err := r.db.Find(&chatusers).Error
	return chatusers, err
}

func (r *ChatUserRepository) GetByID(id uuid.UUID) (*models.ChatUser, error) {
	var chatuser models.ChatUser
	err := r.db.First(&chatuser, "id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return &chatuser, nil
}

func (r *ChatUserRepository) Create(chatuser *models.ChatUser) error {
	return r.db.Create(chatuser).Error
}

func (r *ChatUserRepository) Update(chatuser *models.ChatUser) error {
	return r.db.Save(chatuser).Error
}

func (r *ChatUserRepository) Delete(id uuid.UUID) error {
	return r.db.Delete(&models.ChatUser{}, "id = ?", id).Error
}

func (r *ChatUserRepository) GetByUsername(username string) (*models.ChatUser, error) {
	var chatuser models.ChatUser
	if err := r.db.Where("username = ?", username).First(&chatuser).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &chatuser, nil
}
 
