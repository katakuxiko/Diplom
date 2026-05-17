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

// DeleteCascade removes user along with their chat histories and messages to avoid FK violations.
func (r *ChatUserRepository) DeleteCascade(id uuid.UUID) error {
	return r.db.Transaction(func(tx *gorm.DB) error {
		var historyIDs []uuid.UUID
		if err := tx.Model(&models.ChatHistory{}).
			Where("user_id = ?", id).
			Pluck("id", &historyIDs).Error; err != nil {
			return err
		}

		if len(historyIDs) > 0 {
			if err := tx.Where("chat_history_id IN ?", historyIDs).
				Delete(&models.Message{}).Error; err != nil {
				return err
			}
			if err := tx.Where("id IN ?", historyIDs).
				Delete(&models.ChatHistory{}).Error; err != nil {
				return err
			}
		}

		if err := tx.Delete(&models.ChatUser{}, "id = ?", id).Error; err != nil {
			return err
		}
		return nil
	})
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

func (r *ChatUserRepository) GetByUsernameAndChat(chatID uuid.UUID, username string) (*models.ChatUser, error) {
	var chatuser models.ChatUser
	err := r.db.Where("chat_id = ? AND username = ?", chatID, username).First(&chatuser).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &chatuser, nil
}
