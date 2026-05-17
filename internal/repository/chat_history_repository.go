package repository

import (
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type ChatHistoryRepository struct {
	db *gorm.DB
}

func NewChatHistoryRepository(db *gorm.DB) *ChatHistoryRepository {
	return &ChatHistoryRepository{db: db}
}

// Создать историю чата
func (r *ChatHistoryRepository) Create(history *models.ChatHistory) error {
	return r.db.Create(history).Error
}

// Получить всю историю сообщений по чату
func (r *ChatHistoryRepository) GetMessagesByChatID(chatID string) ([]models.Message, error) {
	var messages []models.Message
	err := r.db.Joins("JOIN chat_histories ON chat_histories.id = messages.chat_history_id").
		Where("chat_histories.chat_id = ?", chatID).
		Order("messages.created_date asc").
		Find(&messages).Error
	return messages, err
}

// Получить истории чата с пользователем и сообщениями
func (r *ChatHistoryRepository) GetHistoriesWithMessages(chatID string) ([]models.ChatHistory, error) {
	var histories []models.ChatHistory
	err := r.db.Preload("User").Preload("Messages", func(db *gorm.DB) *gorm.DB {
		return db.Order("created_date asc")
	}).Where("chat_id = ?", chatID).Find(&histories).Error
	return histories, err
}
