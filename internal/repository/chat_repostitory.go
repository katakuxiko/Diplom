package repository

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type ChatRepository struct {
	db *gorm.DB
}

func NewChatRepository(db *gorm.DB) *ChatRepository {
	return &ChatRepository{db: db}
}

func (r *ChatRepository) Create(chat *models.Chat) error {
	return r.db.Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(chat).Error; err != nil {
			return err
		}

		if chat.AdminID != nil {
			membership := &models.ChatAdmin{
				ChatID:  chat.ID,
				AdminID: *chat.AdminID,
			}
			if err := tx.Create(membership).Error; err != nil {
				return err
			}
		}

		return nil
	})
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
	return r.db.Transaction(func(tx *gorm.DB) error {
		// Сначала удаляем связи админов с чатом, иначе FK может блокировать удаление чата.
		if err := tx.Where("chat_id = ?", id).Delete(&models.ChatAdmin{}).Error; err != nil {
			return err
		}

		// Затем удаляем связанные настройки чата.
		if err := tx.Where("chat_id = ?", id).Delete(&models.ChatSetting{}).Error; err != nil {
			return err
		}

		// После очистки зависимостей удаляем сам чат.
		return tx.Delete(&models.Chat{}, "id = ?", id).Error
	})
}

func (r *ChatRepository) ListByAdmin(adminID string) ([]models.Chat, error) {
	adminUUID, err := uuid.Parse(adminID)
	if err != nil {
		return nil, err
	}

	var chats []models.Chat
	if err := r.db.
		Model(&models.Chat{}).
		Distinct("chats.*").
		Joins("JOIN chat_admins ON chat_admins.chat_id = chats.id").
		Where("chat_admins.admin_id = ?", adminUUID).
		Order("chats.created_date DESC").
		Find(&chats).Error; err != nil {
		return nil, err
	}
	return chats, nil
}

func (r *ChatRepository) Update(chat *models.Chat) error {
	return r.db.Save(chat).Error
}

func (r *ChatRepository) IsCreator(chatID string, adminID string) (bool, error) {
	chatUUID, err := uuid.Parse(chatID)
	if err != nil {
		return false, err
	}
	adminUUID, err := uuid.Parse(adminID)
	if err != nil {
		return false, err
	}

	var count int64
	err = r.db.Model(&models.Chat{}).
		Where("id = ? AND admin_id = ?", chatUUID, adminUUID).
		Count(&count).Error
	if err != nil {
		return false, err
	}

	return count > 0, nil
}

func (r *ChatRepository) IsAdminInChat(chatID string, adminID string) (bool, error) {
	chatUUID, err := uuid.Parse(chatID)
	if err != nil {
		return false, err
	}
	adminUUID, err := uuid.Parse(adminID)
	if err != nil {
		return false, err
	}

	var count int64
	err = r.db.Model(&models.ChatAdmin{}).
		Where("chat_id = ? AND admin_id = ?", chatUUID, adminUUID).
		Count(&count).Error
	if err != nil {
		return false, err
	}

	if count > 0 {
		return true, nil
	}

	// Для совместимости с legacy-данными до backfill.
	err = r.db.Model(&models.Chat{}).
		Where("id = ? AND admin_id = ?", chatUUID, adminUUID).
		Count(&count).Error
	if err != nil {
		return false, err
	}

	return count > 0, nil
}

func (r *ChatRepository) AdminExists(adminID string) (bool, error) {
	adminUUID, err := uuid.Parse(adminID)
	if err != nil {
		return false, err
	}

	var count int64
	err = r.db.Model(&models.Admin{}).
		Where("id = ?", adminUUID).
		Count(&count).Error
	if err != nil {
		return false, err
	}

	return count > 0, nil
}

func (r *ChatRepository) AddAdminToChat(chatID string, adminID string) (bool, error) {
	chatUUID, err := uuid.Parse(chatID)
	if err != nil {
		return false, err
	}
	adminUUID, err := uuid.Parse(adminID)
	if err != nil {
		return false, err
	}

	membership := &models.ChatAdmin{
		ChatID:  chatUUID,
		AdminID: adminUUID,
	}

	result := r.db.
		Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "chat_id"}, {Name: "admin_id"}},
			DoNothing: true,
		}).
		Create(membership)
	if result.Error != nil {
		return false, result.Error
	}

	return result.RowsAffected > 0, nil
}

func (r *ChatRepository) ListAdminsByChat(chatID string) ([]models.Admin, error) {
	chatUUID, err := uuid.Parse(chatID)
	if err != nil {
		return nil, err
	}

	var admins []models.Admin
	err = r.db.Model(&models.Admin{}).
		Distinct("admins.*").
		Joins("JOIN chat_admins ON chat_admins.admin_id = admins.id").
		Where("chat_admins.chat_id = ?", chatUUID).
		Order("admins.username ASC").
		Find(&admins).Error
	if err != nil {
		return nil, err
	}

	return admins, nil
}
