package repository

import (
	"context"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type ChatSettingsRepository struct {
	DB *gorm.DB // Подключение к базе данных
}

func NewChatSettingsRepository(db *gorm.DB) *ChatSettingsRepository {
	return &ChatSettingsRepository{DB: db}
}

// CreateOrUpdateChatSettings создает или обновляет настройки чата по chat_id
func (r *ChatSettingsRepository) CreateOrUpdateChatSettings(ctx context.Context, settings *models.ChatSetting) error {
	// Ищем существующие настройки по chat_id
	existing := &models.ChatSetting{}
	err := r.DB.WithContext(ctx).Where("chat_id = ?", settings.ChatID).First(existing).Error

	if err == gorm.ErrRecordNotFound {
		// Создаем новую запись
		return r.DB.WithContext(ctx).Create(settings).Error
	} else if err != nil {
		return err
	}

	// Обновляем существующую запись
	settings.ID = existing.ID
	settings.CreatedDate = existing.CreatedDate
	return r.DB.WithContext(ctx).Save(settings).Error
}

// GetChatSettingsByID возвращает настройки чата по ID
func (r *ChatSettingsRepository) GetChatSettingsByID(ctx context.Context, id uuid.UUID) (*models.ChatSetting, error) {
	settings := &models.ChatSetting{}
	err := r.DB.WithContext(ctx).First(settings, "id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return settings, nil
}

// GetChatSettingsByChatID возвращает настройки чата по ChatID
func (r *ChatSettingsRepository) GetChatSettingsByChatID(ctx context.Context, chatID uuid.UUID) (*models.ChatSetting, error) {
	settings := &models.ChatSetting{}
	err := r.DB.WithContext(ctx).Where("chat_id = ?", chatID).First(settings).Error
	if err != nil {
		return nil, err
	}
	return settings, nil
}

// UpdateChatSettings обновляет существующие настройки чата
func (r *ChatSettingsRepository) UpdateChatSettings(ctx context.Context, settings *models.ChatSetting) error {
	return r.DB.WithContext(ctx).Save(settings).Error
}

// DeleteChatSettings удаляет настройки чата по ID
func (r *ChatSettingsRepository) DeleteChatSettings(ctx context.Context, id uuid.UUID) error {
	return r.DB.WithContext(ctx).Delete(&models.ChatSetting{}, "id = ?", id).Error
}

// ListChatSettings возвращает все настройки чата
func (r *ChatSettingsRepository) ListChatSettings(ctx context.Context) ([]*models.ChatSetting, error) {
	var settingsList []*models.ChatSetting
	err := r.DB.WithContext(ctx).Find(&settingsList).Error
	if err != nil {
		return nil, err
	}
	return settingsList, nil
}
