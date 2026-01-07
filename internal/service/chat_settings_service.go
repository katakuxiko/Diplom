package service

import (
	"context"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
)

type ChatSettingsService struct {
	Repo *repository.ChatSettingsRepository // Репозиторий настроек чата
}

func NewChatSettingsService(repo *repository.ChatSettingsRepository) *ChatSettingsService {
	return &ChatSettingsService{Repo: repo}
}

// CreateOrUpdate создает или обновляет настройки чата
func (s *ChatSettingsService) CreateOrUpdate(ctx context.Context, settings *models.ChatSetting) error {
	// Валидация или дополнительная логика перед сохранением
	return s.Repo.CreateOrUpdateChatSettings(ctx, settings)
}

// GetByID возвращает настройки чата по ID
func (s *ChatSettingsService) GetByID(ctx context.Context, id uuid.UUID) (*models.ChatSetting, error) {
	return s.Repo.GetChatSettingsByID(ctx, id)
}

// GetByChatID возвращает настройки по ChatID
func (s *ChatSettingsService) GetByChatID(ctx context.Context, chatID uuid.UUID) (*models.ChatSetting, error) {
	return s.Repo.GetChatSettingsByChatID(ctx, chatID)
}

// Update обновляет настройки чата
func (s *ChatSettingsService) Update(ctx context.Context, settings *models.ChatSetting) error {
	return s.Repo.UpdateChatSettings(ctx, settings)
}

// Delete удаляет настройки чата по ID
func (s *ChatSettingsService) Delete(ctx context.Context, id uuid.UUID) error {
	return s.Repo.DeleteChatSettings(ctx, id)
}

// List возвращает все настройки чата
func (s *ChatSettingsService) List(ctx context.Context) ([]*models.ChatSetting, error) {
	return s.Repo.ListChatSettings(ctx)
}
