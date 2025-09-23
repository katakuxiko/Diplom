package service

import (
	"context"

	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
)

type ChatSettingsService struct {
	Repo *repository.ChatSettingsRepository // Репозиторий настроек чата
}

// Create создает новые настройки чата
func (s *ChatSettingsService) Create(ctx context.Context, settings *models.ChatSettings) error {
	// Валидация или дополнительная логика перед сохранением
	return s.Repo.CreateChatSettings(ctx, settings)
}

// GetByID возвращает настройки чата по ID
func (s *ChatSettingsService) GetByID(ctx context.Context, id int) (*models.ChatSettings, error) {
	return s.Repo.GetChatSettingsByID(ctx, id)
}

// Update обновляет настройки чата
func (s *ChatSettingsService) Update(ctx context.Context, settings *models.ChatSettings) error {
	return s.Repo.UpdateChatSettings(ctx, settings)
}

// Delete удаляет настройки чата по ID
func (s *ChatSettingsService) Delete(ctx context.Context, id int) error {
	return s.Repo.DeleteChatSettings(ctx, id)
}

// List возвращает все настройки чата
func (s *ChatSettingsService) List(ctx context.Context) ([]*models.ChatSettings, error) {
	return s.Repo.ListChatSettings(ctx)
}
