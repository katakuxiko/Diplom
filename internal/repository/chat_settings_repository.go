package repository

import (
	"context"
	"database/sql"

	"github.com/katakuxiko/Diplom/internal/models"
)

type ChatSettingsRepository struct {
	DB *sql.DB // Подключение к базе данных
}

// CreateChatSettings создает новую запись настроек чата
func (r *ChatSettingsRepository) CreateChatSettings(ctx context.Context, settings *models.ChatSettings) error {
	// Пример запроса, адаптируйте под вашу схему
	query := `INSERT INTO chat_settings (language, max_messages, is_active) VALUES ($1, $2, $3) RETURNING id`
	return r.DB.QueryRowContext(ctx, query, settings.Language, settings.MaxMessages, settings.IsActive).Scan(&settings.ID)
}

// GetChatSettingsByID возвращает настройки чата по ID
func (r *ChatSettingsRepository) GetChatSettingsByID(ctx context.Context, id int) (*models.ChatSettings, error) {
	settings := &models.ChatSettings{}
	query := `SELECT id, language, max_messages, is_active FROM chat_settings WHERE id = $1`
	row := r.DB.QueryRowContext(ctx, query, id)
	if err := row.Scan(&settings.ID, &settings.Language, &settings.MaxMessages, &settings.IsActive); err != nil {
		return nil, err
	}
	return settings, nil
}

// UpdateChatSettings обновляет существующие настройки чата
func (r *ChatSettingsRepository) UpdateChatSettings(ctx context.Context, settings *models.ChatSettings) error {
	query := `UPDATE chat_settings SET language = $1, max_messages = $2, is_active = $3 WHERE id = $4`
	_, err := r.DB.ExecContext(ctx, query, settings.Language, settings.MaxMessages, settings.IsActive, settings.ID)
	return err
}

// DeleteChatSettings удаляет настройки чата по ID
func (r *ChatSettingsRepository) DeleteChatSettings(ctx context.Context, id int) error {
	query := `DELETE FROM chat_settings WHERE id = $1`
	_, err := r.DB.ExecContext(ctx, query, id)
	return err
}

// ListChatSettings возвращает все настройки чата
func (r *ChatSettingsRepository) ListChatSettings(ctx context.Context) ([]*models.ChatSettings, error) {
	query := `SELECT id, language, max_messages, is_active FROM chat_settings`
	rows, err := r.DB.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var settingsList []*models.ChatSettings
	for rows.Next() {
		settings := &models.ChatSettings{}
		if err := rows.Scan(&settings.ID, &settings.Language, &settings.MaxMessages, &settings.IsActive); err != nil {
			return nil, err
		}
		settingsList = append(settingsList, settings)
	}
	return settingsList, nil
}
