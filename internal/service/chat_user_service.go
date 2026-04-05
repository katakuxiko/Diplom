package service

import (
	"bytes"
	"encoding/csv"
	"fmt"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/utils"
	"github.com/xuri/excelize/v2"
)

type ChatUserService struct {
	repo *repository.ChatUserRepository
}

func NewChatUserService(repo *repository.ChatUserRepository) *ChatUserService {
	return &ChatUserService{repo: repo}
}

func (s *ChatUserService) GetAll() ([]models.ChatUser, error) {
	return s.repo.GetAll()
}

func (s *ChatUserService) GetByID(id uuid.UUID) (*models.ChatUser, error) {
	return s.repo.GetByID(id)
}

// service/chatuser_service.go
func (s *ChatUserService) Create(req *dto.ChatUserCreateRequest) (*models.ChatUser, error) {

	PasswordHash, err := utils.HashPassword(req.Password)
	if err != nil {
		return nil, err
	}

	chatID, err := uuid.Parse(req.ChatID)
	if err != nil {
		return nil, err
	}

	chatuser := &models.ChatUser{
		ChatID:       chatID,
		Username:     req.Username,
		PasswordHash: PasswordHash,
		UserInfo:     req.User_Info,
		AccessLevel:  req.AccessLevel,
		RoleName:     req.User_Role,
	}

	if err := s.repo.Create(chatuser); err != nil {
		return nil, err
	}
	return chatuser, nil
}

func (s *ChatUserService) Update(chatuser *models.ChatUser) error {
	return s.repo.Update(chatuser)
}

func (s *ChatUserService) Delete(id uuid.UUID) error {
	return s.repo.DeleteCascade(id)
}

func (s *ChatUserService) GetByUsername(username string) (*models.ChatUser, error) {
	return s.repo.GetByUsername(username)
}

func (s *ChatUserService) GetByUsernameAndChat(chatID uuid.UUID, username string) (*models.ChatUser, error) {
	return s.repo.GetByUsernameAndChat(chatID, username)
}

func (s *ChatUserService) CheckPassword(password, hash string) bool {
	return utils.CheckPassword(password, hash)
}

func (s *ChatUserService) ImportFromExcel(fileBytes []byte) error {
	f, err := excelize.OpenReader(bytes.NewReader(fileBytes))
	if err != nil {
		return fmt.Errorf("ошибка открытия файла Excel: %w", err)
	}

	sheet := f.GetSheetName(0)
	rows, err := f.GetRows(sheet)
	if err != nil {
		return fmt.Errorf("ошибка чтения листа Excel: %w", err)
	}

	// Проверим, что есть хотя бы заголовки
	if len(rows) < 2 {
		return fmt.Errorf("файл не содержит данных")
	}

	for i, row := range rows[1:] { // пропускаем первую строку (заголовки)
		if len(row) < 4 {
			return fmt.Errorf("строка %d: ожидается минимум 4 колонки (username, password, role, chat_id)", i+2)
		}

		username := row[0]
		password := row[1]
		role := row[2]
		chatID := row[3]

		if chatID == "" {
			return fmt.Errorf("строка %d: chat_id обязателен", i+2)
		}
		parsedChatID, err := uuid.Parse(chatID)
		if err != nil {
			return fmt.Errorf("строка %d: chat_id некорректен: %w", i+2, err)
		}

		// Хэшируем пароль
		hash, err := utils.HashPassword(password)
		if err != nil {
			return fmt.Errorf("ошибка хэширования пароля в строке %d: %w", i+2, err)
		}

		chatUser := &models.ChatUser{
			ChatID:       parsedChatID,
			Username:     username,
			PasswordHash: hash,
			UserInfo:     fmt.Sprintf("Импортированная роль: %s", role),
		}

		// Сохраняем пользователя
		if err := s.repo.Create(chatUser); err != nil {
			return fmt.Errorf("ошибка сохранения пользователя '%s': %w", username, err)
		}
	}

	return nil
}

// ImportFromCSV импортирует пользователей из CSV: username,password,role,chat_id (с заголовком)
func (s *ChatUserService) ImportFromCSV(fileBytes []byte) error {
	reader := csv.NewReader(bytes.NewReader(fileBytes))
	rows, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("ошибка чтения CSV: %w", err)
	}
	if len(rows) < 2 {
		return fmt.Errorf("файл не содержит данных")
	}
	for i, row := range rows[1:] {
		if len(row) < 4 {
			return fmt.Errorf("строка %d: ожидается минимум 4 колонки (username, password, role, chat_id)", i+2)
		}
		username := row[0]
		password := row[1]
		role := row[2]
		chatID := row[3]
		if chatID == "" {
			return fmt.Errorf("строка %d: chat_id обязателен", i+2)
		}
		parsedChatID, err := uuid.Parse(chatID)
		if err != nil {
			return fmt.Errorf("строка %d: chat_id некорректен: %w", i+2, err)
		}
		hash, err := utils.HashPassword(password)
		if err != nil {
			return fmt.Errorf("строка %d: ошибка хэширования пароля: %w", i+2, err)
		}
		chatUser := &models.ChatUser{
			ChatID:       parsedChatID,
			Username:     username,
			PasswordHash: hash,
			UserInfo:     fmt.Sprintf("Импортированная роль: %s", role),
		}
		if err := s.repo.Create(chatUser); err != nil {
			return fmt.Errorf("строка %d: ошибка сохранения пользователя '%s': %w", i+2, username, err)
		}
	}
	return nil
}
