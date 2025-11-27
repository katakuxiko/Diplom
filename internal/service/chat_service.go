package service

import (
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
)

// Интерфейс, который используют хендлеры и тесты
type ChatServiceInterface interface {
	Create(req *dto.ChatCreateRequest) (*models.Chat, error)
	GetByID(id string) (*models.Chat, error)
	ListByAdmin(adminID string) ([]models.Chat, error)
	Delete(id string) error
	Update(id string, req *dto.ChatUpdateRequest) (*models.Chat, error)
}

// Основная реализация сервиса
type ChatService struct {
	repo *repository.ChatRepository
}

func NewChatService(repo *repository.ChatRepository) *ChatService {
	return &ChatService{repo: repo}
}

func (s *ChatService) Create(req *dto.ChatCreateRequest) (*models.Chat, error) {
	chat := &models.Chat{
		AdminID: req.AdminID,
		Name:    req.Name,
		Descr:   req.Descr,
	}
	if err := s.repo.Create(chat); err != nil {
		return nil, err
	}
	return chat, nil
}

func (s *ChatService) GetByID(id string) (*models.Chat, error) {
	return s.repo.GetByID(id)
}

func (s *ChatService) ListByAdmin(adminID string) ([]models.Chat, error) {
	return s.repo.ListByAdmin(adminID)
}

func (s *ChatService) Delete(id string) error {
	return s.repo.Delete(id)
}
func (s *ChatService) Update(id string, req *dto.ChatUpdateRequest) (*models.Chat, error) {
	chat, err := s.repo.GetByID(id)
	if err != nil {
		return nil, err
	}

	if req.Name != nil && *req.Name != "" {
		chat.Name = *req.Name
	}
	if req.Descr != nil && *req.Descr != "" {
		chat.Descr = *req.Descr
	}

	if err := s.repo.Update(chat); err != nil {
		return nil, err
	}
	return chat, nil
}
