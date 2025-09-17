package service

import (
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
)

type ChatService struct {
	repo *repository.ChatRepository
}

func NewChatService(repo *repository.ChatRepository) *ChatService {
	return &ChatService{repo: repo}
}

func (s *ChatService) Create(req *models.ChatCreateRequest) (*models.Chat, error) {
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

func (s *ChatService) List() ([]models.Chat, error) {
	return s.repo.List()
}

func (s *ChatService) Delete(id string) error {
	return s.repo.Delete(id)
}
