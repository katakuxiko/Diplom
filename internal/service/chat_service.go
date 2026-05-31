package service

import (
	"errors"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"gorm.io/gorm"
)

var (
	ErrChatNotFound         = errors.New("chat not found")
	ErrChatAccessDenied     = errors.New("chat access denied")
	ErrOnlyCreatorCanInvite = errors.New("only chat creator can invite admins")
	ErrAdminNotFound        = errors.New("admin not found")
	ErrAdminAlreadyInChat   = errors.New("admin already in chat")
)

// Интерфейс, который используют хендлеры и тесты
type ChatServiceInterface interface {
	Create(req *dto.ChatCreateRequest) (*models.Chat, error)
	CreateForAdmin(adminID string, req *dto.ChatCreateRequest) (*models.Chat, error)
	GetByID(id string) (*models.Chat, error)
	ListAll() ([]models.Chat, error)
	ListByAdmin(adminID string) ([]models.Chat, error)
	EnsureAdminAccess(chatID string, adminID string) error
	InviteAdmin(chatID string, inviterAdminID string, inviteeAdminID string) error
	ListAdmins(chatID string, requesterAdminID string) ([]models.Admin, error)
	Delete(id string) error
	Update(id string, req *dto.ChatUpdateRequest) (*models.Chat, error)
	SetDB(db *gorm.DB)
}

// Основная реализация сервиса
type ChatService struct {
	repo *repository.ChatRepository
	db   *gorm.DB
}

func NewChatService(repo *repository.ChatRepository) *ChatService {
	return &ChatService{repo: repo}
}

func (s *ChatService) SetDB(db *gorm.DB) {
	s.db = db
}

func (s *ChatService) Create(req *dto.ChatCreateRequest) (*models.Chat, error) {
	return s.CreateForAdmin("", req)
}

func (s *ChatService) CreateForAdmin(adminID string, req *dto.ChatCreateRequest) (*models.Chat, error) {
	var adminUUID *uuid.UUID
	if adminID != "" {
		id, err := uuid.Parse(adminID)
		if err != nil {
			return nil, err
		}
		adminUUID = &id
	}

	chat := &models.Chat{
		AdminID: adminUUID,
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

func (s *ChatService) ListAll() ([]models.Chat, error) {
	return s.repo.List()
}

func (s *ChatService) ListByAdmin(adminID string) ([]models.Chat, error) {
	return s.repo.ListByAdmin(adminID)
}

func (s *ChatService) EnsureAdminAccess(chatID string, adminID string) error {
	if _, err := s.repo.GetByID(chatID); err != nil {
		return ErrChatNotFound
	}

	hasAccess, err := s.repo.IsAdminInChat(chatID, adminID)
	if err != nil {
		return err
	}
	if !hasAccess {
		return ErrChatAccessDenied
	}

	return nil
}

func (s *ChatService) InviteAdmin(chatID string, inviterAdminID string, inviteeAdminID string) error {
	if _, err := s.repo.GetByID(chatID); err != nil {
		return ErrChatNotFound
	}

	isCreator, err := s.repo.IsCreator(chatID, inviterAdminID)
	if err != nil {
		return err
	}
	if !isCreator {
		return ErrOnlyCreatorCanInvite
	}

	inviteeExists, err := s.repo.AdminExists(inviteeAdminID)
	if err != nil {
		return err
	}
	if !inviteeExists {
		return ErrAdminNotFound
	}

	hasAccess, err := s.repo.IsAdminInChat(chatID, inviteeAdminID)
	if err != nil {
		return err
	}
	if hasAccess {
		return ErrAdminAlreadyInChat
	}

	created, err := s.repo.AddAdminToChat(chatID, inviteeAdminID)
	if err != nil {
		return err
	}
	if !created {
		return ErrAdminAlreadyInChat
	}

	return nil
}

func (s *ChatService) ListAdmins(chatID string, requesterAdminID string) ([]models.Admin, error) {
	if err := s.EnsureAdminAccess(chatID, requesterAdminID); err != nil {
		return nil, err
	}

	return s.repo.ListAdminsByChat(chatID)
}

func (s *ChatService) Delete(id string) error {
	// Сначала удаляем все документы чата
	if s.db != nil {
		// Удаляем все чанки для документов этого чата
		if err := s.db.Where("chat_id = ?", id).Delete(&models.Chunk{}).Error; err != nil {
			return err
		}
		// Удаляем все документы этого чата
		if err := s.db.Where("chat_id = ?", id).Delete(&models.Document{}).Error; err != nil {
			return err
		}
	}
	// Теперь удаляем сам чат
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
