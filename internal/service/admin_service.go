package service

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/utils"
)

type AdminService struct {
	repo *repository.AdminRepository
}

func NewAdminService(repo *repository.AdminRepository) *AdminService {
	return &AdminService{repo: repo}
}

func (s *AdminService) GetAll() ([]models.Admin, error) {
	return s.repo.GetAll()
}

func (s *AdminService) GetByID(id uuid.UUID) (*models.Admin, error) {
	return s.repo.GetByID(id)
}

// service/admin_service.go
func (s *AdminService) Create(req *dto.AdminCreateRequest) (*models.Admin, error) {

	PasswordHash, err := utils.HashPassword(req.Password)
	if err != nil {
		return nil, err
	}

	admin := &models.Admin{
		Username:     req.Username,
		PasswordHash: PasswordHash,
		IsSuperUser:  req.IsSuper,
	}

	if err := s.repo.Create(admin); err != nil {
		return nil, err
	}
	return admin, nil
}

func (s *AdminService) Update(admin *models.Admin) error {
	return s.repo.Update(admin)
}

func (s *AdminService) Delete(id uuid.UUID) error {
	return s.repo.Delete(id)
}

func (s *AdminService) GetByUsername(username string) (*models.Admin, error) {
	return s.repo.GetByUsername(username)
}

func (s *AdminService) CheckPassword(password, hash string) bool {
	return utils.CheckPassword(password, hash)
}

// GetStats собирает агрегированные метрики для админской панели
func (s *AdminService) GetStats() (*dto.AdminStatsResponse, error) {
	counts, err := s.repo.GetCounts()
	if err != nil {
		return nil, err
	}

	// Собираем статистику по каждому чату
	perChat, err := s.repo.GetCountsPerChat()
	if err != nil {
		return nil, err
	}

	var chatsDto []dto.ChatStatsResponse
	for _, c := range perChat {
		chatsDto = append(chatsDto, dto.ChatStatsResponse{
			ChatID:         c.ChatID,
			Name:           c.Name,
			UsersCount:     c.UsersCount,
			DocumentsCount: c.DocumentsCount,
			MessagesCount:  c.MessagesCount,
			ChunksCount:    c.ChunksCount,
		})
	}

	return &dto.AdminStatsResponse{
		UsersCount:     counts.UsersCount,
		ChatsCount:     counts.ChatsCount,
		DocumentsCount: counts.DocumentsCount,
		MessagesCount:  counts.MessagesCount,
		ChunksCount:    counts.ChunksCount,
		Chats:          chatsDto,
	}, nil
}
