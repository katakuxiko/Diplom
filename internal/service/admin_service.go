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
