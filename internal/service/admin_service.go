package service

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
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

func (s *AdminService) Create(admin *models.Admin) error {
	return s.repo.Create(admin)
}

func (s *AdminService) Update(admin *models.Admin) error {
	return s.repo.Update(admin)
}

func (s *AdminService) Delete(id uuid.UUID) error {
	return s.repo.Delete(id)
}
