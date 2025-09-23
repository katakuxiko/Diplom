package repository

import (
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type AdminRepository struct {
	db *gorm.DB
}

func NewAdminRepository(db *gorm.DB) *AdminRepository {
	return &AdminRepository{db: db}
}

func (r *AdminRepository) GetAll() ([]models.Admin, error) {
	var admins []models.Admin
	err := r.db.Find(&admins).Error
	return admins, err
}

func (r *AdminRepository) GetByID(id uuid.UUID) (*models.Admin, error) {
	var admin models.Admin
	err := r.db.First(&admin, "id = ?", id).Error
	if err != nil {
		return nil, err
	}
	return &admin, nil
}

func (r *AdminRepository) Create(admin *models.Admin) error {
	return r.db.Create(admin).Error
}

func (r *AdminRepository) Update(admin *models.Admin) error {
	return r.db.Save(admin).Error
}

func (r *AdminRepository) Delete(id uuid.UUID) error {
	return r.db.Delete(&models.Admin{}, "id = ?", id).Error
}

func (r *AdminRepository) GetByUsername(username string) (*models.Admin, error) {
	var admin models.Admin
	err := r.db.First(&admin, "username = ?", username).Error
	if err != nil {
		return nil, err
	}
	return &admin, nil
}
