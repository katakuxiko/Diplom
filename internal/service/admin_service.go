package service

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
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

// --- Авторизация ---
var jwtSecret = []byte("your_secret_key") // Замените на свой секрет

type Claims struct {
	AdminID string `json:"admin_id"`
	jwt.RegisteredClaims
}

func (s *AdminService) Authenticate(req *dto.AdminAuthRequest) (*dto.AuthTokensResponse, error) {
	admin, err := s.repo.GetByUsername(req.Username)
	if err != nil || admin == nil {
		return nil, errors.New("invalid credentials")
	}
	if !utils.CheckPassword(req.Password, admin.PasswordHash) {
		return nil, errors.New("invalid credentials")
	}

	accessToken, err := GenerateJWT(admin.ID.String(), 15*time.Minute)
	if err != nil {
		return nil, err
	}
	refreshToken, err := GenerateJWT(admin.ID.String(), 7*24*time.Hour)
	if err != nil {
		return nil, err
	}

	return &dto.AuthTokensResponse{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
	}, nil
}

func GenerateJWT(adminID string, duration time.Duration) (string, error) {
	claims := &Claims{
		AdminID: adminID,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(duration)),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jwtSecret)
}

func ParseJWT(tokenStr string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return jwtSecret, nil
	})
	if err != nil {
		return nil, err
	}
	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}
	return nil, errors.New("invalid token")
}
