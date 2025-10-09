package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/dto"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// ---------- Mock ChatService ----------
type MockChatService struct {
	mock.Mock
}

func (m *MockChatService) Create(req *dto.ChatCreateRequest) (*models.Chat, error) {
	args := m.Called(req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*models.Chat), args.Error(1)
}

func (m *MockChatService) GetByID(id string) (*models.Chat, error) {
	args := m.Called(id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*models.Chat), args.Error(1)
}

func (m *MockChatService) ListByAdmin(adminID string) ([]models.Chat, error) {
	args := m.Called(adminID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]models.Chat), args.Error(1)
}

func (m *MockChatService) Delete(id string) error {
	args := m.Called(id)
	return args.Error(0)
}

// ---------- TESTS ----------

func TestCreateChat(t *testing.T) {
	mockService := new(MockChatService)
	app := fiber.New()
	app.Post("/chats", CreateChat(mockService))

	adminID := uuid.New()
	reqBody := dto.ChatCreateRequest{
		AdminID: &adminID,
		Name:    "Test Chat",
		Descr:   "Test descr",
	}

	chat := &models.Chat{
		ID:          uuid.New(),
		AdminID:     &adminID,
		Name:        reqBody.Name,
		Descr:       reqBody.Descr,
		CreatedDate: time.Now(),
	}

	mockService.On("Create", &reqBody).Return(chat, nil)

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/chats", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, _ := app.Test(req)
	assert.Equal(t, 201, resp.StatusCode)
	mockService.AssertExpectations(t)
}

func TestGetChat_Success(t *testing.T) {
	mockService := new(MockChatService)
	app := fiber.New()
	app.Get("/chats/:id", GetChat(mockService))

	chatID := uuid.New()
	adminID := uuid.New()

	mockChat := &models.Chat{
		ID:          chatID,
		AdminID:     &adminID,
		Name:        "Chat 1",
		Descr:       "Descr",
		CreatedDate: time.Now(),
	}

	mockService.On("GetByID", chatID.String()).Return(mockChat, nil)

	req := httptest.NewRequest("GET", "/chats/"+chatID.String(), nil)
	resp, _ := app.Test(req)
	assert.Equal(t, 200, resp.StatusCode)
	mockService.AssertExpectations(t)
}

func TestGetChat_NotFound(t *testing.T) {
	mockService := new(MockChatService)
	app := fiber.New()
	app.Get("/chats/:id", GetChat(mockService))

	mockService.On("GetByID", "404").Return(nil, errors.New("not found"))

	req := httptest.NewRequest("GET", "/chats/404", nil)
	resp, _ := app.Test(req)
	assert.Equal(t, 404, resp.StatusCode)
	mockService.AssertExpectations(t)
}

func TestListChats(t *testing.T) {
	mockService := new(MockChatService)
	app := fiber.New()

	app.Get("/chats", func(c *fiber.Ctx) error {
		c.Locals("user", jwt.MapClaims{"id": "admin123"})
		return ListChats(mockService)(c)
	})

	chats := []models.Chat{
		{ID: uuid.New(), Name: "Chat1"},
		{ID: uuid.New(), Name: "Chat2"},
	}
	mockService.On("ListByAdmin", "admin123").Return(chats, nil)

	req := httptest.NewRequest("GET", "/chats", nil)
	resp, _ := app.Test(req)
	assert.Equal(t, 200, resp.StatusCode)
	mockService.AssertExpectations(t)
}

func TestDeleteChat_Success(t *testing.T) {
	mockService := new(MockChatService)
	app := fiber.New()
	app.Delete("/chats/:id", DeleteChat(mockService))

	mockService.On("Delete", "123").Return(nil)

	req := httptest.NewRequest("DELETE", "/chats/123", nil)
	resp, _ := app.Test(req)
	assert.Equal(t, 204, resp.StatusCode)
	mockService.AssertExpectations(t)
}

func TestDeleteChat_Error(t *testing.T) {
	mockService := new(MockChatService)
	app := fiber.New()
	app.Delete("/chats/:id", DeleteChat(mockService))

	mockService.On("Delete", "123").Return(errors.New("delete error"))

	req := httptest.NewRequest("DELETE", "/chats/123", nil)
	resp, _ := app.Test(req)
	assert.Equal(t, 500, resp.StatusCode)
	mockService.AssertExpectations(t)
}
