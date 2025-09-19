package models

import (
	"time"

	"github.com/google/uuid"
	"github.com/pgvector/pgvector-go"
)

// Админы
type Admin struct {
	ID           uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	Username     string    `gorm:"unique;not null"`
	PasswordHash string    `gorm:"not null"`
	IsSuperUser  bool      `gorm:"default:false"`
	Chats        []Chat    `gorm:"foreignKey:AdminID" swaggerignore:"true"`
}

// Чаты
type Chat struct {
	ID          uuid.UUID  `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	AdminID     *uuid.UUID `gorm:"type:uuid"`
	Admin       *Admin
	Name        string `gorm:"not null"`
	Descr       string
	CreatedDate time.Time     `gorm:"default:now()"`
	Documents   []Document    `gorm:"foreignKey:ChatID" swaggerignore:"true"`
	Settings    []ChatSetting `gorm:"foreignKey:ChatID" swaggerignore:"true"`
	Roles       []Role        `gorm:"foreignKey:ChatID" swaggerignore:"true"`
	Users       []ChatUser    `gorm:"foreignKey:ChatID" swaggerignore:"true"`
	History     []ChatHistory `gorm:"foreignKey:ChatID" swaggerignore:"true"`
}

// Документы
type Document struct {
	ID          uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey" json:"id"`
	ChatID      uuid.UUID `gorm:"type:uuid;not null" json:"chat_id"`
	Name        string    `gorm:"not null" json:"name"`
	Path        string    `json:"path"`
	Protected   bool      `gorm:"default:false" json:"protected"`
	AccessLevel int       `gorm:"default:0" json:"access_level"`
	CreatedDate time.Time `gorm:"default:now()" json:"created_date"`
	Chat        Chat      `gorm:"constraint:OnUpdate:CASCADE,OnDelete:CASCADE;foreignKey:ChatID" json:"-"`
	Chunks      []Chunk   `gorm:"foreignKey:DocID" swaggerignore:"true" json:"-"`
}

// Чанки
type Chunk struct {
	ID        uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	DocID     uuid.UUID `gorm:"type:uuid;not null"`
	DocName   string
	Text      string
	Embedding pgvector.Vector `gorm:"type:vector(768)" swaggerignore:"true"`
	Filepath  string
	ChunkName string
	Document  Document `gorm:"foreignKey:DocID;references:ID" swaggerignore:"true"`
}

// Настройки чата
type ChatSetting struct {
	ID          uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatID      uuid.UUID `gorm:"type:uuid;not null"`
	Chat        Chat
	HelloText   string
	Name        string
	Descr       string
	URL         string
	CreatedDate time.Time              `gorm:"default:now()"`
	Settings    map[string]interface{} `gorm:"type:jsonb"`
}

// Роли
type Role struct {
	ID          uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatID      uuid.UUID `gorm:"type:uuid;not null"`
	Chat        Chat
	Name        string
	AccessLevel int        `gorm:"default:0"`
	Users       []ChatUser `gorm:"foreignKey:UserRole" swaggerignore:"true"`
}

// Пользователи чата
type ChatUser struct {
	ID           uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatID       uuid.UUID `gorm:"type:uuid;not null"`
	Chat         Chat
	UserRole     *uuid.UUID
	Role         Role `gorm:"foreignKey:UserRole;references:ID"`
	Username     string
	UserInfo     string
	PasswordHash string
	History      []ChatHistory `gorm:"foreignKey:UserID" swaggerignore:"true"`
}

// История чата
type ChatHistory struct {
	ID       uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatID   uuid.UUID `gorm:"type:uuid;not null"`
	Chat     Chat
	UserID   uuid.UUID `gorm:"type:uuid;not null"`
	User     ChatUser
	Messages []Message `gorm:"foreignKey:ChatHistoryID" swaggerignore:"true"`
}

// Сообщения
type Message struct {
	ID            uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatHistoryID uuid.UUID `gorm:"type:uuid;not null"`
	ChatHistory   ChatHistory
	Text          string
	Role          string
	CreatedDate   time.Time `gorm:"default:now()" swaggerignore:"true"`
}
