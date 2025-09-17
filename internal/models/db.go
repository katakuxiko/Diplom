package models

import (
	"time"

	"github.com/google/uuid"
)

// Админы
type Admin struct {
	ID           uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	Username     string    `gorm:"unique;not null"`
	PasswordHash string    `gorm:"not null"`
	IsSuperUser  bool      `gorm:"default:false"`
	Chats        []Chat    `gorm:"foreignKey:AdminID"`
}

// Чаты
type Chat struct {
	ID          uuid.UUID  `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	AdminID     *uuid.UUID `gorm:"type:uuid"`
	Admin       *Admin
	Name        string `gorm:"not null"`
	Descr       string
	CreatedDate time.Time     `gorm:"default:now()"`
	Documents   []Document    `gorm:"foreignKey:ChatID"`
	Settings    []ChatSetting `gorm:"foreignKey:ChatID"`
	Roles       []Role        `gorm:"foreignKey:ChatID"`
	Users       []ChatUser    `gorm:"foreignKey:ChatID"`
	History     []ChatHistory `gorm:"foreignKey:ChatID"`
}

// Документы
type Document struct {
	ID          uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatID      uuid.UUID `gorm:"type:uuid;not null"`
	Chat        Chat
	Name        string `gorm:"not null"`
	Path        string
	Protected   bool      `gorm:"default:false"`
	AccessLevel int       `gorm:"default:0"`
	CreatedDate time.Time `gorm:"default:now()"`
	Chunks      []Chunk   `gorm:"foreignKey:DocID"`
}

// Чанки
type Chunk struct {
	ID        uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	DocID     uuid.UUID `gorm:"type:uuid;not null"`
	Document  Document  `gorm:"foreignKey:DocID;references:ID"`
	DocName   string
	Text      string
	Embedding []float32 `gorm:"type:vector(768)"` // pgvector
	FilePath  string
	ChunkName string
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
	Users       []ChatUser `gorm:"foreignKey:UserRole"`
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
	History      []ChatHistory `gorm:"foreignKey:UserID"`
}

// История чата
type ChatHistory struct {
	ID       uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatID   uuid.UUID `gorm:"type:uuid;not null"`
	Chat     Chat
	UserID   uuid.UUID `gorm:"type:uuid;not null"`
	User     ChatUser
	Messages []Message `gorm:"foreignKey:ChatHistoryID"`
}

// Сообщения
type Message struct {
	ID            uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	ChatHistoryID uuid.UUID `gorm:"type:uuid;not null"`
	ChatHistory   ChatHistory
	Text          string
	Role          string
	CreatedDate   time.Time `gorm:"default:now()"`
}
