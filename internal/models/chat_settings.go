package models

// ХЗ че еще добавить
type ChatSettings struct {
	ID          int    `json:"id" db:"id"`
	Language    string `json:"language" db:"language"`
	MaxMessages int    `json:"max_messages" db:"max_messages"`
	IsActive    bool   `json:"is_active" db:"is_active"`
}
