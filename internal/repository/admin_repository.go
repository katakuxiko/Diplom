package repository

import (
	"database/sql"
	"errors"

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
	if err := r.db.Where("username = ?", username).First(&admin).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &admin, nil
}
 
// StatsCounts хранит агрегированные значения для админской панели
type StatsCounts struct {
	UsersCount     int64
	ChatsCount     int64
	DocumentsCount int64
	MessagesCount  int64
	ChunksCount    int64
}

// ChatStats содержит статистику по одному чату
type ChatStats struct {
	ChatID         uuid.UUID `json:"chat_id"`
	Name           string    `json:"name"`
	UsersCount     int64     `json:"users_count"`
	DocumentsCount int64     `json:"documents_count"`
	MessagesCount  int64     `json:"messages_count"`
	ChunksCount    int64     `json:"chunks_count"`
}

// GetCountsPerChat возвращает агрегации по каждому чату
func (r *AdminRepository) GetCountsPerChat() ([]ChatStats, error) {
	// Получаем список чатов
	var chats []models.Chat
	if err := r.db.Find(&chats).Error; err != nil {
		return nil, err
	}

	// Инициализируем карту результатов
	statsMap := make(map[uuid.UUID]*ChatStats)
	for _, c := range chats {
		copy := c
		statsMap[copy.ID] = &ChatStats{ChatID: copy.ID, Name: copy.Name}
	}

	// Вспомогательная переменная для строк
	var rows *sql.Rows
	var err error

	// Пользователи по чатам
	rows, err = r.db.Model(&models.ChatUser{}).Select("chat_id, COUNT(*) as count").Group("chat_id").Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var chatIDRaw interface{}
		var cnt int64
		if err := rows.Scan(&chatIDRaw, &cnt); err != nil {
			return nil, err
		}
		var chatID uuid.UUID
		switch v := chatIDRaw.(type) {
		case string:
			id, err := uuid.Parse(v)
			if err != nil {
				continue
			}
			chatID = id
		case []byte:
			id, err := uuid.FromBytes(v)
			if err != nil {
				continue
			}
			chatID = id
		default:
			continue
		}
		if _, ok := statsMap[chatID]; !ok {
			statsMap[chatID] = &ChatStats{ChatID: chatID}
		}
		statsMap[chatID].UsersCount = cnt
	}

	// Документы по чатам
	rows, err = r.db.Model(&models.Document{}).Select("chat_id, COUNT(*) as count").Group("chat_id").Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var chatIDRaw interface{}
		var cnt int64
		if err := rows.Scan(&chatIDRaw, &cnt); err != nil {
			return nil, err
		}
		var chatID uuid.UUID
		switch v := chatIDRaw.(type) {
		case string:
			id, err := uuid.Parse(v)
			if err != nil {
				continue
			}
			chatID = id
		case []byte:
			id, err := uuid.FromBytes(v)
			if err != nil {
				continue
			}
			chatID = id
		default:
			continue
		}
		if _, ok := statsMap[chatID]; !ok {
			statsMap[chatID] = &ChatStats{ChatID: chatID}
		}
		statsMap[chatID].DocumentsCount = cnt
	}

	// Сообщения: join messages -> chat_histories
	rows, err = r.db.Table("messages").Select("chat_histories.chat_id as chat_id, COUNT(messages.id) as count").Joins("join chat_histories on messages.chat_history_id = chat_histories.id").Group("chat_histories.chat_id").Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var chatIDRaw interface{}
		var cnt int64
		if err := rows.Scan(&chatIDRaw, &cnt); err != nil {
			return nil, err
		}
		var chatID uuid.UUID
		switch v := chatIDRaw.(type) {
		case string:
			id, err := uuid.Parse(v)
			if err != nil {
				continue
			}
			chatID = id
		case []byte:
			id, err := uuid.FromBytes(v)
			if err != nil {
				continue
			}
			chatID = id
		default:
			continue
		}
		if _, ok := statsMap[chatID]; !ok {
			statsMap[chatID] = &ChatStats{ChatID: chatID}
		}
		statsMap[chatID].MessagesCount = cnt
	}

	// Чанки по чатам
	rows, err = r.db.Model(&models.Chunk{}).Select("chat_id, COUNT(*) as count").Group("chat_id").Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var chatIDRaw interface{}
		var cnt int64
		if err := rows.Scan(&chatIDRaw, &cnt); err != nil {
			return nil, err
		}
		var chatID uuid.UUID
		switch v := chatIDRaw.(type) {
		case string:
			id, err := uuid.Parse(v)
			if err != nil {
				continue
			}
			chatID = id
		case []byte:
			id, err := uuid.FromBytes(v)
			if err != nil {
				continue
			}
			chatID = id
		default:
			continue
		}
		if _, ok := statsMap[chatID]; !ok {
			statsMap[chatID] = &ChatStats{ChatID: chatID}
		}
		statsMap[chatID].ChunksCount = cnt
	}

	// Собираем слайс
	var result []ChatStats
	for _, s := range statsMap {
		result = append(result, *s)
	}

	return result, nil
}

// GetCounts возвращает количество записей по ключевым сущностям
func (r *AdminRepository) GetCounts() (*StatsCounts, error) {
	var usersCount, chatsCount, docsCount, messagesCount, chunksCount int64

	if err := r.db.Model(&models.ChatUser{}).Count(&usersCount).Error; err != nil {
		return nil, err
	}
	if err := r.db.Model(&models.Chat{}).Count(&chatsCount).Error; err != nil {
		return nil, err
	}
	if err := r.db.Model(&models.Document{}).Count(&docsCount).Error; err != nil {
		return nil, err
	}
	if err := r.db.Model(&models.Message{}).Count(&messagesCount).Error; err != nil {
		return nil, err
	}
	if err := r.db.Model(&models.Chunk{}).Count(&chunksCount).Error; err != nil {
		return nil, err
	}

	return &StatsCounts{
		UsersCount:     usersCount,
		ChatsCount:     chatsCount,
		DocumentsCount: docsCount,
		MessagesCount:  messagesCount,
		ChunksCount:    chunksCount,
	}, nil
}
 
