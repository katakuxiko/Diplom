package models

import (
	"time"

	"github.com/google/uuid"
)

// TestQuestion хранит контрольный вопрос для конкретного чата.
type TestQuestion struct {
	ID               uuid.UUID `gorm:"type:uuid;default:gen_random_uuid();primaryKey" json:"id"`
	ChatID           uuid.UUID `gorm:"type:uuid;not null;index" json:"chat_id"`
	Chat             Chat      `gorm:"foreignKey:ChatID;references:ID;constraint:OnDelete:CASCADE" swaggerignore:"true" json:"-"`
	Text             string    `gorm:"type:text;not null" json:"text"`
	Category         string    `gorm:"size:100" json:"category"`
	ExpectedAnswer   string    `gorm:"type:text" json:"expected_answer"`
	ExpectedNoAnswer bool      `gorm:"default:false" json:"expected_no_answer"`
	SourceHint       string    `gorm:"type:text" json:"source_hint"`
	OrderNum         int       `gorm:"default:0" json:"order_num"`
	CreatedAt        time.Time `gorm:"default:now()" json:"created_at"`
}

// EvaluationRun хранит один запуск тестирования по набору вопросов.
type EvaluationRun struct {
	ID             uuid.UUID          `gorm:"type:uuid;default:gen_random_uuid();primaryKey" json:"id"`
	ChatID         uuid.UUID          `gorm:"type:uuid;not null;index" json:"chat_id"`
	Chat           Chat               `gorm:"foreignKey:ChatID;references:ID;constraint:OnDelete:CASCADE" swaggerignore:"true" json:"-"`
	Status         string             `gorm:"size:30;not null;default:in_progress" json:"status"`
	Model          string             `gorm:"size:200" json:"model"`
	TopK           int                `gorm:"default:5" json:"top_k"`
	TotalQuestions int                `gorm:"default:0" json:"total_questions"`
	EvaluatedCount int                `gorm:"default:0" json:"evaluated_count"`
	CorrectCount   int                `gorm:"default:0" json:"correct_count"`
	AvgScore       float64            `gorm:"default:0" json:"avg_score"`
	StartedAt      time.Time          `gorm:"default:now()" json:"started_at"`
	CompletedAt    *time.Time         `json:"completed_at"`
	Results        []EvaluationResult `gorm:"foreignKey:RunID" swaggerignore:"true" json:"results,omitempty"`
}

// EvaluationResult хранит результат ответа модели по одному контрольному вопросу.
type EvaluationResult struct {
	ID                uuid.UUID     `gorm:"type:uuid;default:gen_random_uuid();primaryKey" json:"id"`
	RunID             uuid.UUID     `gorm:"type:uuid;not null;index" json:"run_id"`
	Run               EvaluationRun `gorm:"foreignKey:RunID;references:ID;constraint:OnDelete:CASCADE" swaggerignore:"true" json:"-"`
	QuestionID        uuid.UUID     `gorm:"type:uuid;not null;index" json:"question_id"`
	Question          TestQuestion  `gorm:"foreignKey:QuestionID;references:ID;constraint:OnDelete:CASCADE" json:"question"`
	RetrievedFragment string        `gorm:"type:text" json:"retrieved_fragment"`
	ModelAnswer       string        `gorm:"type:text" json:"model_answer"`
	ResponseTimeMs    int64         `gorm:"default:0" json:"response_time_ms"`
	FallbackUsed      bool          `gorm:"default:false" json:"fallback_used"`
	ErrorMessage      string        `gorm:"type:text" json:"error_message"`
	ExpertScore       *int          `json:"expert_score"`
	ExpertFeedback    string        `gorm:"type:text" json:"expert_feedback"`
	IsCorrect         *bool         `json:"is_correct"`
	EvaluatorAdminID  *uuid.UUID    `gorm:"type:uuid" json:"evaluator_admin_id"`
	EvaluatedAt       *time.Time    `json:"evaluated_at"`
	CreatedAt         time.Time     `gorm:"default:now()" json:"created_at"`
}
