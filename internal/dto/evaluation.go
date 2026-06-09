package dto

import (
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
)

type TestQuestionCreateRequest struct {
	Text             string `json:"text"`
	Category         string `json:"category"`
	ExpectedAnswer   string `json:"expected_answer"`
	ExpectedNoAnswer bool   `json:"expected_no_answer"`
	SourceHint       string `json:"source_hint"`
	OrderNum         int    `json:"order_num"`
}

type TestQuestionResponse struct {
	ID               uuid.UUID `json:"id"`
	ChatID           uuid.UUID `json:"chat_id"`
	Text             string    `json:"text"`
	Category         string    `json:"category"`
	ExpectedAnswer   string    `json:"expected_answer"`
	ExpectedNoAnswer bool      `json:"expected_no_answer"`
	SourceHint       string    `json:"source_hint"`
	OrderNum         int       `json:"order_num"`
	CreatedAt        time.Time `json:"created_at"`
}

type PaginatedTestQuestions struct {
	Data  []TestQuestionResponse `json:"data"`
	Page  int                    `json:"page"`
	Limit int                    `json:"limit"`
	Total int64                  `json:"total"`
}

type EvaluationRunCreateRequest struct {
	ChatID   uuid.UUID           `json:"chat_id"`
	TopK     int                 `json:"top_k"`
	Model    string              `json:"model"`
	Settings *models.AskSettings `json:"settings,omitempty"`
}

type EvaluationResultScoreRequest struct {
	ExpertScore    int    `json:"expert_score"`
	ExpertFeedback string `json:"expert_feedback"`
	IsCorrect      *bool  `json:"is_correct"`
}

type EvaluationResultResponse struct {
	ID                uuid.UUID            `json:"id"`
	RunID             uuid.UUID            `json:"run_id"`
	Question          TestQuestionResponse `json:"question"`
	RetrievedFragment string               `json:"retrieved_fragment"`
	ModelAnswer       string               `json:"model_answer"`
	ResponseTimeMs    int64                `json:"response_time_ms"`
	FallbackUsed      bool                 `json:"fallback_used"`
	ErrorMessage      string               `json:"error_message"`
	ExpertScore       *int                 `json:"expert_score"`
	ExpertFeedback    string               `json:"expert_feedback"`
	IsCorrect         *bool                `json:"is_correct"`
	EvaluatorAdminID  *uuid.UUID           `json:"evaluator_admin_id"`
	EvaluatedAt       *time.Time           `json:"evaluated_at"`
	CreatedAt         time.Time            `json:"created_at"`
}

type EvaluationRunResponse struct {
	ID             uuid.UUID                  `json:"id"`
	ChatID         uuid.UUID                  `json:"chat_id"`
	Status         string                     `json:"status"`
	Model          string                     `json:"model"`
	TopK           int                        `json:"top_k"`
	TotalQuestions int                        `json:"total_questions"`
	EvaluatedCount int                        `json:"evaluated_count"`
	CorrectCount   int                        `json:"correct_count"`
	AvgScore       float64                    `json:"avg_score"`
	StartedAt      time.Time                  `json:"started_at"`
	CompletedAt    *time.Time                 `json:"completed_at"`
	Results        []EvaluationResultResponse `json:"results"`
}

type EvaluationRunListItem struct {
	ID             uuid.UUID  `json:"id"`
	ChatID         uuid.UUID  `json:"chat_id"`
	Status         string     `json:"status"`
	Model          string     `json:"model"`
	TopK           int        `json:"top_k"`
	TotalQuestions int        `json:"total_questions"`
	EvaluatedCount int        `json:"evaluated_count"`
	CorrectCount   int        `json:"correct_count"`
	AvgScore       float64    `json:"avg_score"`
	StartedAt      time.Time  `json:"started_at"`
	CompletedAt    *time.Time `json:"completed_at"`
}

type PaginatedEvaluationRuns struct {
	Data  []EvaluationRunListItem `json:"data"`
	Page  int                     `json:"page"`
	Limit int                     `json:"limit"`
	Total int64                   `json:"total"`
}

type EvaluationMetricsResponse struct {
	RunID              uuid.UUID `json:"run_id"`
	TotalQuestions     int       `json:"total_questions"`
	EvaluatedCount     int       `json:"evaluated_count"`
	CorrectAnswerRate  float64   `json:"correct_answer_rate"`
	CorrectRefusalRate float64   `json:"correct_refusal_rate"`
	HallucinationRate  float64   `json:"hallucination_rate"`
	FallbackRate       float64   `json:"fallback_rate"`
	ErrorRate          float64   `json:"error_rate"`
	AvgLatencyMs       float64   `json:"avg_latency_ms"`
	P95LatencyMs       int64     `json:"p95_latency_ms"`
}

type EvaluationBaselineCompareResponse struct {
	RunID                  uuid.UUID `json:"run_id"`
	QuestionsTotal         int       `json:"questions_total"`
	RAGContextHitRate      float64   `json:"rag_context_hit_rate"`
	BaselineContextHitRate float64   `json:"baseline_context_hit_rate"`
	BaselineAvgSearchMs    float64   `json:"baseline_avg_search_ms"`
	BaselineP95SearchMs    int64     `json:"baseline_p95_search_ms"`
}

type RetrievalCalibrationResponse struct {
	RunID                        uuid.UUID `json:"run_id"`
	TopK                         int       `json:"top_k"`
	SamplesTotal                 int       `json:"samples_total"`
	SamplesUsed                  int       `json:"samples_used"`
	RecommendedMaxCosineDistance float32   `json:"recommended_max_cosine_distance"`
	RecommendedMaxDistanceGap    float32   `json:"recommended_max_distance_gap"`
	Accuracy                     float64   `json:"accuracy"`
	Precision                    float64   `json:"precision"`
	Recall                       float64   `json:"recall"`
	F1                           float64   `json:"f1"`
	TruePositive                 int       `json:"true_positive"`
	TrueNegative                 int       `json:"true_negative"`
	FalsePositive                int       `json:"false_positive"`
	FalseNegative                int       `json:"false_negative"`
}

type RetrievalCalibrationApplyResponse struct {
	RunID             uuid.UUID                    `json:"run_id"`
	ChatID            uuid.UUID                    `json:"chat_id"`
	TopK              int                          `json:"top_k"`
	Applied           bool                         `json:"applied"`
	RetrievalMode     string                       `json:"retrieval_mode"`
	MaxCosineDistance float32                      `json:"max_cosine_distance"`
	MaxDistanceGap    float32                      `json:"max_distance_gap"`
	Calibration       RetrievalCalibrationResponse `json:"calibration"`
}
