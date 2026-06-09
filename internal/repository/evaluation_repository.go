package repository

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"gorm.io/gorm"
)

type EvaluationRepository struct {
	db *gorm.DB
}

func NewEvaluationRepository(db *gorm.DB) *EvaluationRepository {
	return &EvaluationRepository{db: db}
}

func (r *EvaluationRepository) CreateTestQuestion(ctx context.Context, q *models.TestQuestion) error {
	return r.db.WithContext(ctx).Create(q).Error
}

func (r *EvaluationRepository) ListTestQuestionsByChat(ctx context.Context, chatID uuid.UUID, page, limit int) ([]models.TestQuestion, int64, error) {
	if page <= 0 {
		page = 1
	}
	if limit <= 0 {
		limit = 10
	}

	var total int64
	if err := r.db.WithContext(ctx).Model(&models.TestQuestion{}).Where("chat_id = ?", chatID).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var questions []models.TestQuestion
	err := r.db.WithContext(ctx).
		Where("chat_id = ?", chatID).
		Order("order_num asc, created_at asc").
		Offset((page - 1) * limit).
		Limit(limit).
		Find(&questions).Error
	if err != nil {
		return nil, 0, err
	}

	return questions, total, nil
}

func (r *EvaluationRepository) ListAllTestQuestionsByChat(ctx context.Context, chatID uuid.UUID) ([]models.TestQuestion, error) {
	var questions []models.TestQuestion
	err := r.db.WithContext(ctx).
		Where("chat_id = ?", chatID).
		Order("order_num asc, created_at asc").
		Find(&questions).Error
	if err != nil {
		return nil, err
	}
	return questions, nil
}

func (r *EvaluationRepository) DeleteTestQuestion(ctx context.Context, chatID, questionID uuid.UUID) error {
	return r.db.WithContext(ctx).
		Where("chat_id = ? AND id = ?", chatID, questionID).
		Delete(&models.TestQuestion{}).Error
}

func (r *EvaluationRepository) CreateRun(ctx context.Context, run *models.EvaluationRun) error {
	return r.db.WithContext(ctx).Create(run).Error
}

func (r *EvaluationRepository) UpdateRun(ctx context.Context, run *models.EvaluationRun) error {
	return r.db.WithContext(ctx).Save(run).Error
}

func (r *EvaluationRepository) GetRunByID(ctx context.Context, runID uuid.UUID) (*models.EvaluationRun, error) {
	var run models.EvaluationRun
	err := r.db.WithContext(ctx).
		Preload("Results", func(db *gorm.DB) *gorm.DB { return db.Order("created_at asc") }).
		Preload("Results.Question").
		First(&run, "id = ?", runID).Error
	if err != nil {
		return nil, err
	}
	return &run, nil
}

func (r *EvaluationRepository) ListRunsByChat(ctx context.Context, chatID uuid.UUID, page, limit int) ([]models.EvaluationRun, int64, error) {
	if page <= 0 {
		page = 1
	}
	if limit <= 0 {
		limit = 10
	}

	var total int64
	if err := r.db.WithContext(ctx).Model(&models.EvaluationRun{}).Where("chat_id = ?", chatID).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var runs []models.EvaluationRun
	err := r.db.WithContext(ctx).
		Where("chat_id = ?", chatID).
		Order("started_at desc").
		Offset((page - 1) * limit).
		Limit(limit).
		Find(&runs).Error
	if err != nil {
		return nil, 0, err
	}

	return runs, total, nil
}

func (r *EvaluationRepository) CreateResults(ctx context.Context, results []models.EvaluationResult) error {
	if len(results) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Create(&results).Error
}

func (r *EvaluationRepository) UpdateResultScore(ctx context.Context, resultID uuid.UUID, score int, feedback string, isCorrect *bool, evaluatorID *uuid.UUID) (*models.EvaluationResult, error) {
	var result models.EvaluationResult
	if err := r.db.WithContext(ctx).First(&result, "id = ?", resultID).Error; err != nil {
		return nil, err
	}

	now := time.Now()
	result.ExpertScore = &score
	result.ExpertFeedback = feedback
	result.IsCorrect = isCorrect
	result.EvaluatorAdminID = evaluatorID
	result.EvaluatedAt = &now

	if err := r.db.WithContext(ctx).Save(&result).Error; err != nil {
		return nil, err
	}

	if err := r.db.WithContext(ctx).Preload("Question").First(&result, "id = ?", resultID).Error; err != nil {
		return nil, err
	}

	return &result, nil
}

func (r *EvaluationRepository) RecalculateRunStats(ctx context.Context, runID uuid.UUID) error {
	var run models.EvaluationRun
	if err := r.db.WithContext(ctx).First(&run, "id = ?", runID).Error; err != nil {
		return err
	}

	var evaluatedCount int64
	if err := r.db.WithContext(ctx).
		Model(&models.EvaluationResult{}).
		Where("run_id = ? AND expert_score IS NOT NULL", runID).
		Count(&evaluatedCount).Error; err != nil {
		return err
	}

	var correctCount int64
	if err := r.db.WithContext(ctx).
		Model(&models.EvaluationResult{}).
		Where("run_id = ? AND is_correct = ?", runID, true).
		Count(&correctCount).Error; err != nil {
		return err
	}

	var avgScore float64
	if err := r.db.WithContext(ctx).
		Model(&models.EvaluationResult{}).
		Where("run_id = ? AND expert_score IS NOT NULL", runID).
		Select("COALESCE(AVG(expert_score), 0)").
		Scan(&avgScore).Error; err != nil {
		return err
	}

	run.EvaluatedCount = int(evaluatedCount)
	run.CorrectCount = int(correctCount)
	run.AvgScore = avgScore

	return r.db.WithContext(ctx).Save(&run).Error
}
