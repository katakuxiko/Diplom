package service

import (
	"context"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
)

type EvaluationService struct {
	repo *repository.EvaluationRepository
}

func NewEvaluationService(repo *repository.EvaluationRepository) *EvaluationService {
	return &EvaluationService{repo: repo}
}

func (s *EvaluationService) CreateTestQuestion(ctx context.Context, question *models.TestQuestion) error {
	return s.repo.CreateTestQuestion(ctx, question)
}

func (s *EvaluationService) ListTestQuestionsByChat(ctx context.Context, chatID uuid.UUID, page, limit int) ([]models.TestQuestion, int64, error) {
	return s.repo.ListTestQuestionsByChat(ctx, chatID, page, limit)
}

func (s *EvaluationService) ListAllTestQuestionsByChat(ctx context.Context, chatID uuid.UUID) ([]models.TestQuestion, error) {
	return s.repo.ListAllTestQuestionsByChat(ctx, chatID)
}

func (s *EvaluationService) DeleteTestQuestion(ctx context.Context, chatID, questionID uuid.UUID) error {
	return s.repo.DeleteTestQuestion(ctx, chatID, questionID)
}

func (s *EvaluationService) CreateRun(ctx context.Context, run *models.EvaluationRun) error {
	return s.repo.CreateRun(ctx, run)
}

func (s *EvaluationService) UpdateRun(ctx context.Context, run *models.EvaluationRun) error {
	return s.repo.UpdateRun(ctx, run)
}

func (s *EvaluationService) GetRunByID(ctx context.Context, runID uuid.UUID) (*models.EvaluationRun, error) {
	return s.repo.GetRunByID(ctx, runID)
}

func (s *EvaluationService) ListRunsByChat(ctx context.Context, chatID uuid.UUID, page, limit int) ([]models.EvaluationRun, int64, error) {
	return s.repo.ListRunsByChat(ctx, chatID, page, limit)
}

func (s *EvaluationService) CreateResults(ctx context.Context, results []models.EvaluationResult) error {
	return s.repo.CreateResults(ctx, results)
}

func (s *EvaluationService) UpdateResultScore(ctx context.Context, resultID uuid.UUID, score int, feedback string, isCorrect *bool, evaluatorID *uuid.UUID) (*models.EvaluationResult, error) {
	result, err := s.repo.UpdateResultScore(ctx, resultID, score, feedback, isCorrect, evaluatorID)
	if err != nil {
		return nil, err
	}

	if err := s.repo.RecalculateRunStats(ctx, result.RunID); err != nil {
		return nil, err
	}

	return result, nil
}
