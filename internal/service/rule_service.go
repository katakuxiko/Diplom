package service

import (
	"context"

	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
)

type RuleService struct {
	repo *repository.RuleRepository
}

func NewRuleService(repo *repository.RuleRepository) *RuleService {
	return &RuleService{repo: repo}
}

func (s *RuleService) CreateRule(ctx context.Context, rule models.Rule) (int, error) {
	return s.repo.Create(ctx, rule)
}

func (s *RuleService) GetRuleByID(ctx context.Context, id int) (models.Rule, error) {
	return s.repo.GetByID(ctx, id)
}

func (s *RuleService) GetAllRules(ctx context.Context) ([]models.Rule, error) {
	return s.repo.GetAll(ctx)
}

func (s *RuleService) UpdateRule(ctx context.Context, rule models.Rule) error {
	return s.repo.Update(ctx, rule)
}

func (s *RuleService) DeleteRule(ctx context.Context, id int) error {
	return s.repo.Delete(ctx, id)
}
