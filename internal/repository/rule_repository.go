package repository

import (
	"context"
	"database/sql"

	"github.com/katakuxiko/Diplom/internal/models"
)

type RuleRepository struct {
	db *sql.DB
}

func NewRuleRepository(db *sql.DB) *RuleRepository {
	return &RuleRepository{db: db}
}

func (r *RuleRepository) Create(ctx context.Context, rule models.Rule) (int, error) {
	query := "INSERT INTO rules (name, description) VALUES ($1, $2) RETURNING id"
	var id int
	err := r.db.QueryRowContext(ctx, query, rule.Name, rule.Description).Scan(&id)
	return id, err
}

func (r *RuleRepository) GetByID(ctx context.Context, id int) (models.Rule, error) {
	query := "SELECT id, name, description FROM rules WHERE id = $1"
	var rule models.Rule
	err := r.db.QueryRowContext(ctx, query, id).Scan(&rule.ID, &rule.Name, &rule.Description)
	return rule, err
}

func (r *RuleRepository) GetAll(ctx context.Context) ([]models.Rule, error) {
	query := "SELECT id, name, description FROM rules"
	rows, err := r.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var rules []models.Rule
	for rows.Next() {
		var rule models.Rule
		if err := rows.Scan(&rule.ID, &rule.Name, &rule.Description); err != nil {
			return nil, err
		}
		rules = append(rules, rule)
	}
	return rules, nil
}

func (r *RuleRepository) Update(ctx context.Context, rule models.Rule) error {
	query := "UPDATE rules SET name = $1, description = $2 WHERE id = $3"
	_, err := r.db.ExecContext(ctx, query, rule.Name, rule.Description, rule.ID)
	return err
}

func (r *RuleRepository) Delete(ctx context.Context, id int) error {
	query := "DELETE FROM rules WHERE id = $1"
	_, err := r.db.ExecContext(ctx, query, id)
	return err
}
