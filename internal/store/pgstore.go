package store

import (
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/lib/pq"
	"github.com/katakuxiko/Diplom/internal/model"
)

type PgStore struct {
	db *sql.DB
}

func NewPgStore(conn string) (*PgStore, error) {
	db, err := sql.Open("postgres", conn)
	if err != nil {
		return nil, err
	}
	if err := ensureSchema(db); err != nil {
		return nil, err
	}
	return &PgStore{db: db}, nil
}

func (s *PgStore) Add(doc string, c model.Chunk, v []float32) error {
	vec := floatsToPgVectorLiteral(v)
	_, err := s.db.Exec(`
		INSERT INTO chunks (doc_name, chunk_id, text, embedding)
		VALUES ($1, $2, $3, $4::vector)
	`, doc, c.ID, c.Text, vec)
	return err
}

func (s *PgStore) Search(q []float32, k int) ([]model.Chunk, error) {
	vec := floatsToPgVectorLiteral(q)
	rows, err := s.db.Query(`
		SELECT chunk_id, text
		FROM chunks
		ORDER BY embedding <-> $1::vector
		LIMIT $2
	`, vec, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var res []model.Chunk
	for rows.Next() {
		var c model.Chunk
		if err := rows.Scan(&c.ID, &c.Text); err != nil {
			return nil, err
		}
		res = append(res, c)
	}
	return res, rows.Err()
}

func floatsToPgVectorLiteral(v []float32) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, f := range v {
		sb.WriteString(fmt.Sprintf("%.6f", float64(f)))
		if i < len(v)-1 {
			sb.WriteString(",")
		}
	}
	sb.WriteString("]")
	return sb.String()
}
