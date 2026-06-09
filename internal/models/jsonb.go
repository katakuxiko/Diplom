package models

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
)

// JSONB представляет JSONB поле PostgreSQL
type JSONB map[string]interface{}

// Value реализует интерфейс driver.Valuer для записи в БД
func (j JSONB) Value() (driver.Value, error) {
	if j == nil {
		return nil, nil
	}
	return json.Marshal(j)
}

// Scan реализует интерфейс sql.Scanner для чтения из БД
func (j *JSONB) Scan(value interface{}) error {
	if value == nil {
		*j = nil
		return nil
	}

	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}

	result := make(map[string]interface{})
	if err := json.Unmarshal(bytes, &result); err != nil {
		return err
	}

	*j = result
	return nil
}
