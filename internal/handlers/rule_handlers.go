package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/service"
)

// RuleHandler реализует обработчики CRUD операций для сущности "Rule".
type RuleHandler struct {
	Service *service.RuleService
}

// NewRuleHandler создает новый обработчик правил.
func NewRuleHandler(service *service.RuleService) *RuleHandler {
	return &RuleHandler{Service: service}
}

// CreateRule обрабатывает создание нового правила.
// @Summary      Создать правило
// @Description  Принимает JSON с данными правила и сохраняет его в базе данных
// @Tags         rules
// @Accept       json
// @Produce      plain
// @Param        rule body models.Rule true "Данные правила"
// @Success      201 {string} int "ID созданного правила"
// @Failure      400 {string} string "Ошибка декодирования"
// @Failure      500 {string} string "Ошибка сервера"
func (h *RuleHandler) CreateRule(w http.ResponseWriter, r *http.Request) {
	var rule models.Rule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(err.Error()))
		return
	}
	id, err := h.Service.CreateRule(context.Background(), rule)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(http.StatusCreated)
	w.Write([]byte(strconv.Itoa(id)))
}

// GetRule возвращает правило по ID.
// @Summary      Получить правило
// @Description  Возвращает правило по идентификатору
// @Tags         rules
// @Produce      json
// @Param        id query int true "ID правила"
// @Success      200 {object} models.Rule
// @Failure      400 {string} string "Некорректный ID"
// @Failure      404 {string} string "Правило не найдено"
func (h *RuleHandler) GetRule(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("invalid id"))
		return
	}
	rule, err := h.Service.GetRuleByID(context.Background(), id)
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(err.Error()))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rule)
}

// GetAllRules возвращает список всех правил.
// @Summary      Получить все правила
// @Description  Возвращает список всех правил
// @Tags         rules
// @Produce      json
// @Success      200 {array} models.Rule
// @Failure      500 {string} string "Ошибка сервера"
func (h *RuleHandler) GetAllRules(w http.ResponseWriter, r *http.Request) {
	rules, err := h.Service.GetAllRules(context.Background())
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(err.Error()))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rules)
}

// UpdateRule обновляет существующее правило.
// @Summary      Обновить правило
// @Description  Принимает JSON с обновленными данными правила
// @Tags         rules
// @Accept       json
// @Produce      plain
// @Param        rule body models.Rule true "Данные правила"
// @Success      200 {string} string "OK"
// @Failure      400 {string} string "Ошибка декодирования"
// @Failure      500 {string} string "Ошибка сервера"
func (h *RuleHandler) UpdateRule(w http.ResponseWriter, r *http.Request) {
	var rule models.Rule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(err.Error()))
		return
	}
	if err := h.Service.UpdateRule(context.Background(), rule); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(http.StatusOK)
}

// DeleteRule удаляет правило по ID.
// @Summary      Удалить правило
// @Description  Удаляет правило по идентификатору
// @Tags         rules
// @Param        id query int true "ID правила"
// @Success      200 {string} string "OK"
// @Failure      400 {string} string "Некорректный ID"
// @Failure      500 {string} string "Ошибка сервера"
func (h *RuleHandler) DeleteRule(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("invalid id"))
		return
	}
	if err := h.Service.DeleteRule(context.Background(), id); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(http.StatusOK)
}
