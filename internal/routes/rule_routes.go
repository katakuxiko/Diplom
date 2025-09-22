package routes

import (
	"net/http"

	"github.com/katakuxiko/Diplom/internal/handlers"
)

func RegisterRuleRoutes(mux *http.ServeMux, ruleHandler *handlers.RuleHandler) {
	mux.HandleFunc("/rules/create", ruleHandler.CreateRule)
	mux.HandleFunc("/rules/get", ruleHandler.GetRule)
	mux.HandleFunc("/rules/all", ruleHandler.GetAllRules)
	mux.HandleFunc("/rules/update", ruleHandler.UpdateRule)
	mux.HandleFunc("/rules/delete", ruleHandler.DeleteRule)
}
