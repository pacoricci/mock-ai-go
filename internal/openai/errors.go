package openai

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type errorResponse struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

func writeError(w http.ResponseWriter, status int, message, errorType string, param, code *string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	_ = json.NewEncoder(w).Encode(errorResponse{Error: apiError{
		Message: message,
		Type:    errorType,
		Param:   param,
		Code:    code,
	}})
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func writeModelError(w http.ResponseWriter, err error, tokenParam, failureMessage string) {
	switch {
	case errors.Is(err, chat.ErrInvalidCompletionTokens):
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", stringPointer(tokenParam), stringPointer("invalid_value"))
	case errors.Is(err, chat.ErrCompletionTokenLimit):
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", stringPointer(tokenParam), stringPointer("limit_exceeded"))
	default:
		writeError(w, http.StatusInternalServerError, failureMessage, "server_error", nil, nil)
	}
}

func stringPointer(value string) *string {
	return &value
}
