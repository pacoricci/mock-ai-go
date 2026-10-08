// Package anthropic implements a deterministic subset of the Anthropic Messages HTTP API.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

// ErrNilChatRegistry indicates that the Messages handler has no model registry.
var ErrNilChatRegistry = errors.New("anthropic handler chat registry is nil")

// Handler serves stateless Messages requests using a shared model registry.
// Models must support concurrent calls. The handler retains no conversation history.
type Handler struct {
	models   *chat.Registry
	sequence atomic.Uint64
}

// NewHandler constructs a Messages handler without adding models or taking ownership of them.
func NewHandler(models *chat.Registry) (*Handler, error) {
	if models == nil {
		return nil, ErrNilChatRegistry
	}
	return &Handler{models: models}, nil
}

type messageUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type messageResponse struct {
	ID           string       `json:"id"`
	Type         string       `json:"type"`
	Role         string       `json:"role"`
	Model        string       `json:"model"`
	Content      []any        `json:"content"`
	StopReason   *string      `json:"stop_reason"`
	StopSequence *string      `json:"stop_sequence"`
	Usage        messageUsage `json:"usage"`
}

type apiError struct {
	Type      string      `json:"type"`
	Error     errorDetail `json:"error"`
	RequestID string      `json:"request_id"`
}

type errorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// ServeHTTP validates and serves a Messages request, including protocol-specific errors.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	sequence := h.sequence.Add(1)
	requestID := fmt.Sprintf("req_mock_%016x", sequence)
	w.Header().Set("request-id", requestID)
	if r.URL.Path != "/v1/messages" {
		writeError(w, requestID, http.StatusNotFound, "not_found_error", "Unknown Messages endpoint")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, requestID, http.StatusMethodNotAllowed, "invalid_request_error", "Use POST /v1/messages")
		return
	}
	if r.Header.Get("anthropic-version") != "2023-06-01" || r.Header.Get("anthropic-beta") != "" {
		writeError(w, requestID, http.StatusBadRequest, "invalid_request_error", "anthropic-version must be 2023-06-01; beta features are unsupported")
		return
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			writeError(w, requestID, http.StatusRequestEntityTooLarge, "request_too_large", "Request exceeds the mock's 1 MiB body limit")
		} else {
			writeError(w, requestID, http.StatusBadRequest, "invalid_request_error", "Cannot read request body")
		}
		return
	}
	var request messageRequest
	if err := decodeStrict(data, &request); err != nil {
		writeError(w, requestID, http.StatusBadRequest, "invalid_request_error", "Invalid or unsupported request: "+err.Error())
		return
	}
	domain, err := prepareRequest(request)
	if err != nil {
		writeError(w, requestID, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	model, found := h.models.Get(request.Model)
	if !found {
		writeError(w, requestID, http.StatusNotFound, "not_found_error", "Unknown model: "+request.Model)
		return
	}
	domain.StreamNonce = sequence
	plan, err := model.PlanTools(r.Context(), domain)
	if err != nil {
		writeModelError(w, requestID, err)
		return
	}
	if !plan.UseText && plan.Usage.CompletionTokens > *request.MaxTokens {
		writeError(w, requestID, http.StatusBadRequest, "invalid_request_error", "max_tokens is too small for the complete synthetic tool calls; tool truncation is unsupported")
		return
	}
	response := messageResponse{
		ID: fmt.Sprintf("msg_mock_%016x", sequence), Type: "message", Role: "assistant", Model: request.Model, Content: []any{},
	}
	if request.Stream {
		streamMessage(w, r, requestID, response, domain.Request, model, plan)
		return
	}
	completeMessage(w, r, requestID, response, domain.Request, model, plan)
}

func completeMessage(w http.ResponseWriter, r *http.Request, requestID string, response messageResponse, request chat.Request, model chat.Model, plan chat.ToolPlan) {
	reason := "tool_use"
	usage := plan.Usage
	if plan.UseText {
		completion, err := model.Complete(r.Context(), request)
		if err != nil {
			writeModelError(w, requestID, err)
			return
		}
		response.Content = append(response.Content, textBlock{Type: "text", Text: completion.Content})
		usage, reason = completion.Usage, "max_tokens"
	} else {
		for _, call := range plan.Calls {
			var input map[string]any
			if err := json.Unmarshal([]byte(call.Arguments), &input); err != nil || input == nil {
				writeError(w, requestID, http.StatusInternalServerError, "api_error", "Model produced invalid tool input")
				return
			}
			response.Content = append(response.Content, toolUseBlock{Type: "tool_use", ID: toolID(call), Name: call.Name, Input: input})
		}
	}
	response.StopReason = &reason
	response.Usage = toUsage(usage)
	writeJSON(w, http.StatusOK, response)
}

func toolID(call chat.ToolCall) string {
	return "toolu_" + call.CallID
}

func toUsage(usage chat.Usage) messageUsage {
	return messageUsage{InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens}
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, requestID string, status int, kind, message string) {
	writeJSON(w, status, apiError{Type: "error", Error: errorDetail{Type: kind, Message: message}, RequestID: requestID})
}

func isCanceled(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func writeModelError(w http.ResponseWriter, requestID string, err error) {
	if isCanceled(err) {
		return
	}
	if errors.Is(err, chat.ErrCompletionTokenLimit) || errors.Is(err, chat.ErrInvalidCompletionTokens) {
		writeError(w, requestID, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	writeError(w, requestID, http.StatusInternalServerError, "api_error", "Model failed to create a message")
}
