// Package openai implements the OpenAI-compatible HTTP protocol.
package openai

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

// ErrNilChatRegistry indicates that the HTTP handler has no chat models.
var ErrNilChatRegistry = errors.New("OpenAI handler chat registry is nil")

const (
	responseIDPrefix      = "resp_mock_"
	responseIDRandomBytes = 16
)

// Handler serves the OpenAI-compatible HTTP API.
type Handler struct {
	chatModels      *chat.Registry
	mux             *http.ServeMux
	now             func() time.Time
	sequence        atomic.Uint64
	chatIDNamespace string
	responseStore   responseStore
	responseTTL     time.Duration
	newResponseID   func() (string, error)
}

// NewHandler constructs an OpenAI-compatible HTTP handler.
func NewHandler(chatModels *chat.Registry, storeOptions ResponseStoreOptions) (*Handler, error) {
	if chatModels == nil {
		return nil, ErrNilChatRegistry
	}
	chatIDNamespace, err := generateResponseID()
	if err != nil {
		return nil, fmt.Errorf("create chat ID namespace: %w", err)
	}

	handler := &Handler{
		chatModels:      chatModels,
		mux:             http.NewServeMux(),
		now:             time.Now,
		responseTTL:     storeOptions.TTL,
		newResponseID:   generateResponseID,
		chatIDNamespace: strings.TrimPrefix(chatIDNamespace, responseIDPrefix),
	}
	store, err := newMemoryResponseStore(storeOptions, func() time.Time { return handler.now() })
	if err != nil {
		return nil, fmt.Errorf("create response store: %w", err)
	}
	handler.responseStore = store
	handler.mux.HandleFunc("POST /v1/chat/completions", handler.createChatCompletion)
	handler.mux.HandleFunc("POST /v1/responses", handler.createResponse)

	return handler, nil
}

// Chat completion IDs remain unique across handlers, processes, and restarts.
// Keep the existing sequence separate from the namespace so stream generation is unchanged.
func (h *Handler) chatCompletionID(sequence uint64) string {
	return fmt.Sprintf("chatcmpl-mock-%s-%d", h.chatIDNamespace, sequence)
}

func generateResponseID() (string, error) {
	payload := make([]byte, responseIDRandomBytes)
	if _, err := rand.Read(payload); err != nil {
		return "", fmt.Errorf("generate response ID: %w", err)
	}
	return responseIDPrefix + hex.EncodeToString(payload), nil
}

func isResponseID(value string) bool {
	if !strings.HasPrefix(value, responseIDPrefix) || len(value) != len(responseIDPrefix)+responseIDRandomBytes*2 {
		return false
	}
	payload := strings.TrimPrefix(value, responseIDPrefix)
	if payload != strings.ToLower(payload) {
		return false
	}
	_, err := hex.DecodeString(payload)
	return err == nil
}

// ServeHTTP serves an OpenAI-compatible HTTP request.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) resolveChatModel(w http.ResponseWriter, modelID string) (chat.Model, bool) {
	chatModel, found := h.chatModels.Get(modelID)
	if found {
		return chatModel, true
	}

	writeError(
		w,
		http.StatusNotFound,
		fmt.Sprintf("The model %q does not exist", modelID),
		"invalid_request_error",
		stringPointer("model"),
		stringPointer("model_not_found"),
	)
	return nil, false
}
