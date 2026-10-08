package chat

import (
	"errors"
	"fmt"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/model"
)

var (
	// ErrNilModel indicates that a registry received a nil model.
	ErrNilModel = errors.New("chat model is nil")
	// ErrEmptyModelID indicates that a model has no usable identifier.
	ErrEmptyModelID = errors.New("chat model ID is empty")
)

// Registry is an immutable collection of chat models indexed by ID.
// It is safe for concurrent reads after construction.
type Registry struct {
	models map[string]Model
}

// NewRegistry constructs a registry and rejects invalid or duplicate models.
func NewRegistry(models ...Model) (*Registry, error) {
	byID := make(map[string]Model, len(models))

	for _, chatModel := range models {
		if chatModel == nil {
			return nil, ErrNilModel
		}

		info := chatModel.Info()
		if strings.TrimSpace(info.ID) == "" {
			return nil, ErrEmptyModelID
		}
		if info.Kind != model.KindChat {
			return nil, fmt.Errorf("model %q has kind %q, want %q", info.ID, info.Kind, model.KindChat)
		}
		if _, exists := byID[info.ID]; exists {
			return nil, fmt.Errorf("duplicate chat model ID %q", info.ID)
		}

		byID[info.ID] = chatModel
	}

	return &Registry{models: byID}, nil
}

// Get returns the chat model registered with id.
func (r *Registry) Get(id string) (Model, bool) {
	chatModel, ok := r.models[id]
	return chatModel, ok
}

// Len returns the number of registered chat models.
func (r *Registry) Len() int {
	return len(r.models)
}
