package chat

import (
	"context"
	"errors"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/model"
)

func TestRegistry(t *testing.T) {
	t.Parallel()

	chatModel := stubModel{info: model.Info{ID: "mock-chat-model", Kind: model.KindChat}}
	registry, err := NewRegistry(chatModel)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}

	got, ok := registry.Get("mock-chat-model")
	if !ok {
		t.Fatal("Get() did not find registered model")
	}
	if got.Info() != chatModel.Info() {
		t.Fatalf("Get().Info() = %#v, want %#v", got.Info(), chatModel.Info())
	}
	if registry.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", registry.Len())
	}
	if _, ok := registry.Get("missing"); ok {
		t.Fatal("Get() found an unregistered model")
	}
}

func TestRegistryRejectsInvalidModels(t *testing.T) {
	t.Parallel()

	valid := stubModel{info: model.Info{ID: "duplicate", Kind: model.KindChat}}

	tests := []struct {
		name   string
		models []Model
		isErr  error
	}{
		{name: "nil", models: []Model{nil}, isErr: ErrNilModel},
		{name: "empty ID", models: []Model{stubModel{info: model.Info{Kind: model.KindChat}}}, isErr: ErrEmptyModelID},
		{name: "duplicate ID", models: []Model{valid, valid}},
		{name: "wrong kind", models: []Model{stubModel{info: model.Info{ID: "embedding", Kind: "embedding"}}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := NewRegistry(tt.models...)
			if err == nil {
				t.Fatal("NewRegistry() error = nil, want non-nil")
			}
			if tt.isErr != nil && !errors.Is(err, tt.isErr) {
				t.Fatalf("NewRegistry() error = %v, want %v", err, tt.isErr)
			}
		})
	}
}

type stubModel struct {
	info model.Info
}

func (m stubModel) Info() model.Info {
	return m.info
}

func (stubModel) PlanTools(context.Context, ToolRequest) (ToolPlan, error) {
	return ToolPlan{UseText: true}, nil
}

func (stubModel) Complete(context.Context, Request) (Response, error) {
	return Response{}, nil
}

func (stubModel) Stream(context.Context, Request, Emit) (Usage, error) {
	return Usage{}, nil
}

func (stubModel) StreamTools(context.Context, Request, ToolPlan, ToolEmit) error { return nil }
