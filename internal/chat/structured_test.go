package chat_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/pacoricci/mock-ai-go/internal/chat"
	"github.com/pacoricci/mock-ai-go/internal/jsonschema"
)

func TestStructuredBudgetAndStreaming(t *testing.T) {
	t.Parallel()
	m, err := chat.NewMockModel("structured", chat.MockOptions{DefaultCompletionTokens: 64, MaxCompletionTokens: 128, StreamTokensPerChunk: 1})
	if err != nil {
		t.Fatal(err)
	}
	schema, err := jsonschema.Compile([]byte(`{"type":"object","properties":{"name":{"type":"string","const":"München 世界🙂"},"value":{"type":"integer"}}}`), false)
	if err != nil {
		t.Fatal(err)
	}
	request := chat.StructuredRequest{Request: chat.Request{Messages: []chat.Message{{Role: "user", Content: "example"}}}, Schema: schema, Format: "test-format"}
	full, err := m.PlanStructured(context.Background(), request)
	if err != nil || full.Incomplete || !json.Valid([]byte(full.Content)) {
		t.Fatalf("full plan = %+v, %v", full, err)
	}
	for budget := 1; budget <= full.Usage.CompletionTokens+1; budget++ {
		request.MaxCompletionTokens = budget
		plan, err := m.PlanStructured(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(full.Content, plan.Content) || !utf8.ValidString(plan.Content) || len(plan.Content) > budget*4 || plan.Usage.CompletionTokens > budget {
			t.Fatalf("budget %d: %+v", budget, plan)
		}
		if plan.Incomplete != (budget < full.Usage.CompletionTokens) || plan.Usage.TotalTokens != plan.Usage.PromptTokens+plan.Usage.CompletionTokens {
			t.Fatalf("wrong termination or usage: %+v", plan)
		}
		var streamed strings.Builder
		if err := m.StreamStructured(context.Background(), request.Request, plan, func(chunk chat.Chunk) error {
			if !utf8.Valid(chunk.Delta) {
				t.Fatal("split UTF-8 character")
			}
			streamed.Write(chunk.Delta)
			return nil
		}); err != nil || streamed.String() != plan.Content {
			t.Fatalf("stream mismatch: %q, %v", streamed.String(), err)
		}
	}
	request.MaxCompletionTokens = 0
	again, err := m.PlanStructured(context.Background(), request)
	if err != nil || !reflect.DeepEqual(full, again) {
		t.Fatalf("determinism failed: %+v, %v", again, err)
	}
}

func TestStructuredCancellationAndBackpressure(t *testing.T) {
	t.Parallel()
	m, err := chat.NewMockModel("structured", chat.MockOptions{DefaultCompletionTokens: 64, MaxCompletionTokens: 128, StreamTokensPerChunk: 1, StreamDelay: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := m.PlanStructured(context.Background(), chat.StructuredRequest{Format: "json_object"})
	if err != nil {
		t.Fatal(err)
	}
	sentinel := errors.New("consumer stopped")
	if err := m.StreamStructured(context.Background(), chat.Request{}, plan, func(chat.Chunk) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatalf("consumer failure = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.StreamStructured(ctx, chat.Request{}, plan, func(chat.Chunk) error { cancel(); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("stream cancellation = %v", err)
	}
	if _, err := m.PlanStructured(ctx, chat.StructuredRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("plan cancellation = %v", err)
	}
	if _, err := m.PlanStructured(context.Background(), chat.StructuredRequest{Request: chat.Request{MaxCompletionTokens: 129}}); !errors.Is(err, chat.ErrCompletionTokenLimit) {
		t.Fatalf("model limit = %v", err)
	}
}
