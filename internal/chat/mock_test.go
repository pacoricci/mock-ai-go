package chat

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/model"
)

func TestMockModelCompleteIsDeterministic(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	seed := int64(42)
	request := Request{
		Messages:            []Message{{Role: "user", Content: "Hello"}},
		MaxCompletionTokens: 6,
		Seed:                &seed,
	}

	first, err := chatModel.Complete(context.Background(), request)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	second, err := chatModel.Complete(context.Background(), request)
	if err != nil {
		t.Fatalf("Complete() second call error = %v", err)
	}

	if first != second {
		t.Fatalf("Complete() responses differ: %#v and %#v", first, second)
	}
	const wantContent = "shore reef summit indigo stone crisp"
	if first.Content != wantContent {
		t.Fatalf("Complete() content = %q, want legacy golden %q", first.Content, wantContent)
	}
	if len(strings.Fields(first.Content)) != 6 {
		t.Fatalf("Complete() content = %q, want 6 generated words", first.Content)
	}
	wantUsage := Usage{PromptTokens: 9, CompletionTokens: 6, TotalTokens: 15}
	if first.Usage != wantUsage {
		t.Fatalf("Complete() usage = %#v, want %#v", first.Usage, wantUsage)
	}
	if got := chatModel.Info(); got != (model.Info{ID: "mock-chat-model", Kind: model.KindChat}) {
		t.Fatalf("Info() = %#v", got)
	}
}

func TestMockModelSeedChangesContent(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	firstSeed, secondSeed := int64(1), int64(2)
	request := Request{Messages: []Message{{Role: "user", Content: "Hello"}}, MaxCompletionTokens: 16, Seed: &firstSeed}
	first, err := chatModel.Complete(context.Background(), request)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	request.Seed = &secondSeed
	second, err := chatModel.Complete(context.Background(), request)
	if err != nil {
		t.Fatalf("Complete() second seed error = %v", err)
	}
	if first.Content == second.Content {
		t.Fatalf("different seeds produced identical content %q", first.Content)
	}
}

func TestMockModelValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		id      string
		options MockOptions
		wantErr error
	}{
		{name: "empty ID", id: " ", options: testMockOptions(), wantErr: ErrEmptyMockModelID},
		{name: "default tokens", id: "model", options: MockOptions{MaxCompletionTokens: 64, StreamTokensPerChunk: 2}, wantErr: ErrInvalidDefaultCompletionTokens},
		{name: "max below default", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 7, StreamTokensPerChunk: 2}, wantErr: ErrInvalidMaxCompletionTokens},
		{name: "absolute max", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: absoluteMaxCompletionTokens + 1, StreamTokensPerChunk: 2}, wantErr: ErrInvalidMaxCompletionTokens},
		{name: "chunk tokens", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 64}, wantErr: ErrInvalidStreamChunkTokens},
		{name: "delay", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 64, StreamTokensPerChunk: 2, StreamDelay: -1}, wantErr: ErrInvalidStreamDelay},
		{name: "jitter", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 64, StreamTokensPerChunk: 2, StreamDelayJitter: -1}, wantErr: ErrInvalidStreamDelayJitter},
		{name: "timing overflow", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 64, StreamTokensPerChunk: 2, StreamDelay: time.Duration(1<<63 - 1), StreamDelayJitter: 1}, wantErr: ErrInvalidStreamDelayJitter},
		{name: "initial delay", id: "model", options: MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 64, StreamTokensPerChunk: 2, StreamInitialDelay: -1}, wantErr: ErrInvalidStreamInitialDelay},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := NewMockModel(tt.id, tt.options)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("NewMockModel() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestMockModelRequestValidation(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	if _, err := chatModel.Complete(context.Background(), Request{MaxCompletionTokens: -1}); !errors.Is(err, ErrInvalidCompletionTokens) {
		t.Fatalf("Complete() invalid tokens error = %v", err)
	}
	if _, err := chatModel.Complete(context.Background(), Request{MaxCompletionTokens: 65}); !errors.Is(err, ErrCompletionTokenLimit) {
		t.Fatalf("Complete() token limit error = %v", err)
	}
	if _, err := chatModel.Stream(context.Background(), Request{}, nil); !errors.Is(err, ErrNilEmit) {
		t.Fatalf("Stream() nil emit error = %v", err)
	}
}

func TestEstimatePromptTokens(t *testing.T) {
	t.Parallel()

	messages := []Message{{Role: "user", Content: "🙂"}, {Role: "assistant", Content: "12345"}}
	if got, want := EstimatePromptTokens(messages), 14; got != want {
		t.Fatalf("EstimatePromptTokens() = %d, want %d", got, want)
	}
}

func newTestMockModel(t testing.TB, options MockOptions) *MockModel {
	t.Helper()
	chatModel, err := NewMockModel("mock-chat-model", options)
	if err != nil {
		t.Fatalf("NewMockModel() error = %v", err)
	}
	return chatModel
}

func testMockOptions() MockOptions {
	return MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
	}
}
