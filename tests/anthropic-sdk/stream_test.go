package sdk_test

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	mockanthropic "github.com/pacoricci/mock-ai-go/internal/anthropic"
	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type interruptedModel struct {
	chat.Model
	fail    bool
	stopped chan struct{}
}

func (model *interruptedModel) afterChunk(ctx context.Context) error {
	if model.fail {
		return errors.New("controlled failure after a chunk")
	}
	<-ctx.Done()
	return ctx.Err()
}

// Stream observes cancellation or fails immediately after the first text chunk.
func (model *interruptedModel) Stream(ctx context.Context, request chat.Request, emit chat.Emit) (chat.Usage, error) {
	defer close(model.stopped)
	return model.Model.Stream(ctx, request, func(chunk chat.Chunk) error {
		if err := emit(chunk); err != nil {
			return err
		}
		return model.afterChunk(ctx)
	})
}

// StreamTools observes cancellation or fails immediately after the first tool chunk.
func (model *interruptedModel) StreamTools(ctx context.Context, request chat.Request, plan chat.ToolPlan, emit chat.ToolEmit) error {
	defer close(model.stopped)
	return model.Model.StreamTools(ctx, request, plan, func(chunk chat.ToolChunk) error {
		if err := emit(chunk); err != nil {
			return err
		}
		return model.afterChunk(ctx)
	})
}

func interruptedClient(t *testing.T, fail bool) (anthropic.Client, <-chan struct{}) {
	t.Helper()
	base, err := chat.NewMockModel("mock", chat.MockOptions{DefaultCompletionTokens: 32, MaxCompletionTokens: 256, StreamTokensPerChunk: 1})
	if err != nil {
		t.Fatal(err)
	}
	model := &interruptedModel{Model: base, fail: fail, stopped: make(chan struct{})}
	registry, err := chat.NewRegistry(model)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := mockanthropic.NewHandler(registry)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return anthropic.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("dummy"), option.WithMaxRetries(0)), model.stopped
}

func TestStreamInterruption(t *testing.T) {
	for _, content := range []string{"text", "tools"} {
		for _, mode := range []string{"cancel", "close", "deadline", "failure"} {
			t.Run(content+"/"+mode, func(t *testing.T) {
				client, stopped := interruptedClient(t, mode == "failure")
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				streamCtx, stopStream := context.WithCancel(ctx)
				if mode == "deadline" {
					stopStream()
					streamCtx, stopStream = context.WithTimeout(ctx, time.Second)
				}
				defer stopStream()
				params := textParams()
				if content == "tools" {
					params = toolParams()
				}
				stream := client.Messages.NewStreaming(streamCtx, params)
				defer func() { _ = stream.Close() }()
				for stream.Next() {
					if stream.Current().Type == "content_block_delta" {
						break
					}
				}
				if stream.Current().Type != "content_block_delta" {
					t.Fatalf("no content before interruption: %v", stream.Err())
				}
				switch mode {
				case "cancel":
					stopStream()
				case "close":
					if err := stream.Close(); err != nil {
						t.Fatal(err)
					}
				}
				if stream.Next() {
					t.Fatalf("interrupted stream emitted %q", stream.Current().Type)
				}
				switch mode {
				case "cancel":
					if !errors.Is(stream.Err(), context.Canceled) {
						t.Fatalf("cancel error = %v", stream.Err())
					}
				case "deadline":
					if !errors.Is(stream.Err(), context.DeadlineExceeded) {
						t.Fatalf("deadline error = %v", stream.Err())
					}
				case "failure":
					if stream.Err() == nil || !strings.Contains(stream.Err().Error(), "api_error") {
						t.Fatalf("in-stream error = %v", stream.Err())
					}
				}
				select {
				case <-stopped:
				case <-ctx.Done():
					t.Fatal("model did not stop")
				}
			})
		}
	}
}
