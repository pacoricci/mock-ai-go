package chat_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func toolStreamModel(t *testing.T, initial, delay time.Duration) *chat.MockModel {
	t.Helper()
	model, err := chat.NewMockModel("test", chat.MockOptions{DefaultCompletionTokens: 16, MaxCompletionTokens: 64, StreamTokensPerChunk: 1, StreamInitialDelay: initial, StreamDelay: delay})
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestToolStreamReassemblesPlanAndPreservesUTF8(t *testing.T) {
	t.Parallel()
	model := toolStreamModel(t, 0, 0)
	plan := chat.ToolPlan{Calls: []chat.ToolCall{{Arguments: `{"city":"café 世界🙂"}`}, {Arguments: `{"temperature":21}`}}}
	for _, nonce := range []uint64{0, 123} {
		parts := make([]strings.Builder, 2)
		first, last := make([]int, 2), make([]int, 2)
		count := 0
		err := model.StreamTools(context.Background(), chat.Request{StreamNonce: nonce}, plan, func(chunk chat.ToolChunk) error {
			if !utf8.ValidString(chunk.Delta) {
				t.Fatalf("invalid UTF-8: %q", chunk.Delta)
			}
			if chunk.First {
				first[chunk.Index]++
			}
			if chunk.Last {
				last[chunk.Index]++
			}
			parts[chunk.Index].WriteString(chunk.Delta)
			count++
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if count <= len(plan.Calls) {
			t.Fatalf("arguments were not fragmented: %d", count)
		}
		for index, call := range plan.Calls {
			if parts[index].String() != call.Arguments || first[index] != 1 || last[index] != 1 {
				t.Fatalf("call %d: %q, first=%d last=%d", index, parts[index].String(), first[index], last[index])
			}
		}
	}
}

func TestToolStreamPacingAndCancellation(t *testing.T) {
	t.Parallel()
	const delay = 15 * time.Millisecond
	model := toolStreamModel(t, delay, delay)
	plan := chat.ToolPlan{Calls: []chat.ToolCall{{Arguments: `{"x":"long arguments"}`}}}
	start := time.Now()
	previous := start
	chunks := 0
	err := model.StreamTools(context.Background(), chat.Request{}, plan, func(chat.ToolChunk) error {
		now := time.Now()
		if now.Sub(previous) < delay {
			t.Fatalf("fragment arrived without pacing: %s", now.Sub(previous))
		}
		previous = now
		chunks++
		return nil
	})
	if err != nil || chunks < 2 {
		t.Fatalf("chunks=%d err=%v", chunks, err)
	}
	for _, initial := range []bool{true, false} {
		t.Run(map[bool]string{true: "initial", false: "gap"}[initial], func(t *testing.T) {
			initialDelay, gap := time.Duration(0), time.Hour
			if initial {
				initialDelay, gap = time.Hour, 0
			}
			paced := toolStreamModel(t, initialDelay, gap)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if initial {
				cancel()
			}
			seen := 0
			err := paced.StreamTools(ctx, chat.Request{}, plan, func(chat.ToolChunk) error { seen++; cancel(); return nil })
			if !errors.Is(err, context.Canceled) || seen > 1 || (initial && seen != 0) {
				t.Fatalf("chunks=%d err=%v", seen, err)
			}
		})
	}
}

func TestToolStreamStopsOnConsumerError(t *testing.T) {
	t.Parallel()
	model := toolStreamModel(t, 0, 0)
	failure := errors.New("closed writer")
	count := 0
	err := model.StreamTools(context.Background(), chat.Request{}, chat.ToolPlan{Calls: []chat.ToolCall{{Arguments: `{"x":123456}`}}}, func(chat.ToolChunk) error { count++; return failure })
	if !errors.Is(err, failure) || count != 1 {
		t.Fatalf("count=%d err=%v", count, err)
	}
	if err := model.StreamTools(context.Background(), chat.Request{}, chat.ToolPlan{}, nil); !errors.Is(err, chat.ErrNilEmit) {
		t.Fatalf("nil callback: %v", err)
	}
}

func TestToolStreamDeadlineDuringPacing(t *testing.T) {
	t.Parallel()
	for _, initial := range []bool{true, false} {
		t.Run(map[bool]string{true: "initial", false: "gap"}[initial], func(t *testing.T) {
			initialDelay, gap := time.Duration(0), time.Hour
			if initial {
				initialDelay, gap = time.Hour, 0
			}
			model := toolStreamModel(t, initialDelay, gap)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancel()
			chunks := 0
			err := model.StreamTools(ctx, chat.Request{}, chat.ToolPlan{Calls: []chat.ToolCall{{Arguments: `{"x":"long value"}`}}}, func(chat.ToolChunk) error { chunks++; return nil })
			if !errors.Is(err, context.DeadlineExceeded) || (initial && chunks != 0) || (!initial && chunks != 1) {
				t.Fatalf("chunks=%d err=%v", chunks, err)
			}
		})
	}
}
