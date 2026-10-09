package chat_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func TestSampledTextLength(t *testing.T) {
	t.Parallel()
	model, err := chat.NewMockModel("mock", chat.MockOptions{
		DefaultCompletionTokens: 32, MaxCompletionTokens: 4096,
		MinTextTokens: 128, MaxTextTokens: 512, StreamTokensPerChunk: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, limit := range []int{0, 1, 16, 127, 128, 200, 512, 1024} {
		lower, upper := 128, 512
		if limit > 0 {
			upper = min(upper, limit)
			lower = min(lower, upper)
		}
		lengths := make(map[int]bool)
		for seed := int64(0); seed < 64; seed++ {
			request := chat.Request{Messages: []chat.Message{{Role: "user", Content: "Explain HTTP streaming"}}, MaxCompletionTokens: limit, Seed: &seed}
			first, err := model.Complete(context.Background(), request)
			if err != nil {
				t.Fatal(err)
			}
			count := len(strings.Fields(first.Content))
			if count < lower || count > upper || first.Usage.CompletionTokens != count {
				t.Fatalf("limit %d: count %d, usage %+v, want %d..%d", limit, count, first.Usage, lower, upper)
			}
			lengths[count] = true
			var streamed strings.Builder
			usage, err := model.Stream(context.Background(), request, func(chunk chat.Chunk) error {
				streamed.Write(chunk.Delta)
				return nil
			})
			if err != nil || streamed.String() != first.Content || usage != first.Usage {
				t.Fatalf("seed %d, limit %d: stream differs from completion: %v", seed, limit, err)
			}
			second, err := model.Complete(context.Background(), request)
			if err != nil || second != first {
				t.Fatalf("seed %d, limit %d: repeated completion differs: %v", seed, limit, err)
			}
		}
		if lower != upper && len(lengths) < 2 {
			t.Fatalf("limit %d: different seeds produced only one length", limit)
		}
	}
	for _, limit := range []int{-1, 4097} {
		_, err := model.Complete(context.Background(), chat.Request{MaxCompletionTokens: limit})
		want := chat.ErrInvalidCompletionTokens
		if limit > 4096 {
			want = chat.ErrCompletionTokenLimit
		}
		if !errors.Is(err, want) {
			t.Fatalf("limit %d: error %v, want %v", limit, err, want)
		}
	}
}

func TestUnseededTextLengthVaries(t *testing.T) {
	t.Parallel()
	model, err := chat.NewMockModel("mock", chat.MockOptions{
		DefaultCompletionTokens: 32, MaxCompletionTokens: 4096,
		MinTextTokens: 128, MaxTextTokens: 512, StreamTokensPerChunk: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := chat.Request{Messages: []chat.Message{{Role: "user", Content: "Hello"}}}
	lengths := make(map[int]bool)
	for range 64 {
		var content strings.Builder
		usage, err := model.Stream(context.Background(), request, func(chunk chat.Chunk) error {
			content.Write(chunk.Delta)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		count := len(strings.Fields(content.String()))
		if count < 128 || count > 512 || usage.CompletionTokens != count {
			t.Fatalf("count %d, usage %+v", count, usage)
		}
		lengths[count] = true
	}
	if len(lengths) < 2 {
		t.Fatal("repeated unseeded requests produced only one length")
	}
}

func TestTextTokenRangeValidation(t *testing.T) {
	t.Parallel()
	for _, bounds := range [][2]int{{-1, 512}, {0, 512}, {128, 0}, {512, 128}, {128, 4097}} {
		_, err := chat.NewMockModel("mock", chat.MockOptions{
			DefaultCompletionTokens: 32, MaxCompletionTokens: 4096,
			MinTextTokens: bounds[0], MaxTextTokens: bounds[1], StreamTokensPerChunk: 4,
		})
		if !errors.Is(err, chat.ErrInvalidTextTokenRange) {
			t.Fatalf("bounds %v: error %v, want invalid range", bounds, err)
		}
	}
}
