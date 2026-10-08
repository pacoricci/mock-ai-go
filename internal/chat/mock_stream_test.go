package chat

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestMockModelStreamMatchesComplete(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    3,
	})
	request := Request{
		Messages:            []Message{{Role: "developer", Content: "Be concise"}, {Role: "user", Content: "Hello"}},
		MaxCompletionTokens: 7,
	}
	complete, err := chatModel.Complete(context.Background(), request)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	var streamed strings.Builder
	chunks := 0
	usage, err := chatModel.Stream(context.Background(), request, func(chunk Chunk) error {
		chunks++
		streamed.Write(chunk.Delta)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if streamed.String() != complete.Content {
		t.Fatalf("Stream() content = %q, want %q", streamed.String(), complete.Content)
	}
	if usage != complete.Usage {
		t.Fatalf("Stream() usage = %#v, want %#v", usage, complete.Usage)
	}
	if chunks != 3 {
		t.Fatalf("Stream() chunks = %d, want 3", chunks)
	}
}

func TestMockModelStreamStopsOnEmitError(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	wantErr := errors.New("consumer stopped")
	emitted := 0

	usage, err := chatModel.Stream(context.Background(), Request{MaxCompletionTokens: 8}, func(Chunk) error {
		emitted++
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Stream() error = %v, want %v", err, wantErr)
	}
	if emitted != 1 || usage.CompletionTokens != 2 {
		t.Fatalf("Stream() emitted = %d, usage = %#v", emitted, usage)
	}
}

func TestMockModelStreamHonorsCancellationDuringDelay(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
		StreamDelay:             time.Hour,
	})
	ctx, cancel := context.WithCancel(context.Background())

	usage, err := chatModel.Stream(ctx, Request{MaxCompletionTokens: 8}, func(Chunk) error {
		cancel()
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Stream() error = %v, want %v", err, context.Canceled)
	}
	if usage.CompletionTokens != 2 {
		t.Fatalf("Stream() usage = %#v, want 2 completion tokens", usage)
	}
}

func TestMockModelStreamHonorsCancellationDuringInitialDelay(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
		StreamInitialDelay:      time.Hour,
	})
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(10*time.Millisecond, cancel)

	emitted := 0
	usage, err := chatModel.Stream(ctx, Request{MaxCompletionTokens: 8}, func(Chunk) error {
		emitted++
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Stream() error = %v, want %v", err, context.Canceled)
	}
	if emitted != 0 || usage.CompletionTokens != 0 {
		t.Fatalf("Stream() emitted = %d, usage = %#v", emitted, usage)
	}
}

func TestJitteredStreamDelayBoundsAndClamp(t *testing.T) {
	t.Parallel()

	if got := applyJitterSample(50*time.Millisecond, 20*time.Millisecond, 0); got != 30*time.Millisecond {
		t.Fatalf("minimum delay = %s, want 30ms", got)
	}
	if got := applyJitterSample(50*time.Millisecond, 20*time.Millisecond, uint64(40*time.Millisecond)); got != 70*time.Millisecond {
		t.Fatalf("maximum delay = %s, want 70ms", got)
	}
	if got := applyJitterSample(5*time.Millisecond, 20*time.Millisecond, 0); got != 0 {
		t.Fatalf("clamped delay = %s, want 0s", got)
	}
}

func TestTemporalJitterIsRepeatableAndDecorrelated(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    1,
		StreamDelay:             50 * time.Millisecond,
		StreamDelayJitter:       20 * time.Millisecond,
	})
	seed := int64(42)
	request := Request{Messages: []Message{{Role: "user", Content: "Hello"}}, MaxCompletionTokens: 32, Seed: &seed, StreamNonce: 7}
	baseSeed := chatModel.requestSeed(request, request.MaxCompletionTokens)
	first := newXorshift64Star(baseSeed ^ temporalSeedSalt ^ mixSeed(request.StreamNonce))
	second := newXorshift64Star(baseSeed ^ temporalSeedSalt ^ mixSeed(request.StreamNonce))
	different := newXorshift64Star(baseSeed ^ temporalSeedSalt ^ mixSeed(request.StreamNonce+1))

	differenceSeen := false
	for range 32 {
		left := jitteredStreamDelay(chatModel.streamDelay, chatModel.streamDelayJitter, &first)
		right := jitteredStreamDelay(chatModel.streamDelay, chatModel.streamDelayJitter, &second)
		other := jitteredStreamDelay(chatModel.streamDelay, chatModel.streamDelayJitter, &different)
		if left != right {
			t.Fatalf("repeatable delays differ: %s and %s", left, right)
		}
		if left < 30*time.Millisecond || left > 70*time.Millisecond {
			t.Fatalf("delay = %s, want [30ms, 70ms]", left)
		}
		if left != other {
			differenceSeen = true
		}
	}
	if !differenceSeen {
		t.Fatal("different stream nonces produced the same delay sequence")
	}
}

func TestMockModelJitterDoesNotChangeContent(t *testing.T) {
	t.Parallel()

	withoutJitter := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
	})
	withJitter := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
		StreamDelayJitter:       time.Nanosecond,
	})
	seed := int64(42)
	request := Request{Messages: []Message{{Role: "user", Content: "Hello"}}, MaxCompletionTokens: 8, Seed: &seed, StreamNonce: 9}

	stream := func(model *MockModel) string {
		var content strings.Builder
		if _, err := model.Stream(context.Background(), request, func(chunk Chunk) error {
			content.Write(chunk.Delta)
			return nil
		}); err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		return content.String()
	}
	if got, want := stream(withJitter), stream(withoutJitter); got != want {
		t.Fatalf("jittered content = %q, want %q", got, want)
	}
}

func TestMockModelJitterAllocationsArePerStream(t *testing.T) {
	shortModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 2,
		MaxCompletionTokens:     128,
		StreamTokensPerChunk:    1,
		StreamDelay:             2 * time.Nanosecond,
		StreamDelayJitter:       time.Nanosecond,
	})
	longModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 128,
		MaxCompletionTokens:     128,
		StreamTokensPerChunk:    1,
		StreamDelay:             2 * time.Nanosecond,
		StreamDelayJitter:       time.Nanosecond,
	})
	request := Request{StreamNonce: 1}
	measure := func(model *MockModel) float64 {
		return testing.AllocsPerRun(100, func() {
			if _, err := model.Stream(context.Background(), request, func(Chunk) error { return nil }); err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
		})
	}
	shortAllocs := measure(shortModel)
	longAllocs := measure(longModel)
	if longAllocs > shortAllocs {
		t.Fatalf("allocations grow with chunks: short=%g long=%g", shortAllocs, longAllocs)
	}
}
