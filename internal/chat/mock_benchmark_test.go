package chat

import (
	"context"
	"testing"
	"time"
)

var (
	benchmarkResponse Response
	benchmarkRandom   uint64
)

func BenchmarkXorshift64Star(b *testing.B) {
	random := newXorshift64Star(42)
	b.ReportAllocs()
	for range b.N {
		benchmarkRandom = random.next()
	}
}

func BenchmarkMockModelComplete128Tokens(b *testing.B) {
	chatModel := newTestMockModel(b, MockOptions{
		DefaultCompletionTokens: 128,
		MaxCompletionTokens:     4096,
		StreamTokensPerChunk:    4,
	})
	request := Request{Messages: []Message{{Role: "user", Content: "benchmark prompt"}}}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		response, err := chatModel.Complete(context.Background(), request)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResponse = response
	}
}

func BenchmarkMockModelStream128Tokens(b *testing.B) {
	chatModel := newTestMockModel(b, MockOptions{
		DefaultCompletionTokens: 128,
		MaxCompletionTokens:     4096,
		StreamTokensPerChunk:    4,
	})
	request := Request{Messages: []Message{{Role: "user", Content: "benchmark prompt"}}}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := chatModel.Stream(context.Background(), request, func(Chunk) error { return nil }); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStreamDelayCalculation(b *testing.B) {
	for _, benchmark := range []struct {
		name   string
		delay  time.Duration
		jitter time.Duration
	}{
		{name: "disabled", delay: 0, jitter: 0},
		{name: "fixed", delay: 50 * time.Millisecond, jitter: 0},
		{name: "jitter", delay: 50 * time.Millisecond, jitter: 20 * time.Millisecond},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			random := newXorshift64Star(42)
			b.ReportAllocs()
			for range b.N {
				benchmarkRandom = uint64(jitteredStreamDelay(benchmark.delay, benchmark.jitter, &random))
			}
		})
	}
}
