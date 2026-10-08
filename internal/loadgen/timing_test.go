package loadgen_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/loadgen"
)

func TestRunFinishesTimingBeforeFinalMetrics(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var lastScrapeStarted time.Time
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			mu.Lock()
			lastScrapeStarted = time.Now()
			mu.Unlock()
			_, _ = fmt.Fprint(w, `{"values":{"heap":10}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"amber\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	result, err := loadgen.Run(context.Background(), loadgen.Scenario{
		Name:        "timing",
		URL:         server.URL,
		MetricsURL:  server.URL + "/metrics",
		Connections: 1,
		Duration:    loadgen.Duration(time.Second),
		Request: loadgen.Request{
			Model:               "mock-chat-model",
			MaxCompletionTokens: 1,
		},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Completed != 1 || result.Errors != 0 || result.Events == 0 {
		t.Fatalf("unexpected stream result: %#v", result)
	}
	if result.ServerMetrics == nil || result.ServerMetrics.Samples < 2 || result.ServerMetrics.ScrapeErrors != 0 {
		t.Fatalf("initial and final metrics were not collected: %#v", result.ServerMetrics)
	}
	mu.Lock()
	finalScrapeStarted := lastScrapeStarted
	mu.Unlock()
	if !result.FinishedAt.Before(finalScrapeStarted) {
		t.Fatalf("workload finished at %s, after final metrics started at %s", result.FinishedAt, finalScrapeStarted)
	}
	if result.ElapsedSeconds <= 0 || result.ElapsedSeconds >= finalScrapeStarted.Sub(result.StartedAt).Seconds() {
		t.Fatalf("elapsed time %g includes final metrics collection", result.ElapsedSeconds)
	}
	if result.EventsPerSecond != float64(result.Events)/result.ElapsedSeconds ||
		result.BytesPerSecond != float64(result.Bytes)/result.ElapsedSeconds {
		t.Fatal("throughput does not use workload elapsed time")
	}
}
