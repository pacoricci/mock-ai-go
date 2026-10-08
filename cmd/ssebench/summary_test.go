package main

import (
	"bytes"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/loadgen"
)

func TestSummaryReportsCountsRatesAndPercentiles(t *testing.T) {
	result := loadgen.Result{
		Scenario: loadgen.Scenario{Name: "sample"}, ElapsedSeconds: 2,
		Attempted: 10, Connected: 9, Completed: 6, Canceled: 2, Errors: 2,
		InvalidEvents: 1, InvalidUsage: 1, EventsPerSecond: 12.5, BytesPerSecond: 1024,
		OpenLatencyMillis: loadgen.LatencySummary{Count: 9, P50: 1.25, P95: 2.5, P99: 3.75},
		ContentGapMillis:  loadgen.LatencySummary{Count: 20, P50: 10, P95: 15, P99: 20},
	}
	var output bytes.Buffer
	if err := writeSummary(&output, result); err != nil {
		t.Fatal(err)
	}
	const want = "Scenario: sample\n" +
		"Duration: 2.000 s\n" +
		"Streams: attempted=10 connected=9 completed=6 canceled=2 errors=2\n" +
		"Validation: invalid_events=1 invalid_usage=1\n" +
		"Throughput: 12.50 events/s, 1024.00 bytes/s\n" +
		"Open latency: p50=1.250 ms p95=2.500 ms p99=3.750 ms (samples=9)\n" +
		"Content gap: p50=10.000 ms p95=15.000 ms p99=20.000 ms (samples=20)\n"
	if output.String() != want {
		t.Fatalf("summary = %q, want %q", output.String(), want)
	}
}
