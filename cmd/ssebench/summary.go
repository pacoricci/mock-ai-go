package main

import (
	"fmt"
	"io"

	"github.com/pacoricci/mock-ai-go/internal/loadgen"
)

func writeSummary(w io.Writer, result loadgen.Result) error {
	_, err := fmt.Fprintf(w, "Scenario: %s\n"+
		"Duration: %.3f s\n"+
		"Streams: attempted=%d connected=%d completed=%d canceled=%d errors=%d\n"+
		"Validation: invalid_events=%d invalid_usage=%d\n"+
		"Throughput: %.2f events/s, %.2f bytes/s\n"+
		"Open latency: %s\n"+
		"Content gap: %s\n",
		result.Scenario.Name, result.ElapsedSeconds,
		result.Attempted, result.Connected, result.Completed, result.Canceled, result.Errors,
		result.InvalidEvents, result.InvalidUsage,
		result.EventsPerSecond, result.BytesPerSecond,
		formatLatency(result.OpenLatencyMillis), formatLatency(result.ContentGapMillis))
	return err
}

func formatLatency(summary loadgen.LatencySummary) string {
	if summary.Count == 0 {
		return "n/a (no samples)"
	}
	return fmt.Sprintf("p50=%.3f ms p95=%.3f ms p99=%.3f ms (samples=%d)",
		summary.P50, summary.P95, summary.P99, summary.Count)
}
