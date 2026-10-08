package loadgen

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMetricCollector(t *testing.T) {
	t.Parallel()

	value := 1
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"values":{"goroutines":%d,"heap":%d}}`, value, value*10)
		value++
	}))
	defer server.Close()

	collector := newMetricCollector(server.URL)
	if err := collector.initialSample(context.Background()); err != nil {
		t.Fatalf("initialSample() error = %v", err)
	}
	collector.finalSample()
	summary := collector.summary()
	if summary.Samples != 2 || summary.Start["heap"] != 10 || summary.End["heap"] != 20 || summary.Peak["heap"] != 20 || summary.Delta["heap"] != 10 {
		t.Fatalf("summary = %#v", summary)
	}
}
