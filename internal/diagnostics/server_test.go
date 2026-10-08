package diagnostics

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRuntimeMetrics(t *testing.T) {
	t.Parallel()

	request := httptest.NewRequest(http.MethodGet, "/debug/metrics", nil)
	recorder := httptest.NewRecorder()
	NewHandler().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var snapshot runtimeMetricSnapshot
	if err := json.NewDecoder(recorder.Body).Decode(&snapshot); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if snapshot.Timestamp.IsZero() {
		t.Fatal("metrics timestamp is zero")
	}
	if len(snapshot.Values) != len(runtimeMetricNames) {
		t.Fatalf("metric count = %d, want %d", len(snapshot.Values), len(runtimeMetricNames))
	}
}

func TestPprofIndex(t *testing.T) {
	t.Parallel()

	request := httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	recorder := httptest.NewRecorder()
	NewHandler().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
}

func TestDiagnosticsRoutesRejectOtherMethods(t *testing.T) {
	t.Parallel()

	request := httptest.NewRequest(http.MethodPost, "/debug/metrics", nil)
	recorder := httptest.NewRecorder()
	NewHandler().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusMethodNotAllowed)
	}
}
