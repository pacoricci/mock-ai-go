package loadgen

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestMarshalResponseRequestIncludesStore(t *testing.T) {
	t.Parallel()

	store := false
	scenario := Scenario{
		Protocol: ProtocolResponses,
		Request: Request{
			Model:               "mock-chat-model",
			Prompt:              "hello",
			MaxCompletionTokens: 64,
			Store:               &store,
		},
	}
	value, err := marshalRequest(scenario)
	if err != nil {
		t.Fatalf("marshalRequest() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(value, &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if stored, ok := request["store"].(bool); !ok || stored {
		t.Fatalf("store = %#v, want false", request["store"])
	}
}

func TestRunCompletesSSEStreams(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"amber\"}}],\"usage\":null}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	scenario := Scenario{
		Name:        "test-complete",
		URL:         server.URL,
		Connections: 4,
		RampUp:      Duration(10 * time.Millisecond),
		Duration:    Duration(time.Second),
		Request: Request{
			Model:               "mock-chat-model",
			Prompt:              "hello",
			MaxCompletionTokens: 8,
			Seed:                42,
			IncludeUsage:        true,
		},
	}
	result, err := Run(context.Background(), scenario)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Attempted != 4 || result.Connected != 4 || result.Completed != 4 {
		t.Fatalf("connection counts = attempted %d connected %d completed %d", result.Attempted, result.Connected, result.Completed)
	}
	if result.Events != 12 || result.UsageChunks != 4 {
		t.Fatalf("event counts = events %d usage %d", result.Events, result.UsageChunks)
	}
	if result.Errors != 0 || result.InvalidEvents != 0 || result.InvalidUsage != 0 {
		t.Fatalf("validation counts = errors %d invalid events %d invalid usage %d", result.Errors, result.InvalidEvents, result.InvalidUsage)
	}
	if result.OpenLatencyMillis.Count != 4 || result.Bytes == 0 {
		t.Fatalf("metrics = latency %#v bytes %d", result.OpenLatencyMillis, result.Bytes)
	}
}

func TestRunCompletesResponsesSSEStreams(t *testing.T) {
	t.Parallel()

	requestBody := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		requestBody <- body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"delta\":\"amber\"}\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(5 * time.Millisecond)
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"delta\":\" orbit\"}\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":2,\"total_tokens\":4}}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	result, err := Run(context.Background(), Scenario{
		Name:        "test-responses",
		Protocol:    ProtocolResponses,
		URL:         server.URL,
		Connections: 1,
		Duration:    Duration(time.Second),
		Request: Request{
			Model:               "mock-chat-model",
			Prompt:              "hello",
			MaxCompletionTokens: 16,
			Seed:                42,
			IncludeUsage:        true,
		},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Completed != 1 || result.Events != 4 || result.UsageChunks != 1 || result.ContentGapMillis.Count != 1 ||
		result.Errors != 0 || result.InvalidEvents != 0 || result.InvalidUsage != 0 {
		t.Fatalf("result = %#v", result)
	}

	body := <-requestBody
	if body["input"] != "hello" || body["max_output_tokens"] != float64(16) || body["stream"] != true {
		t.Fatalf("request body = %#v", body)
	}
	for _, absent := range []string{"messages", "max_completion_tokens", "seed", "stream_options"} {
		if _, ok := body[absent]; ok {
			t.Fatalf("request body contains %q: %#v", absent, body)
		}
	}
}

func TestRunCancelsLongLivedStreams(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{}}]}\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer server.Close()

	result, err := Run(context.Background(), Scenario{
		Name:               "test-cancel",
		URL:                server.URL,
		Connections:        2,
		Duration:           Duration(time.Second),
		ConnectionLifetime: Duration(20 * time.Millisecond),
		Request: Request{
			Model:               "mock-chat-model",
			MaxCompletionTokens: 8,
		},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Connected != 2 || result.Canceled != 2 || result.Errors != 0 {
		t.Fatalf("result = connected %d canceled %d errors %d", result.Connected, result.Canceled, result.Errors)
	}
}

func TestRunMeasuresContentGaps(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"amber\"}}]}\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(5 * time.Millisecond)
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" orbit\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	result, err := Run(context.Background(), Scenario{
		Name:        "test-content-gaps",
		URL:         server.URL,
		Connections: 1,
		Duration:    Duration(time.Second),
		Request: Request{
			Model:               "mock-chat-model",
			MaxCompletionTokens: 2,
		},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.ContentGapMillis.Count != 1 || result.ContentGapMillis.Min < 1 {
		t.Fatalf("content gap summary = %#v", result.ContentGapMillis)
	}
}

func TestRunRejectsUnsuccessfulResponses(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name        string
		events      []string
		wantErrors  uint64
		wantInvalid uint64
		wantSample  string
	}{
		{
			name:       "failed",
			events:     []string{`{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"Generation failed"}}}`},
			wantErrors: 1, wantSample: "server_error Generation failed",
		},
		{
			name: "error then failed",
			events: []string{
				`{"type":"error","code":"response_store_capacity_exceeded","message":"Response exceeds capacity"}`,
				`{"type":"response.failed","response":{"status":"failed","error":{"code":"response_store_capacity_exceeded","message":"Response exceeds capacity"}}}`,
			},
			wantErrors: 1, wantSample: "response_store_capacity_exceeded Response exceeds capacity",
		},
		{
			name:       "incomplete",
			events:     []string{`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`},
			wantErrors: 1, wantSample: "max_output_tokens",
		},
		{
			name:       "error without terminal response",
			events:     []string{`{"type":"error","code":"server_error","message":"Generation failed"}`},
			wantErrors: 1, wantInvalid: 1, wantSample: "server_error Generation failed",
		},
		{name: "missing terminal response", wantInvalid: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var stream strings.Builder
			events := append([]string{`{"type":"response.created","response":{"status":"in_progress"}}`}, test.events...)
			for sequence, raw := range events {
				var event map[string]any
				if err := json.Unmarshal([]byte(raw), &event); err != nil {
					t.Fatal(err)
				}
				event["sequence_number"] = sequence
				payload, err := json.Marshal(event)
				if err != nil {
					t.Fatal(err)
				}
				fmt.Fprintf(&stream, "event: %s\ndata: %s\n\n", event["type"], payload)
			}
			stream.WriteString("data: [DONE]\n\n")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, stream.String())
			}))
			defer server.Close()

			result, err := Run(context.Background(), Scenario{
				Name: test.name, Protocol: ProtocolResponses, URL: server.URL,
				Connections: 1, Duration: Duration(time.Second),
				Request: Request{Model: "mock-chat-model", MaxCompletionTokens: 16},
			})
			if err != nil {
				t.Fatal(err)
			}
			if result.Completed != 0 || result.Canceled != 0 || result.Errors != test.wantErrors ||
				result.InvalidEvents != test.wantInvalid || result.InvalidUsage != 0 {
				t.Fatalf("unexpected result: %+v", result)
			}
			if test.wantSample != "" && (len(result.ErrorSamples) != 1 || !strings.Contains(result.ErrorSamples[0], test.wantSample)) {
				t.Fatalf("error samples = %v, want %q", result.ErrorSamples, test.wantSample)
			}
		})
	}
}

func TestConnectionHeaderTimeoutIsAnError(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		<-release
	}))
	defer server.Close()
	defer close(release)
	transport := &http.Transport{ResponseHeaderTimeout: 20 * time.Millisecond}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport}
	var counts counters
	runConnection(context.Background(), client, Scenario{URL: server.URL}, []byte(`{}`), &counts)
	if counts.errors.Load() != 1 || counts.canceled.Load() != 0 || counts.connected.Load() != 0 {
		t.Fatalf("errors=%d canceled=%d connected=%d", counts.errors.Load(), counts.canceled.Load(), counts.connected.Load())
	}
	if len(counts.errorSamples) != 1 || !strings.Contains(counts.errorSamples[0], "timeout awaiting response headers") {
		t.Fatalf("error samples = %v", counts.errorSamples)
	}
}
