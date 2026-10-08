package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/loadgen"
)

func TestCommandKeepsJSONSeparateFromSummary(t *testing.T) {
	t.Parallel()

	const chatEvents = "data: {\"choices\":[{\"delta\":{\"content\":\"amber\"}}]}\n\ndata: [DONE]\n\n"
	const responseEvents = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"delta\":\"amber\"}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\ndata: [DONE]\n\n"
	for _, test := range []struct {
		name     string
		protocol loadgen.Protocol
		toFile   bool
		status   int
		events   string
		wantCode int
	}{
		{name: "chat stdout", protocol: loadgen.ProtocolChatCompletions, status: http.StatusOK, events: chatEvents},
		{name: "responses file", protocol: loadgen.ProtocolResponses, toFile: true, status: http.StatusOK, events: responseEvents},
		{name: "failed stream", protocol: loadgen.ProtocolChatCompletions, status: http.StatusServiceUnavailable, wantCode: 1},
		{name: "invalid event", protocol: loadgen.ProtocolChatCompletions, status: http.StatusOK, events: "data: not-json\n\ndata: [DONE]\n\n", wantCode: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(test.status)
				_, _ = fmt.Fprint(w, test.events)
			}))
			defer server.Close()
			directory := t.TempDir()
			scenarioPath := filepath.Join(directory, "scenario.json")
			value, err := json.Marshal(loadgen.Scenario{
				Name: "cli-test", Protocol: test.protocol, URL: server.URL,
				Connections: 1, Duration: loadgen.Duration(time.Second),
				Request: loadgen.Request{Model: "mock-chat-model", MaxCompletionTokens: 16},
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(scenarioPath, value, 0o600); err != nil {
				t.Fatal(err)
			}
			args := []string{"-scenario", scenarioPath}
			outputPath := filepath.Join(directory, "result.json")
			if test.toFile {
				args = append(args, "-output", outputPath)
			}
			var stdout, stderr bytes.Buffer
			if code := runCommand(context.Background(), args, &stdout, &stderr); code != test.wantCode {
				t.Fatalf("exit code = %d, want %d; stderr = %s", code, test.wantCode, stderr.String())
			}
			resultJSON := stdout.Bytes()
			if test.toFile {
				if stdout.Len() != 0 {
					t.Fatalf("stdout should be empty with -output: %s", stdout.String())
				}
				resultJSON, err = os.ReadFile(outputPath)
				if err != nil {
					t.Fatal(err)
				}
			}
			var result loadgen.Result
			if err := json.Unmarshal(resultJSON, &result); err != nil {
				t.Fatalf("output is not a standalone JSON result: %v; output = %s", err, resultJSON)
			}
			if result.Scenario.Name != "cli-test" || result.Attempted != 1 {
				t.Fatalf("unexpected result: %#v", result)
			}
			for _, text := range []string{"Scenario: cli-test", "Duration:", "Streams: attempted=1", "Validation:", "Throughput:", "Open latency:", "Content gap: n/a (no samples)"} {
				if !strings.Contains(stderr.String(), text) {
					t.Errorf("summary missing %q: %s", text, stderr.String())
				}
			}
			if test.status != http.StatusOK && !strings.Contains(stderr.String(), "errors=1") {
				t.Errorf("summary missing stream failure: %s", stderr.String())
			}
			if test.name == "invalid event" && !strings.Contains(stderr.String(), "invalid_events=1") {
				t.Errorf("summary missing protocol failure: %s", stderr.String())
			}
		})
	}
}

func TestCommandUsage(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name string
		args []string
		code int
	}{
		{name: "help", args: []string{"-h"}},
		{name: "missing scenario", code: 2},
		{name: "unknown flag", args: []string{"-unknown"}, code: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runCommand(context.Background(), test.args, &stdout, &stderr); code != test.code {
				t.Fatalf("exit code = %d, want %d", code, test.code)
			}
			if stdout.Len() != 0 || stderr.Len() == 0 {
				t.Fatalf("unexpected usage output: stdout=%q stderr=%q", stdout.String(), stderr.String())
			}
		})
	}
}
