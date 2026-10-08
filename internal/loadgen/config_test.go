package loadgen

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestScenarioDurations(t *testing.T) {
	t.Parallel()

	var scenario Scenario
	err := json.NewDecoder(strings.NewReader(`{
		"name":"test",
		"url":"http://127.0.0.1:8080/v1/chat/completions",
		"connections":10,
		"ramp_up":"250ms",
		"duration":"2s",
		"connection_lifetime":"1s",
		"read_delay":"10ms",
		"request":{"model":"mock","prompt":"hello","max_completion_tokens":64,"seed":42,"include_usage":true}
	}`)).Decode(&scenario)
	if err != nil {
		t.Fatalf("decode scenario: %v", err)
	}
	if time.Duration(scenario.RampUp) != 250*time.Millisecond ||
		time.Duration(scenario.Duration) != 2*time.Second ||
		time.Duration(scenario.ConnectionLifetime) != time.Second ||
		time.Duration(scenario.ReadDelay) != 10*time.Millisecond {
		t.Fatalf("durations = %#v", scenario)
	}
	if err := scenario.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestScenarioValidation(t *testing.T) {
	t.Parallel()

	valid := Scenario{
		Name:        "test",
		URL:         "http://127.0.0.1:8080/v1/chat/completions",
		Connections: 1,
		Duration:    Duration(time.Second),
		Request: Request{
			Model:               "mock",
			MaxCompletionTokens: 1,
		},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if valid.effectiveProtocol() != ProtocolChatCompletions {
		t.Fatalf("effectiveProtocol() = %q, want %q", valid.effectiveProtocol(), ProtocolChatCompletions)
	}

	invalid := valid
	invalid.Connections = 0
	if err := invalid.Validate(); err == nil || !strings.Contains(err.Error(), "connections") {
		t.Fatalf("Validate() error = %v, want connections error", err)
	}

	invalid = valid
	invalid.Protocol = "unknown"
	if err := invalid.Validate(); err == nil || !strings.Contains(err.Error(), "protocol") {
		t.Fatalf("Validate() error = %v, want protocol error", err)
	}

	invalid = valid
	invalid.Protocol = ProtocolResponses
	if err := invalid.Validate(); err == nil || !strings.Contains(err.Error(), "at least 16") {
		t.Fatalf("Validate() error = %v, want responses token minimum error", err)
	}
}

func TestLatencySummary(t *testing.T) {
	t.Parallel()

	summary := summarizeLatencies([]time.Duration{
		100 * time.Millisecond,
		10 * time.Millisecond,
		50 * time.Millisecond,
	})
	if summary.Count != 3 || summary.Min != 10 || summary.P50 != 50 || summary.Max != 100 {
		t.Fatalf("summary = %#v", summary)
	}
}
