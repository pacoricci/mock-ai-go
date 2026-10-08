// Package loadgen implements a black-box SSE load generator.
package loadgen

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// Duration is a time.Duration encoded as a readable JSON string.
type Duration time.Duration

// Protocol identifies the streaming API exercised by a scenario.
type Protocol string

const (
	// ProtocolChatCompletions exercises POST /v1/chat/completions.
	ProtocolChatCompletions Protocol = "chat_completions"
	// ProtocolResponses exercises POST /v1/responses.
	ProtocolResponses Protocol = "responses"
)

func (d Duration) String() string {
	return time.Duration(d).String()
}

// MarshalJSON implements json.Marshaler.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.String())
}

// UnmarshalJSON implements json.Unmarshaler.
func (d *Duration) UnmarshalJSON(value []byte) error {
	var text string
	if err := json.Unmarshal(value, &text); err != nil {
		return fmt.Errorf("duration must be a string: %w", err)
	}
	parsed, err := time.ParseDuration(text)
	if err != nil || parsed < 0 {
		return fmt.Errorf("invalid non-negative duration %q", text)
	}
	*d = Duration(parsed)
	return nil
}

// Request configures the prompt shared by either supported protocol.
type Request struct {
	Model               string `json:"model"`
	Prompt              string `json:"prompt"`
	MaxCompletionTokens int    `json:"max_completion_tokens"`
	Seed                int64  `json:"seed"`
	IncludeUsage        bool   `json:"include_usage"`
	Store               *bool  `json:"store,omitempty"`
}

// Scenario describes one reproducible SSE load test.
type Scenario struct {
	Name               string            `json:"name"`
	Protocol           Protocol          `json:"protocol,omitempty"`
	URL                string            `json:"url"`
	MetricsURL         string            `json:"metrics_url,omitempty"`
	Connections        int               `json:"connections"`
	RampUp             Duration          `json:"ramp_up"`
	Duration           Duration          `json:"duration"`
	ConnectionLifetime Duration          `json:"connection_lifetime,omitempty"`
	ReadDelay          Duration          `json:"read_delay,omitempty"`
	Request            Request           `json:"request"`
	ServerEnv          map[string]string `json:"server_env,omitempty"`
}

// Validate checks whether a scenario can be executed safely.
func (s Scenario) Validate() error {
	if strings.TrimSpace(s.Name) == "" {
		return fmt.Errorf("scenario name must not be empty")
	}
	if protocol := s.effectiveProtocol(); protocol != ProtocolChatCompletions && protocol != ProtocolResponses {
		return fmt.Errorf("protocol must be %q or %q", ProtocolChatCompletions, ProtocolResponses)
	}
	parsedURL, err := url.ParseRequestURI(s.URL)
	if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") || parsedURL.Host == "" {
		return fmt.Errorf("scenario URL must be an absolute HTTP URL")
	}
	if s.MetricsURL != "" {
		parsedMetricsURL, err := url.ParseRequestURI(s.MetricsURL)
		if err != nil || (parsedMetricsURL.Scheme != "http" && parsedMetricsURL.Scheme != "https") || parsedMetricsURL.Host == "" {
			return fmt.Errorf("metrics_url must be an absolute HTTP URL")
		}
	}
	if s.Connections <= 0 {
		return fmt.Errorf("connections must be positive")
	}
	if time.Duration(s.RampUp) < 0 {
		return fmt.Errorf("ramp_up must not be negative")
	}
	if time.Duration(s.Duration) <= 0 {
		return fmt.Errorf("duration must be positive")
	}
	if time.Duration(s.ConnectionLifetime) < 0 {
		return fmt.Errorf("connection_lifetime must not be negative")
	}
	if time.Duration(s.ReadDelay) < 0 {
		return fmt.Errorf("read_delay must not be negative")
	}
	if strings.TrimSpace(s.Request.Model) == "" {
		return fmt.Errorf("request model must not be empty")
	}
	if s.Request.MaxCompletionTokens <= 0 {
		return fmt.Errorf("request max_completion_tokens must be positive")
	}
	if s.effectiveProtocol() == ProtocolResponses && s.Request.MaxCompletionTokens < 16 {
		return fmt.Errorf("request max_completion_tokens must be at least 16 for responses")
	}
	return nil
}

func (s Scenario) effectiveProtocol() Protocol {
	if s.Protocol == "" {
		return ProtocolChatCompletions
	}
	return s.Protocol
}
