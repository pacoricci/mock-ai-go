package config

import (
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/embedding"
)

func TestFromEnv(t *testing.T) {
	t.Setenv("APP_NAME", "custom-name")
	t.Setenv("HTTP_ADDR", "127.0.0.1:9000")
	t.Setenv("PPROF_ADDR", "127.0.0.1:6060")
	t.Setenv("MODEL_ID", "custom-model")
	t.Setenv("EMBEDDING_MODEL_ID", "custom-embedding")
	t.Setenv("EMBEDDING_DIMENSIONS", "256")
	t.Setenv("DEFAULT_COMPLETION_TOKENS", "24")
	t.Setenv("MAX_COMPLETION_TOKENS", "2048")
	t.Setenv("SSE_TOKENS_PER_CHUNK", "3")
	t.Setenv("SSE_DELAY", "25ms")
	t.Setenv("SSE_DELAY_JITTER", "5ms")
	t.Setenv("SSE_INITIAL_DELAY", "100ms")
	t.Setenv("RESPONSE_STORE_TTL", "2h")
	t.Setenv("RESPONSE_STORE_MAX_ENTRIES", "250")
	t.Setenv("RESPONSE_STORE_MAX_BYTES", "1048576")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv() error = %v", err)
	}
	want := Config{
		Name:                    "custom-name",
		HTTPAddress:             "127.0.0.1:9000",
		PPROFAddress:            "127.0.0.1:6060",
		ModelID:                 "custom-model",
		EmbeddingModelID:        "custom-embedding",
		EmbeddingDimensions:     256,
		DefaultCompletionTokens: 24,
		MaxCompletionTokens:     2048,
		StreamTokensPerChunk:    3,
		StreamDelay:             25 * time.Millisecond,
		StreamDelayJitter:       5 * time.Millisecond,
		StreamInitialDelay:      100 * time.Millisecond,
		ResponseStoreTTL:        2 * time.Hour,
		ResponseStoreMaxEntries: 250,
		ResponseStoreMaxBytes:   1 << 20,
	}
	if cfg != want {
		t.Fatalf("FromEnv() = %#v, want %#v", cfg, want)
	}
}

func TestFromEnvUsesDefault(t *testing.T) {
	for _, name := range []string{
		"APP_NAME",
		"HTTP_ADDR",
		"PPROF_ADDR",
		"MODEL_ID",
		"EMBEDDING_MODEL_ID",
		"EMBEDDING_DIMENSIONS",
		"DEFAULT_COMPLETION_TOKENS",
		"MAX_COMPLETION_TOKENS",
		"SSE_TOKENS_PER_CHUNK",
		"SSE_DELAY",
		"SSE_DELAY_JITTER",
		"SSE_INITIAL_DELAY",
		"RESPONSE_STORE_TTL",
		"RESPONSE_STORE_MAX_ENTRIES",
		"RESPONSE_STORE_MAX_BYTES",
	} {
		t.Setenv(name, "")
	}

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv() error = %v", err)
	}
	want := Config{
		Name:                    defaultName,
		HTTPAddress:             defaultHTTPAddress,
		PPROFAddress:            "",
		ModelID:                 defaultModelID,
		EmbeddingModelID:        embedding.DefaultModelID,
		EmbeddingDimensions:     embedding.DefaultDimensions,
		DefaultCompletionTokens: defaultCompletionTokens,
		MaxCompletionTokens:     defaultMaxCompletionTokens,
		StreamTokensPerChunk:    defaultStreamTokensPerChunk,
		StreamDelay:             100 * time.Millisecond,
		StreamDelayJitter:       20 * time.Millisecond,
		StreamInitialDelay:      800 * time.Millisecond,
		ResponseStoreTTL:        defaultResponseStoreTTL,
		ResponseStoreMaxEntries: defaultResponseStoreEntries,
		ResponseStoreMaxBytes:   defaultResponseStoreBytes,
	}
	if cfg != want {
		t.Fatalf("FromEnv() = %#v, want %#v", cfg, want)
	}
}

func TestFromEnvAllowsDisablingStreamPacing(t *testing.T) {
	t.Setenv("SSE_DELAY", "0s")
	t.Setenv("SSE_DELAY_JITTER", "0s")
	t.Setenv("SSE_INITIAL_DELAY", "0s")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv() error = %v", err)
	}
	if cfg.StreamDelay != 0 || cfg.StreamDelayJitter != 0 || cfg.StreamInitialDelay != 0 {
		t.Fatalf("Expected disabled pacing, got delay %s, jitter %s, initial delay %s",
			cfg.StreamDelay, cfg.StreamDelayJitter, cfg.StreamInitialDelay)
	}
}

func TestFromEnvRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name    string
		values  map[string]string
		message string
	}{
		{name: "embedding zero", values: map[string]string{"EMBEDDING_DIMENSIONS": "0"}, message: "EMBEDDING_DIMENSIONS"},
		{name: "embedding too large", values: map[string]string{"EMBEDDING_DIMENSIONS": "4097"}, message: "EMBEDDING_DIMENSIONS"},
		{name: "embedding blank model", values: map[string]string{"EMBEDDING_MODEL_ID": " "}, message: "must not be blank"},
		{name: "default tokens", values: map[string]string{"DEFAULT_COMPLETION_TOKENS": "zero"}, message: "DEFAULT_COMPLETION_TOKENS"},
		{name: "max tokens", values: map[string]string{"MAX_COMPLETION_TOKENS": "0"}, message: "MAX_COMPLETION_TOKENS"},
		{name: "default exceeds max", values: map[string]string{"DEFAULT_COMPLETION_TOKENS": "10", "MAX_COMPLETION_TOKENS": "9"}, message: "must not exceed"},
		{name: "chunk tokens", values: map[string]string{"SSE_TOKENS_PER_CHUNK": "-1"}, message: "SSE_TOKENS_PER_CHUNK"},
		{name: "delay", values: map[string]string{"SSE_DELAY": "yesterday"}, message: "SSE_DELAY"},
		{name: "negative delay", values: map[string]string{"SSE_DELAY": "-1s"}, message: "SSE_DELAY"},
		{name: "jitter", values: map[string]string{"SSE_DELAY_JITTER": "sometimes"}, message: "SSE_DELAY_JITTER"},
		{name: "negative jitter", values: map[string]string{"SSE_DELAY_JITTER": "-1ns"}, message: "SSE_DELAY_JITTER"},
		{name: "initial delay", values: map[string]string{"SSE_INITIAL_DELAY": "later"}, message: "SSE_INITIAL_DELAY"},
		{name: "negative initial delay", values: map[string]string{"SSE_INITIAL_DELAY": "-1ns"}, message: "SSE_INITIAL_DELAY"},
		{name: "delay jitter overflow", values: map[string]string{"SSE_DELAY": "2562047h47m16.854775807s", "SSE_DELAY_JITTER": "1ns"}, message: "exceeds"},
		{name: "response store TTL", values: map[string]string{"RESPONSE_STORE_TTL": "0s"}, message: "RESPONSE_STORE_TTL"},
		{name: "response store entries", values: map[string]string{"RESPONSE_STORE_MAX_ENTRIES": "0"}, message: "RESPONSE_STORE_MAX_ENTRIES"},
		{name: "response store bytes", values: map[string]string{"RESPONSE_STORE_MAX_BYTES": "-1"}, message: "RESPONSE_STORE_MAX_BYTES"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for name, value := range tt.values {
				t.Setenv(name, value)
			}

			_, err := FromEnv()
			if err == nil || !strings.Contains(err.Error(), tt.message) {
				t.Fatalf("FromEnv() error = %v, want message containing %q", err, tt.message)
			}
		})
	}
}
