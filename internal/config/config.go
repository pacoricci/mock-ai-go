// Package config loads and validates application configuration.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/embedding"
)

const (
	defaultName                 = "mock-ai-go"
	defaultHTTPAddress          = ":8080"
	defaultModelID              = "mock-chat-model"
	defaultCompletionTokens     = 32
	defaultMaxCompletionTokens  = 4096
	defaultMinTextTokens        = 128
	defaultMaxTextTokens        = 512
	defaultStreamTokensPerChunk = 4
	defaultStreamDelay          = 100 * time.Millisecond
	defaultStreamDelayJitter    = 20 * time.Millisecond
	defaultStreamInitialDelay   = 800 * time.Millisecond
	defaultResponseStoreTTL     = time.Hour
	defaultResponseStoreEntries = 10_000
	defaultResponseStoreBytes   = 64 << 20
)

// Config contains application configuration.
type Config struct {
	Name                    string
	HTTPAddress             string
	PPROFAddress            string
	ModelID                 string
	EmbeddingModelID        string
	EmbeddingDimensions     int
	DefaultCompletionTokens int
	MaxCompletionTokens     int
	MinTextTokens           int
	MaxTextTokens           int
	StreamTokensPerChunk    int
	StreamDelay             time.Duration
	StreamDelayJitter       time.Duration
	StreamInitialDelay      time.Duration
	ResponseStoreTTL        time.Duration
	ResponseStoreMaxEntries int
	ResponseStoreMaxBytes   int64
}

// FromEnv returns configuration populated from environment variables.
func FromEnv() (Config, error) {
	defaultTokens, err := positiveIntFromEnv("DEFAULT_COMPLETION_TOKENS", defaultCompletionTokens)
	if err != nil {
		return Config{}, err
	}
	maxTokens, err := positiveIntFromEnv("MAX_COMPLETION_TOKENS", defaultMaxCompletionTokens)
	if err != nil {
		return Config{}, err
	}
	if defaultTokens > maxTokens {
		return Config{}, fmt.Errorf("DEFAULT_COMPLETION_TOKENS must not exceed MAX_COMPLETION_TOKENS")
	}
	minTextTokens, err := nonNegativeIntFromEnv("TEXT_MIN_TOKENS", defaultMinTextTokens)
	if err != nil {
		return Config{}, err
	}
	maxTextTokens, err := nonNegativeIntFromEnv("TEXT_MAX_TOKENS", defaultMaxTextTokens)
	if err != nil {
		return Config{}, err
	}
	if (minTextTokens != 0 || maxTextTokens != 0) &&
		(minTextTokens == 0 || maxTextTokens < minTextTokens || maxTextTokens > maxTokens) {
		return Config{}, fmt.Errorf("TEXT_MIN_TOKENS and TEXT_MAX_TOKENS must both be zero or positive, ordered, and within MAX_COMPLETION_TOKENS")
	}
	streamTokens, err := positiveIntFromEnv("SSE_TOKENS_PER_CHUNK", defaultStreamTokensPerChunk)
	if err != nil {
		return Config{}, err
	}
	streamDelay, err := durationFromEnv("SSE_DELAY", defaultStreamDelay)
	if err != nil {
		return Config{}, err
	}
	streamDelayJitter, err := durationFromEnv("SSE_DELAY_JITTER", defaultStreamDelayJitter)
	if err != nil {
		return Config{}, err
	}
	if streamDelayJitter > time.Duration(1<<63-1)-streamDelay {
		return Config{}, fmt.Errorf("SSE_DELAY plus SSE_DELAY_JITTER exceeds the maximum duration")
	}
	streamInitialDelay, err := durationFromEnv("SSE_INITIAL_DELAY", defaultStreamInitialDelay)
	if err != nil {
		return Config{}, err
	}
	responseStoreTTL, err := positiveDurationFromEnv("RESPONSE_STORE_TTL", defaultResponseStoreTTL)
	if err != nil {
		return Config{}, err
	}
	responseStoreMaxEntries, err := positiveIntFromEnv("RESPONSE_STORE_MAX_ENTRIES", defaultResponseStoreEntries)
	if err != nil {
		return Config{}, err
	}
	responseStoreMaxBytes, err := positiveInt64FromEnv("RESPONSE_STORE_MAX_BYTES", defaultResponseStoreBytes)
	if err != nil {
		return Config{}, err
	}

	embeddingDimensions, err := positiveIntFromEnv("EMBEDDING_DIMENSIONS", embedding.DefaultDimensions)
	if err != nil {
		return Config{}, err
	}
	if embeddingDimensions > embedding.MaxDimensions {
		return Config{}, fmt.Errorf("EMBEDDING_DIMENSIONS must not exceed %d", embedding.MaxDimensions)
	}
	embeddingID := envOrDefault("EMBEDDING_MODEL_ID", embedding.DefaultModelID)
	if _, err := embedding.NewMockModel(embeddingID, embeddingDimensions); err != nil {
		return Config{}, err
	}

	return Config{
		EmbeddingModelID:        embeddingID,
		EmbeddingDimensions:     embeddingDimensions,
		Name:                    envOrDefault("APP_NAME", defaultName),
		HTTPAddress:             envOrDefault("HTTP_ADDR", defaultHTTPAddress),
		PPROFAddress:            os.Getenv("PPROF_ADDR"),
		ModelID:                 envOrDefault("MODEL_ID", defaultModelID),
		DefaultCompletionTokens: defaultTokens,
		MaxCompletionTokens:     maxTokens,
		MinTextTokens:           minTextTokens,
		MaxTextTokens:           maxTextTokens,
		StreamTokensPerChunk:    streamTokens,
		StreamDelay:             streamDelay,
		StreamDelayJitter:       streamDelayJitter,
		StreamInitialDelay:      streamInitialDelay,
		ResponseStoreTTL:        responseStoreTTL,
		ResponseStoreMaxEntries: responseStoreMaxEntries,
		ResponseStoreMaxBytes:   responseStoreMaxBytes,
	}, nil
}

func nonNegativeIntFromEnv(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0, fmt.Errorf("%s must be a nonnegative integer", name)
	}
	return parsed, nil
}

func positiveInt64FromEnv(name string, fallback int64) (int64, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}

	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	return parsed, nil
}

func envOrDefault(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func positiveIntFromEnv(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}

	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	return parsed, nil
}

func durationFromEnv(name string, fallback time.Duration) (time.Duration, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}

	parsed, err := time.ParseDuration(value)
	if err != nil || parsed < 0 {
		return 0, fmt.Errorf("%s must be a non-negative duration", name)
	}
	return parsed, nil
}

func positiveDurationFromEnv(name string, fallback time.Duration) (time.Duration, error) {
	value, err := durationFromEnv(name, fallback)
	if err != nil {
		return 0, err
	}
	if value <= 0 {
		return 0, fmt.Errorf("%s must be a positive duration", name)
	}
	return value, nil
}
