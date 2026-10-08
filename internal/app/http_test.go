package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
	"github.com/pacoricci/mock-ai-go/internal/openai"
)

func TestNewHTTPHandlerServesDocumentation(t *testing.T) {
	handler := newHTTPHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}), http.NotFoundHandler(), http.NotFoundHandler())

	tests := []struct {
		name        string
		path        string
		contentType string
		bodyPart    string
	}{
		{name: "Swagger UI", path: "/docs", contentType: "text/html; charset=utf-8", bodyPart: "SwaggerUIBundle"},
		{name: "OpenAPI specification", path: "/openapi.yaml", contentType: "application/yaml; charset=utf-8", bodyPart: "openapi: 3.0.3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodGet, tt.path, nil)

			handler.ServeHTTP(recorder, request)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
			}
			if got := recorder.Header().Get("Content-Type"); got != tt.contentType {
				t.Fatalf("Content-Type = %q, want %q", got, tt.contentType)
			}
			if !strings.Contains(recorder.Body.String(), tt.bodyPart) {
				t.Fatalf("body does not contain %q", tt.bodyPart)
			}
		})
	}
}

func TestNewHTTPHandlerKeepsOpenAIRoute(t *testing.T) {
	handler := newHTTPHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}), http.NotFoundHandler(), http.NotFoundHandler())
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
}

func TestNewHTTPHandlerServesResponsesFunctionCall(t *testing.T) {
	chatModel, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{
		DefaultCompletionTokens: 32,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    4,
	})
	if err != nil {
		t.Fatalf("NewMockModel() error = %v", err)
	}
	registry, err := chat.NewRegistry(chatModel)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	openAIHandler, err := openai.NewHandler(registry, openai.ResponseStoreOptions{
		TTL:        time.Hour,
		MaxEntries: 1_000,
		MaxBytes:   8 << 20,
	})
	if err != nil {
		t.Fatalf("NewHandler() error = %v", err)
	}
	handler := newHTTPHandler(openAIHandler, http.NotFoundHandler(), http.NotFoundHandler())
	body := `{
		"model":"mock-chat-model",
		"input":[{"type":"message","role":"user","content":"What's the weather like in San Francisco?"}],
		"tool_choice":"required",
		"tools":[{
			"type":"function",
			"name":"get_weather",
			"description":"Get the current weather for a location",
			"parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
		}]
	}`
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)),
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"output"`
	}
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(response.Output) != 1 || response.Output[0].Type != "function_call" || response.Output[0].Name != "get_weather" {
		t.Fatalf("output = %#v", response.Output)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(response.Output[0].Arguments), &arguments); err != nil {
		t.Fatalf("arguments = %q: %v", response.Output[0].Arguments, err)
	}
	if _, ok := arguments["location"].(string); !ok {
		t.Fatalf("arguments = %#v", arguments)
	}
}
