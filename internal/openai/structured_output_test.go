package openai_test

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

func TestStructuredFormatValidationBeforeStreaming(t *testing.T) {
	t.Parallel()
	model, err := chat.NewMockModel("mock", chat.MockOptions{DefaultCompletionTokens: 128, MaxCompletionTokens: 256, StreamTokensPerChunk: 4})
	if err != nil {
		t.Fatal(err)
	}
	registry, err := chat.NewRegistry(model)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := openai.NewHandler(registry, openai.ResponseStoreOptions{TTL: time.Hour, MaxEntries: 10, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name, endpoint, field, format string
	}{
		{"unknown chat type", "chat/completions", "response_format", `{"type":"yaml"}`},
		{"null chat format", "chat/completions", "response_format", `null`},
		{"missing chat schema", "chat/completions", "response_format", `{"type":"json_schema"}`},
		{"schema on plain text", "chat/completions", "response_format", `{"type":"text","json_schema":{}}`},
		{"bad name", "chat/completions", "response_format", `{"type":"json_schema","json_schema":{"name":"bad name","schema":{"type":"object"}}}`},
		{"flat chat schema", "chat/completions", "response_format", `{"type":"json_schema","name":"x","schema":{"type":"object"}}`},
		{"null text", "responses", "text", `null`},
		{"null text format", "responses", "text", `{"format":null}`},
		{"unknown responses format", "responses", "text", `{"format":{"type":"yaml"}}`},
		{"schema required", "responses", "text", `{"format":{"type":"json_schema","name":"x"}}`},
		{"nested responses schema", "responses", "text", `{"format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}}`},
		{"strict extras", "responses", "text", `{"format":{"type":"json_schema","name":"x","strict":true,"schema":{"type":"object"}}}`},
		{"settings on json mode", "responses", "text", `{"format":{"type":"json_object","strict":true}}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for _, streaming := range []string{"false", "true"} {
				body := `{"model":"mock","messages":[{"role":"user","content":"JSON"}],"input":"JSON","stream":` + streaming + `,"` + tc.field + `":` + tc.format + `}`
				recorder := httptest.NewRecorder()
				handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/"+tc.endpoint, strings.NewReader(body)))
				if recorder.Code != 400 || recorder.Header().Get("Content-Type") != "application/json" {
					t.Fatalf("invalid format started a stream or succeeded: %d %s", recorder.Code, recorder.Body.String())
				}
				var response struct {
					Error struct {
						Param string `json:"param"`
						Type  string `json:"type"`
					} `json:"error"`
				}
				if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil || !strings.HasPrefix(response.Error.Param, tc.field) || response.Error.Type != "invalid_request_error" {
					t.Fatalf("bad schema error: %v, %s", err, recorder.Body.String())
				}
			}
		})
	}
}
