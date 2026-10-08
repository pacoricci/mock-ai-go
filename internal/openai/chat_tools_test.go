package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func TestChatToolValidation(t *testing.T) {
	t.Parallel()
	const call = `{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}`
	const assistant = `{"role":"assistant","content":null,"tool_calls":[` + call + `]}`
	const output = `{"role":"tool","tool_call_id":"call_1","content":"sunny"}`
	tests := []struct{ name, fields, param string }{
		{"required without tools", `"tool_choice":"required"`, "tool_choice"},
		{"invalid choice", `"tool_choice":"sometimes"`, "tool_choice"},
		{"invalid choice shape", `"tool_choice":42`, "tool_choice"},
		{"unknown named tool", `"tool_choice":{"type":"function","function":{"name":"missing"}},"tools":[{"type":"function","function":{"name":"weather"}}]`, "tool_choice"},
		{"unsupported tool", `"tools":[{"type":"custom","function":{"name":"weather"}}]`, "tools.0.type"},
		{"missing name", `"tools":[{"type":"function","function":{}}]`, "tools.0.function.name"},
		{"duplicate name", `"tools":[{"type":"function","function":{"name":"weather"}},{"type":"function","function":{"name":"weather"}}]`, "tools.1.function.name"},
		{"invalid parameters", `"tools":[{"type":"function","function":{"name":"weather","parameters":[]}}]`, "tools.0.function.parameters"},
		{"legacy functions", `"functions":[]`, "functions"},
		{"legacy selection", `"function_call":"auto"`, "function_call"},
		{"unknown result", `"messages":[` + output + `]`, "messages"},
		{"result before call", `"messages":[` + output + `,` + assistant + `]`, "messages"},
		{"duplicate result", `"messages":[` + assistant + `,` + output + `,` + output + `]`, "messages"},
		{"duplicate call", `"messages":[` + assistant + `,` + assistant + `]`, "messages"},
		{"missing result ID", `"messages":[{"role":"tool","content":"sunny"}]`, "messages.0"},
		{"missing result content", `"messages":[{"role":"tool","tool_call_id":"call_1"}]`, "messages.0"},
		{"calls on user", `"messages":[{"role":"user","tool_calls":[` + call + `]}]`, "messages.0.tool_calls"},
		{"result ID on user", `"messages":[{"role":"user","tool_call_id":"call_1"}]`, "messages.0.tool_call_id"},
		{"malformed arguments", `"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{"}}]}]`, "messages.0.tool_calls.0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, stream := range []bool{false, true} {
				handler := newTestHandler(t, &testChatModel{info: testModelInfo()})
				base := `"model":"mock-chat-model",`
				if !strings.Contains(tt.fields, `"messages"`) {
					base += `"messages":[{"role":"user","content":"Hello"}],`
				}
				if stream {
					base += `"stream":true,`
				}
				recorder := httptest.NewRecorder()
				handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{`+base+tt.fields+`}`)))
				var result errorResponse
				if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				if recorder.Code != http.StatusBadRequest || pointerValue(result.Error.Param) != tt.param {
					t.Fatalf("stream=%v status=%d body=%s", stream, recorder.Code, recorder.Body.String())
				}
			}
		})
	}
}

type cancelToolWriter struct {
	*httptest.ResponseRecorder
	cancel context.CancelFunc
	marker string
}

func (w *cancelToolWriter) Write(data []byte) (int, error) {
	n, err := w.ResponseRecorder.Write(data)
	if strings.Contains(string(data), w.marker) {
		w.cancel()
	}
	return n, err
}

func TestCanceledToolStreamsDoNotComplete(t *testing.T) {
	t.Parallel()
	for _, endpoint := range []string{"/v1/chat/completions", "/v1/responses"} {
		t.Run(endpoint, func(t *testing.T) {
			model, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{DefaultCompletionTokens: 32, MaxCompletionTokens: 64, StreamTokensPerChunk: 1, StreamDelay: time.Hour})
			if err != nil {
				t.Fatal(err)
			}
			handler := newTestHandler(t, model)
			body := `{"model":"mock-chat-model","stream":true,"tool_choice":"required","tools":[{"type":"function","name":"weather","parameters":{"type":"object","properties":{"city":{"const":"long location"}},"required":["city"]}}],"input":"Hello"}`
			marker := "response.function_call_arguments.delta"
			if endpoint == "/v1/chat/completions" {
				body = `{"model":"mock-chat-model","stream":true,"tool_choice":"required","tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"const":"long location"}},"required":["city"]}}}],"messages":[{"role":"user","content":"Hello"}]}`
				marker = `"tool_calls":[`
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			writer := &cancelToolWriter{ResponseRecorder: httptest.NewRecorder(), cancel: cancel, marker: marker}
			handler.ServeHTTP(writer, httptest.NewRequest(http.MethodPost, endpoint, strings.NewReader(body)).WithContext(ctx))
			result := writer.Body.String()
			if !strings.Contains(result, marker) || strings.Contains(result, "[DONE]") || strings.Contains(result, `"finish_reason":"tool_calls"`) || strings.Contains(result, "response.completed") {
				t.Fatalf("canceled stream: %s", result)
			}
			if endpoint == "/v1/responses" {
				events := parseResponseSSEEvents(t, result)
				var created responseLifecycleEvent
				if err := json.Unmarshal([]byte(events[0].data), &created); err != nil {
					t.Fatal(err)
				}
				if _, found, err := handler.responseStore.Get(created.Response.ID); err != nil || found {
					t.Fatalf("canceled response stored: found=%v err=%v", found, err)
				}
			}
		})
	}
}
