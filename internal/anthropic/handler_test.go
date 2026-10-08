package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/anthropic"
	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func newModel(t *testing.T) chat.Model {
	t.Helper()
	model, err := chat.NewMockModel("mock", chat.MockOptions{DefaultCompletionTokens: 8, MaxCompletionTokens: 256, StreamTokensPerChunk: 2})
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func newHandler(t *testing.T, model chat.Model) *anthropic.Handler {
	t.Helper()
	registry, err := chat.NewRegistry(model)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := anthropic.NewHandler(registry)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func perform(handler http.Handler, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	request.Header.Set("anthropic-version", "2023-06-01")
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

func basicRequest(extra string) string {
	return `{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":"Hello 世界"}]` + extra + `}`
}

type response struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Model   string `json:"model"`
	Content []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func decodeResponse(t *testing.T, recorder *httptest.ResponseRecorder) response {
	t.Helper()
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", recorder.Code, recorder.Body.String())
	}
	var value response
	if err := json.Unmarshal(recorder.Body.Bytes(), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestTextNormalizationAndMetadata(t *testing.T) {
	handler := newHandler(t, newModel(t))
	body := basicRequest(`,"system":"Be concise"`)
	first := decodeResponse(t, perform(handler, body))
	equivalent := strings.ReplaceAll(body, `"content":"Hello 世界"`, `"content":[{"type":"text","text":"Hello 世界"}]`)
	equivalent = strings.ReplaceAll(equivalent, `"system":"Be concise"`, `"system":[{"type":"text","text":"Be concise"}]`)
	second := decodeResponse(t, perform(handler, equivalent))
	if first.ID == second.ID || first.Type != "message" || first.Role != "assistant" || first.Model != "mock" || first.StopReason != "max_tokens" {
		t.Fatalf("metadata = %+v / %+v", first, second)
	}
	if !reflect.DeepEqual(first.Content, second.Content) || first.Usage != second.Usage || len(strings.Fields(first.Content[0].Text)) != 32 || first.Usage.OutputTokens != 32 {
		t.Fatalf("normalization or usage mismatch: %+v / %+v", first, second)
	}
	third := decodeResponse(t, perform(handler, basicRequest(`,"system":"Be concise","temperature":0.5,"top_p":0.8,"top_k":10,"metadata":{"user_id":"test"}`)))
	if !reflect.DeepEqual(first.Content, third.Content) || first.Usage != third.Usage {
		t.Fatal("sampling or metadata altered deterministic generation")
	}
}

func TestInvalidRequests(t *testing.T) {
	handler := newHandler(t, newModel(t))
	cases := []struct {
		name string
		body string
		code int
	}{
		{"invalid JSON", `{`, 400},
		{"multiple values", basicRequest("") + `{}`, 400},
		{"null body", `null`, 400},
		{"missing budget", `{"model":"mock","messages":[{"role":"user","content":"hi"}]}`, 400},
		{"zero budget", strings.Replace(basicRequest(""), `:32`, `:0`, 1), 400},
		{"over budget", strings.Replace(basicRequest(""), `:32`, `:257`, 1), 400},
		{"fractional budget", strings.Replace(basicRequest(""), `:32`, `:1.5`, 1), 400},
		{"model", strings.Replace(basicRequest(""), `"mock"`, `"missing"`, 1), 404},
		{"system role", strings.Replace(basicRequest(""), `"user"`, `"system"`, 1), 400},
		{"prefill", strings.Replace(basicRequest(""), `"user"`, `"assistant"`, 1), 400},
		{"thinking", basicRequest(`,"thinking":{"type":"enabled","budget_tokens":1024}`), 400},
		{"cache", basicRequest(`,"cache_control":{"type":"ephemeral"}`), 400},
		{"stops", basicRequest(`,"stop_sequences":["hello"]`), 400},
		{"temperature", basicRequest(`,"temperature":2`), 400},
		{"top p", basicRequest(`,"top_p":-1`), 400},
		{"top k", basicRequest(`,"top_k":-1`), 400},
		{"unknown choice", basicRequest(`,"tool_choice":{"type":"required"}`), 400},
		{"named choice missing name", basicRequest(`,"tool_choice":{"type":"tool"}`), 400},
		{"unexpected name", basicRequest(`,"tool_choice":{"type":"auto","name":"weather"}`), 400},
		{"unknown named tool", basicRequest(tools + `,"tool_choice":{"type":"tool","name":"unknown"}`), 400},
		{"invalid tool name", basicRequest(`,"tools":[{"name":"bad name","input_schema":{"type":"object"}}]`), 400},
		{"duplicate tool", basicRequest(`,"tools":[{"name":"weather","input_schema":{"type":"object"}},{"name":"weather","input_schema":{"type":"object"}}]`), 400},
		{"strict tool", basicRequest(`,"tools":[{"name":"weather","input_schema":{"type":"object"},"strict":true}]`), 400},
		{"array schema", basicRequest(`,"tools":[{"name":"weather","input_schema":{"type":"array"}}]`), 400},
		{"server tool", basicRequest(`,"tools":[{"name":"weather","type":"web_search_20250305"}]`), 400},
		{"empty messages", `{"model":"mock","max_tokens":32,"messages":[]}`, 400},
		{"missing content", `{"model":"mock","max_tokens":32,"messages":[{"role":"user"}]}`, 400},
		{"empty content", `{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":[]}]}`, 400},
		{"null block", `{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":[null]}]}`, 400},
		{"unknown block field", `{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{}}]}]}`, 400},
		{"missing tools", basicRequest(`,"tool_choice":{"type":"any"}`), 400},
		{"image", `{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}`, 400},
		{"body limit", basicRequest(`,"system":"` + strings.Repeat("x", 1<<20) + `"`), 413},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			for _, stream := range []bool{false, true} {
				body := test.body
				if stream && strings.HasPrefix(body, `{"model"`) {
					body = strings.Replace(body, `{"model"`, `{"stream":true,"model"`, 1)
				}
				recorder := perform(handler, body)
				var apiError struct {
					Type      string                         `json:"type"`
					RequestID string                         `json:"request_id"`
					Error     struct{ Type, Message string } `json:"error"`
				}
				if err := json.Unmarshal(recorder.Body.Bytes(), &apiError); err != nil {
					t.Fatal(err)
				}
				if recorder.Code != test.code || apiError.Type != "error" || apiError.Error.Type == "" || apiError.Error.Message == "" || apiError.RequestID == "" || apiError.RequestID != recorder.Header().Get("request-id") {
					t.Fatalf("status = %d, error = %+v", recorder.Code, apiError)
				}
			}
		})
	}
}

func TestHeadersAndRouting(t *testing.T) {
	if _, err := anthropic.NewHandler(nil); !errors.Is(err, anthropic.ErrNilChatRegistry) {
		t.Fatalf("nil registry error = %v", err)
	}
	handler := newHandler(t, newModel(t))
	for _, test := range []struct {
		method, path, version, beta string
		status                      int
	}{
		{"POST", "/v1/messages", "", "", 400},
		{"POST", "/v1/messages", "2020-01-01", "", 400},
		{"POST", "/v1/messages", "2023-06-01", "beta", 400},
		{"GET", "/v1/messages", "2023-06-01", "", 405},
		{"POST", "/v1/messages/count_tokens", "2023-06-01", "", 404},
	} {
		request := httptest.NewRequest(test.method, test.path, strings.NewReader(basicRequest("")))
		request.Header.Set("anthropic-version", test.version)
		request.Header.Set("anthropic-beta", test.beta)
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		if recorder.Code != test.status || recorder.Header().Get("request-id") == "" {
			t.Fatalf("%+v: %d %s", test, recorder.Code, recorder.Body.String())
		}
	}
}

const tools = `,"tools":[{"name":"weather","input_schema":{"type":"object","properties":{"city":{"type":"string","enum":["München 世界"]}},"required":["city"]}},{"name":"clock","input_schema":{"type":"object","properties":{"zone":{"type":"string"}}}}]`

func TestToolSelectionAndBudget(t *testing.T) {
	handler := newHandler(t, newModel(t))
	for _, choice := range []string{`{"type":"any","disable_parallel_tool_use":true}`, `{"type":"tool","name":"weather"}`, `{"type":"none"}`, `{"type":"auto"}`} {
		body := strings.Replace(basicRequest(tools+`,"tool_choice":`+choice), `:32`, `:256`, 1)
		first := decodeResponse(t, perform(handler, body))
		second := decodeResponse(t, perform(handler, body))
		if !reflect.DeepEqual(first.Content, second.Content) || first.Usage != second.Usage {
			t.Fatal("tool planning is not deterministic")
		}
		if strings.Contains(choice, `"any"`) || strings.Contains(choice, `"type":"tool"`) {
			if first.StopReason != "tool_use" || len(first.Content) != 1 || !json.Valid(first.Content[0].Input) || !strings.HasPrefix(first.Content[0].ID, "toolu_") {
				t.Fatalf("tool message = %+v", first)
			}
		}
		if strings.Contains(choice, `"name":"weather"`) && first.Content[0].Name != "weather" {
			t.Fatalf("named selection = %+v", first)
		}
		if strings.Contains(choice, `"none"`) && first.StopReason != "max_tokens" {
			t.Fatalf("none selection = %+v", first)
		}
	}
	for _, stream := range []bool{false, true} {
		body := strings.Replace(basicRequest(tools+fmt.Sprintf(`,"tool_choice":{"type":"any"},"stream":%t`, stream)), `:32`, `:1`, 1)
		if recorder := perform(handler, body); recorder.Code != 400 || strings.Contains(recorder.Header().Get("Content-Type"), "event-stream") {
			t.Fatalf("tool budget response = %d %s", recorder.Code, recorder.Body.String())
		}
	}
}

func TestToolHistoryValidation(t *testing.T) {
	handler := newHandler(t, newModel(t))
	prefix := `{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":"Weather?"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_a","name":"weather","input":{"city":"Rome"}},{"type":"tool_use","id":"toolu_b","name":"weather","input":{"city":"Paris"}}]},`
	results := `{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_b","content":"rain"},{"type":"tool_result","tool_use_id":"toolu_a","content":"sun"}]}]}`
	valid := decodeResponse(t, perform(handler, prefix+results))
	if valid.StopReason != "max_tokens" {
		t.Fatalf("tool replay = %+v", valid)
	}
	for _, empty := range []string{`"content":""`, `"content":[ ]`, `"is_error":false`} {
		decodeResponse(t, perform(handler, prefix+strings.Replace(results, `"content":"rain"`, empty, 1)))
	}
	decodeResponse(t, perform(handler, prefix+strings.Replace(results, `"content":"rain"`, `"content":[{"type":"text","text":"rain"}]`, 1)))
	for _, bad := range []string{
		strings.Replace(results, `"toolu_a"`, `"toolu_b"`, 1),
		strings.Replace(results, `"toolu_a"`, `"unknown"`, 1),
		`{"role":"user","content":"missing results"}]}`,
		strings.Replace(results, `"content":[`, `"content":[{"type":"text","text":"before results"},`, 1),
		strings.Replace(results, `"content":"rain"`, `"content":null`, 1),
	} {
		if recorder := perform(handler, prefix+bad); recorder.Code != 400 {
			t.Fatalf("invalid history accepted: %s", bad)
		}
	}
	errorResult := decodeResponse(t, perform(handler, prefix+strings.Replace(results, `"content":"rain"`, `"content":"rain","is_error":true`, 1)))
	if reflect.DeepEqual(valid.Content, errorResult.Content) {
		t.Fatal("is_error did not contribute to context")
	}
}

type controlledModel struct {
	chat.Model
	failAfterChunk bool
}

func (model *controlledModel) Stream(ctx context.Context, request chat.Request, emit chat.Emit) (chat.Usage, error) {
	if !model.failAfterChunk {
		return chat.Usage{}, errors.New("failure before content")
	}
	if err := emit(chat.Chunk{Delta: []byte("hello")}); err != nil {
		return chat.Usage{}, err
	}
	return chat.Usage{}, errors.New("failure after content")
}

type disconnectedWriter struct {
	header http.Header
	writes int
}

func (writer *disconnectedWriter) Header() http.Header { return writer.header }
func (writer *disconnectedWriter) WriteHeader(int)     {}
func (writer *disconnectedWriter) Flush()              {}
func (writer *disconnectedWriter) Write(data []byte) (int, error) {
	writer.writes++
	if writer.writes >= 3 {
		return 0, io.ErrClosedPipe
	}
	return len(data), nil
}

func TestWriteFailureStopsStreaming(t *testing.T) {
	handler := newHandler(t, newModel(t))
	for _, extra := range []string{"", tools + `,"tool_choice":{"type":"any"}`} {
		request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(strings.Replace(basicRequest(extra+`,"stream":true`), `:32`, `:256`, 1)))
		request.Header.Set("anthropic-version", "2023-06-01")
		writer := &disconnectedWriter{header: make(http.Header)}
		handler.ServeHTTP(writer, request)
		if writer.writes != 3 {
			t.Fatalf("writes after disconnect = %d, want exactly 3", writer.writes)
		}
	}
}

func TestStreamFailures(t *testing.T) {
	for _, after := range []bool{false, true} {
		model := &controlledModel{Model: newModel(t), failAfterChunk: after}
		recorder := perform(newHandler(t, model), basicRequest(`,"stream":true`))
		body := recorder.Body.String()
		if after {
			if recorder.Code != 200 || !strings.Contains(body, "event: error\n") || !strings.Contains(body, "event: content_block_delta\n") || strings.Contains(body, "event: message_stop\n") {
				t.Fatalf("late failure = %d %s", recorder.Code, body)
			}
		} else if recorder.Code != 500 || strings.Contains(body, "event:") {
			t.Fatalf("early failure = %d %s", recorder.Code, body)
		}
	}
}
