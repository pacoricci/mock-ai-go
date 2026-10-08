package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func TestCreateResponse(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info: testModelInfo(),
		response: chat.Response{
			Content: "amber orbit cedar",
			Usage:   chat.Usage{PromptTokens: 21, CompletionTokens: 3, TotalTokens: 24},
		},
	}
	handler := newTestHandler(t, chatModel)
	handler.now = func() time.Time { return time.Unix(1_700_000_000, 0) }

	body := `{
		"model":"mock-chat-model",
		"input":[
			{"type":"message","role":"system","content":"Be concise."},
			{"role":"user","content":[
				{"type":"input_text","text":"Hello "},
				{"type":"input_image","image_url":"https://example.com/image.png"},
				{"type":"input_text","text":"world"}
			]},
			{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"Prior answer"}]}
		],
		"instructions":"Follow instructions.",
		"max_output_tokens":16,
		"metadata":{"suite":"responses"},
		"store":false,
		"tools":[{"type":"function","name":"ignored"}],
		"background":true,
		"temperature":0
	}`
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var response responseResource
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !strings.HasPrefix(response.ID, responseIDPrefix) || response.Object != "response" || response.Status != "completed" {
		t.Fatalf("response identity = %#v", response)
	}
	if response.CreatedAt != 1_700_000_000 || response.CompletedAt == nil || *response.CompletedAt != 1_700_000_000 {
		t.Fatalf("response timestamps = %#v", response)
	}
	if response.Model != "mock-chat-model" || response.Instructions == nil || *response.Instructions != "Follow instructions." {
		t.Fatalf("response metadata = %#v", response)
	}
	if len(response.Output) != 1 || response.Output[0].ID != "msg-mock-1" {
		t.Fatalf("response output = %#v", response.Output)
	}
	message := response.Output[0]
	if message.Type != "message" || message.Status != "completed" || message.Role == nil || *message.Role != "assistant" || message.Content == nil || len(*message.Content) != 1 {
		t.Fatalf("response message = %#v", message)
	}
	if got := (*message.Content)[0]; got.Type != "output_text" || got.Text != "amber orbit cedar" || got.Annotations == nil {
		t.Fatalf("response content = %#v", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 21 || response.Usage.OutputTokens != 3 || response.Usage.TotalTokens != 24 {
		t.Fatalf("response usage = %#v", response.Usage)
	}
	if response.Usage.InputTokensDetails.CachedTokens != 0 || response.Usage.OutputTokensDetails.ReasoningTokens != 0 {
		t.Fatalf("response usage details = %#v", response.Usage)
	}
	if response.Store || response.Background || response.ServiceTier != "default" || response.Text.Format.Type != "text" {
		t.Fatalf("response defaults = %#v", response)
	}
	if response.MaxOutputTokens == nil || *response.MaxOutputTokens != 16 || response.Metadata["suite"] != "responses" {
		t.Fatalf("response echoed fields = %#v", response)
	}
	if len(response.Tools) != 1 || response.Tools[0].Name != "ignored" || response.Tools[0].Strict || response.ToolChoice != "auto" || !response.ParallelToolCalls {
		t.Fatalf("response tool defaults = %#v", response)
	}

	// Text parts are concatenated and non-text parts ignored, so the generation
	// context must match the same conversation written with plain strings.
	gotContext := capturedGenerationContext(t, chatModel)
	equivalent := strings.Replace(body, `[
				{"type":"input_text","text":"Hello "},
				{"type":"input_image","image_url":"https://example.com/image.png"},
				{"type":"input_text","text":"world"}
			]`, `"Hello world"`, 1)
	equivalent = strings.Replace(equivalent, `[{"type":"output_text","text":"Prior answer"}]`, `"Prior answer"`, 1)
	if equivalent == body {
		t.Fatal("equivalent request was not rewritten")
	}
	if want := generationContextFor(t, equivalent); gotContext.Digest() != want.Digest() {
		t.Fatalf("generation context digest = %016x, want %016x", gotContext.Digest(), want.Digest())
	}
	if chatModel.request.MaxCompletionTokens != 16 {
		t.Fatalf("domain max completion tokens = %d, want 16", chatModel.request.MaxCompletionTokens)
	}
}

func TestCreateResponseStringInputUsesDefaults(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{info: testModelInfo(), response: chat.Response{Content: "mock response"}}
	handler := newTestHandler(t, chatModel)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"mock-chat-model","input":"Hello","previous_response_id":null}`)),
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response responseResource
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !response.Store || response.MaxOutputTokens != nil || response.Metadata == nil {
		t.Fatalf("response defaults = %#v", response)
	}
	gotContext := capturedGenerationContext(t, chatModel)
	want := generationContextFor(t, `{"model":"mock-chat-model","input":[{"role":"user","content":"Hello"}]}`)
	if gotContext.Digest() != want.Digest() {
		t.Fatalf("string input context digest = %016x, want user message digest %016x", gotContext.Digest(), want.Digest())
	}
}

// capturedGenerationContext returns the context state of the last text request.
func capturedGenerationContext(t testing.TB, chatModel *testChatModel) chat.ContextState {
	t.Helper()
	if chatModel.request.ContextState == nil {
		t.Fatalf("domain request has no context state: %#v", chatModel.request)
	}
	return *chatModel.request.ContextState
}

// generationContextFor sends body to a fresh handler and returns the context
// state its text generation received.
func generationContextFor(t *testing.T, body string) chat.ContextState {
	t.Helper()
	chatModel := &testChatModel{info: testModelInfo(), response: chat.Response{Content: "mock response"}}
	recorder := httptest.NewRecorder()
	newTestHandler(t, chatModel).ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", recorder.Code, recorder.Body.String())
	}
	return capturedGenerationContext(t, chatModel)
}

func TestCreateResponseToolCallingMatchesUpstreamFixture(t *testing.T) {
	t.Parallel()

	chatModel, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{
		DefaultCompletionTokens: 32,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    4,
	})
	if err != nil {
		t.Fatalf("NewMockModel() error = %v", err)
	}
	handler := newTestHandler(t, chatModel)
	body := `{
		"model":"mock-chat-model",
		"input":[{"type":"message","role":"user","content":"What's the weather like in San Francisco?"}],
		"tool_choice":"required",
		"tools":[{
			"type":"function",
			"name":"get_weather",
			"description":"Get the current weather for a location",
			"parameters":{
				"type":"object",
				"properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"}},
				"required":["location"]
			}
		}]
	}`

	first := createTestResponse(t, handler, body)
	second := createTestResponse(t, handler, body)
	if len(first.Output) != 1 || first.Output[0].Type != "function_call" {
		t.Fatalf("first output = %#v", first.Output)
	}
	call := first.Output[0]
	if call.Status != "completed" || call.Name == nil || *call.Name != "get_weather" || call.CallID == nil || call.Arguments == nil {
		t.Fatalf("function call = %#v", call)
	}
	if first.ID == second.ID || !reflect.DeepEqual(first.Output, second.Output) {
		t.Fatalf("responses are not request-order independent: first=%#v second=%#v", first, second)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(*call.Arguments), &arguments); err != nil {
		t.Fatalf("arguments = %q: %v", *call.Arguments, err)
	}
	if location, ok := arguments["location"].(string); !ok || location == "" {
		t.Fatalf("arguments = %#v", arguments)
	}
	if len(first.Tools) != 1 || first.Tools[0].Name != "get_weather" || first.Tools[0].Strict {
		t.Fatalf("tools = %#v", first.Tools)
	}
	if first.ToolChoice != "required" || !first.ParallelToolCalls || first.MaxToolCalls != nil {
		t.Fatalf("tool settings = %#v", first)
	}
}

func TestCreateResponseRequiredParallelToolCalls(t *testing.T) {
	t.Parallel()

	plan := chat.ToolPlan{
		Calls: []chat.ToolCall{
			{ID: "fc_mock_0", CallID: "call_mock_0", Name: "one", Arguments: `{"value":1}`},
			{ID: "fc_mock_1", CallID: "call_mock_1", Name: "two", Arguments: `{"value":2}`},
		},
		Usage: chat.Usage{PromptTokens: 10, CompletionTokens: 8, TotalTokens: 18},
	}
	chatModel := &testChatModel{info: testModelInfo(), toolPlan: &plan}
	handler := newTestHandler(t, chatModel)
	body := `{
		"model":"mock-chat-model",
		"input":"Use tools",
		"tool_choice":"required",
		"parallel_tool_calls":true,
		"max_tool_calls":2,
		"tools":[
			{"type":"function","name":"one","parameters":{"type":"object"}},
			{"type":"function","name":"two","parameters":{"type":"object"}}
		]
	}`

	response := createTestResponse(t, handler, body)
	if len(response.Output) != 2 || response.Output[0].Type != "function_call" || response.Output[1].Type != "function_call" {
		t.Fatalf("output = %#v", response.Output)
	}
	if response.ToolChoice != "required" || !response.ParallelToolCalls || response.MaxToolCalls == nil || *response.MaxToolCalls != 2 {
		t.Fatalf("tool settings = %#v", response)
	}
	if chatModel.toolRequest.Choice != chat.ToolChoiceRequired || !chatModel.toolRequest.ParallelToolCalls || chatModel.toolRequest.MaxToolCalls != 2 {
		t.Fatalf("domain tool request = %#v", chatModel.toolRequest)
	}
	if response.Usage == nil || response.Usage.InputTokens != 40 || response.Usage.OutputTokens != 8 || response.Usage.TotalTokens != 48 {
		t.Fatalf("usage = %#v", response.Usage)
	}
}

func TestCreateResponseErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		status     int
		param      string
		code       string
		modelError error
	}{
		{name: "invalid JSON", body: `{`, status: http.StatusBadRequest},
		{name: "multiple values", body: `{}` + `{}`, status: http.StatusBadRequest},
		{name: "missing model", body: `{"input":"Hello"}`, status: http.StatusBadRequest, param: "model"},
		{name: "missing input", body: `{"model":"mock-chat-model"}`, status: http.StatusBadRequest, param: "input"},
		{name: "null input", body: `{"model":"mock-chat-model","input":null}`, status: http.StatusBadRequest, param: "input"},
		{name: "empty input", body: `{"model":"mock-chat-model","input":[]}`, status: http.StatusBadRequest, param: "input", code: "invalid_value"},
		{name: "invalid input shape", body: `{"model":"mock-chat-model","input":{}}`, status: http.StatusBadRequest, param: "input", code: "invalid_value"},
		{name: "unsupported role", body: `{"model":"mock-chat-model","input":[{"type":"message","role":"tool","content":"Hello"}]}`, status: http.StatusBadRequest, param: "input.0.role", code: "invalid_value"},
		{name: "missing content", body: `{"model":"mock-chat-model","input":[{"type":"message","role":"user"}]}`, status: http.StatusBadRequest, param: "input.0.content", code: "invalid_value"},
		{name: "missing function arguments", body: `{"model":"mock-chat-model","input":[{"type":"function_call","call_id":"call_1","name":"lookup"}]}`, status: http.StatusBadRequest, param: "input.0.arguments", code: "invalid_value"},
		{name: "missing function output", body: `{"model":"mock-chat-model","input":[{"type":"function_call_output","call_id":"call_1"}]}`, status: http.StatusBadRequest, param: "input.0.output", code: "invalid_value"},
		{name: "invalid reasoning summary", body: `{"model":"mock-chat-model","input":[{"type":"reasoning","summary":[{"type":"input_text","text":"No"}]}]}`, status: http.StatusBadRequest, param: "input.0.summary.0", code: "invalid_value"},
		{name: "unsupported input item", body: `{"model":"mock-chat-model","input":[{"type":"web_search_call"}]}`, status: http.StatusBadRequest, param: "input.0.type", code: "unsupported_value"},
		{name: "invalid output tokens", body: `{"model":"mock-chat-model","input":"Hello","max_output_tokens":15}`, status: http.StatusBadRequest, param: "max_output_tokens", code: "invalid_value"},
		{name: "invalid tool choice", body: `{"model":"mock-chat-model","input":"Hello","tool_choice":"sometimes"}`, status: http.StatusBadRequest, param: "tool_choice", code: "invalid_value"},
		{name: "required without tools", body: `{"model":"mock-chat-model","input":"Hello","tool_choice":"required"}`, status: http.StatusBadRequest, param: "tool_choice", code: "invalid_value"},
		{name: "invalid max tool calls", body: `{"model":"mock-chat-model","input":"Hello","max_tool_calls":0}`, status: http.StatusBadRequest, param: "max_tool_calls", code: "invalid_value"},
		{name: "unsupported tool type", body: `{"model":"mock-chat-model","input":"Hello","tools":[{"type":"web_search","name":"search"}]}`, status: http.StatusBadRequest, param: "tools.0.type", code: "unsupported_value"},
		{name: "empty tool name", body: `{"model":"mock-chat-model","input":"Hello","tools":[{"type":"function","name":" "}]}`, status: http.StatusBadRequest, param: "tools.0.name", code: "invalid_value"},
		{name: "duplicate tool name", body: `{"model":"mock-chat-model","input":"Hello","tools":[{"type":"function","name":"same"},{"type":"function","name":"same"}]}`, status: http.StatusBadRequest, param: "tools.1.name", code: "invalid_value"},
		{name: "invalid tool parameters", body: `{"model":"mock-chat-model","input":"Hello","tools":[{"type":"function","name":"lookup","parameters":[]}]}`, status: http.StatusBadRequest, param: "tools.0.parameters", code: "invalid_value"},
		{name: "malformed previous response", body: `{"model":"mock-chat-model","input":"Hello","previous_response_id":"resp_123"}`, status: http.StatusBadRequest, param: "previous_response_id", code: "invalid_value"},
		{name: "unknown model", body: `{"model":"missing","input":"Hello"}`, status: http.StatusNotFound, param: "model", code: "model_not_found"},
		{name: "model failure", body: `{"model":"mock-chat-model","input":"Hello"}`, status: http.StatusInternalServerError, modelError: errors.New("failed")},
		{name: "model token limit", body: `{"model":"mock-chat-model","input":"Hello","max_output_tokens":100}`, status: http.StatusBadRequest, param: "max_output_tokens", code: "limit_exceeded", modelError: fmt.Errorf("%w: maximum 64", chat.ErrCompletionTokenLimit)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			chatModel := &testChatModel{
				info:        testModelInfo(),
				response:    chat.Response{Content: "mock response"},
				completeErr: tt.modelError,
			}
			handler := newTestHandler(t, chatModel)
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(tt.body)))

			if recorder.Code != tt.status {
				t.Fatalf("status = %d, want %d; body = %s", recorder.Code, tt.status, recorder.Body.String())
			}
			var response errorResponse
			if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Message == "" || response.Error.Type == "" {
				t.Fatalf("error response = %#v", response)
			}
			if pointerValue(response.Error.Param) != tt.param {
				t.Fatalf("error param = %q, want %q", pointerValue(response.Error.Param), tt.param)
			}
			if pointerValue(response.Error.Code) != tt.code {
				t.Fatalf("error code = %q, want %q", pointerValue(response.Error.Code), tt.code)
			}
		})
	}
}

func createTestResponse(t testing.TB, handler http.Handler, body string) responseResource {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)),
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response responseResource
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return response
}

func TestHandlerRejectsUnsupportedResponsesMethod(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, &testChatModel{info: testModelInfo()})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/responses", nil))

	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusMethodNotAllowed)
	}
}
