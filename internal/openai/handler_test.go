package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
	"github.com/pacoricci/mock-ai-go/internal/model"
)

func TestCreateChatCompletion(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info: model.Info{ID: "mock-chat-model", Kind: model.KindChat},
		response: chat.Response{
			Content: "amber orbit cedar",
			Usage:   chat.Usage{PromptTokens: 9, CompletionTokens: 3, TotalTokens: 12},
		},
	}
	handler := newTestHandler(t, chatModel)
	handler.now = func() time.Time { return time.Unix(1_700_000_000, 0) }

	body := `{
		"model":"mock-chat-model",
		"messages":[{"role":"user","content":"Hello"}],
		"max_completion_tokens":3,
		"seed":42,
		"temperature":0
	}`
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var response chatCompletionResponse
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.ID != "chatcmpl-mock-1" || response.Object != "chat.completion" {
		t.Fatalf("response identity = %#v", response)
	}
	if response.Created != 1_700_000_000 || response.Model != "mock-chat-model" {
		t.Fatalf("response metadata = %#v", response)
	}
	if len(response.Choices) != 1 {
		t.Fatalf("len(Choices) = %d, want 1", len(response.Choices))
	}
	choice := response.Choices[0]
	if choice.Index != 0 || choice.Message.Role != "assistant" || pointerValue(choice.Message.Content) != "amber orbit cedar" || choice.FinishReason != "stop" {
		t.Fatalf("choice = %#v", choice)
	}
	if response.Usage != (chatCompletionUsage{PromptTokens: 9, CompletionTokens: 3, TotalTokens: 12}) {
		t.Fatalf("usage = %#v", response.Usage)
	}
	if len(chatModel.request.Messages) != 1 || chatModel.request.Messages[0] != (chat.Message{Role: "user", Content: "Hello"}) {
		t.Fatalf("domain request = %#v", chatModel.request)
	}
	if chatModel.request.MaxCompletionTokens != 3 || chatModel.request.Seed == nil || *chatModel.request.Seed != 42 {
		t.Fatalf("domain generation options = %#v", chatModel.request)
	}
}

func TestCreateChatCompletionErrors(t *testing.T) {
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
		{name: "missing model", body: `{"messages":[{"role":"user","content":"Hello"}]}`, status: http.StatusBadRequest, param: "model"},
		{name: "missing messages", body: `{"model":"mock-chat-model"}`, status: http.StatusBadRequest, param: "messages"},
		{name: "missing role", body: `{"model":"mock-chat-model","messages":[{"content":"Hello"}]}`, status: http.StatusBadRequest, param: "messages.0.role"},
		{name: "invalid completion tokens", body: `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":0}`, status: http.StatusBadRequest, param: "max_completion_tokens", code: "invalid_value"},
		{name: "invalid legacy tokens", body: `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":-1}`, status: http.StatusBadRequest, param: "max_tokens", code: "invalid_value"},
		{name: "conflicting token limits", body: `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"max_tokens":8}`, status: http.StatusBadRequest, param: "max_completion_tokens", code: "invalid_value"},
		{name: "stream options without stream", body: `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"stream_options":{"include_usage":true}}`, status: http.StatusBadRequest, param: "stream_options", code: "invalid_value"},
		{name: "unknown model", body: `{"model":"missing","messages":[{"role":"user","content":"Hello"}]}`, status: http.StatusNotFound, param: "model", code: "model_not_found"},
		{name: "model failure", body: `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}]}`, status: http.StatusInternalServerError, modelError: errors.New("failed")},
		{name: "model token limit", body: `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":100}`, status: http.StatusBadRequest, param: "max_completion_tokens", code: "limit_exceeded", modelError: fmt.Errorf("%w: maximum 64", chat.ErrCompletionTokenLimit)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			chatModel := &testChatModel{
				info:        model.Info{ID: "mock-chat-model", Kind: model.KindChat},
				response:    chat.Response{Content: "mock response"},
				completeErr: tt.modelError,
			}
			handler := newTestHandler(t, chatModel)
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tt.body))
			recorder := httptest.NewRecorder()

			handler.ServeHTTP(recorder, request)

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

func TestHandlerRejectsUnsupportedMethod(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, &testChatModel{info: model.Info{ID: "mock-chat-model", Kind: model.KindChat}})
	request := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusMethodNotAllowed)
	}
}

func TestNewHandlerRejectsNilRegistry(t *testing.T) {
	t.Parallel()

	if _, err := NewHandler(nil, testResponseStoreOptions()); !errors.Is(err, ErrNilChatRegistry) {
		t.Fatalf("NewHandler() error = %v, want %v", err, ErrNilChatRegistry)
	}
}

func TestNewHandlerRejectsInvalidResponseStoreOptions(t *testing.T) {
	t.Parallel()

	registry, err := chat.NewRegistry(&testChatModel{info: testModelInfo()})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	tests := []ResponseStoreOptions{
		{TTL: 0, MaxEntries: 1, MaxBytes: 1},
		{TTL: time.Hour, MaxEntries: 0, MaxBytes: 1},
		{TTL: time.Hour, MaxEntries: 1, MaxBytes: 0},
	}
	for _, options := range tests {
		if _, err := NewHandler(registry, options); err == nil {
			t.Fatalf("NewHandler(%#v) succeeded", options)
		}
	}
}

func newTestHandler(t *testing.T, chatModel chat.Model) *Handler {
	t.Helper()

	registry, err := chat.NewRegistry(chatModel)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	handler, err := NewHandler(registry, testResponseStoreOptions())
	if err != nil {
		t.Fatalf("NewHandler() error = %v", err)
	}
	return handler
}

func testResponseStoreOptions() ResponseStoreOptions {
	return ResponseStoreOptions{
		TTL:        time.Hour,
		MaxEntries: 1_000,
		MaxBytes:   8 << 20,
	}
}

func pointerValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func testModelInfo() model.Info {
	return model.Info{ID: "mock-chat-model", Kind: model.KindChat}
}

type testChatModel struct {
	info                 model.Info
	request              chat.Request
	toolRequest          chat.ToolRequest
	toolPlan             *chat.ToolPlan
	toolPlanErr          error
	response             chat.Response
	completeErr          error
	streamChunks         []chat.Chunk
	streamUsage          chat.Usage
	streamErr            error
	streamErrAfterChunks bool
}

func (m *testChatModel) Info() model.Info {
	return m.info
}

func (m *testChatModel) PlanTools(_ context.Context, request chat.ToolRequest) (chat.ToolPlan, error) {
	m.toolRequest = request
	if m.toolPlan != nil {
		return *m.toolPlan, m.toolPlanErr
	}
	return chat.ToolPlan{UseText: true}, m.toolPlanErr
}

func (m *testChatModel) Complete(_ context.Context, request chat.Request) (chat.Response, error) {
	m.request = request
	return m.response, m.completeErr
}

func (m *testChatModel) Stream(_ context.Context, request chat.Request, emit chat.Emit) (chat.Usage, error) {
	m.request = request
	if m.streamErr != nil && !m.streamErrAfterChunks {
		return chat.Usage{}, m.streamErr
	}
	for _, chunk := range m.streamChunks {
		if err := emit(chunk); err != nil {
			return chat.Usage{}, err
		}
	}
	if m.streamErr != nil {
		return m.streamUsage, m.streamErr
	}
	return m.streamUsage, nil
}

func (m *testChatModel) StreamTools(ctx context.Context, _ chat.Request, plan chat.ToolPlan, emit chat.ToolEmit) error {
	for index, call := range plan.Calls {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := emit(chat.ToolChunk{Index: index, Delta: call.Arguments, First: true, Last: true}); err != nil {
			return err
		}
	}
	return ctx.Err()
}
