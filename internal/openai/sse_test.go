package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func TestStreamingMatchesNonStreamingCompletion(t *testing.T) {
	t.Parallel()

	chatModel, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
	})
	if err != nil {
		t.Fatalf("NewMockModel() error = %v", err)
	}
	handler := newTestHandler(t, chatModel)
	handler.now = func() time.Time { return time.Unix(1_700_000_000, 0) }

	baseBody := `"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":5,"seed":42`
	nonStreamingRecorder := httptest.NewRecorder()
	handler.ServeHTTP(
		nonStreamingRecorder,
		httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{`+baseBody+`}`)),
	)
	if nonStreamingRecorder.Code != http.StatusOK {
		t.Fatalf("non-streaming status = %d; body = %s", nonStreamingRecorder.Code, nonStreamingRecorder.Body.String())
	}
	var nonStreaming chatCompletionResponse
	if err := json.NewDecoder(nonStreamingRecorder.Body).Decode(&nonStreaming); err != nil {
		t.Fatalf("decode non-streaming response: %v", err)
	}

	streamingRecorder := httptest.NewRecorder()
	handler.ServeHTTP(
		streamingRecorder,
		httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{`+baseBody+`,"stream":true,"stream_options":{"include_usage":true}}`)),
	)
	if streamingRecorder.Code != http.StatusOK {
		t.Fatalf("streaming status = %d; body = %s", streamingRecorder.Code, streamingRecorder.Body.String())
	}
	if got := streamingRecorder.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	if got := streamingRecorder.Header().Get("Cache-Control"); got != "no-cache" {
		t.Fatalf("Cache-Control = %q, want no-cache", got)
	}

	events := parseSSEEvents(t, streamingRecorder.Body.String())
	if len(events) != 7 {
		t.Fatalf("SSE events = %d, want 7; body = %s", len(events), streamingRecorder.Body.String())
	}
	if events[len(events)-1] != "[DONE]" {
		t.Fatalf("last SSE event = %q, want [DONE]", events[len(events)-1])
	}

	var streamed strings.Builder
	var finalUsage chatCompletionUsage
	finishSeen := false
	for index, event := range events[:len(events)-1] {
		var raw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(event), &raw); err != nil {
			t.Fatalf("decode SSE event %d: %v", index, err)
		}
		if _, ok := raw["usage"]; !ok {
			t.Fatalf("SSE event %d does not contain usage", index)
		}

		var chunk chatCompletionStreamChunkWithUsage
		if err := json.Unmarshal([]byte(event), &chunk); err != nil {
			t.Fatalf("decode SSE chunk %d: %v", index, err)
		}
		if chunk.Object != "chat.completion.chunk" || chunk.Created != 1_700_000_000 || chunk.Model != "mock-chat-model" {
			t.Fatalf("SSE chunk metadata = %#v", chunk)
		}
		if len(chunk.Choices) == 0 {
			if chunk.Usage == nil {
				t.Fatalf("usage chunk %d has nil usage", index)
			}
			finalUsage = *chunk.Usage
			continue
		}
		choice := chunk.Choices[0]
		if choice.Delta.Content != nil {
			streamed.WriteString(*choice.Delta.Content)
		}
		if choice.FinishReason != nil && *choice.FinishReason == "stop" {
			finishSeen = true
		}
	}

	if got, want := streamed.String(), pointerValue(nonStreaming.Choices[0].Message.Content); got != want {
		t.Fatalf("streamed content = %q, want %q", got, want)
	}
	if finalUsage != nonStreaming.Usage {
		t.Fatalf("streaming usage = %#v, want %#v", finalUsage, nonStreaming.Usage)
	}
	if !finishSeen {
		t.Fatal("stream did not contain a stop finish reason")
	}
}

func TestStreamingOmitsUsageWhenNotRequested(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info:         testModelInfo(),
		streamChunks: []chat.Chunk{{Delta: []byte("amber")}},
		streamUsage:  chat.Usage{PromptTokens: 8, CompletionTokens: 1, TotalTokens: 9},
	}
	handler := newTestHandler(t, chatModel)
	body := `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"stream":true}`
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if chatModel.request.StreamNonce != 1 {
		t.Fatalf("stream nonce = %d, want 1", chatModel.request.StreamNonce)
	}

	events := parseSSEEvents(t, recorder.Body.String())
	if len(events) != 4 {
		t.Fatalf("SSE events = %d, want 4; body = %s", len(events), recorder.Body.String())
	}
	for index, event := range events[:len(events)-1] {
		if strings.Contains(event, `"usage"`) {
			t.Fatalf("SSE event %d unexpectedly contains usage: %s", index, event)
		}
	}
}

func TestStreamingModelValidationErrorUsesJSON(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info:      testModelInfo(),
		streamErr: chat.ErrCompletionTokenLimit,
	}
	handler := newTestHandler(t, chatModel)
	body := `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"stream":true}`
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusBadRequest, recorder.Body.String())
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
}

func TestStreamingEscapesJSONContent(t *testing.T) {
	t.Parallel()

	want := "quote \" line\n snowman ☃"
	chatModel := &testChatModel{
		info:         testModelInfo(),
		streamChunks: []chat.Chunk{{Delta: []byte(want)}},
		streamUsage:  chat.Usage{CompletionTokens: 1, TotalTokens: 1},
	}
	handler := newTestHandler(t, chatModel)
	body := `{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"stream":true}`
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	events := parseSSEEvents(t, recorder.Body.String())
	var chunk chatCompletionStreamChunk
	if err := json.Unmarshal([]byte(events[1]), &chunk); err != nil {
		t.Fatalf("decode content event: %v; event = %s", err, events[1])
	}
	if chunk.Choices[0].Delta.Content == nil || *chunk.Choices[0].Delta.Content != want {
		t.Fatalf("streamed content = %#v, want %q", chunk.Choices[0].Delta.Content, want)
	}
}

func parseSSEEvents(t *testing.T, body string) []string {
	t.Helper()

	blocks := strings.Split(strings.TrimSpace(body), "\n\n")
	events := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if !strings.HasPrefix(block, "data: ") {
			t.Fatalf("invalid SSE block %q", block)
		}
		events = append(events, strings.TrimPrefix(block, "data: "))
	}
	return events
}
