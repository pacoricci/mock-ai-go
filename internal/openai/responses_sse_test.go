package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type parsedResponseSSEEvent struct {
	event string
	data  string
}

func TestStreamingResponseMatchesNonStreamingResponse(t *testing.T) {
	t.Parallel()

	chatModel, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{
		DefaultCompletionTokens: 16,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
	})
	if err != nil {
		t.Fatalf("NewMockModel() error = %v", err)
	}
	handler := newTestHandler(t, chatModel)
	handler.now = func() time.Time { return time.Unix(1_700_000_000, 0) }
	baseBody := `"model":"mock-chat-model","input":[{"type":"message","role":"user","content":"Hello"}],"max_output_tokens":16,"metadata":{"suite":"stream"}`

	nonStreamingRecorder := httptest.NewRecorder()
	handler.ServeHTTP(
		nonStreamingRecorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{`+baseBody+`}`)),
	)
	if nonStreamingRecorder.Code != http.StatusOK {
		t.Fatalf("non-streaming status = %d; body = %s", nonStreamingRecorder.Code, nonStreamingRecorder.Body.String())
	}
	var nonStreaming responseResource
	if err := json.NewDecoder(nonStreamingRecorder.Body).Decode(&nonStreaming); err != nil {
		t.Fatalf("decode non-streaming response: %v", err)
	}

	streamingRecorder := httptest.NewRecorder()
	handler.ServeHTTP(
		streamingRecorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{`+baseBody+`,"stream":true}`)),
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

	events := parseResponseSSEEvents(t, streamingRecorder.Body.String())
	if len(events) != 17 {
		t.Fatalf("SSE events = %d, want 17; body = %s", len(events), streamingRecorder.Body.String())
	}
	if events[len(events)-1].event != "" || events[len(events)-1].data != "[DONE]" {
		t.Fatalf("terminal SSE event = %#v, want [DONE]", events[len(events)-1])
	}

	wantTypes := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
	}
	for range 8 {
		wantTypes = append(wantTypes, "response.output_text.delta")
	}
	wantTypes = append(wantTypes,
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	)

	var streamed strings.Builder
	var completed responseResource
	var responseID, messageID string
	for index, event := range events[:len(events)-1] {
		var envelope struct {
			Type           string             `json:"type"`
			SequenceNumber int                `json:"sequence_number"`
			Response       responseResource   `json:"response"`
			ItemID         string             `json:"item_id"`
			Item           responseOutputItem `json:"item"`
			Delta          string             `json:"delta"`
		}
		if err := json.Unmarshal([]byte(event.data), &envelope); err != nil {
			t.Fatalf("decode SSE event %d: %v", index, err)
		}
		if envelope.Type != wantTypes[index] || event.event != envelope.Type {
			t.Fatalf("event %d type = header %q body %q, want %q", index, event.event, envelope.Type, wantTypes[index])
		}
		if envelope.SequenceNumber != index {
			t.Fatalf("event %d sequence = %d, want %d", index, envelope.SequenceNumber, index)
		}
		switch envelope.Type {
		case "response.created":
			responseID = envelope.Response.ID
			if envelope.Response.Status != "in_progress" || envelope.Response.Output == nil || envelope.Response.Usage != nil {
				t.Fatalf("created response = %#v", envelope.Response)
			}
		case "response.output_item.added":
			messageID = envelope.Item.ID
			if envelope.Item.Status != "in_progress" || envelope.Item.Content == nil {
				t.Fatalf("added item = %#v", envelope.Item)
			}
		case "response.output_text.delta":
			if envelope.ItemID != messageID {
				t.Fatalf("delta item ID = %q, want %q", envelope.ItemID, messageID)
			}
			streamed.WriteString(envelope.Delta)
		case "response.completed":
			completed = envelope.Response
		}
	}

	if responseID == "" || completed.ID != responseID || len(completed.Output) != 1 || completed.Output[0].ID != messageID {
		t.Fatalf("stream identities = response %q message %q completed %#v", responseID, messageID, completed)
	}
	if nonStreaming.Output[0].Content == nil {
		t.Fatalf("non-streaming content = nil")
	}
	if got, want := streamed.String(), (*nonStreaming.Output[0].Content)[0].Text; got != want {
		t.Fatalf("streamed content = %q, want %q", got, want)
	}
	if completed.Output[0].Content == nil || (*completed.Output[0].Content)[0].Text != streamed.String() {
		t.Fatalf("completed content = %#v, want %q", completed.Output[0].Content, streamed.String())
	}
	if completed.Usage == nil || nonStreaming.Usage == nil || *completed.Usage != *nonStreaming.Usage {
		t.Fatalf("streaming usage = %#v, want %#v", completed.Usage, nonStreaming.Usage)
	}
}

func TestStreamingToolResponseMatchesNonStreamingResponse(t *testing.T) {
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
	handler.now = func() time.Time { return time.Unix(1_700_000_000, 0) }
	baseBody := `
		"model":"mock-chat-model",
		"input":[{"type":"message","role":"user","content":"What's the weather like in San Francisco?"}],
		"tool_choice":"required",
		"tools":[{
			"type":"function",
			"name":"get_weather",
			"description":"Get the current weather for a location",
			"parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
		}]
	`

	nonStreaming := createTestResponse(t, handler, `{`+baseBody+`}`)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{`+baseBody+`,"stream":true}`)),
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q", got)
	}

	events := parseResponseSSEEvents(t, recorder.Body.String())
	var completed responseResource
	var arguments strings.Builder
	var types []string
	deltaCount := 0
	for index, event := range events[:len(events)-1] {
		var envelope struct {
			Type           string           `json:"type"`
			SequenceNumber int              `json:"sequence_number"`
			Response       responseResource `json:"response"`
			Delta          string           `json:"delta"`
			Arguments      string           `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(event.data), &envelope); err != nil {
			t.Fatal(err)
		}
		if envelope.Type != event.event || envelope.SequenceNumber != index {
			t.Fatalf("event %d: %+v", index, envelope)
		}
		if event.event == "response.function_call_arguments.delta" {
			arguments.WriteString(envelope.Delta)
			deltaCount++
			continue
		}
		types = append(types, event.event)
		if event.event == "response.function_call_arguments.done" && envelope.Arguments != arguments.String() {
			t.Fatal("argument deltas differ from done")
		}
		if event.event == "response.completed" {
			completed = envelope.Response
		}
	}
	wantTypes := []string{"response.created", "response.in_progress", "response.output_item.added", "response.function_call_arguments.done", "response.output_item.done", "response.completed"}
	if !reflect.DeepEqual(types, wantTypes) || deltaCount < 2 {
		t.Fatalf("types=%v delta count=%d", types, deltaCount)
	}
	if arguments.String() != *nonStreaming.Output[0].Arguments {
		t.Fatal("streamed arguments differ from JSON")
	}
	if events[len(events)-1].data != "[DONE]" {
		t.Fatalf("terminal event = %#v", events[len(events)-1])
	}
	if len(completed.Output) != 1 || !reflect.DeepEqual(completed.Output, nonStreaming.Output) {
		t.Fatalf("completed output = %#v, want %#v", completed.Output, nonStreaming.Output)
	}
	if completed.Usage == nil || nonStreaming.Usage == nil || *completed.Usage != *nonStreaming.Usage {
		t.Fatalf("completed usage = %#v, want %#v", completed.Usage, nonStreaming.Usage)
	}
}

func TestStreamingParallelToolResponseUsesDistinctOutputIndexes(t *testing.T) {
	t.Parallel()

	plan := chat.ToolPlan{
		Calls: []chat.ToolCall{
			{ID: "fc_mock_0", CallID: "call_mock_0", Name: "one", Arguments: `{"value":1}`},
			{ID: "fc_mock_1", CallID: "call_mock_1", Name: "two", Arguments: `{"value":2}`},
		},
		Usage: chat.Usage{PromptTokens: 10, CompletionTokens: 8, TotalTokens: 18},
	}
	handler := newTestHandler(t, &testChatModel{info: testModelInfo(), toolPlan: &plan})
	recorder := httptest.NewRecorder()
	body := `{
		"model":"mock-chat-model",
		"input":"Use tools",
		"stream":true,
		"tool_choice":"required",
		"tools":[
			{"type":"function","name":"one"},
			{"type":"function","name":"two"}
		]
	}`
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	events := parseResponseSSEEvents(t, recorder.Body.String())
	if len(events) != 12 {
		t.Fatalf("events = %d, want 12; body = %s", len(events), recorder.Body.String())
	}
	seenDoneIndexes := make([]int, 0, 2)
	for index, event := range events[:len(events)-1] {
		var envelope struct {
			Type           string `json:"type"`
			SequenceNumber int    `json:"sequence_number"`
			OutputIndex    int    `json:"output_index"`
		}
		if err := json.Unmarshal([]byte(event.data), &envelope); err != nil {
			t.Fatalf("decode event %d: %v", index, err)
		}
		if envelope.SequenceNumber != index {
			t.Fatalf("event %d sequence = %d", index, envelope.SequenceNumber)
		}
		if envelope.Type == "response.output_item.done" {
			seenDoneIndexes = append(seenDoneIndexes, envelope.OutputIndex)
		}
	}
	if !reflect.DeepEqual(seenDoneIndexes, []int{0, 1}) {
		t.Fatalf("done output indexes = %#v", seenDoneIndexes)
	}
}

func TestStreamingResponseValidationErrorUsesJSON(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{info: testModelInfo(), streamErr: chat.ErrCompletionTokenLimit}
	handler := newTestHandler(t, chatModel)
	recorder := httptest.NewRecorder()
	body := `{"model":"mock-chat-model","input":"Hello","stream":true}`

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusBadRequest, recorder.Body.String())
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
}

func TestStreamingResponseFailureAfterStartUsesTerminalEvents(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info:                 testModelInfo(),
		streamChunks:         []chat.Chunk{{Delta: []byte("amber")}},
		streamErr:            errors.New("failed after output"),
		streamErrAfterChunks: true,
	}
	handler := newTestHandler(t, chatModel)
	recorder := httptest.NewRecorder()
	body := `{"model":"mock-chat-model","input":"Hello","stream":true}`

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	events := parseResponseSSEEvents(t, recorder.Body.String())
	if len(events) != 8 {
		t.Fatalf("events = %d, want 8; body = %s", len(events), recorder.Body.String())
	}
	wantTypes := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"error",
		"response.failed",
	}
	for index, wantType := range wantTypes {
		if events[index].event != wantType {
			t.Fatalf("event %d type = %q, want %q", index, events[index].event, wantType)
		}
	}
	if events[len(events)-1].data != "[DONE]" {
		t.Fatalf("terminal event = %#v", events[len(events)-1])
	}

	var failed responseLifecycleEvent
	if err := json.Unmarshal([]byte(events[6].data), &failed); err != nil {
		t.Fatalf("decode failed event: %v", err)
	}
	if failed.Response.Status != "failed" || failed.Response.Error == nil {
		t.Fatalf("failed response = %#v", failed.Response)
	}
}

func TestStreamingResponseEscapesText(t *testing.T) {
	t.Parallel()

	want := "quote \" line\n snowman ☃"
	chatModel := &testChatModel{
		info:         testModelInfo(),
		streamChunks: []chat.Chunk{{Delta: []byte(want)}},
		streamUsage:  chat.Usage{CompletionTokens: 1, TotalTokens: 1},
	}
	handler := newTestHandler(t, chatModel)
	recorder := httptest.NewRecorder()
	body := `{"model":"mock-chat-model","input":"Hello","stream":true}`

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	events := parseResponseSSEEvents(t, recorder.Body.String())
	var delta responseOutputTextDeltaEvent
	if err := json.Unmarshal([]byte(events[4].data), &delta); err != nil {
		t.Fatalf("decode delta event: %v", err)
	}
	if delta.Delta != want {
		t.Fatalf("delta = %q, want %q", delta.Delta, want)
	}
	var done responseOutputTextDoneEvent
	if err := json.Unmarshal([]byte(events[5].data), &done); err != nil {
		t.Fatalf("decode done event: %v", err)
	}
	if done.Text != want {
		t.Fatalf("done text = %q, want %q", done.Text, want)
	}
}

func TestStreamingResponseCancellationDoesNotEmitFailure(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info:                 testModelInfo(),
		streamChunks:         []chat.Chunk{{Delta: []byte("amber")}},
		streamErr:            context.Canceled,
		streamErrAfterChunks: true,
	}
	handler := newTestHandler(t, chatModel)
	recorder := httptest.NewRecorder()
	body := `{"model":"mock-chat-model","input":"Hello","stream":true}`

	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	events := parseResponseSSEEvents(t, recorder.Body.String())
	if len(events) != 5 || events[len(events)-1].event != "response.output_text.delta" {
		t.Fatalf("events after cancellation = %#v", events)
	}
}

func parseResponseSSEEvents(t testing.TB, body string) []parsedResponseSSEEvent {
	t.Helper()

	blocks := strings.Split(strings.TrimSpace(body), "\n\n")
	events := make([]parsedResponseSSEEvent, 0, len(blocks))
	for _, block := range blocks {
		var event parsedResponseSSEEvent
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				event.event = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				event.data = strings.TrimPrefix(line, "data: ")
			}
		}
		if event.data == "" {
			t.Fatalf("SSE block has no data: %q", block)
		}
		events = append(events, event)
	}
	return events
}
