package openai

import (
	"bytes"
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
)

func TestResponseContinuationJSONUsesStoredState(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	first := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"First turn",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	if !first.Store || !isResponseID(first.ID) {
		t.Fatalf("first response identity = %#v", first)
	}

	secondBody := fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Second turn",
		"previous_response_id":%q,
		"max_output_tokens":16,
		"tool_choice":"none"
	}`, first.ID)
	second := createTestResponse(t, handler, secondBody)
	repeated := createTestResponse(t, handler, secondBody)
	if responseText(t, second) != responseText(t, repeated) {
		t.Fatalf("repeated continuation changed output: %q and %q", responseText(t, second), responseText(t, repeated))
	}
	if second.ID == repeated.ID {
		t.Fatalf("repeated requests reused response ID %q", second.ID)
	}
	if second.PreviousResponseID == nil || *second.PreviousResponseID != first.ID {
		t.Fatalf("second previous_response_id = %#v, want %q", second.PreviousResponseID, first.ID)
	}
	if first.Usage == nil || second.Usage == nil || second.Usage.InputTokens <= first.Usage.InputTokens {
		t.Fatalf("continuation usage did not grow: first=%#v second=%#v", first.Usage, second.Usage)
	}

	changedLimit := createTestResponse(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Second turn",
		"previous_response_id":%q,
		"max_output_tokens":32,
		"tool_choice":"none"
	}`, first.ID))
	if changedLimit.Usage == nil || changedLimit.Usage.OutputTokens != 32 {
		t.Fatalf("changed output limit usage = %#v", changedLimit.Usage)
	}
	if len(strings.Fields(responseText(t, changedLimit))) != 32 {
		t.Fatalf("changed output limit text = %q", responseText(t, changedLimit))
	}

	previousID := second.ID
	for turn := 3; turn <= 10; turn++ {
		response := createTestResponse(t, handler, fmt.Sprintf(`{
			"model":"mock-chat-model",
			"input":"Turn %d",
			"previous_response_id":%q,
			"max_output_tokens":16,
			"tool_choice":"none"
		}`, turn, previousID))
		if !isResponseID(response.ID) {
			t.Fatalf("turn %d response ID = %q", turn, response.ID)
		}
		previousID = response.ID
	}
}

func TestResponseStoreFalseIsNotContinuableOverHTTP(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	response := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"Do not retain this turn",
		"max_output_tokens":16,
		"tool_choice":"none",
		"store":false
	}`)
	if response.Store {
		t.Fatalf("store = true, want false")
	}
	assertPreviousResponseErrorCode(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, response.ID), "previous_response_not_found")
}

func TestStreamingResponseContinuationUsesStoredState(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	first := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"First JSON turn",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	body := fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Second SSE turn",
		"previous_response_id":%q,
		"max_output_tokens":16,
		"tool_choice":"none",
		"stream":true
	}`, first.ID)
	second := streamCompletedResponse(t, handler, body)
	repeated := streamCompletedResponse(t, handler, body)
	if responseText(t, second) != responseText(t, repeated) {
		t.Fatalf("repeated SSE continuation changed output: %q and %q", responseText(t, second), responseText(t, repeated))
	}
	if second.ID == repeated.ID {
		t.Fatalf("repeated SSE requests reused response ID %q", second.ID)
	}
	if first.Usage == nil || second.Usage == nil || second.Usage.InputTokens <= first.Usage.InputTokens {
		t.Fatalf("SSE continuation usage did not grow: first=%#v second=%#v", first.Usage, second.Usage)
	}

	notStored := streamCompletedResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"Do not store this stream",
		"max_output_tokens":16,
		"tool_choice":"none",
		"store":false,
		"stream":true
	}`)
	assertPreviousResponseErrorCode(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue missing stream",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, notStored.ID), "previous_response_not_found")
}

func TestResponseContinuationIsProcessLocal(t *testing.T) {
	t.Parallel()

	firstHandler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	first := createTestResponse(t, firstHandler, `{
		"model":"mock-chat-model",
		"input":"First turn",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	secondHandler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	assertPreviousResponseErrorCode(t, secondHandler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after restart",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, first.ID), "previous_response_not_found")
}

func TestResponseContinuationExpiresAndEvicts(t *testing.T) {
	t.Parallel()

	clock := time.Unix(1_700_000_000, 0)
	expiringHandler := newTestHandlerWithOptions(t, newContinuationTestModel(t, "mock-chat-model"), ResponseStoreOptions{
		TTL: time.Minute, MaxEntries: 10, MaxBytes: 1 << 20,
	})
	expiringHandler.now = func() time.Time { return clock }
	expiring := createTestResponse(t, expiringHandler, `{
		"model":"mock-chat-model",
		"input":"Expire this turn",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	clock = clock.Add(time.Minute)
	assertPreviousResponseErrorCode(t, expiringHandler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after expiry",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, expiring.ID), "previous_response_not_found")

	evictingHandler := newTestHandlerWithOptions(t, newContinuationTestModel(t, "mock-chat-model"), ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 1, MaxBytes: 1 << 20,
	})
	first := createTestResponse(t, evictingHandler, `{
		"model":"mock-chat-model",
		"input":"Oldest turn",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	createTestResponse(t, evictingHandler, `{
		"model":"mock-chat-model",
		"input":"Newest turn",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	assertPreviousResponseErrorCode(t, evictingHandler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after eviction",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, first.ID), "previous_response_not_found")
}

func TestResponseContinuationRejectsMalformedMissingAndDifferentModel(t *testing.T) {
	t.Parallel()

	firstModel := newContinuationTestModel(t, "model-one")
	secondModel := newContinuationTestModel(t, "model-two")
	registry, err := chat.NewRegistry(firstModel, secondModel)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	handler, err := NewHandler(registry, testResponseStoreOptions())
	if err != nil {
		t.Fatalf("NewHandler() error = %v", err)
	}
	first := createTestResponse(t, handler, `{
		"model":"model-one",
		"input":"Hello",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)

	assertPreviousResponseErrorCode(t, handler, fmt.Sprintf(`{
		"model":"model-two",
		"input":"Continue",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, first.ID), "invalid_value")
	assertPreviousResponseErrorCode(t, handler, `{
		"model":"model-one",
		"input":"Continue",
		"previous_response_id":"resp_mock_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		"max_output_tokens":16
	}`, "invalid_value")
	assertPreviousResponseErrorCode(t, handler, `{
		"model":"model-one",
		"input":"Continue",
		"previous_response_id":"resp_mock_ABCDEF0123456789abcdef0123456789",
		"max_output_tokens":16
	}`, "invalid_value")
	assertPreviousResponseErrorCode(t, handler, `{
		"model":"model-one",
		"input":"Continue",
		"previous_response_id":"resp_mock_00000000000000000000000000000000",
		"max_output_tokens":16
	}`, "previous_response_not_found")
}

func TestResponseContinuationDoesNotCarryInstructionsButCarriesTools(t *testing.T) {
	t.Parallel()

	chatModel := &testChatModel{
		info:     testModelInfo(),
		response: chat.Response{Content: "fixed output", Usage: chat.Usage{PromptTokens: 8, CompletionTokens: 2, TotalTokens: 10}},
	}
	handler := newTestHandler(t, chatModel)
	first := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"instructions":"Use alpha policy.",
		"input":"Same input",
		"tool_choice":"none"
	}`)
	firstGeneration := *chatModel.request.ContextState
	firstState := loadTestResponseState(t, handler, first.ID)

	second := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"instructions":"Use beta policy.",
		"input":"Same input",
		"tool_choice":"none"
	}`)
	secondGeneration := *chatModel.request.ContextState
	secondState := loadTestResponseState(t, handler, second.ID)
	if firstGeneration.Digest() == secondGeneration.Digest() {
		t.Fatalf("different instructions produced generation digest %016x", firstGeneration.Digest())
	}
	if firstState.FinalContext != secondState.FinalContext {
		t.Fatalf("instructions leaked into continuation state: %#v and %#v", firstState.FinalContext, secondState.FinalContext)
	}

	withTool := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"Same input",
		"tool_choice":"none",
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]
	}`)
	toolState := loadTestResponseState(t, handler, withTool.ID)
	if toolState.FinalContext.Digest() == firstState.FinalContext.Digest() {
		t.Fatalf("tool configuration did not change continuation digest %016x", toolState.FinalContext.Digest())
	}
	withDifferentTool := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"Same input",
		"tool_choice":"none",
		"tools":[{"type":"function","name":"search","parameters":{"type":"object"}}]
	}`)
	differentToolState := loadTestResponseState(t, handler, withDifferentTool.ID)
	if differentToolState.FinalContext.Digest() == toolState.FinalContext.Digest() {
		t.Fatalf("different potential tools produced continuation digest %016x", toolState.FinalContext.Digest())
	}
	withReasoning := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"private reasoning summary"}]},
			{"type":"message","role":"user","content":"Same input"}
		],
		"tool_choice":"none"
	}`)
	reasoningState := loadTestResponseState(t, handler, withReasoning.ID)
	if reasoningState.FinalContext.Digest() == firstState.FinalContext.Digest() {
		t.Fatalf("reasoning did not change continuation digest %016x", reasoningState.FinalContext.Digest())
	}
}

func TestResponseStateSizeExcludesConversationContent(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	short := createTestResponse(t, handler, `{"model":"mock-chat-model","input":"Hi","max_output_tokens":16}`)
	long := createTestResponse(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"instructions":%q,
		"input":%q,
		"max_output_tokens":64
	}`, strings.Repeat("Be thorough. ", 200), strings.Repeat("A long question. ", 2000)))

	shortState := loadTestResponseState(t, handler, short.ID)
	longState := loadTestResponseState(t, handler, long.ID)
	if shortState.LogicalBytes != longState.LogicalBytes {
		t.Fatalf("logical bytes = %d for short and %d for long conversation, want equal", shortState.LogicalBytes, longState.LogicalBytes)
	}
	if longState.FinalContext.Digest() == shortState.FinalContext.Digest() {
		t.Fatal("different conversations share a final context digest")
	}
}

func TestResponseToolConfigurationContributesToContext(t *testing.T) {
	t.Parallel()

	base := functionToolConfig{
		Choice:            chat.ToolChoiceNone,
		ParallelToolCalls: true,
	}
	baseItems, err := functionToolContextItems(base)
	if err != nil {
		t.Fatalf("functionToolContextItems(base) error = %v", err)
	}
	baseState, err := chat.NewContextState().AbsorbAll(baseItems)
	if err != nil {
		t.Fatalf("AbsorbAll(base) error = %v", err)
	}
	maxCalls := 1
	variants := map[string]functionToolConfig{
		"tool choice": {
			Choice:            chat.ToolChoiceAuto,
			ParallelToolCalls: true,
		},
		"parallel calls": {
			Choice:            chat.ToolChoiceNone,
			ParallelToolCalls: false,
		},
		"maximum calls": {
			Choice:            chat.ToolChoiceNone,
			ParallelToolCalls: true,
			MaxToolCalls:      &maxCalls,
		},
	}
	for name, variant := range variants {
		items, err := functionToolContextItems(variant)
		if err != nil {
			t.Fatalf("functionToolContextItems(%s) error = %v", name, err)
		}
		state, err := chat.NewContextState().AbsorbAll(items)
		if err != nil {
			t.Fatalf("AbsorbAll(%s) error = %v", name, err)
		}
		if state.Digest() == baseState.Digest() {
			t.Errorf("%s did not change context digest %016x", name, state.Digest())
		}
	}
}

func TestResponseToolCallUsageIncludesToolConfiguration(t *testing.T) {
	t.Parallel()

	body := `{
		"model":"mock-chat-model",
		"instructions":"Use the configured lookup tool.",
		"input":"Find this value",
		"max_output_tokens":16,
		"tool_choice":"required",
		"parallel_tool_calls":false,
		"max_tool_calls":1,
		"tools":[{
			"type":"function",
			"name":"lookup",
			"description":"Look up a value",
			"parameters":{
				"type":"object",
				"properties":{"query":{"type":"string"}},
				"required":["query"]
			}
		}]
	}`
	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	response := createTestResponse(t, handler, body)

	var request responseRequest
	if err := json.Unmarshal([]byte(body), &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	_, inputItems, toolConfig, requestError := toResponseDomainRequest(request)
	if requestError != nil {
		t.Fatalf("toResponseDomainRequest() error = %s", requestError.message)
	}
	toolItems, err := functionToolContextItems(toolConfig)
	if err != nil {
		t.Fatalf("functionToolContextItems() error = %v", err)
	}
	expectedContext, err := chat.NewContextState().AbsorbAll(toolItems)
	if err != nil {
		t.Fatalf("AbsorbAll(tools) error = %v", err)
	}
	expectedContext, err = expectedContext.Absorb(responseInstructionContextItem(*request.Instructions))
	if err != nil {
		t.Fatalf("Absorb(instructions) error = %v", err)
	}
	expectedContext, err = expectedContext.AbsorbAll(inputItems)
	if err != nil {
		t.Fatalf("AbsorbAll(input) error = %v", err)
	}
	wantInputTokens, err := expectedContext.PromptTokens()
	if err != nil {
		t.Fatalf("PromptTokens() error = %v", err)
	}
	if response.Usage == nil {
		t.Fatal("response usage is nil")
	}
	if response.Usage.InputTokens != wantInputTokens {
		t.Fatalf("input tokens = %d, want %d", response.Usage.InputTokens, wantInputTokens)
	}
	if response.Usage.TotalTokens != wantInputTokens+response.Usage.OutputTokens {
		t.Fatalf(
			"total tokens = %d, want input %d + output %d",
			response.Usage.TotalTokens,
			wantInputTokens,
			response.Usage.OutputTokens,
		)
	}
}

func TestResponseContinuationValidatesFunctionCallOutputs(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	toolResponse := createRequiredToolResponse(t, handler, "First tool response")
	callID := *toolResponse.Output[0].CallID
	textResponse := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"Text only",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	foreignResponse := createRequiredToolResponse(t, handler, "Foreign tool response")
	foreignCallID := *foreignResponse.Output[0].CallID

	tests := []struct {
		name       string
		previousID string
		input      string
	}{
		{name: "after text", previousID: textResponse.ID, input: `[ {"type":"function_call_output","call_id":"call_missing","output":"bad"} ]`},
		{name: "invented", previousID: toolResponse.ID, input: `[ {"type":"function_call_output","call_id":"call_missing","output":"bad"} ]`},
		{name: "foreign", previousID: toolResponse.ID, input: fmt.Sprintf(`[ {"type":"function_call_output","call_id":%q,"output":"bad"} ]`, foreignCallID)},
		{name: "duplicate", previousID: toolResponse.ID, input: fmt.Sprintf(`[
			{"type":"function_call_output","call_id":%q,"output":"first"},
			{"type":"function_call_output","call_id":%q,"output":"second"}
		]`, callID, callID)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertPreviousResponseErrorCode(t, handler, fmt.Sprintf(`{
				"model":"mock-chat-model",
				"previous_response_id":%q,
				"input":%s,
				"max_output_tokens":16,
				"tool_choice":"none"
			}`, tt.previousID, tt.input), "invalid_value")
		})
	}

	validBody := fmt.Sprintf(`{
		"model":"mock-chat-model",
		"previous_response_id":%q,
		"input":[
			{"type":"function_call_output","call_id":%q,"output":"sunny"},
			{"type":"message","role":"user","content":"Answer now"}
		],
		"max_output_tokens":16,
		"tool_choice":"none"
	}`, toolResponse.ID, callID)
	firstBranch := createTestResponse(t, handler, validBody)
	secondBranch := createTestResponse(t, handler, validBody)
	if responseText(t, firstBranch) != responseText(t, secondBranch) {
		t.Fatalf("branching from one tool response changed output")
	}

	explicitReplay := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":[
			{"type":"function_call","call_id":"call_explicit","name":"lookup","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_explicit","output":"ok"},
			{"type":"message","role":"user","content":"Continue"}
		],
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)
	if responseText(t, explicitReplay) == "" {
		t.Fatal("explicit function replay produced empty text")
	}
}

func TestResponseContinuationAcceptsMultipleValidFunctionCallOutputs(t *testing.T) {
	plan := chat.ToolPlan{
		Calls: []chat.ToolCall{
			{ID: "fc_mock_one", CallID: "call_mock_one", Name: "one", Arguments: `{"value":1}`},
			{ID: "fc_mock_two", CallID: "call_mock_two", Name: "two", Arguments: `{"value":2}`},
		},
		Usage: chat.Usage{PromptTokens: 10, CompletionTokens: 8, TotalTokens: 18},
	}
	chatModel := &testChatModel{info: testModelInfo(), toolPlan: &plan}
	handler := newTestHandler(t, chatModel)
	toolResponse := createTestResponse(t, handler, `{
		"model":"mock-chat-model",
		"input":"Call both tools",
		"tool_choice":"required",
		"parallel_tool_calls":true,
		"tools":[
			{"type":"function","name":"one"},
			{"type":"function","name":"two"}
		]
	}`)
	if len(toolResponse.Output) != 2 {
		t.Fatalf("tool outputs = %#v, want two calls", toolResponse.Output)
	}

	chatModel.toolPlan = nil
	chatModel.response = chat.Response{
		Content: "combined",
		Usage:   chat.Usage{PromptTokens: 20, CompletionTokens: 1, TotalTokens: 21},
	}
	continued := createTestResponse(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"previous_response_id":%q,
		"input":[
			{"type":"function_call_output","call_id":"call_mock_one","output":"first"},
			{"type":"function_call_output","call_id":"call_mock_two","output":"second"}
		],
		"tool_choice":"none"
	}`, toolResponse.ID))
	if responseText(t, continued) != "combined" {
		t.Fatalf("continued text = %q", responseText(t, continued))
	}
}

func TestStreamingResponseStateCommitsOnlyOnCompletion(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	completedID := streamResponseID(t, handler, `{
		"model":"mock-chat-model",
		"input":"Stream this turn",
		"max_output_tokens":16,
		"tool_choice":"none",
		"stream":true
	}`)
	createTestResponse(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after stream",
		"previous_response_id":%q,
		"max_output_tokens":16,
		"tool_choice":"none"
	}`, completedID))

	failingModel := &testChatModel{
		info:                 testModelInfo(),
		streamChunks:         []chat.Chunk{{Delta: []byte("partial")}},
		streamErr:            errors.New("failed after output"),
		streamErrAfterChunks: true,
	}
	failingHandler := newTestHandler(t, failingModel)
	failedID := streamResponseID(t, failingHandler, `{
		"model":"mock-chat-model",
		"input":"Fail this stream",
		"stream":true
	}`)
	assertPreviousResponseErrorCode(t, failingHandler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after failure",
		"previous_response_id":%q
	}`, failedID), "previous_response_not_found")

	cancelledModel := &testChatModel{
		info:                 testModelInfo(),
		streamChunks:         []chat.Chunk{{Delta: []byte("partial")}},
		streamErr:            context.Canceled,
		streamErrAfterChunks: true,
	}
	cancelledHandler := newTestHandler(t, cancelledModel)
	cancelledID := streamResponseID(t, cancelledHandler, `{
		"model":"mock-chat-model",
		"input":"Cancel this stream",
		"stream":true
	}`)
	assertPreviousResponseErrorCode(t, cancelledHandler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after cancellation",
		"previous_response_id":%q
	}`, cancelledID), "previous_response_not_found")
}

func TestStreamingResponseDeletesStateWhenCompletedEventWriteFails(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, newContinuationTestModel(t, "mock-chat-model"))
	writer := newFailingResponseWriter([]byte("event: response.completed"))
	handler.ServeHTTP(writer, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"mock-chat-model",
		"input":"Disconnect before completion",
		"max_output_tokens":16,
		"tool_choice":"none",
		"stream":true
	}`)))

	createdID := responseCreatedID(t, writer.body.String())
	assertPreviousResponseErrorCode(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue after disconnect",
		"previous_response_id":%q,
		"max_output_tokens":16
	}`, createdID), "previous_response_not_found")
}

func TestResponseStoreCapacityErrorDoesNotCompleteOrRetainResponse(t *testing.T) {
	t.Parallel()

	handler := newTestHandlerWithOptions(t, newContinuationTestModel(t, "mock-chat-model"), ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 1, MaxBytes: 1,
	})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"mock-chat-model",
		"input":"Too large for the store",
		"max_output_tokens":16,
		"tool_choice":"none"
	}`)))
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusInternalServerError, recorder.Body.String())
	}
	var response errorResponse
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode capacity error: %v", err)
	}
	if pointerValue(response.Error.Code) != "response_store_capacity_exceeded" {
		t.Fatalf("capacity error = %#v", response.Error)
	}

	streamRecorder := httptest.NewRecorder()
	handler.ServeHTTP(streamRecorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"mock-chat-model",
		"input":"Stream too large for the store",
		"max_output_tokens":16,
		"tool_choice":"none",
		"stream":true
	}`)))
	events := parseResponseSSEEvents(t, streamRecorder.Body.String())
	createdID := responseCreatedID(t, streamRecorder.Body.String())
	seenFailure := false
	for _, event := range events {
		if event.event == "response.completed" {
			t.Fatalf("oversized stream emitted response.completed: %s", streamRecorder.Body.String())
		}
		if event.event == "error" {
			var failure responseStreamingErrorEvent
			if err := json.Unmarshal([]byte(event.data), &failure); err != nil {
				t.Fatalf("decode streaming capacity error: %v", err)
			}
			seenFailure = pointerValue(failure.Error.Code) == "response_store_capacity_exceeded"
		}
	}
	if !seenFailure {
		t.Fatalf("stream has no capacity failure: %s", streamRecorder.Body.String())
	}
	assertPreviousResponseErrorCode(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":"Continue oversized stream",
		"previous_response_id":%q
	}`, createdID), "previous_response_not_found")
}

func createRequiredToolResponse(t testing.TB, handler http.Handler, input string) responseResource {
	t.Helper()
	response := createTestResponse(t, handler, fmt.Sprintf(`{
		"model":"mock-chat-model",
		"input":%q,
		"max_output_tokens":16,
		"tool_choice":"required",
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]
	}`, input))
	if len(response.Output) != 1 || response.Output[0].Type != "function_call" || response.Output[0].CallID == nil {
		t.Fatalf("tool response output = %#v", response.Output)
	}
	return response
}

func loadTestResponseState(t testing.TB, handler *Handler, id string) *responseState {
	t.Helper()
	state, found, err := handler.responseStore.Get(id)
	if err != nil {
		t.Fatalf("responseStore.Get(%q) error = %v", id, err)
	}
	if !found {
		t.Fatalf("responseStore.Get(%q) not found", id)
	}
	return state
}

func streamResponseID(t testing.TB, handler http.Handler, body string) string {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("stream status = %d; body = %s", recorder.Code, recorder.Body.String())
	}
	return responseCreatedID(t, recorder.Body.String())
}

func streamCompletedResponse(t testing.TB, handler http.Handler, body string) responseResource {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("stream status = %d; body = %s", recorder.Code, recorder.Body.String())
	}
	for _, event := range parseResponseSSEEvents(t, recorder.Body.String()) {
		if event.event != "response.completed" {
			continue
		}
		var completed responseLifecycleEvent
		if err := json.Unmarshal([]byte(event.data), &completed); err != nil {
			t.Fatalf("decode response.completed: %v", err)
		}
		return completed.Response
	}
	t.Fatalf("stream has no response.completed event: %s", recorder.Body.String())
	return responseResource{}
}

func responseCreatedID(t testing.TB, body string) string {
	t.Helper()
	events := parseResponseSSEEvents(t, body)
	for _, event := range events {
		if event.event != "response.created" {
			continue
		}
		var created responseLifecycleEvent
		if err := json.Unmarshal([]byte(event.data), &created); err != nil {
			t.Fatalf("decode response.created: %v", err)
		}
		return created.Response.ID
	}
	t.Fatalf("stream has no response.created event: %s", body)
	return ""
}

func newTestHandlerWithOptions(t *testing.T, chatModel chat.Model, options ResponseStoreOptions) *Handler {
	t.Helper()
	registry, err := chat.NewRegistry(chatModel)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	handler, err := NewHandler(registry, options)
	if err != nil {
		t.Fatalf("NewHandler() error = %v", err)
	}
	return handler
}

type failingResponseWriter struct {
	header http.Header
	body   bytes.Buffer
	failOn []byte
}

func newFailingResponseWriter(failOn []byte) *failingResponseWriter {
	return &failingResponseWriter{header: make(http.Header), failOn: failOn}
}

func (w *failingResponseWriter) Header() http.Header {
	return w.header
}

func (w *failingResponseWriter) Write(payload []byte) (int, error) {
	if bytes.Contains(payload, w.failOn) {
		return 0, errors.New("simulated client disconnect")
	}
	return w.body.Write(payload)
}

func (w *failingResponseWriter) WriteHeader(_ int) {}

func (w *failingResponseWriter) Flush() {}

func assertPreviousResponseErrorCode(t testing.TB, handler http.Handler, body, code string) {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusBadRequest, recorder.Body.String())
	}
	var response errorResponse
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if pointerValue(response.Error.Param) != "previous_response_id" && code == "previous_response_not_found" {
		t.Fatalf("error param = %#v", response.Error)
	}
	if pointerValue(response.Error.Code) != code {
		t.Fatalf("error = %#v, want code %q", response.Error, code)
	}
}

func newContinuationTestModel(t testing.TB, modelID string) *chat.MockModel {
	t.Helper()
	chatModel, err := chat.NewMockModel(modelID, chat.MockOptions{
		DefaultCompletionTokens: 16,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    4,
	})
	if err != nil {
		t.Fatalf("NewMockModel() error = %v", err)
	}
	return chatModel
}

func responseText(t testing.TB, response responseResource) string {
	t.Helper()
	if len(response.Output) != 1 || response.Output[0].Type != "message" || response.Output[0].Content == nil || len(*response.Output[0].Content) != 1 {
		t.Fatalf("response output = %#v, want one text message", response.Output)
	}
	return (*response.Output[0].Content)[0].Text
}
