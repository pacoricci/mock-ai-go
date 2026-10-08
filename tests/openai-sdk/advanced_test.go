package sdk_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/pacoricci/mock-ai-go/internal/chat"
	mockopenai "github.com/pacoricci/mock-ai-go/internal/openai"
)

func newMock(t *testing.T, id string) *chat.MockModel {
	t.Helper()
	m, err := chat.NewMockModel(id, chat.MockOptions{
		DefaultCompletionTokens: 32, MaxCompletionTokens: 64, StreamTokensPerChunk: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func createResponse(t *testing.T, client openai.Client, ctx context.Context, params responses.ResponseNewParams) *responses.Response {
	t.Helper()
	result, err := client.Responses.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestConcurrentConversationBranches(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	root := createResponse(t, client, ctx, textParams())
	var mu sync.Mutex
	ids := map[string]bool{root.ID: true}
	for worker := range 12 {
		t.Run(fmt.Sprintf("conversation-%d", worker), func(t *testing.T) {
			t.Parallel()
			previous := root
			for turn := range 3 {
				params := textParams()
				params.PreviousResponseID = openai.String(previous.ID)
				params.Input.OfString = openai.String(fmt.Sprintf("Conversation %d, turn %d: continue with café 世界.", worker, turn))
				params.Metadata = map[string]string{"conversation": fmt.Sprint(worker)}
				jsonResult := createResponse(t, client, ctx, params)
				sseResult, _ := streamResponse(t, client, ctx, params)
				assertTextResponse(t, jsonResult)
				assertTextResponse(t, sseResult)
				if jsonResult.OutputText() != sseResult.OutputText() || !reflect.DeepEqual(jsonResult.Usage, sseResult.Usage) {
					t.Fatalf("branch changed under concurrency: JSON=%+v SSE=%+v", jsonResult, sseResult)
				}
				for _, result := range []*responses.Response{jsonResult, sseResult} {
					if result.PreviousResponseID != previous.ID || result.Metadata["conversation"] != fmt.Sprint(worker) || result.Usage.InputTokens <= previous.Usage.InputTokens {
						t.Fatalf("conversation state mixed: previous=%+v result=%+v", previous, result)
					}
					mu.Lock()
					duplicate := ids[result.ID]
					ids[result.ID] = true
					mu.Unlock()
					if duplicate {
						t.Fatalf("duplicate response ID %q", result.ID)
					}
				}
				previous = jsonResult
			}
		})
	}
}

func TestConcurrentChatHistories(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	for worker := range 8 {
		t.Run(fmt.Sprintf("chat-%d", worker), func(t *testing.T) {
			t.Parallel()
			params := openai.ChatCompletionNewParams{
				Model: modelID, MaxCompletionTokens: openai.Int(16), Seed: openai.Int(int64(worker)),
				Messages: []openai.ChatCompletionMessageParamUnion{openai.SystemMessage("Be concise.")},
			}
			var priorUsage int64
			for turn := range 3 {
				params.Messages = append(params.Messages, openai.UserMessage(fmt.Sprintf("Chat %d, turn %d: café 世界.", worker, turn)))
				result, err := client.Chat.Completions.New(ctx, params)
				if err != nil {
					t.Fatal(err)
				}
				if len(result.Choices) != 1 || result.Usage.PromptTokens <= priorUsage {
					t.Fatalf("chat history was lost: %+v", result)
				}
				streamParams := params
				streamParams.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
				stream := client.Chat.Completions.NewStreaming(ctx, streamParams)
				t.Cleanup(func() { _ = stream.Close() })
				var acc openai.ChatCompletionAccumulator
				for stream.Next() {
					if !acc.AddChunk(stream.Current()) {
						t.Fatal("SDK rejected a chat history chunk")
					}
				}
				if err := stream.Err(); err != nil {
					t.Fatal(err)
				}
				if len(acc.Choices) != 1 || acc.Choices[0].Message.Content != result.Choices[0].Message.Content || acc.Usage.PromptTokens != result.Usage.PromptTokens || acc.Usage.CompletionTokens != result.Usage.CompletionTokens || acc.Usage.TotalTokens != result.Usage.TotalTokens {
					t.Fatalf("JSON/SSE history differs at turn %d: JSON prompt tokens=%d SSE prompt tokens=%d", turn, result.Usage.PromptTokens, acc.Usage.PromptTokens)
				}
				params.Messages = append(params.Messages, result.Choices[0].Message.ToParam())
				priorUsage = result.Usage.PromptTokens
			}
		})
	}
}

func parallelToolParams() responses.ResponseNewParams {
	params := textParams()
	params.ParallelToolCalls = openai.Bool(true)
	params.MaxToolCalls = openai.Int(2)
	params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsRequired)}
	for _, name := range []string{"weather", "forecast", "air_quality"} {
		params.Tools = append(params.Tools, responses.ToolUnionParam{OfFunction: &responses.FunctionToolParam{
			Name: name,
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string", "enum": []string{"Città\n\"Roma\" 🌍", "Paris"}},
					"unit":     map[string]any{"const": "metric"},
					"days":     map[string]any{"type": "array", "minItems": 2, "maxItems": 2, "items": map[string]any{"type": "integer", "minimum": 1, "maximum": 7}},
					"options": map[string]any{
						"type": "object", "properties": map[string]any{"alerts": map[string]any{"type": "boolean"}}, "required": []string{"alerts"},
					},
				},
				"required": []string{"location", "unit", "days", "options"},
			},
		}})
	}
	return params
}

func createParallelCalls(t *testing.T, client openai.Client, ctx context.Context) (responses.ResponseNewParams, *responses.Response) {
	t.Helper()
	params := parallelToolParams()
	// Vary deterministic inputs until the mock selects two calls, rather than
	// assuming required selection always invokes every available function.
	for attempt := range 32 {
		params.Input.OfString = openai.String(fmt.Sprintf("Compare Rome and Paris, request %d.", attempt))
		result := createResponse(t, client, ctx, params)
		if len(result.Output) == 0 || len(result.Output) > 2 {
			t.Fatalf("max_tool_calls=2 violated: %+v", result.Output)
		}
		if len(result.Output) == 2 {
			return params, result
		}
	}
	t.Fatal("no deterministic fixture selected two parallel calls")
	return params, nil
}

func functionOutput(callID, output string) responses.ResponseInputItemUnionParam {
	item := responses.ResponseInputItemParamOfFunctionCallOutput(output)
	item.OfFunctionCallOutput.CallID = openai.String(callID)
	return item
}

func TestParallelToolsAndStatelessReplay(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	params, result := createParallelCalls(t, client, ctx)
	names, callIDs := make(map[string]bool), make(map[string]bool)
	for _, item := range result.Output {
		if item.Type != "function_call" {
			t.Fatalf("unexpected output: %+v", item)
		}
		call := item.AsFunctionCall()
		if call.CallID == "" || call.ID == "" || call.Status != "completed" || names[call.Name] || callIDs[call.CallID] {
			t.Fatalf("invalid or duplicated call: %+v", call)
		}
		names[call.Name], callIDs[call.CallID] = true, true
		var args struct {
			Location string `json:"location"`
			Unit     string `json:"unit"`
			Days     []int  `json:"days"`
			Options  struct {
				Alerts *bool `json:"alerts"`
			} `json:"options"`
		}
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
			t.Fatal(err)
		}
		if (args.Location != "Città\n\"Roma\" 🌍" && args.Location != "Paris") || args.Unit != "metric" || len(args.Days) != 2 || args.Options.Alerts == nil {
			t.Fatalf("arguments violate supported schema: %s", call.Arguments)
		}
		for _, day := range args.Days {
			if day < 1 || day > 7 {
				t.Fatalf("day outside schema bounds: %d", day)
			}
		}
	}
	assertParallelToolStream(t, client, ctx, params, result)
	for _, single := range []string{"max_tool_calls", "parallel_tool_calls"} {
		limited := params
		if single == "max_tool_calls" {
			limited.MaxToolCalls = openai.Int(1)
		} else {
			limited.ParallelToolCalls = openai.Bool(false)
		}
		if got := createResponse(t, client, ctx, limited); len(got.Output) != 1 || got.Output[0].Type != "function_call" {
			t.Fatalf("%s did not limit calls: %+v", single, got.Output)
		}
	}
	outputs := []responses.ResponseInputItemUnionParam{
		functionOutput(result.Output[1].AsFunctionCall().CallID, `{"temperature":18}`),
		functionOutput(result.Output[0].AsFunctionCall().CallID, `{"temperature":21}`),
	}
	followup := textParams()
	followup.PreviousResponseID = openai.String(result.ID)
	followup.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: outputs}
	answer := createResponse(t, client, ctx, followup)
	assertTextResponse(t, answer)
	if answer.PreviousResponseID != result.ID {
		t.Fatalf("parallel results lost their conversation: %+v", answer)
	}
	// Resend typed SDK calls and results without relying on a stored response ID.
	replay := []responses.ResponseInputItemUnionParam{responses.ResponseInputItemParamOfMessage(params.Input.OfString.Value, "user")}
	for _, item := range result.Output {
		call := item.AsFunctionCall().ToParam()
		replay = append(replay, responses.ResponseInputItemUnionParam{OfFunctionCall: &call})
	}
	replay = append(replay, outputs...)
	followup = textParams()
	followup.Store = openai.Bool(false)
	followup.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: replay}
	assertTextResponse(t, createResponse(t, client, ctx, followup))
}

func assertParallelToolStream(t *testing.T, client openai.Client, ctx context.Context, params responses.ResponseNewParams, expected *responses.Response) {
	t.Helper()
	stream := client.Responses.NewStreaming(ctx, params)
	t.Cleanup(func() { _ = stream.Close() })
	arguments := make(map[int64]string)
	added, done, argumentDone := make(map[int64]string), make(map[int64]bool), make(map[int64]bool)
	sequence, completions := int64(0), 0
	for stream.Next() {
		event := stream.Current()
		if completions != 0 {
			t.Fatalf("event after completion: %s", event.RawJSON())
		}
		if event.SequenceNumber != sequence {
			t.Fatalf("sequence=%d, want %d", event.SequenceNumber, sequence)
		}
		sequence++
		index := event.OutputIndex
		switch event.Type {
		case "response.output_item.added":
			if added[index] != "" || index < 0 || index >= int64(len(expected.Output)) {
				t.Fatalf("invalid added index %d", index)
			}
			added[index] = event.Item.ID
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			if added[index] == "" || added[index] != event.ItemID || argumentDone[index] {
				t.Fatalf("arguments associated with wrong or completed item: %s", event.RawJSON())
			}
			if event.Type == "response.function_call_arguments.delta" {
				arguments[index] += event.Delta
			} else {
				argumentDone[index] = true
				if event.Arguments != arguments[index] || event.Arguments != expected.Output[index].AsFunctionCall().Arguments {
					t.Fatalf("arguments done differ at index %d", index)
				}
			}
		case "response.output_item.done":
			if done[index] || !argumentDone[index] || added[index] != event.Item.ID {
				t.Fatalf("unexpected item done: %s", event.RawJSON())
			}
			done[index] = true
			if !reflect.DeepEqual(event.Item.AsFunctionCall(), expected.Output[index].AsFunctionCall()) {
				t.Fatalf("SSE call differs from JSON at index %d", index)
			}
		case "response.completed":
			completions++
			if !reflect.DeepEqual(event.Response.Output, expected.Output) || !reflect.DeepEqual(event.Response.Usage, expected.Usage) {
				t.Fatalf("completed tool response differs from JSON: %+v", event.Response)
			}
		case "error", "response.failed", "response.incomplete":
			t.Fatalf("unexpected failure: %s", event.RawJSON())
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if completions != 1 || len(done) != len(expected.Output) {
		t.Fatalf("incomplete parallel stream: completions=%d done=%v", completions, done)
	}
}

func TestInvalidFunctionResultsDoNotCorruptState(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	_, parent := createParallelCalls(t, client, ctx)
	call := parent.Output[0].AsFunctionCall()
	valid := functionOutput(call.CallID, `{"temperature":21}`)
	foreignParams := parallelToolParams()
	foreignParams.Input.OfString = openai.String("An unrelated conversation about Berlin.")
	foreign := createResponse(t, client, ctx, foreignParams)
	foreignCallID := foreign.Output[0].AsFunctionCall().CallID
	if foreignCallID == call.CallID {
		t.Fatal("foreign fixture reused the parent's deterministic call ID")
	}
	for _, invalid := range []struct {
		name  string
		items []responses.ResponseInputItemUnionParam
	}{
		{"unknown call", []responses.ResponseInputItemUnionParam{functionOutput("call_unknown", "bad")}},
		{"foreign call", []responses.ResponseInputItemUnionParam{functionOutput(foreignCallID, "bad")}},
		{"duplicate result", []responses.ResponseInputItemUnionParam{valid, valid}},
		{"duplicate call", []responses.ResponseInputItemUnionParam{{OfFunctionCall: ptrCall(call)}, valid}},
	} {
		t.Run(invalid.name, func(t *testing.T) {
			params := textParams()
			params.PreviousResponseID = openai.String(parent.ID)
			params.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: invalid.items}
			_, err := client.Responses.New(ctx, params)
			assertAPIError(t, err, http.StatusBadRequest, "input", "invalid_value")
			params.Input.OfInputItemList = []responses.ResponseInputItemUnionParam{valid}
			assertTextResponse(t, createResponse(t, client, ctx, params))
		})
	}
	stateless := textParams()
	stateless.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: []responses.ResponseInputItemUnionParam{valid, {OfFunctionCall: ptrCall(call)}}}
	_, err := client.Responses.New(ctx, stateless)
	assertAPIError(t, err, http.StatusBadRequest, "input", "invalid_value")
}

func ptrCall(call responses.ResponseFunctionToolCall) *responses.ResponseFunctionToolCallParam {
	param := call.ToParam()
	return &param
}

func TestResponseStoreEvictionExpiryAndModelIsolation(t *testing.T) {
	t.Parallel()
	for _, scenario := range []string{"eviction", "expiry", "model isolation"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			store := mockopenai.ResponseStoreOptions{TTL: time.Hour, MaxEntries: 2, MaxBytes: 1 << 20}
			if scenario == "expiry" {
				store.TTL = time.Nanosecond
			}
			client, ctx := newClientWithModels(t, store, newMock(t, modelID), newMock(t, "other-model"))
			first := createResponse(t, client, ctx, textParams())
			params := textParams()
			params.PreviousResponseID = openai.String(first.ID)
			params.Store = openai.Bool(false)
			switch scenario {
			case "eviction":
				second := createResponse(t, client, ctx, textParams())
				third, _ := streamResponse(t, client, ctx, textParams())
				_, err := client.Responses.New(ctx, params)
				assertAPIError(t, err, http.StatusBadRequest, "previous_response_id", "previous_response_not_found")
				for _, retained := range []*responses.Response{second, third} {
					params.PreviousResponseID = openai.String(retained.ID)
					assertTextResponse(t, createResponse(t, client, ctx, params))
				}
			case "expiry":
				_, err := client.Responses.New(ctx, params)
				assertAPIError(t, err, http.StatusBadRequest, "previous_response_id", "previous_response_not_found")
			case "model isolation":
				params.Model = "other-model"
				_, err := client.Responses.New(ctx, params)
				assertAPIError(t, err, http.StatusBadRequest, "previous_response_id", "invalid_value")
				params.Model = modelID
				assertTextResponse(t, createResponse(t, client, ctx, params))
			}
		})
	}
}

func TestResponseStoreCapacityFailureAfterDeltas(t *testing.T) {
	t.Parallel()
	client, ctx := newClientWithModels(t, mockopenai.ResponseStoreOptions{TTL: time.Hour, MaxEntries: 1, MaxBytes: 1}, newMock(t, modelID))
	_, err := client.Responses.New(ctx, textParams())
	var apiError *openai.Error
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusInternalServerError || apiError.Code != "response_store_capacity_exceeded" {
		t.Fatalf("capacity error = %v", err)
	}
	stream := client.Responses.NewStreaming(ctx, textParams())
	t.Cleanup(func() { _ = stream.Close() })
	var id string
	var text strings.Builder
	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "response.created":
			id = event.Response.ID
		case "response.output_text.delta":
			text.WriteString(event.Delta)
		case "response.completed":
			t.Fatal("capacity failure was reported as a completed response")
		}
	}
	var streamError *ssestream.StreamError
	if !errors.As(stream.Err(), &streamError) || id == "" || len(strings.Fields(text.String())) != 16 {
		t.Fatalf("expected stream failure after generation: id=%q text=%q err=%v", id, text.String(), stream.Err())
	}
	var payload struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(streamError.Event.Data, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Error.Code != "response_store_capacity_exceeded" {
		t.Fatalf("stream failure code = %q", payload.Error.Code)
	}
	params := textParams()
	params.PreviousResponseID = openai.String(id)
	_, err = client.Responses.New(ctx, params)
	assertAPIError(t, err, http.StatusBadRequest, "previous_response_id", "previous_response_not_found")
	params = textParams()
	params.Store = openai.Bool(false)
	assertTextResponse(t, createResponse(t, client, ctx, params))
}

type blockedStreamModel struct {
	*chat.MockModel
	stopped chan struct{}
}

// Stream pauses after the first emitted chunk until the HTTP request is canceled.
func (m *blockedStreamModel) Stream(ctx context.Context, request chat.Request, emit chat.Emit) (chat.Usage, error) {
	defer close(m.stopped)
	return m.MockModel.Stream(ctx, request, func(chunk chat.Chunk) error {
		if err := emit(chunk); err != nil {
			return err
		}
		<-ctx.Done()
		return ctx.Err()
	})
}

func TestInterruptedResponseStreamsAreNotStored(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"cancel", "close", "deadline"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			model := &blockedStreamModel{MockModel: newMock(t, modelID), stopped: make(chan struct{})}
			client, ctx := newClientWithModels(t, mockopenai.ResponseStoreOptions{TTL: time.Hour, MaxEntries: 100, MaxBytes: 1 << 20}, model)
			streamCtx, cancel := context.WithCancel(ctx)
			if mode == "deadline" {
				cancel()
				streamCtx, cancel = context.WithTimeout(ctx, time.Second)
			}
			defer cancel()
			stream := client.Responses.NewStreaming(streamCtx, textParams())
			t.Cleanup(func() { _ = stream.Close() })
			var id string
			for stream.Next() {
				event := stream.Current()
				if event.Type == "response.created" {
					id = event.Response.ID
				}
				if event.Type == "response.output_text.delta" {
					break
				}
			}
			if id == "" || stream.Current().Type != "response.output_text.delta" {
				t.Fatalf("stream did not reach its first delta: %v", stream.Err())
			}
			switch mode {
			case "cancel":
				cancel()
				if stream.Next() || !errors.Is(stream.Err(), context.Canceled) {
					t.Fatalf("cancel did not stop the SDK stream: %v", stream.Err())
				}
			case "close":
				if err := stream.Close(); err != nil {
					t.Fatal(err)
				}
				if stream.Next() {
					t.Fatal("closed stream yielded another event")
				}
			case "deadline":
				if stream.Next() || !errors.Is(stream.Err(), context.DeadlineExceeded) {
					t.Fatalf("deadline did not cancel the stream: %v", stream.Err())
				}
			}
			select {
			case <-model.stopped:
			case <-ctx.Done():
				t.Fatal("server model did not observe the disconnect")
			}
			params := textParams()
			params.PreviousResponseID = openai.String(id)
			_, err := client.Responses.New(ctx, params)
			assertAPIError(t, err, http.StatusBadRequest, "previous_response_id", "previous_response_not_found")
			params = textParams()
			assertTextResponse(t, createResponse(t, client, ctx, params))
		})
	}
}
