package sdk_test

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

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/pacoricci/mock-ai-go/internal/chat"
	mockopenai "github.com/pacoricci/mock-ai-go/internal/openai"
)

const modelID = "mock-chat-model"

func newClient(t *testing.T) (openai.Client, context.Context) {
	t.Helper()
	model, err := chat.NewMockModel(modelID, chat.MockOptions{
		DefaultCompletionTokens: 32,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    4,
	})
	if err != nil {
		t.Fatal(err)
	}
	return newClientWithModels(t, mockopenai.ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 100, MaxBytes: 1 << 20,
	}, model)
}

func newClientWithModels(t *testing.T, store mockopenai.ResponseStoreOptions, models ...chat.Model) (openai.Client, context.Context) {
	t.Helper()
	registry, err := chat.NewRegistry(models...)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := mockopenai.NewHandler(registry, store)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return openai.NewClient(
		option.WithBaseURL(server.URL+"/v1/"),
		option.WithAPIKey("mock-api-key"),
		option.WithHTTPClient(server.Client()),
		option.WithMaxRetries(0),
	), ctx
}

func TestChatCompletions(t *testing.T) {
	t.Parallel()
	for _, limit := range []string{"default", "max_completion_tokens", "max_tokens"} {
		t.Run(limit, func(t *testing.T) {
			t.Parallel()
			client, ctx := newClient(t)
			params := openai.ChatCompletionNewParams{
				Model: modelID,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage("Be concise."),
					openai.UserMessage("Hello, café 世界"),
				},
				Seed: openai.Int(42),
			}
			wantTokens := int64(32)
			switch limit {
			case "max_completion_tokens":
				params.MaxCompletionTokens = openai.Int(16)
				wantTokens = 16
			case "max_tokens":
				params.MaxTokens = openai.Int(16)
				wantTokens = 16
			}
			result, err := client.Chat.Completions.New(ctx, params)
			if err != nil {
				t.Fatal(err)
			}
			if result.ID == "" || result.Object != "chat.completion" || result.Model != modelID || result.Created <= 0 {
				t.Fatalf("completion metadata = %+v", result)
			}
			if len(result.Choices) != 1 || result.Choices[0].Message.Role != "assistant" || result.Choices[0].FinishReason != "stop" {
				t.Fatalf("choices = %+v", result.Choices)
			}
			text := result.Choices[0].Message.Content
			if int64(len(strings.Fields(text))) != wantTokens || result.Usage.CompletionTokens != wantTokens || result.Usage.PromptTokens <= 0 || result.Usage.TotalTokens != result.Usage.PromptTokens+wantTokens {
				t.Fatalf("text = %q, usage = %+v", text, result.Usage)
			}
			for _, includeUsage := range []bool{false, true} {
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(includeUsage)}
				stream := client.Chat.Completions.NewStreaming(ctx, params)
				t.Cleanup(func() { _ = stream.Close() })
				var acc openai.ChatCompletionAccumulator
				chunks, usageChunks, finishes := 0, 0, 0
				for stream.Next() {
					chunk := stream.Current()
					chunks++
					if chunk.Object != "chat.completion.chunk" || chunk.Model != modelID {
						t.Fatalf("chunk metadata = %+v", chunk)
					}
					if !acc.AddChunk(chunk) {
						t.Fatal("SDK accumulator rejected a chunk")
					}
					if chunk.Usage.CompletionTokens != 0 {
						usageChunks++
						if len(chunk.Choices) != 0 || !reflect.DeepEqual(chunk.Usage, result.Usage) {
							t.Fatalf("usage chunk = %+v", chunk)
						}
					}
					for _, choice := range chunk.Choices {
						if choice.FinishReason != "" {
							finishes++
							if choice.FinishReason != "stop" {
								t.Fatalf("finish reason = %q", choice.FinishReason)
							}
						}
					}
				}
				if err := stream.Err(); err != nil {
					t.Fatal(err)
				}
				wantUsageChunks := 0
				if includeUsage {
					wantUsageChunks = 1
				}
				if chunks < 3 || finishes != 1 || usageChunks != wantUsageChunks || len(acc.Choices) != 1 || acc.Choices[0].Message.Content != text {
					t.Fatalf("chunks=%d finishes=%d usage=%d accumulated=%+v", chunks, finishes, usageChunks, acc.Choices)
				}
			}
		})
	}
}

func textParams() responses.ResponseNewParams {
	return responses.ResponseNewParams{
		Model:           modelID,
		Input:           responses.ResponseNewParamsInputUnion{OfString: openai.String("Hello, café 世界")},
		MaxOutputTokens: openai.Int(16),
	}
}

func TestResponsesText(t *testing.T) {
	t.Parallel()
	for _, input := range []string{"string", "messages"} {
		t.Run(input, func(t *testing.T) {
			t.Parallel()
			client, ctx := newClient(t)
			params := textParams()
			params.Store = openai.Bool(false)
			params.Instructions = openai.String("Be concise.")
			params.Metadata = map[string]string{"suite": "openai-go"}
			if input == "messages" {
				params.Input = responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						{OfMessage: &responses.EasyInputMessageParam{
							Role:    "user",
							Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String("Hello, café 世界")},
						}},
					},
				}
			}
			result, err := client.Responses.New(ctx, params)
			if err != nil {
				t.Fatal(err)
			}
			assertTextResponse(t, result)
			if result.Metadata["suite"] != "openai-go" || result.MaxOutputTokens != 16 {
				t.Fatalf("echoed fields = %+v", result)
			}
			completed, deltas := streamResponse(t, client, ctx, params)
			assertTextResponse(t, completed)
			if deltas != result.OutputText() || completed.OutputText() != result.OutputText() || !reflect.DeepEqual(completed.Usage, result.Usage) {
				t.Fatalf("JSON text=%q usage=%+v, SSE deltas=%q text=%q usage=%+v", result.OutputText(), result.Usage, deltas, completed.OutputText(), completed.Usage)
			}
		})
	}
}

func assertTextResponse(t *testing.T, result *responses.Response) {
	t.Helper()
	if result.ID == "" || result.Object != "response" || result.Status != "completed" || result.Model != modelID || result.CreatedAt <= 0 {
		t.Fatalf("response metadata = %+v", result)
	}
	if len(result.Output) != 1 || result.Output[0].Type != "message" {
		t.Fatalf("output = %+v", result.Output)
	}
	message := result.Output[0].AsMessage()
	if message.Role != "assistant" || message.Status != "completed" || len(message.Content) != 1 || message.Content[0].Type != "output_text" {
		t.Fatalf("message = %+v", message)
	}
	if len(strings.Fields(result.OutputText())) != 16 || result.Usage.OutputTokens != 16 || result.Usage.InputTokens <= 0 || result.Usage.TotalTokens != result.Usage.InputTokens+16 {
		t.Fatalf("text = %q, usage = %+v", result.OutputText(), result.Usage)
	}
}

func streamResponse(t *testing.T, client openai.Client, ctx context.Context, params responses.ResponseNewParams) (*responses.Response, string) {
	t.Helper()
	stream := client.Responses.NewStreaming(ctx, params)
	t.Cleanup(func() { _ = stream.Close() })
	var completed *responses.Response
	var deltas strings.Builder
	var events []string
	for stream.Next() {
		event := stream.Current()
		if event.SequenceNumber != int64(len(events)) {
			t.Fatalf("sequence number = %d, want %d", event.SequenceNumber, len(events))
		}
		events = append(events, event.Type)
		switch event.Type {
		case "response.output_text.delta", "response.function_call_arguments.delta":
			deltas.WriteString(event.Delta)
		case "response.output_text.done":
			if event.AsResponseOutputTextDone().Text != deltas.String() {
				t.Fatalf("text done = %q, deltas = %q", event.Text, deltas.String())
			}
		case "response.function_call_arguments.done":
			if event.AsResponseFunctionCallArgumentsDone().Arguments != deltas.String() {
				t.Fatalf("arguments done = %q, deltas = %q", event.Arguments, deltas.String())
			}
		case "response.completed":
			if completed != nil {
				t.Fatal("duplicate completion event")
			}
			response := event.AsResponseCompleted().Response
			completed = &response
		case "error", "response.failed", "response.incomplete":
			t.Fatalf("unexpected failure event: %s", event.RawJSON())
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if completed == nil || len(events) < 5 || events[0] != "response.created" || events[1] != "response.in_progress" || events[len(events)-1] != "response.completed" || deltas.Len() == 0 {
		t.Fatalf("incomplete stream: events=%v deltas=%q", events, deltas.String())
	}
	return completed, deltas.String()
}

func TestResponsesContinuation(t *testing.T) {
	t.Parallel()
	for _, firstStream := range []bool{false, true} {
		name := "JSON to SSE"
		if firstStream {
			name = "SSE to JSON"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			client, ctx := newClient(t)
			params := textParams()
			var first *responses.Response
			if firstStream {
				first, _ = streamResponse(t, client, ctx, params)
			} else {
				var err error
				first, err = client.Responses.New(ctx, params)
				if err != nil {
					t.Fatal(err)
				}
			}
			params.PreviousResponseID = openai.String(first.ID)
			params.Input.OfString = openai.String("Continue.")
			var second *responses.Response
			if firstStream {
				var err error
				second, err = client.Responses.New(ctx, params)
				if err != nil {
					t.Fatal(err)
				}
			} else {
				second, _ = streamResponse(t, client, ctx, params)
			}
			assertTextResponse(t, second)
			if second.ID == first.ID || second.PreviousResponseID != first.ID || second.Usage.InputTokens <= first.Usage.InputTokens {
				t.Fatalf("continuation: first=%+v second=%+v", first, second)
			}
		})
	}
}

func TestResponsesFunctionCalling(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	params := textParams()
	params.Input.OfString = openai.String("What is the weather in Rome?")
	params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsRequired)}
	params.ParallelToolCalls = openai.Bool(false)
	params.Tools = []responses.ToolUnionParam{{OfFunction: &responses.FunctionToolParam{
		Name: "get_weather",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"location": map[string]any{"type": "string"}},
			"required":   []string{"location"},
		},
	}}}
	result, err := client.Responses.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "completed" || len(result.Output) != 1 || result.Output[0].Type != "function_call" {
		t.Fatalf("function output = %+v", result)
	}
	call := result.Output[0].AsFunctionCall()
	var args struct {
		Location *string `json:"location"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if call.Name != "get_weather" || call.CallID == "" || call.Status != "completed" || args.Location == nil {
		t.Fatalf("function call = %+v", call)
	}
	streamed, deltas := streamResponse(t, client, ctx, params)
	if len(streamed.Output) != 1 || streamed.Output[0].Type != "function_call" || streamed.Status != "completed" {
		t.Fatalf("streamed function output = %+v", streamed)
	}
	streamedCall := streamed.Output[0].AsFunctionCall()
	if !reflect.DeepEqual(streamedCall, call) || deltas != call.Arguments {
		t.Fatalf("JSON call=%+v, SSE call=%+v deltas=%q", call, streamedCall, deltas)
	}
	for _, previous := range []*responses.Response{result, streamed} {
		followup := textParams()
		followup.PreviousResponseID = openai.String(previous.ID)
		followup.Input = responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: openai.String(previous.Output[0].AsFunctionCall().CallID),
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{OfString: openai.String(`{"temperature":21}`)},
				},
			}},
		}
		answer, err := client.Responses.New(ctx, followup)
		if err != nil {
			t.Fatal(err)
		}
		assertTextResponse(t, answer)
	}
}

func TestAPIErrors(t *testing.T) {
	t.Parallel()
	for _, endpoint := range []string{"chat", "responses"} {
		for _, stream := range []bool{false, true} {
			for _, invalid := range []string{"model", "token limit"} {
				name := endpoint + "/JSON/" + invalid
				if stream {
					name = endpoint + "/SSE/" + invalid
				}
				t.Run(name, func(t *testing.T) {
					t.Parallel()
					client, ctx := newClient(t)
					model, limit := modelID, int64(65)
					status, param, code := http.StatusBadRequest, "max_completion_tokens", "limit_exceeded"
					if endpoint == "responses" {
						param = "max_output_tokens"
					}
					if invalid == "model" {
						model, limit = "missing", 16
						status, param, code = http.StatusNotFound, "model", "model_not_found"
					}
					var err error
					if endpoint == "chat" {
						params := openai.ChatCompletionNewParams{Model: model, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Hello")}, MaxCompletionTokens: openai.Int(limit)}
						if stream {
							s := client.Chat.Completions.NewStreaming(ctx, params)
							t.Cleanup(func() { _ = s.Close() })
							if s.Next() {
								t.Fatal("unexpected event on rejected request")
							}
							err = s.Err()
						} else {
							_, err = client.Chat.Completions.New(ctx, params)
						}
					} else {
						params := textParams()
						params.Model = model
						params.MaxOutputTokens = openai.Int(limit)
						if stream {
							s := client.Responses.NewStreaming(ctx, params)
							t.Cleanup(func() { _ = s.Close() })
							if s.Next() {
								t.Fatal("unexpected event on rejected request")
							}
							err = s.Err()
						} else {
							_, err = client.Responses.New(ctx, params)
						}
					}
					assertAPIError(t, err, status, param, code)
				})
			}
		}
	}
}

func assertAPIError(t *testing.T, err error, status int, param, code string) {
	t.Helper()
	var apiError *openai.Error
	if !errors.As(err, &apiError) {
		t.Fatalf("error = %v, want *openai.Error", err)
	}
	if apiError.StatusCode != status || apiError.Type != "invalid_request_error" || apiError.Param != param || apiError.Code != code || apiError.Message == "" {
		t.Fatalf("API error = %+v", apiError)
	}
}

func TestResponsesStoreFalse(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	params := textParams()
	params.Store = openai.Bool(false)
	result, err := client.Responses.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	params.PreviousResponseID = openai.String(result.ID)
	_, err = client.Responses.New(ctx, params)
	assertAPIError(t, err, http.StatusBadRequest, "previous_response_id", "previous_response_not_found")
}
