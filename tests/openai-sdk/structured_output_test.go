package sdk_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/pacoricci/mock-ai-go/internal/chat"
	mockopenai "github.com/pacoricci/mock-ai-go/internal/openai"
)

func structuredClient(t *testing.T) (openai.Client, context.Context) {
	t.Helper()
	model, err := chat.NewMockModel(modelID, chat.MockOptions{DefaultCompletionTokens: 128, MaxCompletionTokens: 1024, StreamTokensPerChunk: 1})
	if err != nil {
		t.Fatal(err)
	}
	return newClientWithModels(t, mockopenai.ResponseStoreOptions{TTL: time.Hour, MaxEntries: 100, MaxBytes: 1 << 20}, model)
}

func structuredSchema(t *testing.T) map[string]any {
	t.Helper()
	var schema map[string]any
	err := json.Unmarshal([]byte(`{
 "type":"object","required":["customer","lines","note","approved"],"additionalProperties":false,
 "properties":{
   "customer":{"type":"string","const":"München 世界🙂"},
   "lines":{"type":"array","minItems":2,"maxItems":2,"items":{"$ref":"#/$defs/line"}},
   "note":{"anyOf":[{"type":"string","enum":["ready","pending"]},{"type":"null"}]},
   "approved":{"type":"boolean"}
 },
 "$defs":{"line":{"type":"object","required":["sku","quantity"],"additionalProperties":false,"properties":{
   "sku":{"type":"string","enum":["SKU-A","SKU-B"]},"quantity":{"type":"integer","minimum":1,"maximum":5}
 }}}
}`), &schema)
	if err != nil {
		t.Fatal(err)
	}
	return schema
}

func chatStructuredParams(t *testing.T) openai.ChatCompletionNewParams {
	t.Helper()
	return openai.ChatCompletionNewParams{
		Model: modelID, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Return an order as JSON.")}, Seed: openai.Int(42),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: "order", Schema: structuredSchema(t), Strict: openai.Bool(true)},
		}},
	}
}

func responseStructuredParams(t *testing.T) responses.ResponseNewParams {
	t.Helper()
	return responses.ResponseNewParams{
		Model: modelID, Input: responses.ResponseNewParamsInputUnion{OfString: openai.String("Return an order as JSON.")},
		Text: responses.ResponseTextConfigParam{Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{Name: "order", Schema: structuredSchema(t), Strict: openai.Bool(true)},
		}},
	}
}

func assertStructuredOrder(t *testing.T, content string) {
	t.Helper()
	var order struct {
		Customer string `json:"customer"`
		Lines    []struct {
			SKU      string `json:"sku"`
			Quantity int    `json:"quantity"`
		} `json:"lines"`
		Note     *string `json:"note"`
		Approved bool    `json:"approved"`
	}
	decoder := json.NewDecoder(strings.NewReader(content))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&order); err != nil {
		t.Fatalf("structured decoding: %v; %s", err, content)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &fields); err != nil || len(fields) != 4 {
		t.Fatalf("missing required properties: %s", content)
	}
	if order.Customer != "München 世界🙂" || len(order.Lines) != 2 {
		t.Fatalf("order violates schema: %s", content)
	}
	for _, line := range order.Lines {
		if line.SKU != "SKU-A" && line.SKU != "SKU-B" || line.Quantity < 1 || line.Quantity > 5 {
			t.Fatalf("line violates schema: %s", content)
		}
	}
	if order.Note != nil && *order.Note != "ready" && *order.Note != "pending" {
		t.Fatalf("note violates schema: %s", content)
	}
}

func TestChatStructuredOutputBudgets(t *testing.T) {
	t.Parallel()
	client, ctx := structuredClient(t)
	params := chatStructuredParams(t)
	full, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	content := full.Choices[0].Message.Content
	assertStructuredOrder(t, content)
	needed := full.Usage.CompletionTokens
	if needed != int64((len(content)+3)/4) || full.Choices[0].FinishReason != "stop" {
		t.Fatalf("full completion: %+v", full)
	}
	for _, budget := range []int64{1, 7, needed - 1, needed, needed + 1} {
		params.MaxCompletionTokens = openai.Int(budget)
		result, err := client.Chat.Completions.New(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		wantReason := "stop"
		if budget < needed {
			wantReason = "length"
		}
		text := result.Choices[0].Message.Content
		if result.Choices[0].FinishReason != wantReason || !strings.HasPrefix(content, text) || !utf8.ValidString(text) || int64(len(text)) > budget*4 {
			t.Fatalf("budget %d: %+v", budget, result)
		}
		if budget < needed && json.Valid([]byte(text)) {
			t.Fatalf("truncated document was repaired: %q", text)
		}
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
		stream := client.Chat.Completions.NewStreaming(ctx, params)
		var accumulated openai.ChatCompletionAccumulator
		finishes, usages := 0, 0
		for stream.Next() {
			chunk := stream.Current()
			if !accumulated.AddChunk(chunk) {
				t.Fatal("SDK rejected chunk")
			}
			if len(chunk.Choices) == 0 {
				usages++
				if finishes != 1 || !reflect.DeepEqual(chunk.Usage, result.Usage) {
					t.Fatalf("bad usage event: %+v", chunk)
				}
			}
			for _, choice := range chunk.Choices {
				if finishes != 0 || !utf8.ValidString(choice.Delta.Content) {
					t.Fatalf("invalid content after finish: %+v", chunk)
				}
				if choice.FinishReason != "" {
					finishes++
				}
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
		if finishes != 1 || usages != 1 || accumulated.Choices[0].Message.Content != text || accumulated.Choices[0].FinishReason != wantReason {
			t.Fatalf("stream mismatch: %+v", accumulated)
		}
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{}
	}
	params.MaxCompletionTokens = openai.Int(0)
	params.MaxTokens = openai.Int(7)
	// Omit the modern field explicitly to exercise the legacy token ceiling.
	legacy, err := client.Chat.Completions.New(ctx, params, option.WithJSONDel("max_completion_tokens"))
	if err != nil || legacy.Choices[0].FinishReason != "length" {
		t.Fatalf("legacy budget: %v, %+v", err, legacy)
	}
}

func TestResponsesStructuredOutputBudgets(t *testing.T) {
	t.Parallel()
	client, ctx := structuredClient(t)
	params := responseStructuredParams(t)
	full, err := client.Responses.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	assertStructuredOrder(t, full.OutputText())
	needed := full.Usage.OutputTokens
	for _, budget := range []int64{16, needed - 1, needed, needed + 1} {
		params.MaxOutputTokens = openai.Int(budget)
		result, err := client.Responses.New(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		wantStatus := "completed"
		if budget < needed {
			wantStatus = "incomplete"
			if result.IncompleteDetails.Reason != "max_output_tokens" || result.CompletedAt != 0 || json.Valid([]byte(result.OutputText())) {
				t.Fatalf("incomplete metadata: %s", result.RawJSON())
			}
		}
		if string(result.Status) != wantStatus || string(result.Output[0].AsMessage().Status) != wantStatus || !strings.HasPrefix(full.OutputText(), result.OutputText()) || result.Usage.OutputTokens > budget {
			t.Fatalf("budget %d: %s", budget, result.RawJSON())
		}
		if result.Text.Format.Type != "json_schema" || result.Text.Format.Name != "order" || !result.Text.Format.Strict || len(result.Text.Format.Schema) == 0 {
			t.Fatalf("format echo lost: %+v", result.Text)
		}
		stream := client.Responses.NewStreaming(ctx, params)
		var deltas strings.Builder
		sequence, terminal := int64(0), 0
		itemDone, textDone := 0, 0
		var streamID string
		for stream.Next() {
			event := stream.Current()
			if terminal != 0 || event.SequenceNumber != sequence {
				t.Fatalf("event ordering: %s", event.RawJSON())
			}
			sequence++
			switch event.Type {
			case "response.output_text.delta":
				if !utf8.ValidString(event.Delta) {
					t.Fatal("invalid UTF-8 delta")
				}
				deltas.WriteString(event.Delta)
			case "response.output_text.done":
				textDone++
				if event.Text != deltas.String() {
					t.Fatal("text done differs from deltas")
				}
			case "response.output_item.done":
				itemDone++
				if string(event.AsResponseOutputItemDone().Item.AsMessage().Status) != wantStatus {
					t.Fatalf("item status mismatch: %s", event.RawJSON())
				}
			case "response.completed", "response.incomplete":
				terminal++
				streamID = event.Response.ID
				if event.Type != "response."+wantStatus || event.Response.OutputText() != result.OutputText() || !reflect.DeepEqual(event.Response.Usage, result.Usage) || event.Response.IncompleteDetails.Reason != result.IncompleteDetails.Reason {
					t.Fatalf("terminal mismatch: %s", event.RawJSON())
				}
			case "error", "response.failed":
				t.Fatalf("unexpected error: %s", event.RawJSON())
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
		if terminal != 1 || itemDone != 1 || textDone != 1 || deltas.String() != result.OutputText() {
			t.Fatalf("stream did not reconstruct response: %q", deltas.String())
		}
		// A terminal incomplete response is stored and can seed the next turn.
		for _, parentID := range []string{result.ID, streamID} {
			followup := responses.ResponseNewParams{Model: modelID, PreviousResponseID: openai.String(parentID), Input: responses.ResponseNewParamsInputUnion{OfString: openai.String("Continue.")}, MaxOutputTokens: openai.Int(16)}
			next, err := client.Responses.New(ctx, followup)
			if err != nil || next.Text.Format.Type != "text" || next.Status != "completed" {
				t.Fatalf("continuation or format lifetime: %v, %+v", err, next)
			}
		}
	}
}

func TestStructuredOutputWithTools(t *testing.T) {
	t.Parallel()
	client, ctx := structuredClient(t)
	params := chatToolParams()
	params.ResponseFormat = chatStructuredParams(t).ResponseFormat
	result, err := client.Chat.Completions.New(ctx, params)
	if err != nil || result.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("structured format prevented tool selection: %v, %+v", err, result)
	}
	params.Messages = append(params.Messages, result.Choices[0].Message.ToParam())
	for _, call := range result.Choices[0].Message.ToolCalls {
		params.Messages = append(params.Messages, openai.ToolMessage(`{"result":"done"}`, call.ID))
	}
	params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("none")}
	answer, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	assertStructuredOrder(t, answer.Choices[0].Message.Content)
}

// StreamStructured pauses after the first delta until the request is canceled.
func (m *blockedStreamModel) StreamStructured(ctx context.Context, request chat.Request, plan chat.StructuredPlan, emit chat.Emit) error {
	defer close(m.stopped)
	return m.MockModel.StreamStructured(ctx, request, plan, func(chunk chat.Chunk) error {
		if err := emit(chunk); err != nil {
			return err
		}
		<-ctx.Done()
		return ctx.Err()
	})
}

func TestStructuredCancellationDoesNotStorePartialResponse(t *testing.T) {
	t.Parallel()
	model := &blockedStreamModel{MockModel: newMock(t, modelID), stopped: make(chan struct{})}
	client, ctx := newClientWithModels(t, mockopenai.ResponseStoreOptions{TTL: time.Hour, MaxEntries: 100, MaxBytes: 1 << 20}, model)
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	params := responseStructuredParams(t)
	stream := client.Responses.NewStreaming(streamCtx, params)
	defer func() { _ = stream.Close() }()
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
		t.Fatalf("stream failed before first delta: %v", stream.Err())
	}
	cancel()
	if stream.Next() || !errors.Is(stream.Err(), context.Canceled) {
		t.Fatalf("cancellation did not stop stream: %v", stream.Err())
	}
	select {
	case <-model.stopped:
	case <-ctx.Done():
		t.Fatal("model did not observe cancellation")
	}
	params.PreviousResponseID = openai.String(id)
	_, err := client.Responses.New(ctx, params)
	assertAPIError(t, err, 400, "previous_response_id", "previous_response_not_found")
}

func TestStructuredJSONModeAndSchemaErrors(t *testing.T) {
	t.Parallel()
	client, ctx := structuredClient(t)
	cp := chatStructuredParams(t)
	cp.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &shared.ResponseFormatJSONObjectParam{}}
	completion, err := client.Chat.Completions.New(ctx, cp)
	if err != nil || !json.Valid([]byte(completion.Choices[0].Message.Content)) {
		t.Fatalf("Chat JSON mode: %v, %+v", err, completion)
	}
	rp := responseStructuredParams(t)
	rp.Text.Format = responses.ResponseFormatTextConfigUnionParam{OfJSONObject: &shared.ResponseFormatJSONObjectParam{}}
	response, err := client.Responses.New(ctx, rp)
	if err != nil || !json.Valid([]byte(response.OutputText())) || response.Text.Format.Type != "json_object" {
		t.Fatalf("Responses JSON mode: %v, %+v", err, response)
	}
	for _, streaming := range []bool{false, true} {
		cp = chatStructuredParams(t)
		cp.ResponseFormat.OfJSONSchema.JSONSchema.Schema = map[string]any{"type": "object", "additionalProperties": false, "required": []string{"x"}, "properties": map[string]any{"x": map[string]any{"type": "string", "pattern": ".+"}}}
		if streaming {
			stream := client.Chat.Completions.NewStreaming(ctx, cp)
			if stream.Next() {
				t.Fatal("invalid schema started a stream")
			}
			err = stream.Err()
			_ = stream.Close()
		} else {
			_, err = client.Chat.Completions.New(ctx, cp)
		}
		assertSchemaAPIError(t, err, "response_format.json_schema.schema.properties.x.pattern")
		rp = responseStructuredParams(t)
		rp.Text.Format.OfJSONSchema.Schema = cp.ResponseFormat.OfJSONSchema.JSONSchema.Schema.(map[string]any)
		if streaming {
			stream := client.Responses.NewStreaming(ctx, rp)
			if stream.Next() {
				t.Fatal("invalid schema started a stream")
			}
			err = stream.Err()
			_ = stream.Close()
		} else {
			_, err = client.Responses.New(ctx, rp)
		}
		assertSchemaAPIError(t, err, "text.format.schema.properties.x.pattern")
	}
}

func assertSchemaAPIError(t *testing.T, err error, param string) {
	t.Helper()
	var apiError *openai.Error
	if !errors.As(err, &apiError) || apiError.StatusCode != 400 || apiError.Param != param || apiError.Type != "invalid_request_error" {
		t.Fatalf("expected typed schema error at %s, got %v", param, err)
	}
}
