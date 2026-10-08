package sdk_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/pacoricci/mock-ai-go/internal/app"
	"github.com/pacoricci/mock-ai-go/internal/config"
)

func newClient(t *testing.T) (anthropic.Client, context.Context) {
	t.Helper()
	server, err := app.Start(io.Discard, config.Config{
		Name: "anthropic-sdk-test", HTTPAddress: "127.0.0.1:0", ModelID: "mock",
		DefaultCompletionTokens: 32, MaxCompletionTokens: 256, StreamTokensPerChunk: 2,
		ResponseStoreTTL: time.Hour, ResponseStoreMaxEntries: 100, ResponseStoreMaxBytes: 1 << 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			t.Error(err)
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return anthropic.NewClient(option.WithBaseURL("http://"+server.HTTPAddress()), option.WithAPIKey("dummy-key"), option.WithMaxRetries(0)), ctx
}

func textParams() anthropic.MessageNewParams {
	return anthropic.MessageNewParams{
		Model: "mock", MaxTokens: 32,
		System:   []anthropic.TextBlockParam{{Text: "Be concise."}},
		Messages: []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("Hello café 世界"))},
	}
}

func accumulated(t *testing.T, client anthropic.Client, ctx context.Context, params anthropic.MessageNewParams) anthropic.Message {
	t.Helper()
	stream := client.Messages.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()
	var message anthropic.Message
	var events []string
	openBlock, nextBlock := false, int64(0)
	for stream.Next() {
		event := stream.Current()
		events = append(events, event.Type)
		switch event.Type {
		case "content_block_start":
			if openBlock || event.Index != nextBlock {
				t.Fatalf("invalid block start: %+v", event)
			}
			openBlock = true
		case "content_block_delta":
			if !openBlock || event.Index != nextBlock {
				t.Fatalf("delta outside its block: %+v", event)
			}
		case "content_block_stop":
			if !openBlock || event.Index != nextBlock {
				t.Fatalf("invalid block stop: %+v", event)
			}
			openBlock = false
			nextBlock++
		}
		if err := message.Accumulate(event); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if len(events) < 5 || events[0] != "message_start" || events[len(events)-2] != "message_delta" || events[len(events)-1] != "message_stop" || openBlock {
		t.Fatalf("event sequence = %v", events)
	}
	return message
}

func equalMessages(t *testing.T, first, second anthropic.Message) {
	t.Helper()
	// SDK raw JSON metadata differs between accumulated and directly decoded blocks.
	a, err := json.Marshal(first.ToParam())
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(second.ToParam())
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(b) || first.StopReason != second.StopReason || first.Usage.InputTokens != second.Usage.InputTokens || first.Usage.OutputTokens != second.Usage.OutputTokens {
		t.Fatalf("JSON/SSE differ: %s / %s; usage %+v / %+v", a, b, first.Usage, second.Usage)
	}
}

func TestTextAndConcurrentReplay(t *testing.T) {
	client, ctx := newClient(t)
	var wg sync.WaitGroup
	ids := make(chan string, 16)
	for conversation := range 8 {
		wg.Go(func() {
			params := textParams()
			params.Messages = append(params.Messages, anthropic.NewUserMessage(anthropic.NewTextBlock(fmt.Sprintf("Conversation %d", conversation))))
			first, err := client.Messages.New(ctx, params)
			if err != nil {
				t.Error(err)
				return
			}
			if first.Type != "message" || first.Role != "assistant" || first.Model != "mock" || first.StopReason != "max_tokens" || len(first.Content) != 1 || len(strings.Fields(first.Content[0].Text)) != 32 || first.Usage.OutputTokens != 32 {
				t.Errorf("message = %+v", first)
				return
			}
			ids <- first.ID
			equalMessages(t, *first, accumulated(t, client, ctx, params))
			params.Messages = append(params.Messages, first.ToParam(), anthropic.NewUserMessage(anthropic.NewTextBlock("Continue")))
			second, err := client.Messages.New(ctx, params)
			if err != nil {
				t.Error(err)
				return
			}
			ids <- second.ID
			if second.Usage.InputTokens <= first.Usage.InputTokens {
				t.Error("replayed history did not increase usage")
			}
			equalMessages(t, *second, accumulated(t, client, ctx, params))
		})
	}
	wg.Wait()
	close(ids)
	seen := map[string]bool{}
	for id := range ids {
		if id == "" || seen[id] {
			t.Fatalf("missing or duplicate ID %q", id)
		}
		seen[id] = true
	}
}

func toolParams() anthropic.MessageNewParams {
	params := textParams()
	params.MaxTokens = 256
	params.ToolChoice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	for _, name := range []string{"weather", "clock", "calendar"} {
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name: name, InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{"city": map[string]any{"type": "string", "enum": []string{"München 世界"}}}, Required: []string{"city"},
			},
		}})
	}
	return params
}

func TestToolsAndSDKReplay(t *testing.T) {
	client, ctx := newClient(t)
	params := toolParams()
	var result *anthropic.Message
	// Find a deterministic parallel plan without relying on a particular PRNG seed.
	for attempt := range 32 {
		params.Messages = []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(fmt.Sprintf("Weather and time %d", attempt)))}
		var err error
		result, err = client.Messages.New(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Content) > 1 {
			break
		}
	}
	if result.StopReason != "tool_use" || len(result.Content) < 2 {
		t.Fatalf("parallel plan = %+v", result)
	}
	equalMessages(t, *result, accumulated(t, client, ctx, params))
	var outputs []anthropic.ContentBlockParamUnion
	for index := len(result.Content) - 1; index >= 0; index-- {
		block := result.Content[index]
		var input map[string]any
		if err := json.Unmarshal(block.Input, &input); err != nil || input["city"] != "München 世界" || block.Type != "tool_use" {
			t.Fatalf("tool = %+v, input = %v", block, input)
		}
		outputs = append(outputs, anthropic.NewToolResultBlock(block.ID, "Sunny, 12:00", false))
	}
	params.Messages = append(params.Messages, result.ToParam(), anthropic.NewUserMessage(outputs...))
	params.ToolChoice = anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
	continued, err := client.Messages.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if continued.StopReason != "max_tokens" || continued.Content[0].Type != "text" {
		t.Fatalf("continued = %+v", continued)
	}
	equalMessages(t, *continued, accumulated(t, client, ctx, params))
}

func TestToolChoicesAndErrors(t *testing.T) {
	client, ctx := newClient(t)
	for _, choice := range []anthropic.ToolChoiceUnionParam{
		{OfAny: &anthropic.ToolChoiceAnyParam{DisableParallelToolUse: anthropic.Bool(true)}},
		anthropic.ToolChoiceParamOfTool("weather"),
		{OfNone: &anthropic.ToolChoiceNoneParam{}},
	} {
		params := toolParams()
		params.ToolChoice = choice
		result, err := client.Messages.New(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Content) != 1 {
			t.Fatalf("single selection = %+v", result)
		}
		equalMessages(t, *result, accumulated(t, client, ctx, params))
	}
	for _, test := range []struct {
		name   string
		budget int64
		status int
	}{{"mock", 0, 400}, {"mock", 257, 400}, {"missing", 32, 404}} {
		params := textParams()
		params.Model, params.MaxTokens = anthropic.Model(test.name), test.budget
		var raw *http.Response
		_, err := client.Messages.New(ctx, params, option.WithResponseInto(&raw))
		var apiError *anthropic.Error
		if !errors.As(err, &apiError) || apiError.StatusCode != test.status || raw.Header.Get("request-id") == "" {
			t.Fatalf("error = %v, response = %+v", err, raw)
		}
		stream := client.Messages.NewStreaming(ctx, params)
		for stream.Next() {
			t.Fatal("invalid request emitted a stream event")
		}
		if !errors.As(stream.Err(), &apiError) || apiError.StatusCode != test.status {
			t.Fatalf("stream error = %v", stream.Err())
		}
		_ = stream.Close()
	}
}
