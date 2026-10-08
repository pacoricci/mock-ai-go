package sdk_test

import (
	"reflect"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

func chatToolParams() openai.ChatCompletionNewParams {
	tools := make([]openai.ChatCompletionToolUnionParam, 0, 3)
	for _, name := range []string{"weather", "traffic", "currency"} {
		tools = append(tools, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{Name: name, Parameters: shared.FunctionParameters{
			"type": "object", "properties": map[string]any{"location": map[string]any{"const": "café 世界🙂 and a long location"}}, "required": []string{"location"},
		}}))
	}
	return openai.ChatCompletionNewParams{Model: modelID, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Check Rome.")}, Tools: tools, ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}, Seed: openai.Int(42), ParallelToolCalls: openai.Bool(true)}
}

func TestChatFunctionCallingRoundTrip(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	params := chatToolParams()
	// Find a deterministic seed that selects multiple distinct tools.
	var result *openai.ChatCompletion
	for seed := int64(0); seed < 64; seed++ {
		params.Seed = openai.Int(seed)
		var err error
		result, err = client.Chat.Completions.New(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Choices[0].Message.ToolCalls) > 1 {
			break
		}
	}
	calls := result.Choices[0].Message.ToolCalls
	if len(calls) < 2 || result.Choices[0].FinishReason != "tool_calls" || result.Choices[0].Message.Content != "" {
		t.Fatalf("tool response: %+v", result)
	}
	seen := map[string]bool{}
	for _, call := range calls {
		if call.ID == "" || call.Type != "function" || seen[call.Function.Name] {
			t.Fatalf("invalid call: %+v", call)
		}
		seen[call.Function.Name] = true
	}
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
	stream := client.Chat.Completions.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()
	var accumulated openai.ChatCompletionAccumulator
	fragments, finish, usage := 0, 0, 0
	for stream.Next() {
		chunk := stream.Current()
		if !accumulated.AddChunk(chunk) {
			t.Fatal("SDK rejected tool chunk")
		}
		if len(chunk.Choices) == 0 {
			usage++
			continue
		}
		choice := chunk.Choices[0]
		fragments += len(choice.Delta.ToolCalls)
		if choice.FinishReason != "" {
			finish++
			if choice.FinishReason != "tool_calls" {
				t.Fatalf("finish=%q", choice.FinishReason)
			}
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if fragments <= len(calls) || finish != 1 || usage != 1 {
		t.Fatalf("fragments=%d finish=%d usage=%d", fragments, finish, usage)
	}
	// SDK response metadata records different raw JSON for fragments; compare semantic fields.
	got := accumulated.Choices[0].Message.ToolCalls
	if len(got) != len(calls) {
		t.Fatalf("calls=%d want=%d", len(got), len(calls))
	}
	for index, call := range calls {
		if got[index].ID != call.ID || got[index].Type != call.Type || got[index].Function.Name != call.Function.Name || got[index].Function.Arguments != call.Function.Arguments {
			t.Fatalf("JSON/SSE differ: %+v / %+v", call, got[index])
		}
	}
	if accumulated.Usage.PromptTokens != result.Usage.PromptTokens || accumulated.Usage.CompletionTokens != result.Usage.CompletionTokens {
		t.Fatalf("usage differs: %+v / %+v", accumulated.Usage, result.Usage)
	}
	for _, message := range []openai.ChatCompletionMessage{result.Choices[0].Message, accumulated.Choices[0].Message} {
		follow := params
		follow.StreamOptions = openai.ChatCompletionStreamOptionsParam{}
		follow.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("none")}
		follow.Messages = append(append([]openai.ChatCompletionMessageParamUnion{}, params.Messages...), message.ToParam())
		for index := len(calls) - 1; index >= 0; index-- {
			follow.Messages = append(follow.Messages, openai.ToolMessage(`{"value":21}`, calls[index].ID))
		}
		answer, err := client.Chat.Completions.New(ctx, follow)
		if err != nil {
			t.Fatal(err)
		}
		if answer.Choices[0].FinishReason != "stop" || answer.Choices[0].Message.Content == "" || answer.Usage.PromptTokens <= result.Usage.PromptTokens {
			t.Fatalf("follow-up: %+v", answer)
		}
		streamed := client.Chat.Completions.NewStreaming(ctx, follow)
		var acc openai.ChatCompletionAccumulator
		for streamed.Next() {
			acc.AddChunk(streamed.Current())
		}
		_ = streamed.Close()
		if err := streamed.Err(); err != nil {
			t.Fatal(err)
		}
		if acc.Choices[0].Message.Content != answer.Choices[0].Message.Content {
			t.Fatal("follow-up JSON/SSE text differs")
		}
		follow.Messages[len(follow.Messages)-1] = openai.ToolMessage(`{"value":9999}`, calls[0].ID)
		changed, err := client.Chat.Completions.New(ctx, follow)
		if err != nil {
			t.Fatal(err)
		}
		if changed.Choices[0].Message.Content == answer.Choices[0].Message.Content {
			t.Fatal("tool result did not affect generation")
		}
	}
}

func TestChatToolChoices(t *testing.T) {
	t.Parallel()
	client, ctx := newClient(t)
	params := chatToolParams()
	params.ParallelToolCalls = openai.Bool(false)
	result, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Choices[0].Message.ToolCalls) != 1 {
		t.Fatal("parallel=false must emit one call")
	}
	params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{Function: openai.ChatCompletionNamedToolChoiceFunctionParam{Name: "traffic"}}}
	result, err = client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Choices[0].Message.ToolCalls) != 1 || result.Choices[0].Message.ToolCalls[0].Function.Name != "traffic" {
		t.Fatalf("named choice: %+v", result)
	}
	params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("auto")}
	first, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	second, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.Choices, second.Choices) {
		t.Fatal("auto choice is not deterministic")
	}
	params.Seed = openai.Int(999)
	changed, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(first.Choices, changed.Choices) {
		t.Fatal("seed did not affect tool-enabled generation")
	}
}
