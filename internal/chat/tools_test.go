package chat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestMockModelPlanToolsOfficialFixtureUsesFunctionCall(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, MockOptions{
		DefaultCompletionTokens: 32,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    4,
	})
	request := ToolRequest{
		Request: Request{Messages: []Message{{
			Role:    "user",
			Content: "What's the weather like in San Francisco?",
		}}},
		Tools: []FunctionTool{{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
				},
				"required": []any{"location"},
			},
		}},
		Choice:            ToolChoiceAuto,
		ParallelToolCalls: true,
	}

	first, err := chatModel.PlanTools(context.Background(), request)
	if err != nil {
		t.Fatalf("PlanTools() error = %v", err)
	}
	second, err := chatModel.PlanTools(context.Background(), request)
	if err != nil {
		t.Fatalf("PlanTools() second error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("PlanTools() plans differ: %#v and %#v", first, second)
	}
	if first.UseText || len(first.Calls) != 1 {
		t.Fatalf("PlanTools() = %#v, want one function call", first)
	}
	call := first.Calls[0]
	if call.Name != "get_weather" || !strings.HasPrefix(call.ID, "fc_mock_") || !strings.HasPrefix(call.CallID, "call_mock_") {
		t.Fatalf("tool call = %#v", call)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		t.Fatalf("arguments JSON = %q: %v", call.Arguments, err)
	}
	if location, ok := arguments["location"].(string); !ok || location == "" {
		t.Fatalf("arguments = %#v, want string location", arguments)
	}
	if first.Usage.PromptTokens == 0 || first.Usage.CompletionTokens == 0 {
		t.Fatalf("usage = %#v", first.Usage)
	}
}

func TestMockModelPlanToolsIsIndependentFromStreamNonceAndMapOrder(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	seed := int64(42)
	first := ToolRequest{
		Request: Request{
			Messages:    []Message{{Role: "user", Content: "Use a tool."}},
			Seed:        &seed,
			StreamNonce: 1,
		},
		Tools: []FunctionTool{{
			Name: "lookup",
			Parameters: map[string]any{
				"required": []any{"query"},
				"properties": map[string]any{
					"query": map[string]any{"minLength": float64(4), "type": "string"},
				},
				"type": "object",
			},
		}},
		Choice:            ToolChoiceRequired,
		ParallelToolCalls: true,
	}
	second := first
	second.StreamNonce = 999
	second.Tools = []FunctionTool{{
		Name: "lookup",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "minLength": float64(4)},
			},
			"required": []any{"query"},
		},
	}}

	firstPlan, err := chatModel.PlanTools(context.Background(), first)
	if err != nil {
		t.Fatalf("PlanTools() error = %v", err)
	}
	secondPlan, err := chatModel.PlanTools(context.Background(), second)
	if err != nil {
		t.Fatalf("PlanTools() reordered error = %v", err)
	}
	if !reflect.DeepEqual(firstPlan, secondPlan) {
		t.Fatalf("plans differ by nonce or map order: %#v and %#v", firstPlan, secondPlan)
	}
}

func TestMockModelPlanToolsChoices(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	base := ToolRequest{
		Request: Request{Messages: []Message{{Role: "user", Content: "Hello"}}},
		Tools:   []FunctionTool{{Name: "lookup"}},
	}

	none := base
	none.Choice = ToolChoiceNone
	plan, err := chatModel.PlanTools(context.Background(), none)
	if err != nil || !plan.UseText || len(plan.Calls) != 0 {
		t.Fatalf("PlanTools(none) = %#v, %v", plan, err)
	}

	autoWithoutTools := base
	autoWithoutTools.Tools = nil
	autoWithoutTools.Choice = ToolChoiceAuto
	plan, err = chatModel.PlanTools(context.Background(), autoWithoutTools)
	if err != nil || !plan.UseText {
		t.Fatalf("PlanTools(auto without tools) = %#v, %v", plan, err)
	}

	requiredWithoutTools := base
	requiredWithoutTools.Tools = nil
	requiredWithoutTools.Choice = ToolChoiceRequired
	if _, err := chatModel.PlanTools(context.Background(), requiredWithoutTools); !errors.Is(err, ErrRequiredToolMissing) {
		t.Fatalf("PlanTools(required without tools) error = %v", err)
	}

	invalid := base
	invalid.Choice = "sometimes"
	if _, err := chatModel.PlanTools(context.Background(), invalid); !errors.Is(err, ErrInvalidToolChoice) {
		t.Fatalf("PlanTools(invalid choice) error = %v", err)
	}
}

func TestMockModelPlanToolsAutoCoversTextAndToolBranches(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	sawText := false
	sawTools := false
	for index := range 64 {
		plan, err := chatModel.PlanTools(context.Background(), ToolRequest{
			Request: Request{Messages: []Message{{Role: "user", Content: fmt.Sprintf("request %d", index)}}},
			Tools:   []FunctionTool{{Name: "lookup"}},
			Choice:  ToolChoiceAuto,
		})
		if err != nil {
			t.Fatalf("PlanTools(%d) error = %v", index, err)
		}
		sawText = sawText || plan.UseText
		sawTools = sawTools || len(plan.Calls) > 0
	}
	if !sawText || !sawTools {
		t.Fatalf("auto branches: text=%t tools=%t", sawText, sawTools)
	}
}

func TestMockModelPlanToolsParallelCallsAreDistinctAndBounded(t *testing.T) {
	t.Parallel()

	chatModel := newTestMockModel(t, testMockOptions())
	tools := []FunctionTool{{Name: "one"}, {Name: "two"}, {Name: "three"}, {Name: "four"}}
	sawMultiple := false
	for seedValue := int64(0); seedValue < 64; seedValue++ {
		seed := seedValue
		plan, err := chatModel.PlanTools(context.Background(), ToolRequest{
			Request:           Request{Messages: []Message{{Role: "user", Content: "Use tools."}}, Seed: &seed},
			Tools:             tools,
			Choice:            ToolChoiceRequired,
			ParallelToolCalls: true,
			MaxToolCalls:      3,
		})
		if err != nil {
			t.Fatalf("PlanTools(%d) error = %v", seedValue, err)
		}
		if len(plan.Calls) < 1 || len(plan.Calls) > 3 {
			t.Fatalf("calls = %d, want 1..3", len(plan.Calls))
		}
		sawMultiple = sawMultiple || len(plan.Calls) > 1
		names := make(map[string]struct{}, len(plan.Calls))
		for _, call := range plan.Calls {
			if _, exists := names[call.Name]; exists {
				t.Fatalf("duplicate call name %q in %#v", call.Name, plan.Calls)
			}
			names[call.Name] = struct{}{}
		}
	}
	if !sawMultiple {
		t.Fatal("parallel selection never produced multiple calls")
	}

	plan, err := chatModel.PlanTools(context.Background(), ToolRequest{
		Request:           Request{Messages: []Message{{Role: "user", Content: "Use a tool."}}},
		Tools:             tools,
		Choice:            ToolChoiceRequired,
		ParallelToolCalls: false,
		MaxToolCalls:      3,
	})
	if err != nil || len(plan.Calls) != 1 {
		t.Fatalf("non-parallel plan = %#v, %v", plan, err)
	}
}

func TestSynthesizeToolArgumentsCoversPragmaticSchemaSubset(t *testing.T) {
	t.Parallel()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"mode":    map[string]any{"type": "string", "enum": []any{"fast", "safe"}},
			"enabled": map[string]any{"type": "boolean"},
			"count":   map[string]any{"type": "integer", "minimum": float64(2), "maximum": float64(4)},
			"ratio":   map[string]any{"type": "number", "minimum": float64(0), "maximum": float64(1)},
			"tags":    map[string]any{"type": "array", "minItems": float64(2), "items": map[string]any{"type": "string"}},
			"target": map[string]any{"oneOf": []any{
				map[string]any{"type": "string"},
				map[string]any{"type": "integer"},
			}},
		},
		"required": []any{"mode", "enabled", "count", "ratio", "tags", "target"},
	}
	firstRandom := newXorshift64Star(42)
	secondRandom := newXorshift64Star(42)
	first := synthesizeToolArguments(schema, &firstRandom)
	second := synthesizeToolArguments(schema, &secondRandom)
	if first != second {
		t.Fatalf("arguments differ: %q and %q", first, second)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(first), &arguments); err != nil {
		t.Fatalf("decode arguments: %v", err)
	}
	if _, ok := arguments["mode"].(string); !ok {
		t.Fatalf("mode = %#v", arguments["mode"])
	}
	if _, ok := arguments["enabled"].(bool); !ok {
		t.Fatalf("enabled = %#v", arguments["enabled"])
	}
	if count, ok := arguments["count"].(float64); !ok || count < 2 || count > 4 || count != float64(int(count)) {
		t.Fatalf("count = %#v", arguments["count"])
	}
	if tags, ok := arguments["tags"].([]any); !ok || len(tags) != 2 {
		t.Fatalf("tags = %#v", arguments["tags"])
	}
	if _, exists := arguments["target"]; !exists {
		t.Fatalf("target missing from %#v", arguments)
	}
}

func TestMockModelPlanToolsHonorsCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	chatModel := newTestMockModel(t, testMockOptions())
	_, err := chatModel.PlanTools(ctx, ToolRequest{Choice: ToolChoiceNone})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("PlanTools() error = %v, want context canceled", err)
	}
}
