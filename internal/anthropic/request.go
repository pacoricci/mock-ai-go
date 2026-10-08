package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

const maxRequestBytes = 1 << 20

type messageRequest struct {
	Model       string           `json:"model"`
	Messages    []inputMessage   `json:"messages"`
	MaxTokens   *int             `json:"max_tokens"`
	System      json.RawMessage  `json:"system"`
	Stream      bool             `json:"stream"`
	Tools       []toolDefinition `json:"tools"`
	ToolChoice  *toolChoice      `json:"tool_choice"`
	Temperature *float64         `json:"temperature"`
	TopP        *float64         `json:"top_p"`
	TopK        *int             `json:"top_k"`
	Metadata    *struct {
		UserID *string `json:"user_id"`
	} `json:"metadata"`
	StopSequences []string `json:"stop_sequences"`
}

type inputMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type toolDefinition struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
	Strict      bool           `json:"strict"`
}

type toolChoice struct {
	Type               string `json:"type"`
	Name               string `json:"name"`
	DisableParallelUse bool   `json:"disable_parallel_tool_use"`
}

func decodeStrict(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("expected a single JSON value")
	}
	return nil
}

func prepareRequest(request messageRequest) (chat.ToolRequest, error) {
	if strings.TrimSpace(request.Model) == "" || len(request.Messages) == 0 {
		return chat.ToolRequest{}, fmt.Errorf("model and nonempty messages are required")
	}
	if request.MaxTokens == nil || *request.MaxTokens <= 0 {
		return chat.ToolRequest{}, fmt.Errorf("max_tokens must be positive; cache warming with zero tokens is unsupported")
	}
	if len(request.StopSequences) != 0 {
		return chat.ToolRequest{}, fmt.Errorf("stop_sequences are unsupported")
	}
	if request.Temperature != nil && (*request.Temperature < 0 || *request.Temperature > 1) {
		return chat.ToolRequest{}, fmt.Errorf("temperature must be between 0 and 1")
	}
	if request.TopP != nil && (*request.TopP < 0 || *request.TopP > 1) {
		return chat.ToolRequest{}, fmt.Errorf("top_p must be between 0 and 1")
	}
	if request.TopK != nil && *request.TopK < 0 {
		return chat.ToolRequest{}, fmt.Errorf("top_k must be nonnegative")
	}
	domain, items, err := prepareTools(request.Tools, request.ToolChoice)
	if err != nil {
		return chat.ToolRequest{}, err
	}
	if len(request.System) > 0 {
		parts, err := textParts(request.System)
		if err != nil {
			return chat.ToolRequest{}, fmt.Errorf("system: %w", err)
		}
		items = append(items, chat.ContextItem{Type: chat.ContextItemMessage, Role: "system", Content: parts})
	}
	history, lastHasResults, err := contextItems(request.Messages)
	if err != nil {
		return chat.ToolRequest{}, err
	}
	items = append(items, history...)
	state, err := chat.NewContextState().AbsorbAll(items)
	if err != nil {
		return chat.ToolRequest{}, err
	}
	domain.Request = chat.Request{MaxCompletionTokens: *request.MaxTokens, ContextState: &state}
	// Auto completes a client tool round-trip with text instead of an unbounded loop.
	if lastHasResults && domain.Choice == chat.ToolChoiceAuto {
		domain.Choice = chat.ToolChoiceNone
	}
	return domain, nil
}

func prepareTools(tools []toolDefinition, selection *toolChoice) (chat.ToolRequest, []chat.ContextItem, error) {
	choice := toolChoice{Type: "auto"}
	if selection != nil {
		choice = *selection
	}
	domain := chat.ToolRequest{ParallelToolCalls: !choice.DisableParallelUse}
	switch choice.Type {
	case "auto":
		domain.Choice = chat.ToolChoiceAuto
	case "none":
		domain.Choice = chat.ToolChoiceNone
	case "any", "tool":
		domain.Choice = chat.ToolChoiceRequired
	default:
		return domain, nil, fmt.Errorf("tool_choice.type must be auto, none, any, or tool")
	}
	if (choice.Type == "tool") != (choice.Name != "") {
		return domain, nil, fmt.Errorf("tool_choice.name is required only for type tool")
	}
	items := make([]chat.ContextItem, 0, len(tools)+1)
	seen := make(map[string]bool, len(tools))
	for _, tool := range tools {
		if !validToolName(tool.Name) || seen[tool.Name] {
			return domain, nil, fmt.Errorf("tool names must be unique, 1-64 characters, and contain only letters, digits, underscores, or hyphens")
		}
		seen[tool.Name] = true
		if tool.Type != "" && tool.Type != "custom" {
			return domain, nil, fmt.Errorf("only custom client tools are supported")
		}
		if tool.Strict || tool.InputSchema == nil || tool.InputSchema["type"] != "object" {
			return domain, nil, fmt.Errorf("tools require an object input_schema; strict schema enforcement is unsupported")
		}
		encoded, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return domain, nil, fmt.Errorf("encode tool schema: %w", err)
		}
		items = append(items, chat.ContextItem{Type: chat.ContextItemFunctionTool, Name: tool.Name, Content: []chat.ContextContent{
			{Type: "description", Text: tool.Description}, {Type: "parameters", Text: string(encoded)},
		}})
		if choice.Type != "tool" || choice.Name == tool.Name {
			domain.Tools = append(domain.Tools, chat.FunctionTool{Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema})
		}
	}
	if domain.Choice == chat.ToolChoiceRequired && len(domain.Tools) == 0 {
		return domain, nil, fmt.Errorf("tool_choice requires a matching tool definition")
	}
	encoded, err := json.Marshal(choice)
	if err != nil {
		return domain, nil, err
	}
	items = append(items, chat.ContextItem{Type: chat.ContextItemToolConfiguration, Content: []chat.ContextContent{{Type: "selection", Text: string(encoded)}}})
	return domain, items, nil
}

func validToolName(name string) bool {
	if name == "" || len(name) > 64 {
		return false
	}
	for _, char := range name {
		switch {
		case char >= 'a' && char <= 'z', char >= 'A' && char <= 'Z', char >= '0' && char <= '9', char == '_', char == '-':
		default:
			return false
		}
	}
	return true
}
