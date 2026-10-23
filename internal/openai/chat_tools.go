package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type chatFunctionTool struct {
	Type     string                 `json:"type"`
	Function functionToolDefinition `json:"function"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func chatMessageText(message chatCompletionMessage) string {
	if message.Content == nil {
		return ""
	}
	return *message.Content
}

func normalizeChatTools(request chatCompletionRequest) (functionToolConfig, *responseRequestError) {
	for _, field := range []struct {
		name string
		raw  json.RawMessage
	}{{"functions", request.Functions}, {"function_call", request.FunctionCall}} {
		if len(field.raw) > 0 {
			return functionToolConfig{}, newResponseRequestError("Legacy function calling is unsupported; use tools and tool_choice", field.name, "unsupported_value")
		}
	}
	choice := "auto"
	forced := ""
	raw := bytes.TrimSpace(request.ToolChoice)
	if len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		if raw[0] == '"' {
			if err := json.Unmarshal(raw, &choice); err != nil {
				return functionToolConfig{}, newResponseRequestError("Invalid tool_choice", "tool_choice", "invalid_value")
			}
		} else {
			var named struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if err := json.Unmarshal(raw, &named); err != nil || named.Type != "function" || strings.TrimSpace(named.Function.Name) == "" {
				return functionToolConfig{}, newResponseRequestError("tool_choice must be none, auto, required, or a named function", "tool_choice", "invalid_value")
			}
			choice = "required"
			forced = named.Function.Name
		}
	}
	tools := make([]functionToolDefinition, len(request.Tools))
	for index, tool := range request.Tools {
		tools[index] = tool.Function
		tools[index].Type = tool.Type
	}
	config, requestError := normalizeFunctionTools(tools, &choice, request.ParallelToolCalls, nil)
	if requestError != nil {
		// Chat nests function definitions under tools[index].function.
		parts := strings.Split(requestError.param, ".")
		if len(parts) == 3 && parts[0] == "tools" && parts[2] != "type" {
			requestError.param = strings.Join(parts[:2], ".") + ".function." + parts[2]
		}
		return functionToolConfig{}, requestError
	}
	if forced != "" {
		for _, tool := range config.DomainTools {
			if tool.Name == forced {
				config.DomainTools = []chat.FunctionTool{tool}
				return config, nil
			}
		}
		return functionToolConfig{}, newResponseRequestError("The selected function must be present in tools", "tool_choice", "invalid_value")
	}
	return config, nil
}

func prepareChatTools(request chatCompletionRequest, domain chat.Request) (chat.ToolRequest, *responseRequestError, error) {
	config, requestError := normalizeChatTools(request)
	if requestError != nil {
		return chat.ToolRequest{}, requestError, nil
	}
	hasToolHistory := false
	for _, message := range request.Messages {
		if message.Role == "tool" || len(message.ToolCalls) > 0 || message.ToolCallID != "" {
			hasToolHistory = true
			break
		}
	}
	if len(request.Tools) == 0 && !hasToolHistory {
		return chat.ToolRequest{Request: domain, Choice: config.Choice, ParallelToolCalls: config.ParallelToolCalls}, nil, nil
	}
	items, requestError := chatContextItems(request.Messages)
	if requestError != nil {
		return chat.ToolRequest{}, requestError, nil
	}
	toolItems, err := functionToolContextItems(config)
	if err != nil {
		return chat.ToolRequest{}, nil, err
	}
	state, err := chat.NewContextState().AbsorbAll(toolItems)
	if err != nil {
		return chat.ToolRequest{}, nil, err
	}
	state, err = state.AbsorbAll(items)
	if err != nil {
		return chat.ToolRequest{}, nil, err
	}
	domain.ContextState = &state
	return chat.ToolRequest{Request: domain, Tools: config.DomainTools, Choice: config.Choice, ParallelToolCalls: config.ParallelToolCalls}, nil, nil
}

func chatContextItems(messages []chatCompletionMessage) ([]chat.ContextItem, *responseRequestError) {
	items := make([]chat.ContextItem, 0, len(messages))
	for index, message := range messages {
		param := fmt.Sprintf("messages.%d", index)
		if len(message.ToolCalls) > 0 && message.Role != "assistant" {
			return nil, newResponseRequestError("Only assistant messages may contain tool_calls", param+".tool_calls", "invalid_value")
		}
		if message.Role == "tool" {
			if strings.TrimSpace(message.ToolCallID) == "" || message.Content == nil {
				return nil, newResponseRequestError("Tool messages require tool_call_id and string content", param, "invalid_value")
			}
			items = append(items, chat.ContextItem{Type: chat.ContextItemFunctionCallOutput, CallID: message.ToolCallID, Content: []chat.ContextContent{{Type: "output", Text: *message.Content}}})
			continue
		}
		if message.ToolCallID != "" {
			return nil, newResponseRequestError("tool_call_id requires role tool", param+".tool_call_id", "invalid_value")
		}
		if message.Content != nil || len(message.ToolCalls) == 0 {
			items = append(items, chat.ContextItem{Type: chat.ContextItemMessage, Role: message.Role, Content: []chat.ContextContent{{Type: "text", Text: chatMessageText(message)}}})
		}
		for callIndex, call := range message.ToolCalls {
			if call.Type != "function" || strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Function.Name) == "" || !json.Valid([]byte(call.Function.Arguments)) {
				return nil, newResponseRequestError("Function calls require an ID, name, and JSON arguments", fmt.Sprintf("%s.tool_calls.%d", param, callIndex), "invalid_value")
			}
			items = append(items, chat.ContextItem{Type: chat.ContextItemFunctionCall, CallID: call.ID, Name: call.Function.Name, Content: []chat.ContextContent{{Type: "arguments", Text: call.Function.Arguments}}})
		}
	}
	if requestError := validateFunctionOutputs(nil, items, "messages"); requestError != nil {
		return nil, requestError
	}
	return items, nil
}

func (h *Handler) completeChatTools(w http.ResponseWriter, request chatCompletionRequest, plan chat.ToolPlan) {
	calls := make([]chatToolCall, len(plan.Calls))
	for index, call := range plan.Calls {
		calls[index] = chatToolCall{ID: call.CallID, Type: "function", Function: chatFunctionCall{Name: call.Name, Arguments: call.Arguments}}
	}
	writeJSON(w, chatCompletionResponse{
		ID: h.chatCompletionID(h.sequence.Add(1)), Object: "chat.completion", Created: h.now().Unix(), Model: request.Model,
		Choices: []chatCompletionChoice{{Index: 0, Message: chatCompletionMessage{Role: "assistant", ToolCalls: calls}, FinishReason: "tool_calls"}}, Usage: toAPIUsage(plan.Usage),
	})
}
