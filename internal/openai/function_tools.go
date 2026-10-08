package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type functionToolDefinition struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description *string         `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict"`
}

type functionToolConfig struct {
	DomainTools       []chat.FunctionTool
	ResourceTools     []responseFunctionToolResource
	Choice            chat.ToolChoice
	ParallelToolCalls bool
	MaxToolCalls      *int
}

func normalizeFunctionTools(tools []functionToolDefinition, toolChoice *string, parallelToolCalls *bool, maxToolCalls *int) (functionToolConfig, *responseRequestError) {
	choice := chat.ToolChoiceAuto
	if toolChoice != nil {
		choice = chat.ToolChoice(*toolChoice)
	}
	if choice != chat.ToolChoiceNone && choice != chat.ToolChoiceAuto && choice != chat.ToolChoiceRequired {
		return functionToolConfig{}, newResponseRequestError(
			"tool_choice must be one of none, auto, or required",
			"tool_choice",
			"invalid_value",
		)
	}
	if maxToolCalls != nil && *maxToolCalls < 1 {
		return functionToolConfig{}, newResponseRequestError(
			"max_tool_calls must be at least 1",
			"max_tool_calls",
			"invalid_value",
		)
	}
	if choice == chat.ToolChoiceRequired && len(tools) == 0 {
		return functionToolConfig{}, newResponseRequestError(
			"tool_choice required needs at least one function tool",
			"tool_choice",
			"invalid_value",
		)
	}

	parallel := true
	if parallelToolCalls != nil {
		parallel = *parallelToolCalls
	}
	config := functionToolConfig{
		DomainTools:       make([]chat.FunctionTool, 0, len(tools)),
		ResourceTools:     make([]responseFunctionToolResource, 0, len(tools)),
		Choice:            choice,
		ParallelToolCalls: parallel,
		MaxToolCalls:      maxToolCalls,
	}
	seenNames := make(map[string]struct{}, len(tools))
	for index, tool := range tools {
		if tool.Type != "function" {
			return functionToolConfig{}, newResponseRequestError(
				"Only function tools are supported",
				fmt.Sprintf("tools.%d.type", index),
				"unsupported_value",
			)
		}
		if strings.TrimSpace(tool.Name) == "" {
			return functionToolConfig{}, newResponseRequestError(
				"Function tool name must not be empty",
				fmt.Sprintf("tools.%d.name", index),
				"invalid_value",
			)
		}
		if _, exists := seenNames[tool.Name]; exists {
			return functionToolConfig{}, newResponseRequestError(
				"Function tool names must be unique",
				fmt.Sprintf("tools.%d.name", index),
				"invalid_value",
			)
		}
		seenNames[tool.Name] = struct{}{}

		parameters, requestError := decodeFunctionToolParameters(tool.Parameters, index)
		if requestError != nil {
			return functionToolConfig{}, requestError
		}
		description := ""
		if tool.Description != nil {
			description = *tool.Description
		}
		config.DomainTools = append(config.DomainTools, chat.FunctionTool{
			Name:        tool.Name,
			Description: description,
			Parameters:  parameters,
		})
		config.ResourceTools = append(config.ResourceTools, responseFunctionToolResource{
			Type:        "function",
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  parameters,
			Strict:      false,
		})
	}
	return config, nil
}

func decodeFunctionToolParameters(raw json.RawMessage, toolIndex int) (map[string]any, *responseRequestError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	if trimmed[0] != '{' {
		return nil, newResponseRequestError(
			"Function tool parameters must be a JSON object or null",
			fmt.Sprintf("tools.%d.parameters", toolIndex),
			"invalid_value",
		)
	}
	var parameters map[string]any
	if err := json.Unmarshal(trimmed, &parameters); err != nil {
		return nil, newResponseRequestError(
			"Function tool parameters must be a valid JSON Schema object",
			fmt.Sprintf("tools.%d.parameters", toolIndex),
			"invalid_value",
		)
	}
	return parameters, nil
}

func functionToolContextItems(toolConfig functionToolConfig) ([]chat.ContextItem, error) {
	items := make([]chat.ContextItem, 0, len(toolConfig.ResourceTools)+1)
	for _, tool := range toolConfig.ResourceTools {
		parameters, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, fmt.Errorf("canonicalize tool %q parameters: %w", tool.Name, err)
		}
		description := ""
		if tool.Description != nil {
			description = *tool.Description
		}
		items = append(items, chat.ContextItem{
			Type: chat.ContextItemFunctionTool,
			Name: tool.Name,
			Content: []chat.ContextContent{
				{Type: "description", Text: description},
				{Type: "parameters", Text: string(parameters)},
				{Type: "strict", Text: strconv.FormatBool(tool.Strict)},
			},
		})
	}
	maxToolCalls := "unbounded"
	if toolConfig.MaxToolCalls != nil {
		maxToolCalls = strconv.Itoa(*toolConfig.MaxToolCalls)
	}
	items = append(items, chat.ContextItem{
		Type: chat.ContextItemToolConfiguration,
		Content: []chat.ContextContent{
			{Type: "tool_choice", Text: string(toolConfig.Choice)},
			{Type: "parallel_tool_calls", Text: strconv.FormatBool(toolConfig.ParallelToolCalls)},
			{Type: "max_tool_calls", Text: maxToolCalls},
		},
	})
	return items, nil
}

func validateFunctionOutputs(previousCalls map[string]struct{}, items []chat.ContextItem, param string) *responseRequestError {
	availableCalls := make(map[string]struct{}, len(previousCalls))
	for id := range previousCalls {
		availableCalls[id] = struct{}{}
	}
	seenOutputs := make(map[string]struct{})
	for _, item := range items {
		switch item.Type {
		case chat.ContextItemFunctionCall:
			if _, exists := availableCalls[item.CallID]; exists {
				return newResponseRequestError(
					fmt.Sprintf("Function call %q is duplicated", item.CallID),
					param,
					"invalid_value",
				)
			}
			availableCalls[item.CallID] = struct{}{}
		case chat.ContextItemFunctionCallOutput:
			if _, duplicate := seenOutputs[item.CallID]; duplicate {
				return newResponseRequestError(
					fmt.Sprintf("Function call output %q is duplicated", item.CallID),
					param,
					"invalid_value",
				)
			}
			if _, exists := availableCalls[item.CallID]; !exists {
				return newResponseRequestError(
					fmt.Sprintf("Function call output %q has no matching function call", item.CallID),
					param,
					"invalid_value",
				)
			}
			seenOutputs[item.CallID] = struct{}{}
		}
	}
	return nil
}
