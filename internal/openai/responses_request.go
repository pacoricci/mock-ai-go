package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

const minResponseMaxOutputTokens = 16

type responseRequest struct {
	Text               json.RawMessage          `json:"text"`
	Model              string                   `json:"model"`
	Input              json.RawMessage          `json:"input"`
	PreviousResponseID *string                  `json:"previous_response_id"`
	Instructions       *string                  `json:"instructions"`
	MaxOutputTokens    *int                     `json:"max_output_tokens"`
	Stream             bool                     `json:"stream"`
	Metadata           map[string]string        `json:"metadata"`
	Store              *bool                    `json:"store"`
	Tools              []functionToolDefinition `json:"tools"`
	ToolChoice         *string                  `json:"tool_choice"`
	ParallelToolCalls  *bool                    `json:"parallel_tool_calls"`
	MaxToolCalls       *int                     `json:"max_tool_calls"`
}

type responseInputItem struct {
	Type             string                `json:"type"`
	Role             string                `json:"role"`
	CallID           string                `json:"call_id"`
	Name             string                `json:"name"`
	Arguments        *string               `json:"arguments"`
	Content          json.RawMessage       `json:"content"`
	Output           json.RawMessage       `json:"output"`
	Summary          []responseContentPart `json:"summary"`
	EncryptedContent *string               `json:"encrypted_content"`
}

type responseContentPart struct {
	Type    string  `json:"type"`
	Text    *string `json:"text"`
	Refusal *string `json:"refusal"`
}

func toResponseDomainRequest(
	request responseRequest,
) (chat.Request, []chat.ContextItem, functionToolConfig, *responseRequestError) {
	if strings.TrimSpace(request.Model) == "" {
		return chat.Request{}, nil, functionToolConfig{}, newResponseRequestError("Missing required parameter: model", "model", "")
	}
	if request.MaxOutputTokens != nil && *request.MaxOutputTokens < minResponseMaxOutputTokens {
		return chat.Request{}, nil, functionToolConfig{}, newResponseRequestError(
			fmt.Sprintf("max_output_tokens must be at least %d", minResponseMaxOutputTokens),
			"max_output_tokens",
			"invalid_value",
		)
	}
	toolConfig, requestError := normalizeFunctionTools(request.Tools, request.ToolChoice, request.ParallelToolCalls, request.MaxToolCalls)
	if requestError != nil {
		return chat.Request{}, nil, functionToolConfig{}, requestError
	}

	items, requestError := decodeResponseInput(request.Input)
	if requestError != nil {
		return chat.Request{}, nil, functionToolConfig{}, requestError
	}

	// Generation and usage read the conversation from ContextState, which the
	// caller derives from these items, so Messages stays empty.
	var domainRequest chat.Request
	if request.MaxOutputTokens != nil {
		domainRequest.MaxCompletionTokens = *request.MaxOutputTokens
	}
	return domainRequest, items, toolConfig, nil
}

func decodeResponseInput(raw json.RawMessage) ([]chat.ContextItem, *responseRequestError) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, newResponseRequestError("Missing required parameter: input", "input", "")
	}

	switch trimmed[0] {
	case '"':
		var input string
		if err := json.Unmarshal(trimmed, &input); err != nil {
			return nil, newResponseRequestError("input must be a string or an array", "input", "invalid_value")
		}
		return []chat.ContextItem{{
			Type: chat.ContextItemMessage,
			Role: "user",
			Content: []chat.ContextContent{{
				Type: "text",
				Text: input,
			}},
		}}, nil
	case '[':
		var items []responseInputItem
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return nil, newResponseRequestError("input must contain valid response items", "input", "invalid_value")
		}
		if len(items) == 0 {
			return nil, newResponseRequestError("input must not be empty", "input", "invalid_value")
		}

		input := make([]chat.ContextItem, 0, len(items))
		for index, item := range items {
			itemType := item.Type
			if itemType == "" {
				itemType = "message"
			}
			switch itemType {
			case "message":
				if !isResponseMessageRole(item.Role) {
					param := fmt.Sprintf("input.%d.role", index)
					return nil, newResponseRequestError("Message role is not supported", param, "invalid_value")
				}
				content, requestError := decodeResponseMessageContent(item.Content, index)
				if requestError != nil {
					return nil, requestError
				}
				input = append(input, chat.ContextItem{
					Type: chat.ContextItemMessage,
					Role: item.Role,
					Content: []chat.ContextContent{{
						Type: "text",
						Text: content,
					}},
				})
			case "function_call":
				if strings.TrimSpace(item.CallID) == "" {
					return nil, newResponseRequestError("Function call call_id is required", fmt.Sprintf("input.%d.call_id", index), "invalid_value")
				}
				if strings.TrimSpace(item.Name) == "" {
					return nil, newResponseRequestError("Function call name is required", fmt.Sprintf("input.%d.name", index), "invalid_value")
				}
				if item.Arguments == nil {
					return nil, newResponseRequestError("Function call arguments are required", fmt.Sprintf("input.%d.arguments", index), "invalid_value")
				}
				input = append(input, chat.ContextItem{
					Type:   chat.ContextItemFunctionCall,
					CallID: item.CallID,
					Name:   item.Name,
					Content: []chat.ContextContent{{
						Type: "arguments",
						Text: *item.Arguments,
					}},
				})
			case "function_call_output":
				if strings.TrimSpace(item.CallID) == "" {
					return nil, newResponseRequestError("Function call output call_id is required", fmt.Sprintf("input.%d.call_id", index), "invalid_value")
				}
				output, requestError := decodeResponseFunctionOutput(item.Output, index)
				if requestError != nil {
					return nil, requestError
				}
				input = append(input, chat.ContextItem{
					Type:   chat.ContextItemFunctionCallOutput,
					CallID: item.CallID,
					Content: []chat.ContextContent{{
						Type: "output",
						Text: output,
					}},
				})
			case "reasoning":
				content := make([]chat.ContextContent, 0, len(item.Summary)+1)
				for partIndex, part := range item.Summary {
					if part.Type != "summary_text" || part.Text == nil {
						return nil, newResponseRequestError(
							"Reasoning summary must contain summary_text parts",
							fmt.Sprintf("input.%d.summary.%d", index, partIndex),
							"invalid_value",
						)
					}
					content = append(content, chat.ContextContent{Type: "summary_text", Text: *part.Text})
				}
				if item.EncryptedContent != nil {
					content = append(content, chat.ContextContent{Type: "encrypted_content", Text: *item.EncryptedContent})
				}
				input = append(input, chat.ContextItem{Type: chat.ContextItemReasoning, Content: content})
			default:
				return nil, newResponseRequestError(
					"Input item type is not supported by this mock",
					fmt.Sprintf("input.%d.type", index),
					"unsupported_value",
				)
			}
		}
		return input, nil
	default:
		return nil, newResponseRequestError("input must be a string or an array", "input", "invalid_value")
	}
}

func decodeResponseFunctionOutput(raw json.RawMessage, itemIndex int) (string, *responseRequestError) {
	trimmed := bytes.TrimSpace(raw)
	param := fmt.Sprintf("input.%d.output", itemIndex)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", newResponseRequestError("Function call output is required", param, "invalid_value")
	}
	if trimmed[0] == '"' {
		var output string
		if err := json.Unmarshal(trimmed, &output); err != nil {
			return "", newResponseRequestError("Function call output must be a string or an array", param, "invalid_value")
		}
		return output, nil
	}
	if trimmed[0] != '[' {
		return "", newResponseRequestError("Function call output must be a string or an array", param, "invalid_value")
	}
	var parts []responseContentPart
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return "", newResponseRequestError("Function call output must contain valid content parts", param, "invalid_value")
	}
	var output strings.Builder
	for partIndex, part := range parts {
		if part.Type != "input_text" || part.Text == nil {
			return "", newResponseRequestError(
				"Only input_text function call output content is supported",
				fmt.Sprintf("input.%d.output.%d", itemIndex, partIndex),
				"unsupported_value",
			)
		}
		output.WriteString(*part.Text)
	}
	return output.String(), nil
}

func decodeResponseMessageContent(raw json.RawMessage, itemIndex int) (string, *responseRequestError) {
	trimmed := bytes.TrimSpace(raw)
	param := fmt.Sprintf("input.%d.content", itemIndex)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", newResponseRequestError("Message content is required", param, "invalid_value")
	}

	if trimmed[0] == '"' {
		var content string
		if err := json.Unmarshal(trimmed, &content); err != nil {
			return "", newResponseRequestError("Message content must be a string or an array", param, "invalid_value")
		}
		return content, nil
	}
	if trimmed[0] != '[' {
		return "", newResponseRequestError("Message content must be a string or an array", param, "invalid_value")
	}

	var parts []responseContentPart
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return "", newResponseRequestError("Message content must contain valid content parts", param, "invalid_value")
	}
	var content strings.Builder
	for partIndex, part := range parts {
		switch part.Type {
		case "input_text", "output_text", "text", "summary_text", "reasoning_text":
			if part.Text == nil {
				partParam := fmt.Sprintf("input.%d.content.%d.text", itemIndex, partIndex)
				return "", newResponseRequestError("Text content is required", partParam, "invalid_value")
			}
			content.WriteString(*part.Text)
		case "refusal":
			if part.Refusal == nil {
				partParam := fmt.Sprintf("input.%d.content.%d.refusal", itemIndex, partIndex)
				return "", newResponseRequestError("Refusal content is required", partParam, "invalid_value")
			}
			content.WriteString(*part.Refusal)
		}
	}
	return content.String(), nil
}

func isResponseMessageRole(role string) bool {
	switch role {
	case "user", "system", "developer", "assistant":
		return true
	default:
		return false
	}
}
