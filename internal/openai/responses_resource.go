package openai

import (
	"errors"
	"fmt"
	"math"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type responseFunctionToolResource struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description *string        `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict"`
}

type responseResource struct {
	ID                 string                         `json:"id"`
	Object             string                         `json:"object"`
	CreatedAt          int64                          `json:"created_at"`
	CompletedAt        *int64                         `json:"completed_at"`
	Status             string                         `json:"status"`
	IncompleteDetails  any                            `json:"incomplete_details"`
	Model              string                         `json:"model"`
	PreviousResponseID *string                        `json:"previous_response_id"`
	Instructions       *string                        `json:"instructions"`
	Output             []responseOutputItem           `json:"output"`
	Error              any                            `json:"error"`
	Tools              []responseFunctionToolResource `json:"tools"`
	ToolChoice         string                         `json:"tool_choice"`
	Truncation         string                         `json:"truncation"`
	ParallelToolCalls  bool                           `json:"parallel_tool_calls"`
	Text               responseTextConfig             `json:"text"`
	TopP               float64                        `json:"top_p"`
	PresencePenalty    float64                        `json:"presence_penalty"`
	FrequencyPenalty   float64                        `json:"frequency_penalty"`
	TopLogprobs        int                            `json:"top_logprobs"`
	Temperature        float64                        `json:"temperature"`
	Reasoning          any                            `json:"reasoning"`
	Usage              *responseUsage                 `json:"usage"`
	MaxOutputTokens    *int                           `json:"max_output_tokens"`
	MaxToolCalls       *int                           `json:"max_tool_calls"`
	Store              bool                           `json:"store"`
	Background         bool                           `json:"background"`
	ServiceTier        string                         `json:"service_tier"`
	Metadata           map[string]string              `json:"metadata"`
	SafetyIdentifier   *string                        `json:"safety_identifier"`
	PromptCacheKey     *string                        `json:"prompt_cache_key"`
}

type responseOutputItem struct {
	ID        string                   `json:"id"`
	Type      string                   `json:"type"`
	Status    string                   `json:"status"`
	Role      *string                  `json:"role,omitempty"`
	Content   *[]responseOutputContent `json:"content,omitempty"`
	CallID    *string                  `json:"call_id,omitempty"`
	Name      *string                  `json:"name,omitempty"`
	Arguments *string                  `json:"arguments,omitempty"`
}

type responseOutputContent struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

type responseTextConfig struct {
	Format outputFormat `json:"format"`
}

type responseUsage struct {
	InputTokens         int                        `json:"input_tokens"`
	InputTokensDetails  responseInputTokenDetails  `json:"input_tokens_details"`
	OutputTokens        int                        `json:"output_tokens"`
	OutputTokensDetails responseOutputTokenDetails `json:"output_tokens_details"`
	TotalTokens         int                        `json:"total_tokens"`
}

type responseInputTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type responseOutputTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type responseGenerationMetadata struct {
	TextFormat         outputFormat
	ResponseID         string
	MessageID          string
	CreatedAt          int64
	Model              string
	PreviousResponseID *string
	Instructions       *string
	MaxOutputTokens    *int
	Store              bool
	Metadata           map[string]string
	Tools              []responseFunctionToolResource
	ToolChoice         string
	ParallelTools      bool
	MaxToolCalls       *int
}

func newResponseGenerationMetadata(
	h *Handler,
	request responseRequest,
	toolConfig functionToolConfig,
	responseID string,
	nonce uint64,
) responseGenerationMetadata {
	metadata := request.Metadata
	if metadata == nil {
		metadata = make(map[string]string)
	}
	store := true
	if request.Store != nil {
		store = *request.Store
	}

	return responseGenerationMetadata{
		TextFormat:         outputFormat{Type: "text"},
		ResponseID:         responseID,
		MessageID:          fmt.Sprintf("msg-mock-%d", nonce),
		CreatedAt:          h.now().Unix(),
		Model:              request.Model,
		PreviousResponseID: request.PreviousResponseID,
		Instructions:       request.Instructions,
		MaxOutputTokens:    request.MaxOutputTokens,
		Store:              store,
		Metadata:           metadata,
		Tools:              toolConfig.ResourceTools,
		ToolChoice:         string(toolConfig.Choice),
		ParallelTools:      toolConfig.ParallelToolCalls,
		MaxToolCalls:       toolConfig.MaxToolCalls,
	}
}

func (m responseGenerationMetadata) resource(status string) responseResource {
	return responseResource{
		ID:                 m.ResponseID,
		Object:             "response",
		CreatedAt:          m.CreatedAt,
		CompletedAt:        nil,
		Status:             status,
		IncompleteDetails:  nil,
		Model:              m.Model,
		PreviousResponseID: m.PreviousResponseID,
		Instructions:       m.Instructions,
		Output:             make([]responseOutputItem, 0),
		Error:              nil,
		Tools:              m.Tools,
		ToolChoice:         m.ToolChoice,
		Truncation:         "disabled",
		ParallelToolCalls:  m.ParallelTools,
		Text:               responseTextConfig{Format: m.TextFormat},
		TopP:               1,
		PresencePenalty:    0,
		FrequencyPenalty:   0,
		TopLogprobs:        0,
		Temperature:        1,
		Reasoning:          nil,
		Usage:              nil,
		MaxOutputTokens:    m.MaxOutputTokens,
		MaxToolCalls:       m.MaxToolCalls,
		Store:              m.Store,
		Background:         false,
		ServiceTier:        "default",
		Metadata:           m.Metadata,
		SafetyIdentifier:   nil,
		PromptCacheKey:     nil,
	}
}

func (m responseGenerationMetadata) outputItem(status, content string) responseOutputItem {
	parts := make([]responseOutputContent, 0)
	if status != "in_progress" {
		parts = append(parts, responseOutputContent{
			Type:        "output_text",
			Text:        content,
			Annotations: make([]any, 0),
		})
	}
	role := "assistant"
	return responseOutputItem{
		ID:      m.MessageID,
		Type:    "message",
		Status:  status,
		Role:    &role,
		Content: &parts,
	}
}

func responseToolOutputItem(call chat.ToolCall, status, arguments string) responseOutputItem {
	callID := call.CallID
	name := call.Name
	return responseOutputItem{
		ID:        call.ID,
		Type:      "function_call",
		Status:    status,
		CallID:    &callID,
		Name:      &name,
		Arguments: &arguments,
	}
}

func (m responseGenerationMetadata) textResource(content string, usage chat.Usage, completedAt int64, incomplete bool) responseResource {
	resource := m.resource("completed")
	resource.CompletedAt = &completedAt
	resource.Output = []responseOutputItem{m.outputItem("completed", content)}
	resource.Usage = toResponseUsage(usage)
	if incomplete {
		resource.Status = "incomplete"
		resource.CompletedAt = nil
		resource.IncompleteDetails = map[string]string{"reason": "max_output_tokens"}
		resource.Output[0].Status = "incomplete"
	}
	return resource
}

func (m responseGenerationMetadata) completedToolResource(
	calls []chat.ToolCall,
	usage chat.Usage,
	completedAt int64,
) responseResource {
	resource := m.resource("completed")
	resource.CompletedAt = &completedAt
	resource.Output = make([]responseOutputItem, 0, len(calls))
	for _, call := range calls {
		resource.Output = append(resource.Output, responseToolOutputItem(call, "completed", call.Arguments))
	}
	resource.Usage = toResponseUsage(usage)
	return resource
}

func toResponseUsage(usage chat.Usage) *responseUsage {
	return &responseUsage{
		InputTokens:        usage.PromptTokens,
		InputTokensDetails: responseInputTokenDetails{CachedTokens: 0},
		OutputTokens:       usage.CompletionTokens,
		OutputTokensDetails: responseOutputTokenDetails{
			ReasoningTokens: 0,
		},
		TotalTokens: usage.TotalTokens,
	}
}

func responseToolUsage(usage chat.Usage, contextState chat.ContextState) (chat.Usage, error) {
	if usage.CompletionTokens < 0 {
		return chat.Usage{}, errors.New("tool completion token count cannot be negative")
	}
	promptTokens, err := contextState.PromptTokens()
	if err != nil {
		return chat.Usage{}, err
	}
	if usage.CompletionTokens > math.MaxInt-promptTokens {
		return chat.Usage{}, fmt.Errorf("%w: tool usage total token count overflows int", chat.ErrInvalidContextState)
	}
	usage.PromptTokens = promptTokens
	usage.TotalTokens = promptTokens + usage.CompletionTokens
	return usage, nil
}
