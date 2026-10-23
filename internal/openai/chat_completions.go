package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type chatCompletionRequest struct {
	ResponseFormat      json.RawMessage              `json:"response_format"`
	Model               string                       `json:"model"`
	Messages            []chatCompletionMessage      `json:"messages"`
	MaxCompletionTokens *int                         `json:"max_completion_tokens"`
	MaxTokens           *int                         `json:"max_tokens"`
	Seed                *int64                       `json:"seed"`
	Stream              bool                         `json:"stream"`
	StreamOptions       *chatCompletionStreamOptions `json:"stream_options"`
	Tools               []chatFunctionTool           `json:"tools"`
	ToolChoice          json.RawMessage              `json:"tool_choice"`
	ParallelToolCalls   *bool                        `json:"parallel_tool_calls"`
	Functions           json.RawMessage              `json:"functions"`
	FunctionCall        json.RawMessage              `json:"function_call"`
}

type chatCompletionStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatCompletionMessage struct {
	Role       string         `json:"role"`
	Content    *string        `json:"content"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   chatCompletionUsage    `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      chatCompletionMessage `json:"message"`
	Logprobs     any                   `json:"logprobs"`
	FinishReason string                `json:"finish_reason"`
}

type chatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (h *Handler) createChatCompletion(w http.ResponseWriter, r *http.Request) {
	request, ok := decodeJSONRequest[chatCompletionRequest](w, r)
	if !ok {
		return
	}
	if !validateChatCompletionRequest(w, request) {
		return
	}

	chatModel, found := h.resolveChatModel(w, request.Model)
	if !found {
		return
	}

	format, formatError := normalizeChatFormat(request.ResponseFormat)
	if formatError != nil {
		writeResponseRequestError(w, formatError)
		return
	}
	domainRequest := toDomainRequest(request)
	toolRequest, requestError, err := prepareChatTools(request, domainRequest)
	if requestError != nil {
		writeResponseRequestError(w, requestError)
		return
	}
	if err != nil {
		writeChatModelError(w, err)
		return
	}
	plan, err := chatModel.PlanTools(r.Context(), toolRequest)
	if err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			writeChatModelError(w, err)
		}
		return
	}
	domainRequest = toolRequest.Request
	if !plan.UseText {
		if request.Stream {
			h.streamChatTools(w, r, request, domainRequest, chatModel, plan)
		} else {
			h.completeChatTools(w, request, plan)
		}
		return
	}
	generation, requestError, err := prepareTextGeneration(r.Context(), chatModel, domainRequest, format)
	if requestError != nil {
		writeResponseRequestError(w, requestError)
		return
	}
	if err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			writeChatModelError(w, err)
		}
		return
	}
	if request.Stream {
		h.streamChatCompletion(w, r, request, domainRequest, generation)
		return
	}
	h.completeChatCompletion(w, r, request, domainRequest, generation)
}

func (h *Handler) completeChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	request chatCompletionRequest,
	domainRequest chat.Request,
	chatModel textGeneration,
) {
	completion, err := chatModel.Complete(r.Context(), domainRequest)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		writeChatModelError(w, err)
		return
	}

	response := chatCompletionResponse{
		ID:      h.chatCompletionID(h.sequence.Add(1)),
		Object:  "chat.completion",
		Created: h.now().Unix(),
		Model:   request.Model,
		Choices: []chatCompletionChoice{{
			Index: 0,
			Message: chatCompletionMessage{
				Role:    "assistant",
				Content: stringPointer(completion.Content),
			},
			Logprobs:     nil,
			FinishReason: chatModel.finishReason(),
		}},
		Usage: toAPIUsage(completion.Usage),
	}

	writeJSON(w, response)
}

func validateChatCompletionRequest(w http.ResponseWriter, request chatCompletionRequest) bool {
	if strings.TrimSpace(request.Model) == "" {
		writeError(w, http.StatusBadRequest, "Missing required parameter: model", "invalid_request_error", stringPointer("model"), nil)
		return false
	}
	if len(request.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "Missing required parameter: messages", "invalid_request_error", stringPointer("messages"), nil)
		return false
	}
	for index, message := range request.Messages {
		if strings.TrimSpace(message.Role) == "" {
			param := fmt.Sprintf("messages.%d.role", index)
			writeError(w, http.StatusBadRequest, "Message role must not be empty", "invalid_request_error", &param, nil)
			return false
		}
	}
	if request.MaxCompletionTokens != nil && *request.MaxCompletionTokens <= 0 {
		writeError(w, http.StatusBadRequest, "max_completion_tokens must be positive", "invalid_request_error", stringPointer("max_completion_tokens"), stringPointer("invalid_value"))
		return false
	}
	if request.MaxTokens != nil && *request.MaxTokens <= 0 {
		writeError(w, http.StatusBadRequest, "max_tokens must be positive", "invalid_request_error", stringPointer("max_tokens"), stringPointer("invalid_value"))
		return false
	}
	if request.MaxCompletionTokens != nil && request.MaxTokens != nil {
		writeError(w, http.StatusBadRequest, "max_completion_tokens and max_tokens cannot both be set", "invalid_request_error", stringPointer("max_completion_tokens"), stringPointer("invalid_value"))
		return false
	}
	if request.StreamOptions != nil && !request.Stream {
		writeError(w, http.StatusBadRequest, "stream_options requires stream=true", "invalid_request_error", stringPointer("stream_options"), stringPointer("invalid_value"))
		return false
	}

	return true
}

func toDomainRequest(request chatCompletionRequest) chat.Request {
	domainRequest := chat.Request{
		Messages: make([]chat.Message, len(request.Messages)),
		Seed:     request.Seed,
	}
	if request.MaxCompletionTokens != nil {
		domainRequest.MaxCompletionTokens = *request.MaxCompletionTokens
	} else if request.MaxTokens != nil {
		domainRequest.MaxCompletionTokens = *request.MaxTokens
	}
	for index, message := range request.Messages {
		domainRequest.Messages[index] = chat.Message{
			Role:    message.Role,
			Content: chatMessageText(message),
		}
	}
	return domainRequest
}

func toAPIUsage(usage chat.Usage) chatCompletionUsage {
	return chatCompletionUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}
}

func writeChatModelError(w http.ResponseWriter, err error) {
	writeModelError(w, err, "max_completion_tokens", "The chat model failed to create a completion")
}
