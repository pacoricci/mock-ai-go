package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type chatCompletionStreamChunk struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []chatCompletionStreamChoice `json:"choices"`
}

type chatCompletionStreamChunkWithUsage struct {
	chatCompletionStreamChunk
	Usage *chatCompletionUsage `json:"usage"`
}

type chatCompletionStreamChoice struct {
	Index        int                       `json:"index"`
	Delta        chatCompletionStreamDelta `json:"delta"`
	Logprobs     any                       `json:"logprobs"`
	FinishReason *string                   `json:"finish_reason"`
}

type chatCompletionStreamDelta struct {
	Role      *string         `json:"role,omitempty"`
	Content   *string         `json:"content,omitempty"`
	ToolCalls []chatToolDelta `json:"tool_calls,omitempty"`
}

type streamMetadata struct {
	ID      string
	Created int64
	Model   string
}

type sseEncoder struct {
	*sseWriter
	metadata     streamMetadata
	includeUsage bool
}

func (h *Handler) streamChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	request chatCompletionRequest,
	domainRequest chat.Request,
	chatModel textGeneration,
) {
	streamNonce := h.sequence.Add(1)
	domainRequest.StreamNonce = streamNonce
	metadata := streamMetadata{
		ID:      fmt.Sprintf("chatcmpl-mock-%d", streamNonce),
		Created: h.now().Unix(),
		Model:   request.Model,
	}
	includeUsage := request.StreamOptions != nil && request.StreamOptions.IncludeUsage
	encoder := newSSEEncoder(w, http.NewResponseController(w), metadata, includeUsage)
	started := false

	start := func() error {
		if started {
			return nil
		}
		started = true
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		return encoder.writeInitial()
	}

	usage, err := chatModel.Stream(r.Context(), domainRequest, func(chunk chat.Chunk) error {
		if err := start(); err != nil {
			return err
		}
		return encoder.writeContent(chunk.Delta)
	})
	if err != nil {
		if !started && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			writeChatModelError(w, err)
		}
		return
	}
	if err := start(); err != nil {
		return
	}
	if err := encoder.writeFinishReason(chatModel.finishReason()); err != nil {
		return
	}
	if includeUsage {
		if err := encoder.writeUsage(toAPIUsage(usage)); err != nil {
			return
		}
	}
	_ = encoder.writeDone()
}

func newSSEEncoder(
	w io.Writer,
	controller *http.ResponseController,
	metadata streamMetadata,
	includeUsage bool,
) *sseEncoder {
	return &sseEncoder{
		sseWriter:    newSSEWriter(w, controller),
		metadata:     metadata,
		includeUsage: includeUsage,
	}
}

func (e *sseEncoder) writeInitial() error {
	e.beginEvent()
	e.buffer = append(e.buffer, `[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]`...)
	e.appendNullUsage()
	return e.endEvent()
}

func (e *sseEncoder) writeContent(content []byte) error {
	e.beginEvent()
	e.buffer = append(e.buffer, `[{"index":0,"delta":{"content":`...)
	e.buffer = appendJSONStringBytes(e.buffer, content)
	e.buffer = append(e.buffer, `},"logprobs":null,"finish_reason":null}]`...)
	e.appendNullUsage()
	return e.endEvent()
}

func (e *sseEncoder) writeFinish() error {
	return e.writeFinishReason("stop")
}

func (e *sseEncoder) writeFinishReason(reason string) error {
	e.beginEvent()
	e.buffer = append(e.buffer, `[{"index":0,"delta":{},"logprobs":null,"finish_reason":`...)
	e.buffer = strconv.AppendQuote(e.buffer, reason)
	e.buffer = append(e.buffer, '}', ']')
	e.appendNullUsage()
	return e.endEvent()
}

func (e *sseEncoder) writeUsage(usage chatCompletionUsage) error {
	e.beginEvent()
	e.buffer = append(e.buffer, `[],"usage":{"prompt_tokens":`...)
	e.buffer = strconv.AppendInt(e.buffer, int64(usage.PromptTokens), 10)
	e.buffer = append(e.buffer, `,"completion_tokens":`...)
	e.buffer = strconv.AppendInt(e.buffer, int64(usage.CompletionTokens), 10)
	e.buffer = append(e.buffer, `,"total_tokens":`...)
	e.buffer = strconv.AppendInt(e.buffer, int64(usage.TotalTokens), 10)
	e.buffer = append(e.buffer, '}')
	return e.endEvent()
}

func (e *sseEncoder) writeDone() error {
	return e.sseWriter.writeDone()
}

func (e *sseEncoder) beginEvent() {
	e.buffer = append(e.buffer[:0], `data: {"id":`...)
	e.buffer = strconv.AppendQuote(e.buffer, e.metadata.ID)
	e.buffer = append(e.buffer, `,"object":"chat.completion.chunk","created":`...)
	e.buffer = strconv.AppendInt(e.buffer, e.metadata.Created, 10)
	e.buffer = append(e.buffer, `,"model":`...)
	e.buffer = strconv.AppendQuote(e.buffer, e.metadata.Model)
	e.buffer = append(e.buffer, `,"choices":`...)
}

func (e *sseEncoder) appendNullUsage() {
	if e.includeUsage {
		e.buffer = append(e.buffer, `,"usage":null`...)
	}
}

func (e *sseEncoder) endEvent() error {
	e.buffer = append(e.buffer, '}', '\n', '\n')
	return e.writeAndFlush()
}
