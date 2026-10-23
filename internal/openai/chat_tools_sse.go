package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type chatToolDelta struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function chatFunctionDelta `json:"function"`
}

type chatFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

func (h *Handler) streamChatTools(w http.ResponseWriter, r *http.Request, request chatCompletionRequest, domain chat.Request, model chat.Model, plan chat.ToolPlan) {
	sequence := h.sequence.Add(1)
	domain.StreamNonce = sequence
	includeUsage := request.StreamOptions != nil && request.StreamOptions.IncludeUsage
	encoder := newSSEEncoder(w, http.NewResponseController(w), streamMetadata{ID: h.chatCompletionID(sequence), Created: h.now().Unix(), Model: request.Model}, includeUsage)
	started := false
	err := model.StreamTools(r.Context(), domain, plan, func(chunk chat.ToolChunk) error {
		if !started {
			started = true
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			if err := encoder.writeToolInitial(); err != nil {
				return err
			}
		}
		return encoder.writeToolDelta(plan.Calls[chunk.Index], chunk)
	})
	if err != nil {
		if !started && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			writeChatModelError(w, err)
		}
		return
	}
	if r.Context().Err() != nil {
		return
	}
	if err := encoder.writeToolFinish(); err != nil {
		return
	}
	if includeUsage {
		if err := encoder.writeUsage(toAPIUsage(plan.Usage)); err != nil {
			return
		}
	}
	_ = encoder.writeDone()
}

func (e *sseEncoder) writeToolInitial() error {
	e.beginEvent()
	e.buffer = append(e.buffer, `[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]`...)
	e.appendNullUsage()
	return e.endEvent()
}

func (e *sseEncoder) writeToolDelta(call chat.ToolCall, chunk chat.ToolChunk) error {
	delta := chatToolDelta{Index: chunk.Index, Function: chatFunctionDelta{Arguments: chunk.Delta}}
	if chunk.First {
		delta.ID = call.CallID
		delta.Type = "function"
		delta.Function.Name = call.Name
	}
	encoded, err := json.Marshal(delta)
	if err != nil {
		return err
	}
	e.beginEvent()
	e.buffer = append(e.buffer, `[{"index":0,"delta":{"tool_calls":[`...)
	e.buffer = append(e.buffer, encoded...)
	e.buffer = append(e.buffer, `]},"logprobs":null,"finish_reason":null}]`...)
	e.appendNullUsage()
	return e.endEvent()
}

func (e *sseEncoder) writeToolFinish() error {
	e.beginEvent()
	e.buffer = append(e.buffer, `[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]`...)
	e.appendNullUsage()
	return e.endEvent()
}
