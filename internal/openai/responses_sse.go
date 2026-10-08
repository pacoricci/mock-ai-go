package openai

import (
	"context"
	"errors"
	"net/http"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type responseLifecycleEvent struct {
	Type           string           `json:"type"`
	SequenceNumber int              `json:"sequence_number"`
	Response       responseResource `json:"response"`
}

type responseOutputItemEvent struct {
	Type           string             `json:"type"`
	SequenceNumber int                `json:"sequence_number"`
	OutputIndex    int                `json:"output_index"`
	Item           responseOutputItem `json:"item"`
}

type responseContentPartEvent struct {
	Type           string                `json:"type"`
	SequenceNumber int                   `json:"sequence_number"`
	ItemID         string                `json:"item_id"`
	OutputIndex    int                   `json:"output_index"`
	ContentIndex   int                   `json:"content_index"`
	Part           responseOutputContent `json:"part"`
}

type responseOutputTextDeltaEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	ContentIndex   int    `json:"content_index"`
	Delta          string `json:"delta"`
}

type responseOutputTextDoneEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	ContentIndex   int    `json:"content_index"`
	Text           string `json:"text"`
}

type responseFunctionCallArgumentsDeltaEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	Delta          string `json:"delta"`
}

type responseFunctionCallArgumentsDoneEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	Arguments      string `json:"arguments"`
}

type responseStreamingErrorEvent struct {
	Type           string                        `json:"type"`
	SequenceNumber int                           `json:"sequence_number"`
	Error          responseStreamingErrorPayload `json:"error"`
}

type responseStreamingErrorPayload struct {
	Type    string  `json:"type"`
	Code    *string `json:"code"`
	Message string  `json:"message"`
	Param   *string `json:"param"`
}

type responseFailure struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type responseSSEEncoder struct {
	*sseWriter
	metadata       responseGenerationMetadata
	sequenceNumber int
	content        []byte
}

func (h *Handler) streamResponse(
	w http.ResponseWriter,
	r *http.Request,
	domainRequest chat.Request,
	chatModel textGeneration,
	metadata responseGenerationMetadata,
	carryContext chat.ContextState,
) {
	encoder := newResponseSSEEncoder(w, http.NewResponseController(w), metadata)
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
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		if !started {
			writeResponseModelError(w, err)
			return
		}
		if encoder.writeFailure("server_error", "The chat model failed to create a response") == nil {
			_ = encoder.writeDone()
		}
		return
	}
	if err := start(); err != nil {
		return
	}
	completedAt := h.now()
	content := string(encoder.content)
	finalContext, err := carryContext.AbsorbAssistantText(content)
	if err != nil {
		if encoder.writeFailure("server_error", "The completed response context is invalid") == nil {
			_ = encoder.writeDone()
		}
		return
	}
	response := metadata.textResource(content, usage, completedAt.Unix(), chatModel.incomplete())
	if err := encoder.writeTextFinishedItems(chatModel.outputStatus()); err != nil {
		return
	}
	if err := h.storeFinishedResponse(metadata, response.Output, finalContext, completedAt); err != nil {
		code, message := responseStoreErrorDetails(err)
		if encoder.writeFailure(code, message) == nil {
			_ = encoder.writeDone()
		}
		return
	}
	if err := encoder.writeLifecycle("response."+response.Status, response); err != nil {
		if metadata.Store {
			_ = h.responseStore.Delete(metadata.ResponseID)
		}
		return
	}
	_ = encoder.writeDone()
}

func (h *Handler) streamToolResponse(
	w http.ResponseWriter,
	r *http.Request,
	domain chat.Request,
	model chat.Model,
	plan chat.ToolPlan,
	metadata responseGenerationMetadata,
	carryContext chat.ContextState,
) {
	encoder := newResponseSSEEncoder(w, http.NewResponseController(w), metadata)
	started := false
	err := model.StreamTools(r.Context(), domain, plan, func(chunk chat.ToolChunk) error {
		if !started {
			started = true
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			if err := encoder.writeLifecycleInitial(); err != nil {
				return err
			}
		}
		return encoder.writeToolChunk(plan.Calls[chunk.Index], chunk)
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		if !started {
			writeResponseModelError(w, err)
			return
		}
		if encoder.writeFailure("server_error", "The chat model failed to stream function arguments") == nil {
			_ = encoder.writeDone()
		}
		return
	}
	if r.Context().Err() != nil {
		return
	}
	finalContext, err := carryContext.AbsorbToolCalls(plan.Calls)
	if err != nil {
		if encoder.writeFailure("server_error", "The completed response context is invalid") == nil {
			_ = encoder.writeDone()
		}
		return
	}
	completedAt := h.now()
	response := metadata.completedToolResource(plan.Calls, plan.Usage, completedAt.Unix())
	if err := h.storeFinishedResponse(metadata, response.Output, finalContext, completedAt); err != nil {
		code, message := responseStoreErrorDetails(err)
		if encoder.writeFailure(code, message) == nil {
			_ = encoder.writeDone()
		}
		return
	}
	if err := encoder.writeCompletedResource(response); err != nil {
		if metadata.Store {
			_ = h.responseStore.Delete(metadata.ResponseID)
		}
		return
	}
	_ = encoder.writeDone()
}

func newResponseSSEEncoder(
	w http.ResponseWriter,
	controller *http.ResponseController,
	metadata responseGenerationMetadata,
) *responseSSEEncoder {
	return &responseSSEEncoder{
		sseWriter: newSSEWriter(w, controller),
		metadata:  metadata,
		content:   make([]byte, 0, initialSSEBufferCapacity),
	}
}

func (e *responseSSEEncoder) writeInitial() error {
	if err := e.writeLifecycleInitial(); err != nil {
		return err
	}
	if err := e.writeOutputItem("response.output_item.added", 0, e.metadata.outputItem("in_progress", "")); err != nil {
		return err
	}
	return e.writeContentPart("response.content_part.added", responseOutputContent{
		Type:        "output_text",
		Text:        "",
		Annotations: make([]any, 0),
	})
}

func (e *responseSSEEncoder) writeLifecycleInitial() error {
	resource := e.metadata.resource("in_progress")
	if err := e.writeLifecycle("response.created", resource); err != nil {
		return err
	}
	return e.writeLifecycle("response.in_progress", resource)
}

func (e *responseSSEEncoder) writeContent(delta []byte) error {
	e.content = append(e.content, delta...)
	event := responseOutputTextDeltaEvent{
		Type:           "response.output_text.delta",
		SequenceNumber: e.nextSequenceNumber(),
		ItemID:         e.metadata.MessageID,
		OutputIndex:    0,
		ContentIndex:   0,
		Delta:          string(delta),
	}
	return e.writeJSONEvent(event.Type, event)
}

func (e *responseSSEEncoder) writeTextFinishedItems(status string) error {
	content := string(e.content)
	done := responseOutputTextDoneEvent{
		Type:           "response.output_text.done",
		SequenceNumber: e.nextSequenceNumber(),
		ItemID:         e.metadata.MessageID,
		OutputIndex:    0,
		ContentIndex:   0,
		Text:           content,
	}
	if err := e.writeJSONEvent(done.Type, done); err != nil {
		return err
	}
	part := responseOutputContent{
		Type:        "output_text",
		Text:        content,
		Annotations: make([]any, 0),
	}
	if err := e.writeContentPart("response.content_part.done", part); err != nil {
		return err
	}
	return e.writeOutputItem("response.output_item.done", 0, e.metadata.outputItem(status, content))
}

func (e *responseSSEEncoder) writeToolChunk(call chat.ToolCall, chunk chat.ToolChunk) error {
	if chunk.First {
		if err := e.writeOutputItem("response.output_item.added", chunk.Index, responseToolOutputItem(call, "in_progress", "")); err != nil {
			return err
		}
	}
	delta := responseFunctionCallArgumentsDeltaEvent{Type: "response.function_call_arguments.delta", SequenceNumber: e.nextSequenceNumber(), ItemID: call.ID, OutputIndex: chunk.Index, Delta: chunk.Delta}
	if err := e.writeJSONEvent(delta.Type, delta); err != nil {
		return err
	}
	if chunk.Last {
		done := responseFunctionCallArgumentsDoneEvent{Type: "response.function_call_arguments.done", SequenceNumber: e.nextSequenceNumber(), ItemID: call.ID, OutputIndex: chunk.Index, Arguments: call.Arguments}
		if err := e.writeJSONEvent(done.Type, done); err != nil {
			return err
		}
		return e.writeOutputItem("response.output_item.done", chunk.Index, responseToolOutputItem(call, "completed", call.Arguments))
	}
	return nil
}

func (e *responseSSEEncoder) writeCompletedResource(response responseResource) error {
	return e.writeLifecycle("response.completed", response)
}

func (e *responseSSEEncoder) writeFailure(code, message string) error {
	errorEvent := responseStreamingErrorEvent{
		Type:           "error",
		SequenceNumber: e.nextSequenceNumber(),
		Error: responseStreamingErrorPayload{
			Type:    "server_error",
			Code:    &code,
			Message: message,
			Param:   nil,
		},
	}
	if err := e.writeJSONEvent(errorEvent.Type, errorEvent); err != nil {
		return err
	}
	resource := e.metadata.resource("failed")
	resource.Error = responseFailure{Code: code, Message: message}
	return e.writeLifecycle("response.failed", resource)
}

func (e *responseSSEEncoder) writeLifecycle(eventType string, resource responseResource) error {
	event := responseLifecycleEvent{
		Type:           eventType,
		SequenceNumber: e.nextSequenceNumber(),
		Response:       resource,
	}
	return e.writeJSONEvent(event.Type, event)
}

func (e *responseSSEEncoder) writeOutputItem(eventType string, outputIndex int, item responseOutputItem) error {
	event := responseOutputItemEvent{
		Type:           eventType,
		SequenceNumber: e.nextSequenceNumber(),
		OutputIndex:    outputIndex,
		Item:           item,
	}
	return e.writeJSONEvent(event.Type, event)
}

func (e *responseSSEEncoder) writeContentPart(eventType string, part responseOutputContent) error {
	event := responseContentPartEvent{
		Type:           eventType,
		SequenceNumber: e.nextSequenceNumber(),
		ItemID:         e.metadata.MessageID,
		OutputIndex:    0,
		ContentIndex:   0,
		Part:           part,
	}
	return e.writeJSONEvent(event.Type, event)
}

func (e *responseSSEEncoder) nextSequenceNumber() int {
	sequenceNumber := e.sequenceNumber
	e.sequenceNumber++
	return sequenceNumber
}
