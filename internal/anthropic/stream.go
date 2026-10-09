package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type streamEvent struct {
	Type         string           `json:"type"`
	Message      *messageResponse `json:"message,omitempty"`
	Index        *int             `json:"index,omitempty"`
	ContentBlock any              `json:"content_block,omitempty"`
	Delta        any              `json:"delta,omitempty"`
	Usage        *messageUsage    `json:"usage,omitempty"`
}

type textDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type inputDelta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

type messageDelta struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type eventWriter struct {
	w       http.ResponseWriter
	started bool
	failed  bool
}

func (writer *eventWriter) write(kind string, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if !writer.started {
		writer.w.Header().Set("Content-Type", "text/event-stream")
		writer.w.Header().Set("Cache-Control", "no-cache")
		writer.started = true
	}
	payload := append([]byte("event: "+kind+"\ndata: "), encoded...)
	payload = append(payload, '\n', '\n')
	n, err := writer.w.Write(payload)
	if err == nil && n != len(payload) {
		err = io.ErrShortWrite
	}
	if err == nil {
		err = http.NewResponseController(writer.w).Flush()
	}
	writer.failed = err != nil
	return err
}

func (writer *eventWriter) event(event streamEvent) error {
	return writer.write(event.Type, event)
}

func streamMessage(w http.ResponseWriter, r *http.Request, requestID string, response messageResponse, request chat.Request, model chat.Model, plan chat.ToolPlan) {
	writer := &eventWriter{w: w}
	inputTokens, err := request.ContextState.PromptTokens()
	if err != nil {
		writeModelError(w, requestID, err)
		return
	}
	response.Usage.InputTokens = inputTokens
	start := func() error {
		if writer.started {
			return nil
		}
		return writer.event(streamEvent{Type: "message_start", Message: &response})
	}
	usage, reason := plan.Usage, "tool_use"
	if plan.UseText {
		index := 0
		usage, err = model.Stream(r.Context(), request, func(chunk chat.Chunk) error {
			if !writer.started {
				if err := start(); err != nil {
					return err
				}
				if err := writer.event(streamEvent{Type: "content_block_start", Index: &index, ContentBlock: textBlock{Type: "text"}}); err != nil {
					return err
				}
			}
			return writer.event(streamEvent{Type: "content_block_delta", Index: &index, Delta: textDelta{Type: "text_delta", Text: string(chunk.Delta)}})
		})
		if err == nil {
			// The mock always emits text, but an alternate model may emit no chunks.
			if !writer.started {
				err = start()
				if err == nil {
					err = writer.event(streamEvent{Type: "content_block_start", Index: &index, ContentBlock: textBlock{Type: "text"}})
				}
			}
			if err == nil {
				err = writer.event(streamEvent{Type: "content_block_stop", Index: &index})
			}
		}
		reason = textStopReason(request, usage)
	} else {
		err = model.StreamTools(r.Context(), request, plan, func(chunk chat.ToolChunk) error {
			if chunk.Index < 0 || chunk.Index >= len(plan.Calls) {
				return fmt.Errorf("model emitted invalid tool index")
			}
			if err := start(); err != nil {
				return err
			}
			if chunk.First {
				call := plan.Calls[chunk.Index]
				block := toolUseBlock{Type: "tool_use", ID: toolID(call), Name: call.Name, Input: map[string]any{}}
				if err := writer.event(streamEvent{Type: "content_block_start", Index: &chunk.Index, ContentBlock: block}); err != nil {
					return err
				}
			}
			if err := writer.event(streamEvent{Type: "content_block_delta", Index: &chunk.Index, Delta: inputDelta{Type: "input_json_delta", PartialJSON: chunk.Delta}}); err != nil {
				return err
			}
			if chunk.Last {
				return writer.event(streamEvent{Type: "content_block_stop", Index: &chunk.Index})
			}
			return nil
		})
	}
	if err != nil {
		if !writer.started {
			writeModelError(w, requestID, err)
		} else if !writer.failed && !isCanceled(err) {
			_ = writer.write("error", apiError{Type: "error", Error: errorDetail{Type: "api_error", Message: "Model failed during streaming"}, RequestID: requestID})
		}
		return
	}
	if err := start(); err != nil {
		return
	}
	finalUsage := toUsage(usage)
	if err := writer.event(streamEvent{Type: "message_delta", Delta: messageDelta{StopReason: reason}, Usage: &finalUsage}); err != nil {
		return
	}
	_ = writer.event(streamEvent{Type: "message_stop"})
}
