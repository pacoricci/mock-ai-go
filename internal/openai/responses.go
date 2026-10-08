package openai

import (
	"context"
	"errors"
	"net/http"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func (h *Handler) createResponse(w http.ResponseWriter, r *http.Request) {
	request, ok := decodeJSONRequest[responseRequest](w, r)
	if !ok {
		return
	}

	domainRequest, contextItems, toolConfig, requestError := toResponseDomainRequest(request)
	if requestError != nil {
		writeResponseRequestError(w, requestError)
		return
	}

	format, formatError := normalizeResponseFormat(request.Text)
	if formatError != nil {
		writeResponseRequestError(w, formatError)
		return
	}

	chatModel, found := h.resolveChatModel(w, request.Model)
	if !found {
		return
	}

	previousState, requestError, err := h.loadResponseState(request.PreviousResponseID, request.Model)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		writeResponseModelError(w, err)
		return
	}
	if requestError != nil {
		writeResponseRequestError(w, requestError)
		return
	}
	if requestError := validateResponseFunctionOutputs(previousState, contextItems); requestError != nil {
		writeResponseRequestError(w, requestError)
		return
	}

	contexts, requestError, err := prepareResponseContexts(previousState, request.Instructions, contextItems, toolConfig)
	if requestError != nil {
		writeResponseRequestError(w, requestError)
		return
	}
	if err != nil {
		writeResponseModelError(w, err)
		return
	}
	domainRequest.ContextState = &contexts.planning

	maxToolCalls := 0
	if toolConfig.MaxToolCalls != nil {
		maxToolCalls = *toolConfig.MaxToolCalls
	}
	toolPlan, err := chatModel.PlanTools(r.Context(), chat.ToolRequest{
		Request:           domainRequest,
		Tools:             toolConfig.DomainTools,
		Choice:            toolConfig.Choice,
		ParallelToolCalls: toolConfig.ParallelToolCalls,
		MaxToolCalls:      maxToolCalls,
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		writeResponseModelError(w, err)
		return
	}
	if !toolPlan.UseText {
		toolPlan.Usage, err = responseToolUsage(toolPlan.Usage, contexts.generation)
		if err != nil {
			writeResponseModelError(w, err)
			return
		}
	} else {
		domainRequest.ContextState = &contexts.generation
	}
	generation := textGeneration{Model: chatModel}
	if toolPlan.UseText {
		generation, requestError, err = prepareTextGeneration(r.Context(), chatModel, domainRequest, format)
		if requestError != nil {
			writeResponseRequestError(w, requestError)
			return
		}
		if err != nil {
			if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				writeResponseModelError(w, err)
			}
			return
		}
	}
	sequence := h.sequence.Add(1)
	responseID, err := h.newResponseID()
	if err != nil {
		writeResponseModelError(w, err)
		return
	}
	metadata := newResponseGenerationMetadata(h, request, toolConfig, responseID, sequence)
	metadata.TextFormat = format.format
	if request.Stream {
		if !toolPlan.UseText {
			domainRequest.StreamNonce = sequence
			h.streamToolResponse(w, r, domainRequest, chatModel, toolPlan, metadata, contexts.carry)
			return
		}
		domainRequest.StreamNonce = sequence
		h.streamResponse(w, r, domainRequest, generation, metadata, contexts.carry)
		return
	}
	if !toolPlan.UseText {
		h.completeToolResponse(w, toolPlan, metadata, contexts.carry)
		return
	}

	h.completeResponse(w, r, domainRequest, generation, metadata, contexts.carry)
}

func (h *Handler) completeResponse(
	w http.ResponseWriter,
	r *http.Request,
	domainRequest chat.Request,
	chatModel textGeneration,
	metadata responseGenerationMetadata,
	carryContext chat.ContextState,
) {
	completion, err := chatModel.Complete(r.Context(), domainRequest)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		writeResponseModelError(w, err)
		return
	}

	completedAt := h.now()
	response := metadata.textResource(completion.Content, completion.Usage, completedAt.Unix(), chatModel.incomplete())
	finalContext, err := carryContext.AbsorbAssistantText(completion.Content)
	if err != nil {
		writeResponseModelError(w, err)
		return
	}
	if err := h.storeFinishedResponse(metadata, response.Output, finalContext, completedAt); err != nil {
		writeResponseStoreError(w, err)
		return
	}
	writeJSON(w, response)
}

func (h *Handler) completeToolResponse(
	w http.ResponseWriter,
	plan chat.ToolPlan,
	metadata responseGenerationMetadata,
	carryContext chat.ContextState,
) {
	if plan.Usage.CompletionTokens < 0 {
		writeResponseModelError(w, errors.New("tool plan has a negative completion token count"))
		return
	}
	finalContext, err := carryContext.AbsorbToolCalls(plan.Calls)
	if err != nil {
		writeResponseModelError(w, err)
		return
	}
	completedAt := h.now()
	response := metadata.completedToolResource(plan.Calls, plan.Usage, completedAt.Unix())
	if err := h.storeFinishedResponse(metadata, response.Output, finalContext, completedAt); err != nil {
		writeResponseStoreError(w, err)
		return
	}
	writeJSON(w, response)
}
