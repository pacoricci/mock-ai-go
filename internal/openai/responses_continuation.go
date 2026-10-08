package openai

import (
	"fmt"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func (h *Handler) loadResponseState(
	previousResponseID *string,
	model string,
) (*responseState, *responseRequestError, error) {
	if previousResponseID == nil {
		return nil, nil, nil
	}
	if !isResponseID(*previousResponseID) {
		return nil, newResponseRequestError(
			"previous_response_id is malformed or unsupported",
			"previous_response_id",
			"invalid_value",
		), nil
	}
	state, found, err := h.responseStore.Get(*previousResponseID)
	if err != nil {
		return nil, nil, fmt.Errorf("load previous response state: %w", err)
	}
	if !found {
		return nil, newResponseRequestError(
			fmt.Sprintf("Previous response with id %q was not found", *previousResponseID),
			"previous_response_id",
			"previous_response_not_found",
		), nil
	}
	if state.Model != model {
		return nil, newResponseRequestError(
			"previous_response_id was created for a different model",
			"previous_response_id",
			"invalid_value",
		), nil
	}
	return state, nil, nil
}

func validateResponseFunctionOutputs(previous *responseState, items []chat.ContextItem) *responseRequestError {
	var previousCalls map[string]struct{}
	if previous != nil {
		previousCalls = previous.EmittedFunctionCalls
	}
	return validateFunctionOutputs(previousCalls, items, "input")
}

func (h *Handler) storeFinishedResponse(
	metadata responseGenerationMetadata,
	output []responseOutputItem,
	finalContext chat.ContextState,
	completedAt time.Time,
) error {
	if !metadata.Store {
		return nil
	}
	state := &responseState{
		ID:                   metadata.ResponseID,
		Model:                metadata.Model,
		EmittedFunctionCalls: emittedFunctionCallIDs(output),
		FinalContext:         finalContext,
		ExpiresAt:            completedAt.Add(h.responseTTL),
	}
	state.LogicalBytes = responseStateLogicalBytes(state)
	return h.responseStore.Put(state)
}

func emittedFunctionCallIDs(output []responseOutputItem) map[string]struct{} {
	callIDs := make(map[string]struct{})
	for _, item := range output {
		if item.Type == "function_call" && item.CallID != nil {
			callIDs[*item.CallID] = struct{}{}
		}
	}
	return callIDs
}

// responseStateFixedBytes covers the context digest, context units, and expiry.
const responseStateFixedBytes = 24

func responseStateLogicalBytes(state *responseState) int64 {
	size := int64(responseStateFixedBytes + len(state.ID) + len(state.Model))
	for callID := range state.EmittedFunctionCalls {
		size += int64(len(callID))
	}
	return size
}
