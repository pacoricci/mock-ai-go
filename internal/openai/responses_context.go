package openai

import (
	"fmt"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

// Response contexts share the completed conversation prefix but serve different purposes.
// Item order contributes to the deterministic digest and must be preserved.
type responseContexts struct {
	// Planning includes current instructions and input. PlanTools hashes the tool
	// configuration separately, so it must not also appear in this context.
	planning chat.ContextState
	// Generation includes current tools, instructions, and input for text and usage.
	generation chat.ContextState
	// Carry includes current tools and input, but excludes request-only instructions.
	// Completed output is added before this context is stored for continuation.
	carry chat.ContextState
}

// Context preparation separates request validation failures from internal errors.
func prepareResponseContexts(
	previousState *responseState,
	instructions *string,
	contextItems []chat.ContextItem,
	toolConfig functionToolConfig,
) (responseContexts, *responseRequestError, error) {
	var err error
	baseContext := chat.NewContextState()
	if previousState != nil {
		baseContext = previousState.FinalContext
	}
	planningContext := baseContext
	if instructions != nil {
		planningContext, err = planningContext.Absorb(responseInstructionContextItem(*instructions))
		if err != nil {
			return responseContexts{}, newResponseRequestError(
				fmt.Sprintf("instructions cannot be added to the conversation context: %v", err),
				"instructions",
				"invalid_value",
			), nil
		}
	}
	planningContext, err = planningContext.AbsorbAll(contextItems)
	if err != nil {
		return responseContexts{}, newResponseRequestError(
			fmt.Sprintf("input cannot be added to the conversation context: %v", err),
			"input",
			"invalid_value",
		), nil
	}
	toolContextItems, err := functionToolContextItems(toolConfig)
	if err != nil {
		return responseContexts{}, nil, err
	}
	carryContext, err := baseContext.AbsorbAll(toolContextItems)
	if err != nil {
		return responseContexts{}, newResponseRequestError(
			fmt.Sprintf("tools cannot be added to the conversation context: %v", err),
			"tools",
			"invalid_value",
		), nil
	}
	carryContext, err = carryContext.AbsorbAll(contextItems)
	if err != nil {
		return responseContexts{}, newResponseRequestError(
			fmt.Sprintf("input cannot be added to the conversation context: %v", err),
			"input",
			"invalid_value",
		), nil
	}
	generationContext, err := baseContext.AbsorbAll(toolContextItems)
	if err != nil {
		return responseContexts{}, nil, err
	}
	if instructions != nil {
		generationContext, err = generationContext.Absorb(responseInstructionContextItem(*instructions))
		if err != nil {
			return responseContexts{}, newResponseRequestError(
				fmt.Sprintf("instructions cannot be added to the conversation context: %v", err),
				"instructions",
				"invalid_value",
			), nil
		}
	}
	generationContext, err = generationContext.AbsorbAll(contextItems)
	if err != nil {
		return responseContexts{}, nil, err
	}
	return responseContexts{planning: planningContext, generation: generationContext, carry: carryContext}, nil, nil
}

func responseInstructionContextItem(instructions string) chat.ContextItem {
	return chat.ContextItem{
		Type: chat.ContextItemMessage,
		Role: "developer",
		Content: []chat.ContextContent{{
			Type: "text",
			Text: instructions,
		}},
	}
}
