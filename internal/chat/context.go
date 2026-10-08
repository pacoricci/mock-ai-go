package chat

import (
	"errors"
	"fmt"
	"math"
)

const (
	contextDigestDomain = "mock-ai-go/context/v1"
	contextItemStart    = 0x01
	contextItemEnd      = 0x02
)

var (
	// ErrInvalidContextState indicates context accounting that cannot be represented safely.
	ErrInvalidContextState = errors.New("chat context state is invalid")
	// ErrInvalidContextItem indicates a context item without a supported type.
	ErrInvalidContextItem = errors.New("chat context item is invalid")
)

// ContextItemType identifies one semantic item in a conversation.
type ContextItemType string

const (
	// ContextItemMessage is a user, developer, system, or assistant message.
	ContextItemMessage ContextItemType = "message"
	// ContextItemFunctionCall is a function call emitted by the model.
	ContextItemFunctionCall ContextItemType = "function_call"
	// ContextItemFunctionCallOutput is the result returned for a function call.
	ContextItemFunctionCallOutput ContextItemType = "function_call_output"
	// ContextItemReasoning is reasoning state supplied as conversation context.
	ContextItemReasoning ContextItemType = "reasoning"
	// ContextItemFunctionTool is one function definition available for generation.
	ContextItemFunctionTool ContextItemType = "function_tool"
	// ContextItemToolConfiguration is the selection policy applied to function tools.
	ContextItemToolConfiguration ContextItemType = "tool_configuration"
)

// ContextContent is one normalized content part belonging to a context item.
type ContextContent struct {
	Type string
	Text string
}

// ContextItem is one normalized item that contributes to conversation state.
// Protocol-only identifiers, timestamps, and transport metadata must not be
// included in these fields.
type ContextItem struct {
	Type    ContextItemType
	Role    string
	CallID  string
	Name    string
	Content []ContextContent
}

// ContextState is a fixed-size digest and token estimate for an ordered
// conversation prefix. It contains no transcript or item content.
type ContextState struct {
	digest uint64
	units  uint64
}

// NewContextState returns the empty, versioned conversation state.
func NewContextState() ContextState {
	digest := uint64(fnvOffsetBasis)
	hashString(&digest, contextDigestDomain)
	return ContextState{digest: digest}
}

// Digest returns the incremental digest of all absorbed context items.
func (s ContextState) Digest() uint64 {
	return s.digest
}

// Units returns the cumulative item units before the fixed reply overhead.
func (s ContextState) Units() uint64 {
	return s.units
}

// PromptTokens returns the simulated input-token count represented by the state.
func (s ContextState) PromptTokens() (int, error) {
	if s.units > uint64(math.MaxInt-replyTokenOverhead) {
		return 0, fmt.Errorf("%w: prompt token count overflows int", ErrInvalidContextState)
	}
	return int(s.units) + replyTokenOverhead, nil
}

// Absorb folds one context item into a new state without retaining the item.
func (s ContextState) Absorb(item ContextItem) (ContextState, error) {
	if !isSupportedContextItemType(item.Type) {
		return ContextState{}, fmt.Errorf("%w: unsupported type %q", ErrInvalidContextItem, item.Type)
	}

	additionalUnits, err := contextItemUnits(item)
	if err != nil {
		return ContextState{}, err
	}
	if additionalUnits > math.MaxUint64-s.units {
		return ContextState{}, fmt.Errorf("%w: context units overflow uint64", ErrInvalidContextState)
	}

	digest := s.digest
	hashByte(&digest, contextItemStart)
	hashString(&digest, string(item.Type))
	hashString(&digest, item.Role)
	hashString(&digest, item.CallID)
	hashString(&digest, item.Name)
	hashUint64(&digest, uint64(len(item.Content)))
	for _, part := range item.Content {
		hashString(&digest, part.Type)
		hashString(&digest, part.Text)
	}
	hashByte(&digest, contextItemEnd)

	state := ContextState{digest: digest, units: s.units + additionalUnits}
	if _, err := state.PromptTokens(); err != nil {
		return ContextState{}, err
	}
	return state, nil
}

// AbsorbAll folds an ordered list of context items into a new state.
func (s ContextState) AbsorbAll(items []ContextItem) (ContextState, error) {
	var err error
	for _, item := range items {
		s, err = s.Absorb(item)
		if err != nil {
			return ContextState{}, err
		}
	}
	return s, nil
}

// AbsorbAssistantText folds one completed assistant text message into the state.
func (s ContextState) AbsorbAssistantText(content string) (ContextState, error) {
	return s.Absorb(ContextItem{
		Type: ContextItemMessage,
		Role: "assistant",
		Content: []ContextContent{{
			Type: "text",
			Text: content,
		}},
	})
}

// AbsorbToolCalls folds completed model function calls into the state in output order.
func (s ContextState) AbsorbToolCalls(calls []ToolCall) (ContextState, error) {
	items := make([]ContextItem, 0, len(calls))
	for _, call := range calls {
		items = append(items, ContextItem{
			Type:   ContextItemFunctionCall,
			CallID: call.CallID,
			Name:   call.Name,
			Content: []ContextContent{{
				Type: "arguments",
				Text: call.Arguments,
			}},
		})
	}
	return s.AbsorbAll(items)
}

func isSupportedContextItemType(itemType ContextItemType) bool {
	switch itemType {
	case ContextItemMessage,
		ContextItemFunctionCall,
		ContextItemFunctionCallOutput,
		ContextItemReasoning,
		ContextItemFunctionTool,
		ContextItemToolConfiguration:
		return true
	default:
		return false
	}
}

func contextItemUnits(item ContextItem) (uint64, error) {
	contentBytes := uint64(len(item.Name))
	for _, part := range item.Content {
		if uint64(len(part.Text)) > math.MaxUint64-contentBytes {
			return 0, fmt.Errorf("%w: context content length overflows uint64", ErrInvalidContextState)
		}
		contentBytes += uint64(len(part.Text))
	}
	contentTokens := uint64(0)
	if contentBytes > 0 {
		contentTokens = (contentBytes + bytesPerToken - 1) / bytesPerToken
	}
	return messageTokenOverhead + contentTokens, nil
}
