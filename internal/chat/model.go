// Package chat defines the chat model capability and its domain types.
package chat

import (
	"context"

	"github.com/pacoricci/mock-ai-go/internal/model"
)

// Message is one message in a chat conversation.
type Message struct {
	Role    string
	Content string
}

// Request contains the input for a chat completion.
type Request struct {
	Messages            []Message
	MaxCompletionTokens int
	Seed                *int64
	// ContextState supplies aggregate conversation state. When it is
	// non-nil, generation and usage use it instead of re-reading Messages.
	ContextState *ContextState
	// StreamNonce decorrelates timing between otherwise identical streams. It
	// never participates in completion content generation.
	StreamNonce uint64
}

// ToolChoice controls whether the model may emit function calls.
type ToolChoice string

const (
	// ToolChoiceNone always selects text generation.
	ToolChoiceNone ToolChoice = "none"
	// ToolChoiceAuto deterministically selects text or function calls.
	ToolChoiceAuto ToolChoice = "auto"
	// ToolChoiceRequired always selects one or more function calls.
	ToolChoiceRequired ToolChoice = "required"
)

// FunctionTool describes one function available to the model.
type FunctionTool struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// ToolRequest contains the normalized inputs used to plan function calls.
type ToolRequest struct {
	Request
	Tools             []FunctionTool
	Choice            ToolChoice
	ParallelToolCalls bool
	// MaxToolCalls is zero when no explicit limit was supplied.
	MaxToolCalls int
}

// ToolCall is one deterministic request to invoke a function.
type ToolCall struct {
	ID        string
	CallID    string
	Name      string
	Arguments string
}

// ToolPlan selects text generation or a completed set of function calls.
type ToolPlan struct {
	UseText bool
	Calls   []ToolCall
	Usage   Usage
}

// Usage contains simulated token accounting for a completion.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// Response contains a completed chat response.
type Response struct {
	Content string
	Usage   Usage
}

// Chunk contains one incremental part of a streamed response.
type Chunk struct {
	// Delta is valid only until the Emit callback returns. Consumers that retain
	// it must make a copy.
	Delta []byte
}

// Emit consumes a streamed chunk synchronously. Returning an error stops the
// stream and propagates backpressure to the model.
type Emit func(Chunk) error

// Model plans function calls and generates complete or streamed chat responses.
type Model interface {
	// StreamTools emits an already planned set of calls in order, without modifying
	// the plan. The caller owns the plan and must not mutate it until this returns.
	StreamTools(context.Context, Request, ToolPlan, ToolEmit) error
	Info() model.Info
	PlanTools(context.Context, ToolRequest) (ToolPlan, error)
	Complete(context.Context, Request) (Response, error)
	Stream(context.Context, Request, Emit) (Usage, error)
}

// ToolChunk is an incremental argument fragment for one call in a ToolPlan.
// Index identifies the call; First and Last delimit its argument stream.
// Delta is immutable and may be retained by the consumer.
type ToolChunk struct {
	Index int
	Delta string
	First bool
	Last  bool
}

// ToolEmit consumes a tool fragment synchronously. Returning an error stops
// generation immediately and propagates backpressure to the model.
type ToolEmit func(ToolChunk) error
