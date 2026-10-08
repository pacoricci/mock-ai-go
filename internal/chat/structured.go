package chat

import (
	"context"
	"fmt"
	"unicode/utf8"

	"github.com/pacoricci/mock-ai-go/internal/jsonschema"
)

// StructuredRequest describes structured text generation. Schema is immutable and
// must remain available until planning finishes. A nil Schema selects JSON mode.
type StructuredRequest struct {
	Request
	Schema *jsonschema.Schema
	// Format supplies the normalized format bytes for deterministic seeding and usage.
	Format string
}

// StructuredPlan owns generated text, usage, and its terminal state. Incomplete
// means the token ceiling cut the document short; Content may then be invalid JSON.
type StructuredPlan struct {
	Content    string
	Usage      Usage
	Incomplete bool
}

// StructuredModel optionally extends Model with structured output generation.
// StreamStructured borrows an immutable plan until it returns, emits synchronously,
// honors cancellation and backpressure, and never changes its content or usage.
type StructuredModel interface {
	PlanStructured(context.Context, StructuredRequest) (StructuredPlan, error)
	StreamStructured(context.Context, Request, StructuredPlan, Emit) error
}

// PlanStructured generates JSON independently of the output budget, then takes a
// UTF-8-safe prefix within that budget. One simulated token represents four bytes.
func (m *MockModel) PlanStructured(ctx context.Context, request StructuredRequest) (StructuredPlan, error) {
	if err := ctx.Err(); err != nil {
		return StructuredPlan{}, err
	}
	budget, err := m.completionTokens(request.Request)
	if err != nil {
		return StructuredPlan{}, err
	}
	hash := m.requestSeed(request.Request, 0)
	hashString(&hash, "mock-ai-go/structured-output/v1")
	hashString(&hash, request.Format)
	random := newXorshift64Star(hash)
	var content []byte
	if request.Schema == nil {
		content = []byte(fmt.Sprintf(`{"message":"mock_%016x","value":%d}`, random.next(), random.next()%1000))
	} else {
		content, err = request.Schema.Generate(ctx, random.next)
		if err != nil {
			return StructuredPlan{}, err
		}
	}
	end := min(len(content), budget*bytesPerToken)
	for end < len(content) && end > 0 && !utf8.RuneStart(content[end]) {
		end--
	}
	text := string(content[:end])
	usage, err := requestUsage(request.Request, estimateTextTokens(text))
	if err != nil {
		return StructuredPlan{}, err
	}
	formatTokens := estimateTextTokens(request.Format)
	if formatTokens > int(^uint(0)>>1)-usage.TotalTokens {
		return StructuredPlan{}, ErrInvalidContextState
	}
	usage.PromptTokens += formatTokens
	usage.TotalTokens += formatTokens
	return StructuredPlan{Content: text, Usage: usage, Incomplete: end < len(content)}, nil
}

// StreamStructured emits a planned JSON prefix using the configured pacing.
func (m *MockModel) StreamStructured(ctx context.Context, request Request, plan StructuredPlan, emit Emit) error {
	if emit == nil {
		return ErrNilEmit
	}
	return m.StreamTools(ctx, request, ToolPlan{Calls: []ToolCall{{Arguments: plan.Content}}}, func(chunk ToolChunk) error {
		return emit(Chunk{Delta: []byte(chunk.Delta)})
	})
}
