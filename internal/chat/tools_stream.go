package chat

import (
	"context"
	"fmt"
	"unicode/utf8"
)

// StreamTools emits UTF-8-safe fragments of planned function arguments using
// the model's initial delay, inter-fragment delay, and jitter. One configured
// token corresponds to approximately four argument bytes. It never replans calls
// or starts a goroutine; content and usage remain those of the supplied plan.
func (m *MockModel) StreamTools(ctx context.Context, request Request, plan ToolPlan, emit ToolEmit) error {
	if emit == nil {
		return ErrNilEmit
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(plan.Calls) == 0 {
		return nil
	}
	completionTokens, err := m.completionTokens(request)
	if err != nil {
		return err
	}
	random := newXorshift64Star(m.requestSeed(request, completionTokens) ^ temporalSeedSalt ^ mixSeed(request.StreamNonce))
	var timer reusableTimer
	defer timer.stop()
	if err := timer.wait(ctx, m.streamInitialDelay); err != nil {
		return err
	}
	first := true
	// Bound multiplication even when a caller supplies an unusually large chunk size.
	chunkBytes := min(m.streamTokensPerChunk, absoluteMaxCompletionTokens) * bytesPerToken
	for index, call := range plan.Calls {
		for offset := 0; ; {
			if !first {
				if err := timer.wait(ctx, jitteredStreamDelay(m.streamDelay, m.streamDelayJitter, &random)); err != nil {
					return err
				}
			}
			first = false
			if err := ctx.Err(); err != nil {
				return err
			}
			end := min(offset+chunkBytes, len(call.Arguments))
			for end < len(call.Arguments) && !utf8.RuneStart(call.Arguments[end]) {
				end++
			}
			last := end == len(call.Arguments)
			if err := emit(ToolChunk{Index: index, Delta: call.Arguments[offset:end], First: offset == 0, Last: last}); err != nil {
				return fmt.Errorf("emit tool chunk: %w", err)
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if last {
				break
			}
			offset = end
		}
	}
	return nil
}
